"""Lower an Event Tensor megakernel graph to a single persistent Triton kernel.

Sibling of :mod:`compgen.ir.tile.lower_triton` and
:mod:`compgen.ir.tile.lower_exo`.  Consumes an ``event.graph`` op that has
already been annotated by
:mod:`compgen.ir.payload.passes.megakernel_static_schedule` (Algorithm 1 of
the Event Tensor Compiler paper) and produces a single ``@triton.jit``
(or ``@triton.autotune`` when :attr:`MegakernelLoweringSpec.tune_config` is
non-empty) function whose grid equals the target's SM count.

Code-generation strategy:

    * Allocate one ``i32``/``i64`` tensor per Event Tensor in the graph,
      zero-initialised at host launch time and seeded with ``wait_count``.
    * Embed the per-SM task queue as a flat ``tl.constexpr`` table.
    * Per-SM body is a ``while task_idx < my_queue_len`` loop that fetches
      ``(task_id, task_type)`` from the table, dispatches into a device-
      function-specific branch, and advances.
    * Each branch invokes a real per-device-function ``@triton.jit`` body
      supplied by the caller via :class:`DeviceFunctionSpec`.  Bodies
      receive the task id, every data pointer, every event pointer, and
      every constexpr arg declared by the caller.
    * ``event.notify`` calls inside a body lower to
      ``tl.atomic_add(E_ptr + linear_idx, -k)``.
    * ``event.wait`` calls inside a body lower to a spin-poll
      ``while tl.atomic_or(E_ptr + linear_idx, 0) > 0: pass``.

When :paramref:`device_functions` is omitted the emitter falls back to
empty-body placeholders (used by structural tests); a real workload must
supply bodies for every device function referenced by the graph.

 will add the dynamic push/pop scheduler in a sibling emitter.
"""

from __future__ import annotations

import itertools
import json
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import IntegerAttr, StringAttr

from compgen.ir.event.attrs import EventTensorTypeAttr
from compgen.ir.event.ops import CallDeviceOp, EventTensorOp, GraphOp

_SCHEDULE_ATTR = "compgen.static_schedule"


@dataclass(frozen=True)
class DeviceFunctionSpec:
    """A real ``@triton.jit`` body for one device function in the graph.

    Attributes:
        name:        The device-function symbol (matches
                     ``CallDeviceOp.device_func``).
        body_source: Indented Triton source for the function body.  May
                     reference any data pointer in
                     :attr:`MegakernelLoweringSpec.data_pointers`, any
                     event-tensor pointer declared on the graph, any
                     constexpr arg in
                     :attr:`MegakernelLoweringSpec.constexpr_args`, and
                     ``task_id`` (the int task coordinate).  Each line
                     must already be indented relative to the function
                     body (4 spaces) -- the emitter does NOT re-indent.
    """

    name: str
    body_source: str


@dataclass(frozen=True)
class MegakernelLoweringSpec:
    """Caller-supplied wiring around the megakernel emitter.

    Attributes:
        data_pointers:    Names of pointer args (e.g. ``"A_ptr"``).
                          Passed positionally to every device function
                          body and to the megakernel itself.
        constexpr_args:   Names of ``tl.constexpr`` args (e.g. ``"M"``,
                          ``"BLOCK_M"``).
        device_functions: One :class:`DeviceFunctionSpec` per device
                          function referenced by the graph.  When empty
                          the emitter falls back to ``pass``-bodied
                          stubs.
        num_warps:        Triton ``num_warps`` for the launch.
        num_stages:       Triton ``num_stages`` for the launch.
    """

    data_pointers: tuple[str, ...] = ()
    constexpr_args: tuple[str, ...] = ()
    device_functions: tuple[DeviceFunctionSpec, ...] = ()
    num_warps: int = 4
    num_stages: int = 3
    tune_config: dict[str, tuple[int, ...]] = field(default_factory=dict)
    """Autotuning sweep axes.  Keys in :attr:`constexpr_args` become
    ``tl.constexpr`` sweeps; ``"num_warps"`` / ``"num_stages"`` become
    Triton launch-param sweeps.  Empty dict (default) → plain
    ``@triton.jit``, no autotuning overhead.

    Example::

        tune_config={"BLOCK_M": (16, 32, 64), "num_warps": (2, 4, 8)}
    """


@dataclass(frozen=True)
class MegakernelLoweringResult:
    """Output of lowering an event.graph to a persistent Triton kernel.

    Attributes:
        kernel_name:    name of the emitted ``@triton.jit`` function.
        kernel_source:  full Python source (including the @triton.jit decorator).
        launch_config:  ``{"grid": int, "num_warps": int, "num_stages": int}``
                        consumable by the host-side launcher.
        event_layout:   one entry per Event Tensor describing its size +
                        dtype + initial wait count, used by the host to
                        allocate and seed the global int tensors.
        task_queue:     per-SM task list (``sm_idx -> [(task_id, kind), ...]``)
                        baked into the kernel as a constexpr.
        device_function_table: ``{kind_int: device_func_name}`` -- the
                        order callers must use when filling per-task
                        ``task_kind`` entries in ``QUEUE_PTR``.
        diagnostics:    non-fatal warnings produced during lowering.
    """

    kernel_name: str
    kernel_source: str
    launch_config: dict[str, Any] = field(default_factory=dict)
    event_layout: list[dict[str, Any]] = field(default_factory=list)
    task_queue: dict[int, list[tuple[str, int]]] = field(default_factory=dict)
    device_function_table: dict[int, str] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)


def _event_layout(graph: GraphOp) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    for op in graph.body.ops:
        if not isinstance(op, EventTensorOp):
            continue
        et: EventTensorTypeAttr = op.event_type
        shape = [d.value.data for d in et.shape.data if isinstance(d, IntegerAttr)]
        size = 1
        for d in shape:
            size *= max(d, 1)
        layout.append(
            {
                "name": op.sym_name.data,
                "shape": shape,
                "size": size,
                "wait_count": op.wait_count.value.data,
                "scope": et.scope.data,
                "counter_dtype": et.counter_dtype.data,
            },
        )
    return layout


def _kernel_name(graph: GraphOp) -> str:
    return f"megakernel_{graph.sym_name.data}"


def _emit_task_table(
    per_sm_order: dict[str, list[str]],
    task_kind_map: dict[str, int],
) -> tuple[str, dict[int, list[tuple[str, int]]]]:
    """Emit a Python-source table + return a structured copy.

    Each per-SM list becomes an entry of the form::

        ((task_id_int, kind_int), ...)

    where ``task_id_int`` is the index into the *flat* task list (used by
    each device branch to pick its task coordinate).
    """
    lines: list[str] = ["TASK_TABLE = ["]
    structured: dict[int, list[tuple[str, int]]] = {}
    for sm_str, queue in sorted(per_sm_order.items(), key=lambda kv: int(kv[0])):
        sm_idx = int(sm_str)
        sm_entries: list[tuple[str, int]] = []
        encoded_entries: list[str] = []
        for tid in queue:
            kind = task_kind_map.get(tid, 0)
            sm_entries.append((tid, kind))
            encoded_entries.append(f'("{tid}", {kind})')
        structured[sm_idx] = sm_entries
        lines.append(f"    [{', '.join(encoded_entries)}],   # SM {sm_idx}")
    lines.append("]")
    return "\n".join(lines), structured


def _signature_args(
    data_pointers: Sequence[str],
    event_names: Sequence[str],
    constexpr_args: Sequence[str],
) -> tuple[str, str]:
    """Build the function signature + the call-site argument list.

    Returns ``(decl, call)`` where ``decl`` is suitable for placement
    inside ``def f(<decl>):`` and ``call`` is suitable for ``f(<call>)``.
    """
    decl_parts: list[str] = []
    call_parts: list[str] = []
    for ptr in data_pointers:
        decl_parts.append(ptr)
        call_parts.append(ptr)
    for ev in event_names:
        decl_parts.append(f"{ev}_ptr")
        call_parts.append(f"{ev}_ptr")
    for ce in constexpr_args:
        decl_parts.append(f"{ce}: tl.constexpr")
        call_parts.append(ce)
    return ", ".join(decl_parts), ", ".join(call_parts)


def _device_function_table(
    funcs: Sequence[str],
    spec: MegakernelLoweringSpec,
) -> dict[str, DeviceFunctionSpec]:
    by_name = {df.name: df for df in spec.device_functions}
    out: dict[str, DeviceFunctionSpec] = {}
    for fn in funcs:
        if fn in by_name:
            out[fn] = by_name[fn]
        else:
            out[fn] = DeviceFunctionSpec(
                name=fn,
                body_source="    pass  # stub: no DeviceFunctionSpec supplied",
            )
    return out


def _emit_dispatch_branches(
    funcs: Sequence[str],
    func_to_kind: dict[str, int],
    call_args: str,
) -> str:
    branches: list[str] = []
    for k, fn in enumerate(funcs):
        kind = func_to_kind[fn]
        keyword = "elif" if k else "if"
        branches.append(f"        {keyword} task_kind == {kind}:")
        branches.append(f"            _run_{fn}(task_id, {call_args})")
    if not branches:
        branches = ["        pass  # no tasks"]
    return "\n".join(branches)


def _build_autotune_configs(spec: MegakernelLoweringSpec) -> str:
    """Build the source code for a ``triton.Config`` list from ``spec.tune_config``.

    Returns source for::

        [
            triton.Config({'BLOCK_M': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
            ...
        ]
    """
    keys = list(spec.tune_config.keys())
    value_lists = [spec.tune_config[k] for k in keys]
    config_lines: list[str] = []
    for combo in itertools.product(*value_lists):
        kwargs: dict[str, int] = {}
        nw = spec.num_warps
        ns = spec.num_stages
        for k, v in zip(keys, combo):
            if k == "num_warps":
                nw = v
            elif k == "num_stages":
                ns = v
            else:
                kwargs[k] = v
        kwargs_parts = ", ".join(f"{k!r}: {v}" for k, v in kwargs.items())
        config_lines.append(
            f"        triton.Config({{{kwargs_parts}}}, num_warps={nw}, num_stages={ns}),"
        )
    return "[\n" + "\n".join(config_lines) + "\n    ]"


def _regroup_per_sm_order(
    graph: GraphOp,
    per_sm_order: dict[str, list[str]],
    sm_count: int,
) -> dict[str, list[str]]:
    """Post-process schedule: move tasks so every producer & its consumer
    share a SM, making all events local.

    Uses the IR's ``CallDeviceOp`` out_edges / in_edges to determine
    producer-consumer relationships.  For the row-sum pattern this means
    all ``partial_sum`` tasks for a row-block land on the same SM as
    that row's ``final_sum``.
    """
    # Extract producer/consumer event indices from IR
    # Map: event_idx → set of (task_id_prefix, task_i)
    ev_producers: dict[int, list[tuple[str, int]]] = {}
    ev_consumers: dict[int, list[tuple[str, int]]] = {}
    for op in graph.body.ops:
        if not isinstance(op, CallDeviceOp):
            continue
        func = op.device_func.root_reference.data
        if op.out_edges is not None:
            for edge in op.out_edges:
                for c in edge.indices.data:  # type: ignore[union-attr]
                    ev_idx = int(c.data)
                    ev_producers.setdefault(ev_idx, []).append((func, ev_idx))
        if op.in_edges is not None:
            task_count = int(op.task_shape.data[0].value.data)  # type: ignore[union-attr]
            n_events = sum(1 for _ in op.in_edges)
            events_per_task = n_events // task_count if task_count else 1
            for edge in op.in_edges:
                for c in edge.indices.data:  # type: ignore[union-attr]
                    ev_idx = int(c.data)
                    consumer_i = ev_idx // events_per_task if events_per_task else 0
                    ev_consumers.setdefault(ev_idx, []).append((func, consumer_i))

    # Build task_id → sm_id lookup from original schedule
    task_to_sm: dict[str, int] = {}
    for sm_str, tasks in per_sm_order.items():
        for tid in tasks:
            task_to_sm[tid] = int(sm_str)

    if not ev_producers or not ev_consumers:
        return per_sm_order

    # For each event, if producer and consumer are on different SMs,
    # move the producer to the consumer's SM.
    new_per_sm: dict[int, list[str]] = {s: list(tasks) for s_str, tasks in per_sm_order.items() if (s := int(s_str)) >= 0}
    # Flatten for rebuilding
    for ev_idx, producers in ev_producers.items():
        consumers = ev_consumers.get(ev_idx, [])
        for p_func, p_i in producers:
            p_tid = f"{p_func}:{p_i}"
            for c_func, c_i in consumers:
                c_tid = f"{c_func}:{c_i}"
                if p_tid not in task_to_sm or c_tid not in task_to_sm:
                    continue
                p_sm = task_to_sm[p_tid]
                c_sm = task_to_sm[c_tid]
                if p_sm != c_sm:
                    # Move producer to consumer's SM — insert BEFORE consumer
                    # so the producer executes first (dependency order).
                    for sm, tasks in list(new_per_sm.items()):
                        if p_tid in tasks:
                            tasks.remove(p_tid)
                    c_queue = new_per_sm.setdefault(c_sm, [])
                    try:
                        c_pos = c_queue.index(c_tid)
                        c_queue.insert(c_pos, p_tid)
                    except ValueError:
                        c_queue.append(p_tid)
                    task_to_sm[p_tid] = c_sm  # update mapping

    # Convert back to str-keyed dict
    result: dict[str, list[str]] = {str(s): tasks for s, tasks in new_per_sm.items()}
    for s in range(sm_count):
        result.setdefault(str(s), [])
    return result


def _build_event_locality(
    graph: GraphOp,
    per_sm_order: dict[str, list[str]],
) -> tuple[int, str, str]:
    """Analyse which events are local (producer & consumer on same SM).

    Returns ``(n_events, is_local_tuple_src, n_events_src)``.
    """
    task_to_sm: dict[str, int] = {}
    for sm_str, tasks in per_sm_order.items():
        sm = int(sm_str)
        for tid in tasks:
            task_to_sm[tid] = sm

    n_events = 0
    for op in graph.body.ops:
        if isinstance(op, EventTensorOp):
            et: EventTensorTypeAttr = op.event_type
            shape = [int(d.value.data) for d in et.shape.data if isinstance(d, IntegerAttr)]
            n_events = 1
            for d in shape:
                n_events *= max(d, 1)
            break

    if n_events == 0:
        return 0, "()", "0"

    producers: dict[str, int] = {}   # task_id → ev_idx
    consumers: dict[str, list[int]] = {}  # task_id → [ev_indices]

    for op in graph.body.ops:
        if not isinstance(op, CallDeviceOp):
            continue
        func = op.device_func.root_reference.data
        task_count = int(op.task_shape.data[0].value.data)  # type: ignore[union-attr]

        if op.out_edges is not None:
            for edge in op.out_edges:
                for c in edge.indices.data:  # type: ignore[union-attr]
                    ev_idx = int(c.data)
                    producers[f"{func}:{ev_idx}"] = ev_idx

        if op.in_edges is not None:
            events_per_task = n_events // task_count if task_count else n_events
            for task_i in range(task_count):
                start = task_i * events_per_task
                consumers[f"{func}:{task_i}"] = list(range(start, start + events_per_task))

    is_local_parts: list[str] = []
    for ev_idx in range(n_events):
        sms: set[int] = set()
        for tid, idx in producers.items():
            if idx == ev_idx and tid in task_to_sm:
                sms.add(task_to_sm[tid])
        for tid, ev_list in consumers.items():
            if ev_idx in ev_list and tid in task_to_sm:
                sms.add(task_to_sm[tid])
        is_local_parts.append("True" if len(sms) <= 1 else "False")

    return n_events, "(" + ", ".join(is_local_parts) + ",)", str(n_events)


def lower_megakernel(
    graph: GraphOp,
    spec: MegakernelLoweringSpec | None = None,
) -> MegakernelLoweringResult:
    """Lower an annotated ``event.graph`` to persistent-Triton source.

    Raises ``ValueError`` if the graph has not been annotated with
    ``compgen.static_schedule``; callers must run
    :class:`StaticMegakernelSchedule` first.
    """
    if _SCHEDULE_ATTR not in graph.attributes:
        raise ValueError(
            f"event.graph {graph.sym_name.data!r} is missing the "
            f"{_SCHEDULE_ATTR!r} attribute; run StaticMegakernelSchedule first"
        )
    payload_attr = graph.attributes[_SCHEDULE_ATTR]
    if not isinstance(payload_attr, StringAttr):
        raise ValueError(f"{_SCHEDULE_ATTR} must be a StringAttr, got {type(payload_attr).__name__}")
    schedule = json.loads(payload_attr.data)
    if schedule.get("status") != "ok":
        return MegakernelLoweringResult(
            kernel_name=_kernel_name(graph),
            kernel_source="",
            diagnostics=[f"static schedule rejected: {schedule.get('errors', [])}"],
        )

    if spec is None:
        spec = MegakernelLoweringSpec()

    events = _event_layout(graph)
    event_names = [e["name"] for e in events]
    sm_count = int(schedule["sm_count"])
    per_sm_order = {str(k): list(v) for k, v in schedule["per_sm_order"].items()}
    assignment = schedule["assignment"]

    # ── Regroup: co-locate producer & consumer tasks on same SM ──
    per_sm_order = _regroup_per_sm_order(graph, per_sm_order, sm_count)

    funcs = sorted({tid.split(":")[0] for tid in assignment})
    func_to_kind = {fn: i for i, fn in enumerate(funcs)}
    task_kind_map: dict[str, int] = {tid: func_to_kind[tid.split(":")[0]] for tid in assignment}

    task_table_src, structured_queue = _emit_task_table(per_sm_order, task_kind_map)
    dispatch_decl, dispatch_call = _signature_args(spec.data_pointers, event_names, spec.constexpr_args)
    body_table = _device_function_table(funcs, spec)
    dispatch_branches = _emit_dispatch_branches(funcs, func_to_kind, dispatch_call)
    kernel_name = _kernel_name(graph)

    # Analyse event locality: if producer & consumer are on the same SM
    # the event is "local" and atomics can be skipped.
    n_events, is_local_src, n_events_src = _build_event_locality(graph, per_sm_order)

    lines: list[str] = []
    lines.append("import triton")
    lines.append("import triton.language as tl")
    lines.append("")
    lines.append("# Per-SM task table baked into the megakernel at compile time.")
    lines.append(task_table_src)
    lines.append("")
    lines.append("# --- event locality table ---")
    lines.append(f"N_EVENTS = tl.constexpr({n_events_src})")
    lines.append(f"EVENT_IS_LOCAL = tl.constexpr({is_local_src})")
    lines.append("")
    lines.append("# --- atomic notify / wait helpers (locality-aware) ---")
    lines.append("@triton.jit")
    lines.append("def _event_notify(E_ptr, ev_idx, sm_id):")
    lines.append("    for k in tl.static_range(0, N_EVENTS):")
    lines.append("        if ev_idx == k:")
    lines.append("            if EVENT_IS_LOCAL[k]:")
    lines.append("                tl.store(E_ptr + k, 0)  # same SM, non-atomic")
    lines.append("            else:")
    lines.append("                tl.atomic_add(E_ptr + k, -1)  # cross-SM")
    lines.append("")
    lines.append("@triton.jit")
    lines.append("def _event_wait(E_ptr, ev_idx, sm_id):")
    lines.append("    for k in tl.static_range(0, N_EVENTS):")
    lines.append("        if ev_idx == k and not EVENT_IS_LOCAL[k]:")
    lines.append("            counter = tl.atomic_or(E_ptr + k, 0)")
    lines.append("            while counter > 0:")
    lines.append("                counter = tl.atomic_or(E_ptr + k, 0)")
    lines.append("")
    lines.append("# --- per-device-function bodies ---")
    for fn in funcs:
        body = body_table[fn]
        lines.append("@triton.jit")
        lines.append(f"def _run_{fn}(task_id, {dispatch_decl}):")
        body_text = textwrap.dedent(body.body_source).strip("\n")
        if not body_text.strip():
            body_text = "pass"
        # Indent every line of the body by 4 spaces (function-body indent).
        indented = textwrap.indent(body_text, "    ")
        lines.append(indented)
        lines.append("")

    # Megakernel signature: (data_ptrs, event_ptrs, QUEUE, QUEUE_LEN,
    # then all constexprs including user constexprs + SM_COUNT, MAX_QLEN).
    mk_decl_parts: list[str] = []
    for ptr in spec.data_pointers:
        mk_decl_parts.append(ptr)
    for ev in event_names:
        mk_decl_parts.append(f"{ev}_ptr")
    mk_decl_parts.append("QUEUE_PTR")
    mk_decl_parts.append("QUEUE_LEN_PTR")
    for ce in spec.constexpr_args:
        mk_decl_parts.append(f"{ce}: tl.constexpr")
    mk_decl_parts.append("SM_COUNT: tl.constexpr")
    mk_decl_parts.append("MAX_QLEN: tl.constexpr")

    lines.append("# --- persistent megakernel: grid = SM_COUNT ---")
    if spec.tune_config:
        configs_src = _build_autotune_configs(spec)
        key_args = [k for k in spec.tune_config if k not in ("num_warps", "num_stages")]
        if not key_args:
            key_args = ["SM_COUNT"]  # fallback: always present in megakernel signature
        lines.append("@triton.autotune(")
        lines.append(f"    configs={configs_src},")
        lines.append(f"    key=[{', '.join(repr(k) for k in key_args)}],")
        lines.append("    warmup=25, rep=100,")
        lines.append(")")
    lines.append("@triton.jit")
    lines.append(f"def {kernel_name}(")
    for part in mk_decl_parts:
        lines.append(f"    {part},")
    lines.append("):")
    lines.append('    """Persistent megakernel emitted by ETC Algorithm 1.')
    lines.append("")
    lines.append("    grid = (SM_COUNT,); each program walks its precomputed queue.")
    lines.append('    """')
    lines.append("    sm_id = tl.program_id(0)")
    lines.append("    qlen = tl.load(QUEUE_LEN_PTR + sm_id)")
    lines.append("    task_idx = 0")
    lines.append("    while task_idx < qlen:")
    lines.append("        task_id = tl.load(QUEUE_PTR + (sm_id * MAX_QLEN + task_idx) * 2 + 0)")
    lines.append("        task_kind = tl.load(QUEUE_PTR + (sm_id * MAX_QLEN + task_idx) * 2 + 1)")
    lines.append(dispatch_branches)
    lines.append("        task_idx += 1")
    lines.append("")

    kernel_source = "\n".join(lines)

    launch_config = {
        "grid": sm_count,
        "num_warps": spec.num_warps,
        "num_stages": spec.num_stages,
    }

    device_function_table = {func_to_kind[fn]: fn for fn in funcs}

    return MegakernelLoweringResult(
        kernel_name=kernel_name,
        kernel_source=kernel_source,
        launch_config=launch_config,
        event_layout=events,
        task_queue=structured_queue,
        device_function_table=device_function_table,
    )


__all__ = [
    "DeviceFunctionSpec",
    "MegakernelLoweringResult",
    "MegakernelLoweringSpec",
    "lower_megakernel",
]
