"""Auto-generate Triton device-function bodies for dynamic megakernels.

Bridges the gap between:

1. **Template provider** — :class:`TritonTemplateProvider` generates
   complete ``@triton.jit`` kernels for known op families.
2. **Tile IR lowering** — :func:`lower_tile_to_triton` generates
   Triton statement fragments from Tile IR ops.
3. **Dynamic megakernel** — :class:`DynamicDeviceFunctionSpec`
   expects a pure-compute Triton body (no event coordination).

This module provides:

- :func:`generate_body_from_op` — one call to go from op name +
  shapes to a body ready for ``DynamicDeviceFunctionSpec``.
- :func:`generate_bodies_for_graph` — batch-generate all bodies
  needed for an event graph's ``CallDeviceOp`` set.

Usage in place of hand-written bodies::

    from compgen.transforms.emit_triton_bodies import generate_body_from_op

    spec = DynamicDeviceFunctionSpec(
        name="compute_scores",
        body_source=generate_body_from_op(
            op="softmax_attn_scores",
            tile_m=16, tile_n=32,
            inputs={"Q_ptr": "Q", "K_ptr": "K"},
            outputs={"P_ptr": "P"},
        ),
    )
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Triton body templates for common op families
# ---------------------------------------------------------------------------
# Each template is a function that takes keyword args (tile dims,
# pointer names, etc.) and returns an indented Triton body string.
# The body must be pure compute — no atomic_add / spin-wait.
# The dynamic scheduler wrapper adds event coordination automatically.

_TEMPLATES: dict[str, str] = {}


def _register(op: str, template: str) -> None:
    _TEMPLATES[op] = template


# ---- matmul: C[m,n] += A[m,k] × B[k,n] ----

_register("matmul", """\
# Auto-generated matmul body: {name}
# Tile: {tile_m}×{tile_n}×{tile_k}
m_offs = task_id * {tile_m} + tl.arange(0, {tile_m})
n_offs = tl.arange(0, {tile_n})
k_offs = tl.arange(0, {tile_k})

a_ptrs = {A_ptr} + m_offs[:, None] * {K} + k_offs[None, :]
b_ptrs = {B_ptr} + k_offs[:, None] * {N} + n_offs[None, :]

acc = tl.zeros(({tile_m}, {tile_n}), dtype=tl.float32)
for k in range(0, {K}, {tile_k}):
    a = tl.load(a_ptrs + k * {A_stride_k})
    b = tl.load(b_ptrs + k * {B_stride_k})
    acc += tl.dot(a, b)

c_ptrs = {C_ptr} + m_offs[:, None] * {N} + n_offs[None, :]
tl.store(c_ptrs, acc)
""")


# ---- elementwise relu ----

_register("relu", """\
# Auto-generated relu body: {name}
n_elems = {total_elems}
offs = task_id * n_elems + tl.arange(0, n_elems)
mask = offs < {total_elems}
x = tl.load({in_ptr} + offs, mask=mask)
tl.store({out_ptr} + offs, tl.maximum(x, 0.0), mask=mask)
""")


# ---- elementwise gelu ----

_register("gelu", """\
# Auto-generated gelu body: {name}
n_elems = {total_elems}
offs = task_id * n_elems + tl.arange(0, n_elems)
mask = offs < {total_elems}
x = tl.load({in_ptr} + offs, mask=mask)
# GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
gelu = 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
tl.store({out_ptr} + offs, gelu, mask=mask)
""")


# ---- elementwise add ----

_register("add", """\
# Auto-generated add body: {name}
n_elems = {total_elems}
offs = task_id * n_elems + tl.arange(0, n_elems)
mask = offs < {total_elems}
a = tl.load({A_ptr} + offs, mask=mask)
b = tl.load({B_ptr} + offs, mask=mask)
tl.store({C_ptr} + offs, a + b, mask=mask)
""")


# ---- softmax (used in attention scores) ----

_register("softmax_per_row", """\
# Auto-generated softmax body: {name}
row = task_id
cols = tl.arange(0, {N})
ptrs = {in_ptr} + row * {N} + cols
vals = tl.load(ptrs)

row_max = tl.max(vals, axis=0)
vals = vals - row_max
exps = tl.exp(vals)
denom = tl.sum(exps, axis=0)
probs = exps / denom

out_ptrs = {out_ptr} + row * {N} + cols
tl.store(out_ptrs, probs)
""")


# ---- attention values (P × V → A) ----

_register("attention_values", """\
# Auto-generated attention-values body: {name}
h = task_id // {Q_TILES}
q_tile = task_id % {Q_TILES}
q_rows = q_tile * {BLOCK_M} + tl.arange(0, {BLOCK_M})
kv_rows = tl.arange(0, {S})
d_cols = tl.arange(0, {D_HEAD})

p_ptrs = {P_ptr} + h * ({S} * {S}) + q_rows[:, None] * {S} + kv_rows[None, :]
p = tl.load(p_ptrs)
v_ptrs = {V_ptr} + h * ({S} * {D_HEAD}) + kv_rows[:, None] * {D_HEAD} + d_cols[None, :]
v = tl.load(v_ptrs)

out = tl.dot(p, v)
a_ptrs = {A_ptr} + h * ({S} * {D_HEAD}) + q_rows[:, None] * {D_HEAD} + d_cols[None, :]
tl.store(a_ptrs, out)
""")


# ---- MLP projection (gate/up) ----

_register("mlp_proj", """\
# Auto-generated MLP projection body: {name}
m_tile = task_id // {I_TILES}
i_tile = task_id % {I_TILES}
m_rows = m_tile * {BLOCK_M} + tl.arange(0, {BLOCK_M})
i_cols = i_tile * {BLOCK_I} + tl.arange(0, {BLOCK_I})
k_idx = tl.arange(0, {D_HIDDEN})

hi_ptrs = {HI_ptr} + m_rows[:, None] * {D_HIDDEN} + k_idx[None, :]
hi = tl.load(hi_ptrs)
w_ptrs = {W_ptr} + i_cols[:, None] * {D_HIDDEN} + k_idx[None, :]
w = tl.load(w_ptrs)

proj = tl.dot(hi, tl.trans(w))
{gated}

out_ptrs = {OUT_ptr} + m_rows[:, None] * {I} + i_cols[None, :]
tl.store(out_ptrs, {store_val})
""")


# ---- MLP down projection ----

_register("mlp_down_proj", """\
# Auto-generated MLP down-projection body: {name}
m_tile = task_id // {N_TILES}
n_tile = task_id % {N_TILES}
m_rows = m_tile * {BLOCK_M} + tl.arange(0, {BLOCK_M})
n_cols = n_tile * {BLOCK_N} + tl.arange(0, {BLOCK_N})
i_idx = tl.arange(0, {I})

g_ptrs = {G_ptr} + m_rows[:, None] * {I} + i_idx[None, :]
u_ptrs = {U_ptr} + m_rows[:, None] * {I} + i_idx[None, :]
g = tl.load(g_ptrs)
u = tl.load(u_ptrs)
hid = g * u

wd_ptrs = {WD_ptr} + n_cols[:, None] * {I} + i_idx[None, :]
wd = tl.load(wd_ptrs)
mlp_out = tl.dot(hid, tl.trans(wd))

hi_ptrs = {HI_ptr} + m_rows[:, None] * {D_HIDDEN} + n_cols[None, :]
hi = tl.load(hi_ptrs)
y_ptrs = {Y_ptr} + m_rows[:, None] * {D_HIDDEN} + n_cols[None, :]
tl.store(y_ptrs, hi + mlp_out)
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class _SafeDict(dict):
    """A dict that returns the key name wrapped in braces for missing keys.

    Used by :func:`generate_body_from_op` so callers can pass a shared
    kwargs dict across multiple templates without KeyError — missing
    placeholders are left as ``{key_name}`` in the output.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def generate_body_from_op(
    op: str,
    *,
    name: str = "",
    **kwargs: Any,
) -> str:
    """Generate a Triton device-function body for a given op family.

    Args:
        op: Op family name (``"matmul"``, ``"relu"``, ``"gelu"``,
            ``"add"``, ``"softmax_per_row"``, ``"attention_values"``,
            ``"mlp_proj"``, ``"mlp_down_proj"``).
        name: Device function name (used in generated comments).
        **kwargs: Op-specific parameters (tile dims, pointer names, etc.).

    Returns:
        Indented Triton body string ready for
        :class:`DynamicDeviceFunctionSpec.body_source`.
    """
    template = _TEMPLATES.get(op)
    if template is None:
        raise ValueError(
            f"Unknown op family: {op!r}. Available: {sorted(_TEMPLATES.keys())}"
        )
    # Use format_map with a dict that leaves missing keys as-is
    # so callers can pass shared kwargs without KeyError.
    return template.format_map(_SafeDict(kwargs, name=name or op))


def generate_body_via_provider(
    op_family: str,
    *,
    shapes: tuple[tuple[int, ...], ...],
    dtypes: tuple[str, ...] = ("float32",),
    target_name: str = "cuda_a100",
) -> str | None:
    """Try to generate a Triton body via the template provider.

    Args:
        op_family: e.g. ``"matmul_bias_gelu"``.
        shapes: Input/output shapes.
        dtypes: Data types.
        target_name: Target profile name.

    Returns:
        Extracted compute body from the generated Triton kernel,
        or ``None`` if the provider couldn't generate one.
    """
    try:
        from compgen.kernels.provider import KernelContract, SearchBudget
        from compgen.kernels.providers.triton_templates import TritonTemplateProvider

        contract = KernelContract(
            region_id="auto_0",
            op_family=op_family,
            input_shapes=shapes,
            output_shapes=((1,),),  # dummy
            dtypes=dtypes,
            target_name=target_name,
        )
        provider = TritonTemplateProvider()
        result = provider.search(contract, SearchBudget(max_iterations=1, max_time_ms=5000))
        if result.found:
            return _extract_compute_body(result.kernel_code)
        return None
    except Exception:
        log.warning("generate_body_via_provider_failed", op_family=op_family)
        return None


def _extract_compute_body(kernel_source: str) -> str:
    """Extract the compute body from a full @triton.jit kernel source.

    Returns the indented body content without the function signature
    and decorator, suitable for injection into the dynamic scheduler.
    """
    lines = kernel_source.splitlines()
    # Find the function body start (first indented line after "def ")
    in_body = False
    body_lines: list[str] = []
    for line in lines:
        if not in_body:
            if line.strip().startswith("def "):
                in_body = True
            continue
        if line.strip() == "":
            continue
        # Body lines are indented; function body ends at unindented line
        if line and not line[0].isspace():
            break
        body_lines.append(line)
    return "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Integration helper: generate all bodies for an event graph
# ---------------------------------------------------------------------------


def generate_bodies_for_graph(
    *,
    op_map: dict[str, str],
    **common_kwargs: Any,
) -> dict[str, str]:
    """Batch-generate bodies for a set of device functions.

    Args:
        op_map: ``{device_func_name → op_family}`` mapping.
        **common_kwargs: Passed to every template.
            Override per-body with ``<name>_<key>`` keys,
            e.g. ``compute_scores_tile_m=16``.

    Returns:
        ``{device_func_name → body_source}`` ready for
        ``DynamicDeviceFunctionSpec``.
    """
    bodies: dict[str, str] = {}
    for func_name, op in op_map.items():
        # Merge common kwargs with per-function overrides
        kwargs = dict(common_kwargs)
        prefix = f"{func_name}_"
        for k, v in common_kwargs.items():
            if prefix + k in common_kwargs:
                kwargs[k] = common_kwargs[prefix + k]
        bodies[func_name] = generate_body_from_op(op, name=func_name, **kwargs)
    return bodies


__all__ = [
    "generate_body_from_op",
    "generate_body_via_provider",
    "generate_bodies_for_graph",
]
