"""Quick test: verify Triton megakernel generation works correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from user_perspective.prototypes.triton_megakernel_api import (
    UnsupportedModel,
    compile_to_triton_megakernel,
    match_model,
)


class Diamond(nn.Module):
    def __init__(self, in_dim=256, out_dim=128):
        super().__init__()
        self.linear_a = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_b = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, x):
        return (self.linear_a(x) + self.linear_b(x)).relu()


class FFN(nn.Module):
    def __init__(self, in_dim=256, hidden=512, out_dim=128):
        super().__init__()
        self.linear_up = nn.Linear(in_dim, hidden, bias=False)
        self.linear_down = nn.Linear(hidden, out_dim, bias=False)
    def forward(self, x):
        return self.linear_down(torch.relu(self.linear_up(x)))


class NotSupported(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(256, 128, bias=False)
        self.b = nn.Linear(256, 128, bias=False)
        self.c = nn.Linear(128, 64, bias=False)
    def forward(self, x):
        return self.c(self.a(x) + self.b(x))


def check(name, condition, detail=""):
    if condition:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}  FAILED {detail}")
    return condition


def main():
    passed = 0
    B, IN, OUT = 4, 256, 128

    # ── Diamond ──
    print("=" * 55)
    print("Diamond: match & generate")
    model = Diamond(IN, OUT)
    x = torch.randn(B, IN)
    dag = match_model(model, B, sm_count=2)
    passed += check("pattern=diamond", dag.pattern_name == "diamond")
    passed += check("4 ops", len(dag.ops) == 4)
    passed += check("names correct",
                    [o.name for o in dag.ops] == ["mm_a","mm_b","add","relu"])

    bundle = compile_to_triton_megakernel(model, (x,), sm_count=2)
    src = bundle.kernel_source
    passed += check("source > 500 chars", len(src) > 500)
    passed += check("has @triton.jit", "@triton.jit" in src)
    passed += check("has megakernel def", "def megakernel(" in src)
    passed += check("has QUEUE_PTR", "QUEUE_PTR" in src)
    passed += check("has tl.atomic_or", "atomic_or" in src)
    passed += check("has tl.atomic_add", "atomic_add" in src)
    passed += check("no break statement", "break" not in src)
    passed += check("no _OP_TBL (no list index)", "_OP_TBL" not in src)
    passed += check("mm_a uses w_a_ptr", "w_a_ptr" in dag.ops[0].triton_body)
    passed += check("mm_b uses w_b_ptr", "w_b_ptr" in dag.ops[1].triton_body)

    # ── FFN ──
    print()
    print("=" * 55)
    print("FFN: match & generate")
    model = FFN(IN, 512, OUT)
    x = torch.randn(B, IN)
    dag = match_model(model, B, sm_count=2)
    passed += check("pattern=ffn", dag.pattern_name == "ffn")
    passed += check("3 ops", len(dag.ops) == 3)

    bundle = compile_to_triton_megakernel(model, (x,), sm_count=2)
    src = bundle.kernel_source
    passed += check("source > 200 chars", len(src) > 200)
    passed += check("has @triton.jit", "@triton.jit" in src)

    # ── Unsupported ──
    print()
    print("=" * 55)
    print("Unsupported: should reject")
    model = NotSupported()
    x = torch.randn(B, IN)
    try:
        match_model(model, B)
        passed += check("rejects 3-linear", False, "should have raised")
    except UnsupportedModel as e:
        passed += check("rejects 3-linear", "NotSupported" in str(e))

    # ── GPU correctness ──
    print()
    print("=" * 55)
    if torch.cuda.is_available():
        bundle_d = compile_to_triton_megakernel(Diamond(IN, OUT).eval(), (torch.randn(B, IN),), sm_count=2)
        if bundle_d.kernel_fn is not None:
            print("GPU correctness test")
            model_d = Diamond(IN, OUT).eval().cuda()
            x_d = torch.randn(B, IN, device="cuda")
            bundle_d = compile_to_triton_megakernel(model_d, (x_d,), sm_count=2)

            from user_perspective.prototypes.triton_megakernel_api import _build_queue_tensors
            TM = TN = TK = 32
            tiles_per_row = (OUT + TN - 1) // TN
            dev = x_d.device
            buffers = [
                x_d, model_d.linear_a.weight.data, model_d.linear_b.weight.data,
                torch.zeros(B, OUT, device=dev), torch.zeros(B, OUT, device=dev),
                torch.zeros(B, OUT, device=dev), torch.zeros(B, OUT, device=dev),
            ]
            queue, lens = _build_queue_tensors(bundle_d.sm_queues, 2, dev)
            n_ev = len(bundle_d.dag.ops) * bundle_d.dag.total_tiles
            E = torch.full((n_ev,), 1, dtype=torch.int32, device=dev)
            ce = {"TM":TM,"TN":TN,"TK":TK,"TILES_PER_ROW":tiles_per_row,"K_DIM":IN,"N_DIM":OUT}
            bundle_d.kernel_fn[(2,)](*buffers, E, queue, lens, **ce)
            torch.cuda.synchronize()
            with torch.no_grad():
                eager = model_d(x_d)
            err = (buffers[-1] - eager).abs().max().item()
            passed += check(f"GPU error {err:.2e} < 1e-3", err < 1e-3)
        else:
            print("  skipped (Triton not available)")
    else:
        print("  skipped (no CUDA)")

    total = 16
    print(f"\n{'='*55}")
    print(f"{passed}/{total} checks passed")
    print("ALL OK" if passed == total else "SOME FAILED")


if __name__ == "__main__":
    main()
