"""Triton body auto-generation tests.

Covers :mod:`compgen.transforms.emit_triton_bodies`:

- Every registered op family produces valid non-empty bodies.
- Generated bodies contain expected Triton constructs.
- Unknown op family raises ``ValueError`` with available names.
- ``generate_body_via_provider`` path (if Triton available).
- ``generate_bodies_for_graph`` batch API.
- Bodies are pure compute — no atomic_add / spin-wait (scheduler
  handles coordination).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Individual op templates
# ---------------------------------------------------------------------------

class TestGenerateBodyFromOp:
    """Test each registered op family."""

    def test_matmul_body(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_from_op

        body = generate_body_from_op(
            "matmul",
            name="test_matmul",
            tile_m=16, tile_n=32, tile_k=32,
            K=128, N=64,
            A_ptr="A_ptr", B_ptr="B_ptr", C_ptr="C_ptr",
            A_stride_k=1, B_stride_k=1,
        )
        assert "tl.zeros" in body
        assert "tl.dot" in body
        assert "tl.load" in body
        assert "tl.store" in body
        assert "A_ptr" in body
        assert "C_ptr" in body
        assert len(body) > 100

    def test_relu_body(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_from_op

        body = generate_body_from_op(
            "relu", name="test_relu",
            total_elems=1024, in_ptr="in_ptr", out_ptr="out_ptr",
        )
        assert "tl.maximum" in body
        assert "tl.load" in body
        assert "tl.store" in body
        assert "in_ptr" in body
        assert "out_ptr" in body

    def test_gelu_body(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_from_op

        body = generate_body_from_op(
            "gelu", name="test_gelu",
            total_elems=512, in_ptr="in_ptr", out_ptr="out_ptr",
        )
        assert "tl.math.tanh" in body
        assert "0.5" in body
        assert "tl.load" in body
        assert "tl.store" in body

    def test_add_body(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_from_op

        body = generate_body_from_op(
            "add", name="test_add",
            total_elems=256,
            A_ptr="A_ptr", B_ptr="B_ptr", C_ptr="C_ptr",
        )
        assert "tl.load" in body
        assert "tl.store" in body
        assert "A_ptr" in body
        assert "B_ptr" in body
        assert "C_ptr" in body

    def test_softmax_per_row_body(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_from_op

        body = generate_body_from_op(
            "softmax_per_row", name="test_softmax",
            N=32, in_ptr="in_ptr", out_ptr="out_ptr",
        )
        assert "tl.max" in body
        assert "tl.exp" in body
        assert "tl.sum" in body
        assert "tl.load" in body
        assert "tl.store" in body

    def test_attention_values_body(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_from_op

        body = generate_body_from_op(
            "attention_values", name="apply_values",
            Q_TILES=2, BLOCK_M=16, S=32, D_HEAD=32,
            P_ptr="P_ptr", V_ptr="V_ptr", A_ptr="A_ptr",
        )
        assert "tl.dot" in body
        assert "tl.load" in body
        assert "tl.store" in body
        assert "P_ptr" in body
        assert "V_ptr" in body
        assert "A_ptr" in body
        assert "task_id" in body

    def test_mlp_proj_body(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_from_op

        body = generate_body_from_op(
            "mlp_proj", name="mlp_gate",
            BLOCK_M=16, BLOCK_I=32, D_HIDDEN=128, I=64,
            HI_ptr="HI_ptr", W_ptr="WG_ptr", OUT_ptr="G_ptr",
            I_TILES=2, gated="gate * tl.sigmoid(gate)", store_val="gated",
        )
        assert "tl.dot" in body
        assert "tl.trans" in body
        assert "tl.load" in body
        assert "tl.store" in body
        assert "HI_ptr" in body
        assert "WG_ptr" in body
        assert "G_ptr" in body
        # The gated/store_val expressions appear literally
        assert "gate * tl.sigmoid(gate)" in body

    def test_mlp_down_proj_body(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_from_op

        body = generate_body_from_op(
            "mlp_down_proj", name="mlp_down",
            N_TILES=4, BLOCK_M=16, BLOCK_N=32,
            D_HIDDEN=128, I=64,
            G_ptr="G_ptr", U_ptr="U_ptr", WD_ptr="WD_ptr",
            HI_ptr="HI_ptr", Y_ptr="Y_ptr",
        )
        assert "tl.dot" in body
        assert "tl.trans" in body
        assert "tl.load" in body
        assert "tl.store" in body
        assert "G_ptr" in body
        assert "U_ptr" in body
        assert "Y_ptr" in body

    def test_bodies_are_pure_compute(self) -> None:
        """Generated bodies must NOT contain event coordination primitives.
        The dynamic scheduler adds these automatically."""
        from compgen.transforms.emit_triton_bodies import generate_body_from_op

        for op in ("matmul", "relu", "gelu", "add", "attention_values",
                    "mlp_proj", "mlp_down_proj", "softmax_per_row"):
            body = generate_body_from_op(
                op, name="test",
                tile_m=16, tile_n=32, tile_k=32,
                K=64, N=64,
                A_ptr="A", B_ptr="B", C_ptr="C",
                in_ptr="in", out_ptr="out",
                total_elems=256,
                Q_TILES=1, BLOCK_M=16, BLOCK_I=32, BLOCK_N=32,
                S=32, D_HEAD=32, D_HIDDEN=64, I=32,
                HI_ptr="HI", W_ptr="W", OUT_ptr="OUT",
                WD_ptr="WD", G_ptr="G", U_ptr="U", Y_ptr="Y",
                I_TILES=1, N_TILES=2,
                gated="g", store_val="g",
                A_stride_k=1, B_stride_k=1,
            )
            assert "atomic_add" not in body, f"{op} body contains atomic_add"
            assert "atomic_or" not in body, f"{op} body contains atomic_or"
            assert "while" not in body, f"{op} body contains spin-wait loop (while)"


class TestUnknownOp:
    """Edge cases."""

    def test_unknown_op_raises(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_from_op

        with pytest.raises(ValueError, match="Unknown op family"):
            generate_body_from_op("nonexistent_op", name="test")

    def test_error_message_lists_available(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_from_op

        with pytest.raises(ValueError) as exc_info:
            generate_body_from_op("bad_op", name="test")
        assert "Available:" in str(exc_info.value)
        assert "matmul" in str(exc_info.value)
        assert "relu" in str(exc_info.value)


class TestGenerateBodiesForGraph:
    """Batch generation API."""

    def test_batch_generates_all(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_bodies_for_graph

        bodies = generate_bodies_for_graph(
            op_map={
                "compute_scores": "softmax_per_row",
                "apply_values": "attention_values",
                "mlp_gate": "mlp_proj",
                "mlp_down": "mlp_down_proj",
            },
            S=32, D_HEAD=32, D_HIDDEN=128, I=64,
            BLOCK_M=16, BLOCK_I=32, BLOCK_N=32,
            Q_TILES=2, I_TILES=2, N_TILES=4,
            P_ptr="P", V_ptr="V", A_ptr="A",
            HI_ptr="HI", W_ptr="W", OUT_ptr="OUT",
            WD_ptr="WD", G_ptr="G", U_ptr="U", Y_ptr="Y",
            gated="g", store_val="g",
        )
        assert set(bodies.keys()) == {"compute_scores", "apply_values", "mlp_gate", "mlp_down"}
        for name, body in bodies.items():
            assert len(body) > 50, f"{name} body too short: {len(body)} chars"
            assert "tl." in body, f"{name} body has no Triton ops"


# ---------------------------------------------------------------------------
# Provider path (requires Triton)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_triton(), reason="Triton not available")
class TestGenerateBodyViaProvider:
    """Test the TritonTemplateProvider integration path."""

    def test_matmul_bias_gelu_via_provider(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_via_provider

        body = generate_body_via_provider(
            "matmul_bias_gelu",
            shapes=((64, 128), (128, 256)),
        )
        if body is not None:
            assert len(body) > 50
            assert "tl." in body
        # Provider may return None if Triton isn't installed or
        # the op family isn't matched — that's expected.

    def test_matmul_via_provider(self) -> None:
        from compgen.transforms.emit_triton_bodies import generate_body_via_provider

        body = generate_body_via_provider(
            "matmul",
            shapes=((32, 64), (64, 32)),
        )
        if body is not None:
            assert len(body) > 50
            assert "tl." in body


# ---------------------------------------------------------------------------
# Template registry completeness
# ---------------------------------------------------------------------------

class TestTemplateRegistry:
    """Structural checks on the template registry."""

    def test_all_op_families_covered_for_event_tensor_examples(self) -> None:
        """The event_tensor transformer_block example uses these ops.
        All must have templates so hand-written bodies can be removed."""
        from compgen.transforms.emit_triton_bodies import _TEMPLATES

        required = {"matmul", "relu", "gelu", "add", "softmax_per_row",
                     "attention_values", "mlp_proj", "mlp_down_proj"}
        missing = required - set(_TEMPLATES)
        assert not missing, f"Missing templates for: {missing}"

    def test_every_template_is_non_empty(self) -> None:
        from compgen.transforms.emit_triton_bodies import _TEMPLATES

        for name, template in _TEMPLATES.items():
            assert template.strip(), f"Template {name!r} is empty"
            assert "{" in template, f"Template {name!r} has no format placeholders"

    def test_every_template_generates_valid_body(self) -> None:
        """Every template must produce a non-empty body with defaults."""
        from compgen.transforms.emit_triton_bodies import generate_body_from_op, _TEMPLATES

        defaults: dict[str, dict] = {
            "matmul": dict(tile_m=8, tile_n=8, tile_k=8, K=32, N=32,
                           A_ptr="A", B_ptr="B", C_ptr="C",
                           A_stride_k=1, B_stride_k=1),
            "relu": dict(total_elems=64, in_ptr="in", out_ptr="out"),
            "gelu": dict(total_elems=64, in_ptr="in", out_ptr="out"),
            "add": dict(total_elems=64, A_ptr="A", B_ptr="B", C_ptr="C"),
            "softmax_per_row": dict(N=16, in_ptr="in", out_ptr="out"),
            "attention_values": dict(Q_TILES=1, BLOCK_M=8, S=16, D_HEAD=16,
                                      P_ptr="P", V_ptr="V", A_ptr="A"),
            "mlp_proj": dict(BLOCK_M=8, BLOCK_I=16, D_HIDDEN=64, I=32,
                             HI_ptr="HI", W_ptr="W", OUT_ptr="O",
                             I_TILES=1, gated="g", store_val="g"),
            "mlp_down_proj": dict(N_TILES=2, BLOCK_M=8, BLOCK_N=16,
                                   D_HIDDEN=64, I=32,
                                   G_ptr="G", U_ptr="U", WD_ptr="WD",
                                   HI_ptr="HI", Y_ptr="Y"),
        }

        for name in _TEMPLATES:
            kwargs = defaults.get(name, {})
            body = generate_body_from_op(name, name=name, **kwargs)
            assert len(body) > 20, f"Template {name!r} produced body of {len(body)} chars"
