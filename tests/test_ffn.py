"""
FFN Chunking Test Suite for noflash-attention

Tests:
  1. Detection (CPU) — structural pre-filter correctness
  2. Verification (GPU) — runtime map-property proof
  3. Correctness (GPU) — chunked output == unchunked output
  4. Edge cases — guards and fallbacks
  5. Lifecycle — patch/remove/weakref cleanup
  6. Memory — peak VRAM reduction
"""

import torch
import torch.nn as nn
import pytest
import gc
import weakref

from noflash_attention._detect import (
    is_chunkable_candidate,
    is_moe_module,
    has_internal_normalization,
    is_attention_module,
    find_linear_layers,
    expansion_ratio,
)
from noflash_attention.ffn import (
    patch_ffn,
    ChunkedFFNWrapper,
    FFNPatchHandle,
    _VerifyState,
)


# ============================================================
# Synthetic module definitions for testing
# ============================================================

class SequentialFFN(nn.Module):
    """ComfyUI / LTX-2.3 style: self.net = Sequential(Linear, GELU, Linear)"""
    def __init__(self, dim=256, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )
    def forward(self, x):
        return self.net(x)


class GatedSwiGLU(nn.Module):
    """LLaMA / Mistral style: gate_proj + up_proj -> SiLU -> down_proj"""
    def __init__(self, dim=256, expansion=4):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim * expansion, bias=False)
        self.up_proj = nn.Linear(dim, dim * expansion, bias=False)
        self.down_proj = nn.Linear(dim * expansion, dim, bias=False)
    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class WanStyleFFN(nn.Module):
    """Wan 2.2 style: self.ffn = Sequential(Linear, GELU, Linear)"""
    def __init__(self, dim=256):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    def forward(self, x):
        return self.ffn(x)


class MoEModule(nn.Module):
    """Mixture of Experts: gate + expert list"""
    def __init__(self, dim=256, num_experts=4):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            for _ in range(num_experts)
        ])
        self.top_k = 2
    def forward(self, x):
        scores = self.gate(x)
        # Simplified routing (real MoE uses top-k)
        weights = torch.softmax(scores, dim=-1)
        out = sum(w.unsqueeze(-1) * expert(x) for w, expert in
                  zip(weights.unbind(-1), self.experts))
        return out


class FFNWithNorm(nn.Module):
    """FFN with internal LayerNorm — MUST NOT be chunked"""
    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    def forward(self, x):
        return self.proj(x)


class AttentionModule(nn.Module):
    """Attention module — MUST NOT be chunked"""
    def __init__(self, dim=256):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5), dim=-1)
        return self.to_out(attn @ v)


class TinyProjector(nn.Module):
    """Small embedder — expansion < 1.5x, should be skipped"""
    def __init__(self, dim=256):
        super().__init__()
        self.in_layer = nn.Linear(dim, dim)
        self.out_layer = nn.Linear(dim, dim)
    def forward(self, x):
        return self.out_layer(nn.functional.silu(self.in_layer(x)))


class CrossTokenFFN(nn.Module):
    """FFN with hidden cross-token op — verification MUST catch this"""
    def __init__(self, dim=256):
        super().__init__()
        self.up = nn.Linear(dim, dim * 4)
        self.down = nn.Linear(dim * 4, dim)
    def forward(self, x):
        h = self.up(x)
        h = h + h.mean(dim=1, keepdim=True)  # cross-token!
        return self.down(torch.relu(h))


class TransformerBlock(nn.Module):
    """Full block with attention + FFN for integration testing"""
    def __init__(self, dim=256):
        super().__init__()
        self.attn = AttentionModule(dim)
        self.ff = SequentialFFN(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class SimpleModel(nn.Module):
    """Multi-block model for patch_ffn integration testing"""
    def __init__(self, dim=256, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(num_blocks)])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ============================================================
# 1. Detection Tests (CPU, no GPU needed)
# ============================================================

class TestDetection:
    """Structural pre-filter correctness."""

    def test_sequential_ffn_detected(self):
        m = SequentialFFN()
        ok, reason = is_chunkable_candidate(m)
        assert ok, f"SequentialFFN should be candidate, got: {reason}"

    def test_gated_swiglu_detected(self):
        m = GatedSwiGLU()
        ok, reason = is_chunkable_candidate(m)
        assert ok, f"GatedSwiGLU should be candidate, got: {reason}"

    def test_wan_ffn_detected(self):
        m = WanStyleFFN()
        # WanStyleFFN wraps Sequential in .ffn — the Sequential itself is the FFN
        ok, reason = is_chunkable_candidate(m.ffn)
        assert ok, f"Wan Sequential FFN should be candidate, got: {reason}"

    def test_moe_rejected(self):
        m = MoEModule()
        ok, reason = is_chunkable_candidate(m)
        assert not ok, "MoE should NOT be candidate"
        assert "moe" in reason.lower()

    def test_ffn_with_norm_rejected(self):
        m = FFNWithNorm()
        ok, reason = is_chunkable_candidate(m)
        assert not ok, "FFN with internal norm should NOT be candidate"
        assert "normalization" in reason.lower()

    def test_attention_rejected(self):
        m = AttentionModule()
        ok, reason = is_chunkable_candidate(m)
        assert not ok, "Attention should NOT be candidate"
        assert "attention" in reason.lower()

    def test_tiny_projector_rejected(self):
        m = TinyProjector()
        ok, reason = is_chunkable_candidate(m)
        assert not ok, "Tiny projector should NOT be candidate (expansion < 1.5x)"
        assert "expansion" in reason.lower() or "insufficient" in reason.lower()

    def test_cross_token_passes_structural_filter(self):
        """CrossTokenFFN passes structural filter — runtime verification catches it."""
        m = CrossTokenFFN()
        ok, reason = is_chunkable_candidate(m)
        assert ok, "CrossTokenFFN passes structural filter (caught by runtime verify)"


class TestMoEDetection:
    """Thorough MoE structural detection."""

    def test_standard_moe(self):
        m = MoEModule()
        is_moe, _ = is_moe_module(m)
        assert is_moe

    def test_moe_with_routing_attrs(self):
        """Module with top_k/num_experts attrs."""
        m = nn.Module()
        m.experts = nn.ModuleList([nn.Linear(10, 10) for _ in range(4)])
        m.top_k = 2
        is_moe, _ = is_moe_module(m)
        assert is_moe

    def test_non_moe_with_modulelist(self):
        """ModuleList alone is not MoE (no gate, no routing attrs)."""
        m = nn.Module()
        m.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(4)])
        is_moe, _ = is_moe_module(m)
        assert not is_moe

    def test_sequential_ffn_not_moe(self):
        m = SequentialFFN()
        is_moe, _ = is_moe_module(m)
        assert not is_moe


class TestNormDetection:
    def test_layernorm_inside(self):
        m = FFNWithNorm()
        assert has_internal_normalization(m)

    def test_no_norm(self):
        m = SequentialFFN()
        assert not has_internal_normalization(m)

    def test_rmsnorm_detected(self):
        """Custom RMSNorm class detected by name."""
        class RMSNorm(nn.Module):
            def __init__(self): super().__init__()
        m = nn.Module()
        m.norm = RMSNorm()
        m.linear = nn.Linear(10, 10)
        assert has_internal_normalization(m)


class TestAttentionDetection:
    def test_attention_detected(self):
        m = AttentionModule()
        assert is_attention_module(m)

    def test_ffn_not_attention(self):
        m = SequentialFFN()
        assert not is_attention_module(m)

    def test_gated_not_attention(self):
        m = GatedSwiGLU()
        assert not is_attention_module(m)


class TestExpansionRatio:
    def test_4x_expansion(self):
        linears = find_linear_layers(SequentialFFN(dim=256, expansion=4))
        assert abs(expansion_ratio(linears) - 4.0) < 0.1

    def test_1x_no_expansion(self):
        linears = find_linear_layers(TinyProjector())
        assert expansion_ratio(linears) < 1.5


# ============================================================
# 2. Runtime Verification Tests (GPU)
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
class TestVerification:

    def test_safe_module_passes(self):
        m = SequentialFFN(dim=128).cuda().half().eval()
        w = ChunkedFFNWrapper(m, num_chunks=4)
        x = torch.randn(1, 512, 128, device='cuda', dtype=torch.float16)
        assert w._verify(x) is True

    def test_cross_token_fails(self):
        """Runtime verification catches CrossTokenFFN."""
        m = CrossTokenFFN(dim=128).cuda().half().eval()
        w = ChunkedFFNWrapper(m, num_chunks=4)
        x = torch.randn(1, 512, 128, device='cuda', dtype=torch.float16)
        assert w._verify(x) is False

    def test_gated_swiglu_passes(self):
        m = GatedSwiGLU(dim=128).cuda().half().eval()
        w = ChunkedFFNWrapper(m, num_chunks=4)
        x = torch.randn(1, 512, 128, device='cuda', dtype=torch.float16)
        assert w._verify(x) is True


# ============================================================
# 3. Correctness Tests (GPU)
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
class TestCorrectness:

    def _check_exact(self, module_cls, dim=128, seq=1024, chunks=4, dtype=torch.float16):
        m = module_cls(dim=dim).cuda().to(dtype).eval()
        x = torch.randn(2, seq, dim, device='cuda', dtype=dtype)
        with torch.no_grad():
            ref = m(x)
            w = ChunkedFFNWrapper(m, num_chunks=chunks, min_seq=1)
            # Force verification to pass
            w._state = _VerifyState.SAFE
            out = w(x)
        assert torch.allclose(ref, out, atol=1e-3 if dtype == torch.float16 else 1e-5,
                              rtol=1e-2 if dtype == torch.float16 else 1e-4), \
            f"Max diff: {(ref - out).abs().max().item()}"

    def test_sequential_fp16(self):
        self._check_exact(SequentialFFN, dtype=torch.float16)

    def test_sequential_fp32(self):
        self._check_exact(SequentialFFN, dtype=torch.float32)

    def test_gated_swiglu_fp16(self):
        self._check_exact(GatedSwiGLU, dtype=torch.float16)

    def test_uneven_chunks(self):
        """seq=1000 with 3 chunks -> 334, 334, 332."""
        self._check_exact(SequentialFFN, seq=1000, chunks=3)

    def test_single_token_chunk(self):
        """Extreme: seq=8, chunks=4 -> 2 tokens per chunk."""
        self._check_exact(SequentialFFN, seq=8, chunks=4)

    def test_large_seq(self):
        """Realistic video generation seq length."""
        self._check_exact(SequentialFFN, dim=128, seq=4096, chunks=8)


# ============================================================
# 4. Edge Case Tests
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
class TestEdgeCases:

    def test_2d_input_passthrough(self):
        """2D input should pass through without chunking."""
        m = SequentialFFN(dim=128).cuda().eval()
        w = ChunkedFFNWrapper(m, num_chunks=4)
        x = torch.randn(128, 128, device='cuda')  # 2D
        with torch.no_grad():
            out = w(x)
        assert out.shape == (128, 128)

    def test_short_seq_passthrough(self):
        """Short sequence below min_seq should not chunk."""
        m = SequentialFFN(dim=128).cuda().eval()
        w = ChunkedFFNWrapper(m, num_chunks=4, min_seq=512)
        w._state = _VerifyState.SAFE
        x = torch.randn(1, 64, 128, device='cuda')
        with torch.no_grad():
            out = w(x)
        assert out.shape == x.shape

    def test_unsafe_module_passthrough(self):
        """Unsafe-verified module always passes through."""
        m = CrossTokenFFN(dim=128).cuda().half().eval()
        w = ChunkedFFNWrapper(m, num_chunks=4, min_seq=1)
        x = torch.randn(1, 512, 128, device='cuda', dtype=torch.float16)
        with torch.no_grad():
            out = w(x)
        assert w._state == _VerifyState.UNSAFE
        assert out.shape == x.shape


# ============================================================
# 5. Lifecycle Tests
# ============================================================

class TestLifecycle:

    def test_patch_and_remove(self):
        model = SimpleModel(dim=128, num_blocks=2)
        handle = patch_ffn(model, num_chunks=4, verbose=False)
        assert handle.num_wrapped > 0

        # Check FFN modules are wrapped
        for block in model.blocks:
            assert isinstance(block.ff, ChunkedFFNWrapper)

        handle.remove()

        # Check originals restored
        for block in model.blocks:
            assert isinstance(block.ff, SequentialFFN)
        assert handle.num_wrapped == 0

    def test_double_patch_skips(self):
        model = SimpleModel(dim=128, num_blocks=2)
        h1 = patch_ffn(model, num_chunks=4, verbose=False)
        count1 = h1.num_wrapped
        h2 = patch_ffn(model, num_chunks=4, verbose=False)
        assert h2.num_wrapped == 0  # already wrapped, skip

        h1.remove()

    def test_weakref_cleanup(self):
        model = SimpleModel(dim=128, num_blocks=2)
        handle = patch_ffn(model, num_chunks=4, verbose=False)
        assert handle.active

        ref = weakref.ref(model)
        del model
        gc.collect()

        assert ref() is None
        assert not handle.active
        assert handle.num_wrapped == 0

    def test_attention_not_wrapped(self):
        """Attention modules in transformer blocks should NOT be wrapped."""
        model = SimpleModel(dim=128, num_blocks=2)
        handle = patch_ffn(model, num_chunks=4, verbose=False)

        for block in model.blocks:
            assert not isinstance(block.attn, ChunkedFFNWrapper), \
                "Attention module should not be wrapped"

        handle.remove()


# ============================================================
# 6. Integration Tests (GPU)
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
class TestIntegration:

    def test_full_model_inference(self):
        """Patched model produces same output as unpatched."""
        model = SimpleModel(dim=128, num_blocks=4).cuda().half().eval()
        x = torch.randn(1, 512, 128, device='cuda', dtype=torch.float16)

        with torch.no_grad():
            ref = model(x)

        handle = patch_ffn(model, num_chunks=4, verbose=False)
        with torch.no_grad():
            out = model(x)

        # Note: first call triggers verification (3 extra FFN calls per wrapper).
        # Output should still match because chunking is mathematically identical.
        assert torch.allclose(ref, out, atol=1e-2, rtol=1e-1), \
            f"Max diff: {(ref - out).abs().max().item()}"

        handle.remove()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
