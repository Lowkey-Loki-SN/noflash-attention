"""
Comprehensive correctness test suite for gfx906 SDPA patch.
Every test compares OUTPUT VALUES against a reference (Math SDPA or manual).
Not just "doesn't crash" — actual numerical verification.
"""
import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

try:
    import noflash_attention
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import noflash_attention

original = noflash_attention._original_sdpa
device = "cuda"
dtype = torch.float16
TOLERANCE = 0.05


def _check(name, out, ref, tol=TOLERANCE):
    assert out.shape == ref.shape, f"{name}: shape mismatch {out.shape} vs {ref.shape}"
    assert not torch.isnan(out).any().item(), f"{name}: NaN in output"
    max_diff = (out - ref).abs().max().item()
    denom = ref.abs().mean().item()
    rel_err = max_diff / denom if denom > 0 else max_diff
    assert rel_err <= tol, f"{name}: rel={rel_err:.4f} > tol={tol:.4f}, max_diff={max_diff:.6f}"


def _gqa_ref(Q, K, V, **kw):
    G = Q.shape[1] // K.shape[1]
    return original(Q, K.repeat_interleave(G, dim=1), V.repeat_interleave(G, dim=1), **kw)


# =====================================================
# 1. Standard MHA — 9 configs
# =====================================================
@pytest.mark.parametrize("B,H,seq,D", [
    (1, 16, 256, 64),
    (1, 16, 1024, 64),
    (1, 16, 4096, 64),
    (2, 8, 512, 64),
    (4, 4, 256, 64),
    (1, 8, 512, 80),
    (1, 8, 512, 96),
    (1, 8, 512, 128),
    (1, 32, 512, 128),
])
def test_mha(B, H, seq, D):
    Q = torch.randn(B, H, seq, D, device=device, dtype=dtype)
    K = torch.randn(B, H, seq, D, device=device, dtype=dtype)
    V = torch.randn(B, H, seq, D, device=device, dtype=dtype)
    ref = original(Q, K, V)
    out = F.scaled_dot_product_attention(Q, K, V)
    _check(f"B={B} H={H} seq={seq} D={D}", out, ref)


# =====================================================
# 2. Cross-attention (seq_q != seq_kv) — 5 configs
# =====================================================
@pytest.mark.parametrize("sq,skv", [
    (64, 1024), (256, 4096), (1, 2048), (512, 512), (1024, 256),
])
def test_cross_attention(sq, skv):
    Q = torch.randn(1, 16, sq, 64, device=device, dtype=dtype)
    K = torch.randn(1, 16, skv, 64, device=device, dtype=dtype)
    V = torch.randn(1, 16, skv, 64, device=device, dtype=dtype)
    ref = original(Q, K, V)
    out = F.scaled_dot_product_attention(Q, K, V)
    _check(f"seq_q={sq} seq_kv={skv}", out, ref)


# =====================================================
# 3. is_causal — 3 configs
# =====================================================
@pytest.mark.parametrize("B,H,seq,D", [
    (1, 16, 256, 64), (1, 8, 1024, 128), (2, 4, 512, 64),
])
def test_causal(B, H, seq, D):
    Q = torch.randn(B, H, seq, D, device=device, dtype=dtype)
    K = torch.randn(B, H, seq, D, device=device, dtype=dtype)
    V = torch.randn(B, H, seq, D, device=device, dtype=dtype)
    ref = original(Q, K, V, is_causal=True)
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    _check(f"causal B={B} H={H} seq={seq} D={D}", out, ref)


# =====================================================
# 4. Bool attention mask — 2 configs
# =====================================================
@pytest.mark.parametrize("seq", [256, 1024])
def test_bool_mask(seq):
    Q = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    mask = torch.ones(seq, seq, device=device, dtype=torch.bool).tril()
    ref = original(Q, K, V, attn_mask=mask)
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    _check(f"bool_mask seq={seq}", out, ref)


# =====================================================
# 5. Float attention mask — 2 configs
# =====================================================
@pytest.mark.parametrize("seq", [256, 1024])
def test_float_mask(seq):
    Q = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    bmask = torch.ones(seq, seq, device=device, dtype=torch.bool).tril()
    fmask = torch.zeros(seq, seq, device=device, dtype=dtype)
    fmask.masked_fill_(~bmask, float("-inf"))
    ref = original(Q, K, V, attn_mask=fmask)
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=fmask)
    _check(f"float_mask seq={seq}", out, ref)


# =====================================================
# 6. 4D attention mask (per-head) — 1 config
# =====================================================
def test_4d_mask():
    Q = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    mask4d = torch.zeros(1, 8, 256, 256, device=device, dtype=dtype)
    mask4d[:, :, :, 128:] = float("-inf")
    ref = original(Q, K, V, attn_mask=mask4d)
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask4d)
    _check("4d_mask (1,8,256,256)", out, ref)


# =====================================================
# 7. GQA — 5 configs
# =====================================================
@pytest.mark.parametrize("H_q,H_kv,seq,D", [
    (32, 8, 512, 64),
    (14, 2, 512, 64),
    (32, 1, 256, 64),
    (16, 8, 1024, 128),
    (32, 8, 256, 128),
])
def test_gqa(H_q, H_kv, seq, D):
    Q = torch.randn(1, H_q, seq, D, device=device, dtype=dtype)
    K = torch.randn(1, H_kv, seq, D, device=device, dtype=dtype)
    V = torch.randn(1, H_kv, seq, D, device=device, dtype=dtype)
    ref = _gqa_ref(Q, K, V)
    out = F.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
    _check(f"GQA {H_q}q/{H_kv}kv seq={seq} D={D}", out, ref)


# =====================================================
# 8. GQA + causal — 1 config
# =====================================================
def test_gqa_causal():
    Q = torch.randn(1, 32, 512, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    ref = _gqa_ref(Q, K, V, is_causal=True)
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)
    _check("GQA 32q/8kv + causal", out, ref)


# =====================================================
# 9. GQA + bool mask — 1 config
# =====================================================
def test_gqa_bool_mask():
    Q = torch.randn(1, 32, 256, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    mask = torch.ones(256, 256, device=device, dtype=torch.bool).tril()
    ref = _gqa_ref(Q, K, V, attn_mask=mask)
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, enable_gqa=True)
    _check("GQA 32q/8kv + bool_mask", out, ref)


# =====================================================
# 10. GQA + cross-attention — 1 config
# =====================================================
def test_gqa_cross_attention():
    Q = torch.randn(1, 32, 128, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    ref = _gqa_ref(Q, K, V)
    out = F.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
    _check("GQA cross-attn 128q/512kv", out, ref)


# =====================================================
# 11. BF16 — 1 config (wider tolerance: BF16->FP16 conversion)
# =====================================================
def test_bf16():
    Q = torch.randn(1, 16, 512, 64, device=device, dtype=torch.bfloat16)
    K = torch.randn(1, 16, 512, 64, device=device, dtype=torch.bfloat16)
    V = torch.randn(1, 16, 512, 64, device=device, dtype=torch.bfloat16)
    ref = original(Q, K, V)
    out = F.scaled_dot_product_attention(Q, K, V)
    _check("BF16 seq=512", out, ref, tol=0.10)


# =====================================================
# 12. Fully masked row — zeros, no NaN — 1 config
# =====================================================
def test_masked_row_no_nan():
    Q = torch.randn(1, 4, 8, 64, device=device, dtype=dtype)
    K = torch.randn(1, 4, 8, 64, device=device, dtype=dtype)
    V = torch.randn(1, 4, 8, 64, device=device, dtype=dtype)
    mask = torch.ones(8, 8, device=device, dtype=torch.bool)
    mask[3, :] = False
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    assert not torch.isnan(out).any().item(), "NaN in masked output"
    assert out[:, :, 3, :].abs().max().item() < 0.01, "Fully masked row not zero"


# =====================================================
# 13. Custom scale — 1 config
# =====================================================
def test_custom_scale():
    Q = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    ref = original(Q, K, V, scale=0.05)
    out = F.scaled_dot_product_attention(Q, K, V, scale=0.05)
    _check("custom scale=0.05", out, ref)


# =====================================================
# 14. Decode path (seq_q=1) — 3 + 1 configs
# =====================================================
@pytest.mark.parametrize("skv", [256, 2048, 8192])
def test_decode(skv):
    Q = torch.randn(1, 16, 1, 64, device=device, dtype=dtype)
    K = torch.randn(1, 16, skv, 64, device=device, dtype=dtype)
    V = torch.randn(1, 16, skv, 64, device=device, dtype=dtype)
    ref = original(Q, K, V)
    out = F.scaled_dot_product_attention(Q, K, V)
    _check(f"decode seq_kv={skv}", out, ref)


def test_gqa_decode():
    Q = torch.randn(1, 14, 1, 64, device=device, dtype=dtype)
    K = torch.randn(1, 2, 2048, 64, device=device, dtype=dtype)
    V = torch.randn(1, 2, 2048, 64, device=device, dtype=dtype)
    ref = _gqa_ref(Q, K, V)
    out = F.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
    _check("GQA decode 14q/2kv kv=2048", out, ref)
