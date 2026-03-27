"""
Tests for v10 online softmax (Tier 2) — direct invocation to verify
the new K-tiled streaming attention produces correct results.
"""
import pytest
import torch
import torch.nn.functional as F
import math
import sys
from pathlib import Path

try:
    import noflash_attention
    from noflash_attention.patch import (
        chunked_sdpa, chunked_sdpa_online, chunked_sdpa_inplace,
        _qk_matmul, _pv_matmul, _original_sdpa, SOFTMAX_FTZ_THRESHOLD,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import noflash_attention
    from noflash_attention.patch import (
        chunked_sdpa, chunked_sdpa_online, chunked_sdpa_inplace,
        _qk_matmul, _pv_matmul, _original_sdpa, SOFTMAX_FTZ_THRESHOLD,
    )

device = "cuda"
dtype = torch.float16
TOLERANCE = 0.05


def _check(name, out, ref, tol=TOLERANCE):
    assert out.shape == ref.shape, f"{name}: shape {out.shape} vs {ref.shape}"
    assert not torch.isnan(out).any().item(), f"{name}: NaN in output"
    max_diff = (out - ref).abs().max().item()
    denom = ref.abs().mean().item()
    rel_err = max_diff / denom if denom > 0 else max_diff
    assert rel_err <= tol, f"{name}: rel={rel_err:.4f} > tol={tol:.4f}"


def _gqa_ref(Q, K, V, **kw):
    G = Q.shape[1] // K.shape[1]
    return _original_sdpa(Q, K.repeat_interleave(G, dim=1), V.repeat_interleave(G, dim=1), **kw)


# ==========================================================
# 1. Online vs Tier 1 — standard MHA, 6 configs
# ==========================================================
@pytest.mark.parametrize("B,H,seq,D,bm,bn", [
    (1, 8, 256, 64, 128, 64),
    (1, 16, 1024, 64, 256, 128),
    (1, 8, 512, 128, 256, 256),
    (1, 16, 4096, 64, 512, 256),
    (2, 4, 512, 64, 256, 128),
    (1, 8, 512, 80, 256, 128),
])
def test_online_vs_tier1(B, H, seq, D, bm, bn):
    Q = torch.randn(B, H, seq, D, device=device, dtype=dtype)
    K = torch.randn(B, H, seq, D, device=device, dtype=dtype)
    V = torch.randn(B, H, seq, D, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    ref = chunked_sdpa(Q, K, V, scale, block_m=bm)
    out = chunked_sdpa_online(Q, K, V, scale, block_m=bm, block_n=bn)
    _check(f"B={B} H={H} seq={seq} D={D} bm={bm} bn={bn}", out, ref)


# ==========================================================
# 2. Online vs Math SDPA — ground truth, 3 configs
# ==========================================================
@pytest.mark.parametrize("seq", [256, 1024, 4096])
def test_online_vs_math_sdpa(seq):
    Q = torch.randn(1, 16, seq, 64, device=device, dtype=dtype)
    K = torch.randn(1, 16, seq, 64, device=device, dtype=dtype)
    V = torch.randn(1, 16, seq, 64, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(64)
    ref = _original_sdpa(Q, K, V)
    out = chunked_sdpa_online(Q, K, V, scale, block_m=256, block_n=128)
    _check(f"vs Math SDPA seq={seq}", out, ref)


# ==========================================================
# 3. Online with causal mask — 3 configs
# ==========================================================
@pytest.mark.parametrize("seq,bm,bn", [
    (256, 128, 64), (1024, 256, 128), (512, 256, 256),
])
def test_online_causal(seq, bm, bn):
    Q = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(64)
    ref = _original_sdpa(Q, K, V, is_causal=True)
    out = chunked_sdpa_online(Q, K, V, scale, bm, bn, is_causal=True)
    _check(f"causal seq={seq} bm={bm} bn={bn}", out, ref)


# ==========================================================
# 4. Online with bool mask — 2 configs
# ==========================================================
@pytest.mark.parametrize("seq", [256, 1024])
def test_online_bool_mask(seq):
    Q = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, seq, 64, device=device, dtype=dtype)
    mask = torch.ones(seq, seq, device=device, dtype=torch.bool).tril()
    scale = 1.0 / math.sqrt(64)
    ref = _original_sdpa(Q, K, V, attn_mask=mask)
    out = chunked_sdpa_online(Q, K, V, scale, 256, 128, attn_mask=mask)
    _check(f"bool_mask seq={seq}", out, ref)


# ==========================================================
# 5. Online with float mask — 1 config
# ==========================================================
def test_online_float_mask():
    Q = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    bmask = torch.ones(512, 512, device=device, dtype=torch.bool).tril()
    fmask = torch.zeros(512, 512, device=device, dtype=dtype)
    fmask.masked_fill_(~bmask, float("-inf"))
    scale = 1.0 / math.sqrt(64)
    ref = _original_sdpa(Q, K, V, attn_mask=fmask)
    out = chunked_sdpa_online(Q, K, V, scale, 256, 128, attn_mask=fmask)
    _check("float_mask seq=512", out, ref)


# ==========================================================
# 6. Online with 4D mask — 1 config
# ==========================================================
def test_online_4d_mask():
    Q = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    mask4d = torch.zeros(1, 8, 256, 256, device=device, dtype=dtype)
    mask4d[:, :, :, 128:] = float("-inf")
    scale = 1.0 / math.sqrt(64)
    ref = _original_sdpa(Q, K, V, attn_mask=mask4d)
    out = chunked_sdpa_online(Q, K, V, scale, 128, 64, attn_mask=mask4d)
    _check("4d_mask (1,8,256,256)", out, ref)


# ==========================================================
# 7. Online with GQA — 4 configs
# ==========================================================
@pytest.mark.parametrize("H_q,H_kv,seq", [
    (32, 8, 512), (14, 2, 512), (32, 1, 256), (16, 8, 1024),
])
def test_online_gqa(H_q, H_kv, seq):
    Q = torch.randn(1, H_q, seq, 64, device=device, dtype=dtype)
    K = torch.randn(1, H_kv, seq, 64, device=device, dtype=dtype)
    V = torch.randn(1, H_kv, seq, 64, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(64)
    ref = _gqa_ref(Q, K, V)
    out = chunked_sdpa_online(Q, K, V, scale, 256, 128)
    _check(f"GQA {H_q}q/{H_kv}kv seq={seq}", out, ref)


# ==========================================================
# 8. Online with GQA + causal — 1 config
# ==========================================================
def test_online_gqa_causal():
    Q = torch.randn(1, 32, 512, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(64)
    ref = _gqa_ref(Q, K, V, is_causal=True)
    out = chunked_sdpa_online(Q, K, V, scale, 256, 128, is_causal=True)
    _check("GQA 32q/8kv + causal", out, ref)


# ==========================================================
# 9. Online: cross-attention — 3 configs
# ==========================================================
@pytest.mark.parametrize("sq,skv", [(64, 1024), (256, 4096), (512, 256)])
def test_online_cross_attention(sq, skv):
    Q = torch.randn(1, 8, sq, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, skv, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, skv, 64, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(64)
    ref = _original_sdpa(Q, K, V)
    out = chunked_sdpa_online(Q, K, V, scale, 128, 128)
    _check(f"cross sq={sq} skv={skv}", out, ref)


# ==========================================================
# 10. Fully masked row — no NaN — 1 config
# ==========================================================
def test_online_masked_row_no_nan():
    Q = torch.randn(1, 4, 8, 64, device=device, dtype=dtype)
    K = torch.randn(1, 4, 8, 64, device=device, dtype=dtype)
    V = torch.randn(1, 4, 8, 64, device=device, dtype=dtype)
    mask = torch.ones(8, 8, device=device, dtype=torch.bool)
    mask[3, :] = False
    scale = 1.0 / math.sqrt(64)
    out = chunked_sdpa_online(Q, K, V, scale, 4, 4, attn_mask=mask)
    assert not torch.isnan(out).any().item(), "NaN in fully-masked-row output"


# ==========================================================
# 11. block_n=1 (extreme K-tiling) — 1 config
# ==========================================================
def test_online_block_n_1():
    Q = torch.randn(1, 4, 16, 64, device=device, dtype=dtype)
    K = torch.randn(1, 4, 16, 64, device=device, dtype=dtype)
    V = torch.randn(1, 4, 16, 64, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(64)
    ref = _original_sdpa(Q, K, V)
    out = chunked_sdpa_online(Q, K, V, scale, 16, 1)
    _check("block_n=1 (extreme streaming)", out, ref, tol=0.10)


# ==========================================================
# 12. Flattened GQA matmuls — 2 configs
# ==========================================================
def test_qk_matmul_gqa():
    B, H_q, H_kv, seq_q, seq_kv, D = 1, 14, 2, 256, 512, 64
    Q = torch.randn(B, H_q, seq_q, D, device=device, dtype=dtype)
    K_t = torch.randn(B, H_kv, D, seq_kv, device=device, dtype=dtype)
    out = _qk_matmul(Q, K_t, H_q, H_kv, B, seq_q, seq_kv)
    G = H_q // H_kv
    ref = torch.matmul(Q, K_t.repeat_interleave(G, dim=1))
    _check("_qk_matmul 14q/2kv", out, ref, tol=0.001)


def test_pv_matmul_gqa():
    B, H_q, H_kv, seq_q, seq_kv, D = 1, 14, 2, 256, 512, 64
    V = torch.randn(B, H_kv, seq_kv, D, device=device, dtype=dtype)
    P = torch.randn(B, H_q, seq_q, seq_kv, device=device, dtype=dtype)
    out = _pv_matmul(P, V, H_q, H_kv, B, seq_q, D)
    G = H_q // H_kv
    ref = torch.matmul(P, V.repeat_interleave(G, dim=1))
    _check("_pv_matmul 14q/2kv", out, ref, tol=0.001)


# ==========================================================
# 13. Tier 3 (in-place) with FTZ — 1 config
# ==========================================================
def test_tier3_inplace_ftz():
    Q = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 512, 64, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(64)
    ref = _original_sdpa(Q, K, V)
    out = chunked_sdpa_inplace(Q, K, V, scale, 256)
    _check("inplace+FTZ seq=512", out, ref)
