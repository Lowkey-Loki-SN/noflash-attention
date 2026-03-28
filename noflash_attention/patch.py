"""
SDPA Patch for noflash-attention

Three-tier chunked attention with online softmax and dynamic OOM fallback:
  Tier 1: Standard chunked (PyTorch softmax + Tensile GEMMs — fastest, most memory)
  Tier 2: Online chunked (K-tiled streaming softmax — moderate speed, much less memory)
  Tier 3: In-place chunked (manual FP16 softmax — slowest, least memory)

Supports: GQA, MQA, standard MHA, attn_mask (bool/float), is_causal, BF16 input.
Forward-compatible with future PyTorch SDPA API changes via **kwargs.
"""
import torch
import torch.nn.functional as F
import math
import os

from noflash_attention.detect_gpu import should_activate

__version__ = "1.0.1"

# Support both new and legacy env var names (backward compat)
_threshold_str = os.environ.get("NOFLASH_THRESHOLD",
                                os.environ.get("GFX906_FLASH_THRESHOLD", "0"))
FLASH_THRESHOLD = int(_threshold_str)

SOFTMAX_FTZ_THRESHOLD = -20.0  # exp(-20) ≈ 2e-9, below FP16 denormal range. Inspired by llama.cpp.
_fp32_warned = False

# Guard against double-import: don't save our own function as "original"
_PATCH_MARKER = "_noflash_attention"
if hasattr(F.scaled_dot_product_attention, _PATCH_MARKER):
    _original_sdpa = F.scaled_dot_product_attention._original
else:
    _original_sdpa = F.scaled_dot_product_attention


def _auto_block_m(B, N_kv, num_heads_q, dtype, budget_factor=2):
    """Select block_m to keep per-chunk buffers under ~1.5GB.
    budget_factor=2 for standard (S+P), =1 for in-place/online."""
    elem_bytes = 2 if dtype == torch.float16 else 4
    budget = 1.5 * 1024**3
    denom = budget_factor * B * num_heads_q * N_kv * elem_bytes
    if denom == 0:
        return 128
    max_bm = int(budget // denom)
    for bm in [4096, 2048, 1024, 512, 256, 128]:
        if bm <= max_bm:
            return bm
    return 128


def _auto_block_n(B, block_m, num_heads_q, dtype, budget_mb=256):
    """Select block_n for online softmax K-tiling. Keeps P_ij (FP32) under budget."""
    budget = budget_mb * 1024**2
    denom = B * num_heads_q * block_m * 4  # FP32 for P_ij
    if denom == 0:
        return 256
    max_bn = int(budget // denom)
    for bn in [1024, 512, 256, 128, 64]:
        if bn <= max_bn:
            return bn
    return 64


def _apply_mask(S, attn_mask, N_q, i, end, is_causal, N_kv):
    """Apply attention mask and/or causal mask to score matrix S (in-place).
    Used by Tier 1 and Tier 3 which see the full K dimension."""
    if attn_mask is not None:
        m = attn_mask
        if m.ndim >= 2 and m.shape[-2] == N_q:
            m = m[..., i:end, :]
        if m.dtype == torch.bool:
            S.masked_fill_(~m, float("-inf"))
        else:
            S += m.to(S.dtype)
    if is_causal:
        rows = torch.arange(i, end, device=S.device).unsqueeze(1)
        cols = torch.arange(N_kv, device=S.device).unsqueeze(0)
        S.masked_fill_(rows < cols, float("-inf"))


def _apply_mask_block(S, attn_mask, N_q, i, end_q, j, end_k, is_causal):
    """Apply attention mask for a K-block [j:end_k] in online softmax (Tier 2).
    Slices both Q and K dimensions from the full mask."""
    if attn_mask is not None:
        m = attn_mask
        if m.ndim >= 2 and m.shape[-2] == N_q:
            m = m[..., i:end_q, j:end_k]
        else:
            m = m[..., j:end_k]
        if m.dtype == torch.bool:
            S.masked_fill_(~m, float("-inf"))
        else:
            S += m.to(S.dtype)
    if is_causal:
        rows = torch.arange(i, end_q, device=S.device).unsqueeze(1)
        cols = torch.arange(j, end_k, device=S.device).unsqueeze(0)
        S.masked_fill_(rows < cols, float("-inf"))


def _qk_matmul(q_chunk, K_t, H_q, H_kv, B, chunk_len, N_kv):
    """QK^T matmul with adaptive GQA. Returns (B, H_q, chunk_len, N_kv).
    Uses flattened GEMM when G*chunk_len is large enough to benefit from reduced
    batch overhead; falls back to broadcast for smaller sizes where batched
    GEMMs are more efficient with fewer, larger matmuls."""
    if H_kv == H_q:
        return torch.matmul(q_chunk, K_t)
    G = H_q // H_kv
    q_cont = q_chunk.contiguous() if not q_chunk.is_contiguous() else q_chunk
    if G * chunk_len >= 16384:
        # Flat: one large GEMM [B, H_kv, G*chunk_len, D] × [B, H_kv, D, N_kv]
        q_flat = q_cont.view(B, H_kv, G * chunk_len, -1)
        S = torch.matmul(q_flat, K_t)
        return S.view(B, H_kv, G, chunk_len, -1).reshape(B, H_q, chunk_len, -1)
    else:
        # Broadcast: G batched GEMMs [B, H_kv, G, chunk_len, D] × [B, H_kv, 1, D, N_kv]
        q_grouped = q_cont.view(B, H_kv, G, chunk_len, -1)
        S = torch.matmul(q_grouped, K_t.unsqueeze(2))
        return S.reshape(B, H_q, chunk_len, N_kv)


def _pv_matmul(P, V, H_q, H_kv, B, chunk_len, D):
    """PV matmul with adaptive GQA. Returns (B, H_q, chunk_len, D).
    Same adaptive strategy as _qk_matmul."""
    if H_kv == H_q:
        return torch.matmul(P, V)
    G = H_q // H_kv
    P_cont = P.contiguous() if not P.is_contiguous() else P
    if G * chunk_len >= 16384:
        P_flat = P_cont.view(B, H_kv, G * chunk_len, -1)
        O = torch.matmul(P_flat, V)
        return O.view(B, H_kv, G, chunk_len, D).reshape(B, H_q, chunk_len, D)
    else:
        P_grouped = P_cont.view(B, H_kv, G, chunk_len, -1)
        O = torch.matmul(P_grouped, V.unsqueeze(2))
        return O.reshape(B, H_q, chunk_len, D)


# ---------------------------------------------------------------------------
# Tier 1: Standard chunked attention
# ---------------------------------------------------------------------------
def chunked_sdpa(Q, K, V, scale=None, block_m=1024, attn_mask=None, is_causal=False):
    """Tier 1: Standard chunked attention. PyTorch softmax (fastest).
    Flattened GQA/MQA via single large GEMM (zero extra memory for K/V)."""
    B, H_q, N_q, D = Q.shape
    H_kv = K.shape[1]
    N_kv = K.shape[2]
    has_mask = attn_mask is not None or is_causal
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)
    K_t = K.transpose(-2, -1)

    for i in range(0, N_q, block_m):
        end = min(i + block_m, N_q)
        chunk_len = end - i
        q_chunk = Q[:, :, i:end, :] * scale
        S = _qk_matmul(q_chunk, K_t, H_q, H_kv, B, chunk_len, N_kv)
        if has_mask:
            _apply_mask(S, attn_mask, N_q, i, end, is_causal, N_kv)
        P = torch.softmax(S, dim=-1)
        del S
        if has_mask:
            P = torch.nan_to_num(P)
        O[:, :, i:end, :] = _pv_matmul(P, V, H_q, H_kv, B, chunk_len, D)
        del P

    return O


# ---------------------------------------------------------------------------
# Tier 2: Online chunked attention (K-tiled streaming softmax)
# ---------------------------------------------------------------------------
def chunked_sdpa_online(Q, K, V, scale=None, block_m=1024, block_n=256,
                        attn_mask=None, is_causal=False):
    """Tier 2: Online chunked attention with K-dimension tiling.
    O(block_m × block_n) peak memory via streaming softmax with running max/sum.
    FP32 accumulation, flattened GQA, FTZ numerical hardening.
    Inspired by FlashAttention-2 (Dao et al.) and llama.cpp online softmax.

    FA2 optimizations applied:
    - Causal early termination: K iteration bounded to min(N_kv, end_q) (avoids loop overhead)
    - Mask/no-mask separation: skip mask ops for K-blocks fully below causal diagonal"""
    B, H_q, N_q, D = Q.shape
    H_kv = K.shape[1]
    N_kv = K.shape[2]
    has_attn_mask = attn_mask is not None
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)

    for i in range(0, N_q, block_m):
        end_q = min(i + block_m, N_q)
        chunk_len = end_q - i
        q_chunk = Q[:, :, i:end_q, :] * scale

        # Running accumulators (FP32 for numerical precision across many K-blocks)
        m_i = torch.full((B, H_q, chunk_len, 1), float('-inf'),
                         device=Q.device, dtype=torch.float32)
        l_i = torch.zeros((B, H_q, chunk_len, 1),
                          device=Q.device, dtype=torch.float32)
        o_i = torch.zeros((B, H_q, chunk_len, D),
                          device=Q.device, dtype=torch.float32)

        # FA2-style causal early termination: don't iterate K blocks past the diagonal.
        # For causal, query positions i..end_q-1 can only attend to key positions < end_q.
        kv_end = min(N_kv, end_q) if is_causal else N_kv

        for j in range(0, kv_end, block_n):
            end_k = min(j + block_n, kv_end)
            bn = end_k - j

            K_block_t = K[:, :, j:end_k, :].transpose(-2, -1)
            V_block = V[:, :, j:end_k, :]

            # S_ij: [B, H_q, chunk_len, bn]
            S_ij = _qk_matmul(q_chunk, K_block_t, H_q, H_kv, B, chunk_len, bn)

            # FA2-style mask/no-mask separation: only apply causal mask for the
            # boundary block (where the diagonal crosses). Blocks fully below the
            # diagonal need no causal masking — all positions are valid.
            block_needs_causal = is_causal and end_k > i  # diagonal crosses this block
            if has_attn_mask or block_needs_causal:
                _apply_mask_block(S_ij, attn_mask, N_q, i, end_q, j, end_k,
                                  is_causal=block_needs_causal)

            # --- Online softmax ---
            m_ij = S_ij.amax(dim=-1, keepdim=True).float()  # [B, H_q, chunk_len, 1]
            m_new = torch.maximum(m_i, m_ij)

            # Rescale old accumulators by exp(m_old - m_new)
            correction = torch.exp(m_i - m_new)
            correction.nan_to_num_(nan=0.0)  # handles -inf - (-inf) → NaN → 0
            o_i *= correction
            l_i *= correction

            # Compute globally-correct attention weights: P_ij = exp(S_ij - m_new)
            # FP32 for precision, FTZ to prevent denormals
            P_ij = S_ij.float()
            del S_ij
            P_ij -= m_new
            P_ij.clamp_(min=SOFTMAX_FTZ_THRESHOLD)
            P_ij.exp_()
            P_ij.nan_to_num_(nan=0.0)  # masked -inf scores → 0

            # Accumulate
            l_i += P_ij.sum(dim=-1, keepdim=True)
            o_i += _pv_matmul(P_ij.to(Q.dtype), V_block, H_q, H_kv, B, chunk_len, D).float()
            del P_ij

            m_i = m_new

        # Final normalization: O = o_i / l_i
        l_safe = torch.where(l_i == 0, torch.ones_like(l_i), l_i)  # avoid 0/0
        O[:, :, i:end_q, :] = (o_i / l_safe).to(Q.dtype)
        del m_i, l_i, o_i

    return O


# ---------------------------------------------------------------------------
# Tier 3: In-place chunked attention
# ---------------------------------------------------------------------------
def chunked_sdpa_inplace(Q, K, V, scale=None, block_m=1024, attn_mask=None, is_causal=False):
    """Tier 3: In-place chunked attention. Manual FP16 softmax — no P allocation.
    Inference only (in-place ops incompatible with autograd).
    Flattened GQA, FTZ numerical hardening."""
    B, H_q, N_q, D = Q.shape
    H_kv = K.shape[1]
    N_kv = K.shape[2]
    has_mask = attn_mask is not None or is_causal
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)
    K_t = K.transpose(-2, -1)

    for i in range(0, N_q, block_m):
        end = min(i + block_m, N_q)
        chunk_len = end - i
        q_chunk = Q[:, :, i:end, :] * scale
        S = _qk_matmul(q_chunk, K_t, H_q, H_kv, B, chunk_len, N_kv)
        if has_mask:
            _apply_mask(S, attn_mask, N_q, i, end, is_causal, N_kv)
        S.sub_(S.max(dim=-1, keepdim=True)[0])
        S.clamp_(min=SOFTMAX_FTZ_THRESHOLD)
        S.exp_()
        S.div_(S.sum(dim=-1, keepdim=True))
        if has_mask:
            S.nan_to_num_(0.0)
        O[:, :, i:end, :] = _pv_matmul(S, V, H_q, H_kv, B, chunk_len, D)
        del S

    return O


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False, **kwargs):
    seq_kv = key.shape[-2]
    H_q = query.shape[1] if query.ndim == 4 else 0
    H_kv = key.shape[1] if key.ndim == 4 else 0
    gqa_needed = H_q != H_kv and H_kv > 0

    use_chunked = (
        query.is_cuda
        and query.ndim == 4
        and not getattr(query, "is_nested", False)
        and dropout_p == 0.0
        and seq_kv > FLASH_THRESHOLD
        and (not gqa_needed or enable_gqa)
        and (not gqa_needed or (H_q % H_kv == 0))
    )

    if use_chunked:
        global _fp32_warned
        orig_dtype = query.dtype
        if orig_dtype == torch.bfloat16:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)
            if attn_mask is not None and attn_mask.dtype == torch.bfloat16:
                attn_mask = attn_mask.to(torch.float16)
        elif orig_dtype == torch.float32 and not _fp32_warned:
            import warnings
            warnings.warn(
                "[noflash-attn] FP32 input detected. Consider using dtype=torch.float16 "
                "for 2x faster attention. Proceeding with FP32.",
                stacklevel=3,
            )
            _fp32_warned = True

        B = query.shape[0]
        seq_q = query.shape[2]

        # Disable autocast inside our path — we manage dtypes explicitly.
        # Without this, autocast recasts our FP16 tensors to BF16 inside matmul,
        # which crashes on GPUs without native BF16 hardware.
        with torch.amp.autocast(device_type="cuda", enabled=False):

            # --- Fast path: decode / tiny seq_q → Math SDPA (fused C++ kernel) ---
            s_bytes = B * H_q * seq_q * seq_kv * 4  # FP32 in Math SDPA
            if seq_q <= 8 and s_bytes < 256 * 1024 * 1024:
                try:
                    out = _original_sdpa(query, key, value, attn_mask=attn_mask,
                                         dropout_p=0.0, is_causal=is_causal,
                                         scale=scale, enable_gqa=enable_gqa)
                    return out.to(orig_dtype) if orig_dtype != query.dtype else out
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()

            # --- Tier 1: Standard chunked (fast, most memory) ---
            block_m = _auto_block_m(B, seq_kv, H_q, query.dtype, budget_factor=2)
            for bm in [block_m, block_m // 2, block_m // 4, 128]:
                if bm < 64:
                    break
                try:
                    out = chunked_sdpa(query, key, value, scale, bm,
                                       attn_mask=attn_mask, is_causal=is_causal)
                    return out.to(orig_dtype) if orig_dtype != query.dtype else out
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()

            # --- Tier 2: Online chunked (K-tiled, much less memory) ---
            block_m_ol = _auto_block_m(B, seq_kv, H_q, query.dtype, budget_factor=1)
            block_n = _auto_block_n(B, block_m_ol, H_q, query.dtype)
            for bm in [block_m_ol, block_m_ol // 2, 128]:
                if bm < 64:
                    break
                try:
                    out = chunked_sdpa_online(query, key, value, scale, bm, block_n,
                                              attn_mask=attn_mask, is_causal=is_causal)
                    return out.to(orig_dtype) if orig_dtype != query.dtype else out
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()

            # --- Tier 3: In-place chunked (slowest, least memory, inference only) ---
            requires_grad = query.requires_grad or key.requires_grad or value.requires_grad
            if not requires_grad:
                block_m_ip = _auto_block_m(B, seq_kv, H_q, query.dtype, budget_factor=1)
                for bm in [block_m_ip, block_m_ip // 2, block_m_ip // 4, 128]:
                    if bm < 64:
                        break
                    try:
                        out = chunked_sdpa_inplace(query, key, value, scale, bm,
                                                   attn_mask=attn_mask, is_causal=is_causal)
                        return out.to(orig_dtype) if orig_dtype != query.dtype else out
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()

        raise torch.cuda.OutOfMemoryError("All attention paths exhausted")

    else:
        return _original_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, enable_gqa=enable_gqa, **kwargs)


def disable():
    """Restore original SDPA."""
    F.scaled_dot_product_attention = _original_sdpa
    print("[noflash-attn] Patch disabled, original SDPA restored")


def enable():
    """Re-enable the patch after disable()."""
    F.scaled_dot_product_attention = patched_sdpa
    print("[noflash-attn] Patch re-enabled")


def is_enabled():
    """Check if the patch is currently active."""
    return F.scaled_dot_product_attention is patched_sdpa


# Mark our function so we can detect double-import and other patches can detect us
patched_sdpa._noflash_attention = True
patched_sdpa._original = _original_sdpa

# Auto-activate based on GPU detection
if should_activate():
    F.scaled_dot_product_attention = patched_sdpa
    print("[noflash-attn] v%s — SDPA patched (three-tier chunked attention)" % __version__)
else:
    print("[noflash-attn] v%s — Efficient SDPA available, patch not applied. Set NOFLASH_FORCE_PATCH=1 to override." % __version__)
