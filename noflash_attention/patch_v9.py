"""
SDPA Patch v9 for AMD MI50 (gfx906)

Two-tier chunked attention with dynamic OOM fallback:
  Tier 1: Standard chunked (PyTorch softmax - fastest, ~2x S+P memory per chunk)
  Tier 2: In-place chunked (manual softmax - 50% less memory, inference only)

Supports: GQA, MQA, standard MHA, attn_mask (bool/float), is_causal, BF16 input.
Forward-compatible with future PyTorch SDPA API changes via **kwargs.

v9 fixes from v8:
- **kwargs for forward compatibility (PyTorch API changes won't crash us)
- enable_gqa validation (only broadcast when explicitly enabled)
- Non-contiguous tensor guards for GQA view() operations
- Tier 2 skipped when tensors require grad (in-place breaks autograd)
- NaN guard for fully-masked rows (masked softmax 0/0 = NaN -> 0)
- Batch size accounted for in auto_block_m
- torch.autocast protection (prevents BF16 recast inside chunked path)
- disable()/enable() for runtime toggling without restart
- Smart fast path: small attention uses Math SDPA (faster fused C++ kernel)
- __version__ and is_enabled() for introspection
"""
import torch
import torch.nn.functional as F
import math
import os

__version__ = "0.9.2"
FLASH_THRESHOLD = int(os.environ.get("GFX906_FLASH_THRESHOLD", "0"))
_fp32_warned = False

# Guard against double-import: don't save our own function as "original"
_PATCH_MARKER = "_gfx906_chunked_attn"
if hasattr(F.scaled_dot_product_attention, _PATCH_MARKER):
    _original_sdpa = F.scaled_dot_product_attention._original
else:
    _original_sdpa = F.scaled_dot_product_attention


def _is_gfx906():
    """Check if the current GPU is gfx906 (MI50/MI60)."""
    if os.environ.get("GFX906_FORCE_PATCH"):
        return True
    if not torch.cuda.is_available():
        return False
    try:
        gcn = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        return "gfx906" in gcn
    except Exception:
        return False


def _auto_block_m(B, N_kv, num_heads_q, dtype, budget_factor=2):
    """Select block_m to keep per-chunk buffers under ~1.5GB.
    budget_factor=2 for standard (S+P), =1 for in-place (S only).
    Accounts for batch size B."""
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


def _apply_mask(S, attn_mask, N_q, i, end, is_causal, N_kv):
    """Apply attention mask and/or causal mask to score matrix S (in-place)."""
    if attn_mask is not None:
        m = attn_mask
        if m.ndim >= 3 and m.shape[-2] == N_q:
            m = m[..., i:end, :]
        elif m.ndim == 4 and m.shape[-2] == N_q:
            m = m[:, :, i:end, :]
        if m.dtype == torch.bool:
            S.masked_fill_(~m, float("-inf"))
        else:
            S += m.to(S.dtype)
    if is_causal:
        rows = torch.arange(i, end, device=S.device).unsqueeze(1)
        cols = torch.arange(N_kv, device=S.device).unsqueeze(0)
        S.masked_fill_(rows < cols, float("-inf"))


def _qk_matmul(q_chunk, K_t, H_q, H_kv, B, chunk_len, N_kv):
    """QK^T matmul with GQA broadcast support. Returns (B, H_q, chunk_len, N_kv)."""
    if H_kv == H_q:
        return torch.matmul(q_chunk, K_t)
    G = H_q // H_kv
    q_cont = q_chunk.contiguous() if not q_chunk.is_contiguous() else q_chunk
    q_grouped = q_cont.view(B, H_kv, G, chunk_len, -1)
    S = torch.matmul(q_grouped, K_t.unsqueeze(2))
    return S.reshape(B, H_q, chunk_len, N_kv)


def _pv_matmul(P, V, H_q, H_kv, B, chunk_len, D):
    """PV matmul with GQA broadcast support. Returns (B, H_q, chunk_len, D)."""
    if H_kv == H_q:
        return torch.matmul(P, V)
    G = H_q // H_kv
    P_cont = P.contiguous() if not P.is_contiguous() else P
    P_grouped = P_cont.view(B, H_kv, G, chunk_len, -1)
    O = torch.matmul(P_grouped, V.unsqueeze(2))
    return O.reshape(B, H_q, chunk_len, D)


def chunked_sdpa(Q, K, V, scale=None, block_m=1024, attn_mask=None, is_causal=False):
    """Tier 1: Standard chunked attention. PyTorch softmax (fastest).
    Supports GQA/MQA via grouped broadcast (zero extra memory for K/V)."""
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


def chunked_sdpa_inplace(Q, K, V, scale=None, block_m=1024, attn_mask=None, is_causal=False):
    """Tier 2: In-place chunked attention. Manual FP16 softmax - no P allocation.
    Inference only (in-place ops incompatible with autograd)."""
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
        S.exp_()
        S.div_(S.sum(dim=-1, keepdim=True))
        if has_mask:
            S.nan_to_num_(0.0)
        O[:, :, i:end, :] = _pv_matmul(S, V, H_q, H_kv, B, chunk_len, D)
        del S

    return O


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
                "[gfx906-attn] FP32 input detected. Consider using dtype=torch.float16 "
                "for 2x faster attention on MI50. Proceeding with FP32.",
                stacklevel=3,
            )
            _fp32_warned = True

        B = query.shape[0]
        seq_q = query.shape[2]

        # Disable autocast inside our path — we manage dtypes explicitly.
        # Without this, autocast recasts our FP16 tensors to BF16 inside matmul,
        # which crashes on gfx906 (no BF16 hardware).
        with torch.amp.autocast(device_type="cuda", enabled=False):

            # --- Fast path: decode / tiny seq_q → Math SDPA (fused C++ kernel) ---
            # When seq_q is small, our Python overhead dominates. Math SDPA's
            # fused C++ kernel is faster for decode (seq_q=1) and small prefills.
            # We still get BF16→FP16 + autocast protection.
            # For larger seq_q (>= 32), our chunked approach wins via better
            # memory tiling even at moderate sequence lengths.
            s_bytes = B * H_q * seq_q * seq_kv * 4  # FP32 in Math SDPA
            if seq_q <= 8 and s_bytes < 256 * 1024 * 1024:
                try:
                    out = _original_sdpa(query, key, value, attn_mask=attn_mask,
                                         dropout_p=0.0, is_causal=is_causal,
                                         scale=scale, enable_gqa=enable_gqa)
                    return out.to(orig_dtype) if orig_dtype != query.dtype else out
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()

            # --- Tier 1: Standard chunked (fast, more memory) ---
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

            # --- Tier 2: In-place chunked (slower, 50% less memory, inference only) ---
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
    """Restore original SDPA. Call flash_attn_sdpa_patch.disable() to turn off."""
    F.scaled_dot_product_attention = _original_sdpa
    print("[gfx906-attn] Patch disabled, original SDPA restored")


def enable():
    """Re-enable the patch after disable()."""
    F.scaled_dot_product_attention = patched_sdpa
    print("[gfx906-attn] Patch re-enabled")


def is_enabled():
    """Check if the patch is currently active."""
    return F.scaled_dot_product_attention is patched_sdpa


# Mark our function so we can detect double-import and other patches can detect us
patched_sdpa._gfx906_chunked_attn = True
patched_sdpa._original = _original_sdpa

# Only activate on gfx906 (MI50/MI60) — on other GPUs, native chunked attention is faster
if _is_gfx906():
    F.scaled_dot_product_attention = patched_sdpa
    print("[gfx906-attn] v%s — SDPA patched for gfx906 (two-tier chunked attention)" % __version__)
else:
    print("[gfx906-attn] v%s — Not gfx906, patch not applied. Set GFX906_FORCE_PATCH=1 to override." % __version__)
