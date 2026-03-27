"""
noflash-attention GPU detection

Determines whether the current GPU needs the SDPA patch.

Detection strategy (in order):
  1. Environment variable overrides (NOFLASH_FORCE_PATCH / NOFLASH_DISABLE)
  2. Fast-path: known AMD architectures without fused attention
  3. Fast-path: NVIDIA compute capability < 80 (pre-Ampere, no Flash SDP)
  4. Runtime test: actually probe whether PyTorch has an efficient SDPA backend

The runtime test is the ground truth — it asks PyTorch directly. The fast paths
exist to avoid the ~10ms CUDA initialization cost for known hardware.

Results are cached: detection runs once per process.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("noflash-attn")

# ───────────────────── Known architectures ─────────────────────

# AMD GPUs with confirmed fused attention (CK / AOTriton).
# Everything NOT in this set is assumed to need the patch.
_AMD_HAS_FUSED = frozenset({
    # CDNA (all have MFMA → CK fused attention)
    "gfx908",   # MI100
    "gfx90a",   # MI210, MI250, MI250X
    "gfx940",   # MI300A
    "gfx941",   # MI300X
    "gfx942",   # MI300X (variant)
    # RDNA 3+ (CK fused attention support)
    "gfx1100",  # RX 7900 XTX / 7900 XT
    "gfx1101",  # RX 7800 XT / 7700 XT
    "gfx1102",  # RX 7600
    "gfx1103",  # Phoenix APU
    "gfx1150",  # Strix Point APU
    "gfx1151",  # Strix Point (variant)
    # RDNA 4
    "gfx1200",  # RX 9070 XT
    "gfx1201",  # RX 9070
})

# Minimum NVIDIA SM version for PyTorch's built-in Flash SDP.
# SM 80 = Ampere (A100, RTX 3090). SM 75 = Turing (RTX 2080, T4).
# Flash Attention (Dao AI) supports SM 75+, but PyTorch's built-in
# Flash SDP requires SM 80+. We use 80 as the threshold.
_NVIDIA_MIN_SM_FOR_FLASH = 80


# ───────────────────── Detection logic ─────────────────────

_cached_result = None


def should_activate() -> bool:
    """
    Determine whether the SDPA patch should be activated.

    Returns True if the current GPU lacks efficient SDPA and would benefit
    from noflash-attention. Results are cached per process.
    """
    global _cached_result
    if _cached_result is not None:
        return _cached_result

    _cached_result = _detect()
    return _cached_result


def _detect() -> bool:
    """Core detection logic. Called once, result cached."""

    # ── 1. Environment overrides ──
    if os.environ.get("NOFLASH_FORCE_PATCH", "").strip() not in ("", "0"):
        logger.info("Force-enabled via NOFLASH_FORCE_PATCH")
        return True

    if os.environ.get("NOFLASH_DISABLE", "").strip() not in ("", "0"):
        logger.info("Disabled via NOFLASH_DISABLE")
        return False

    # Legacy env var support (v0.x compatibility)
    if os.environ.get("GFX906_FORCE_PATCH", "").strip() not in ("", "0"):
        logger.info("Force-enabled via GFX906_FORCE_PATCH (legacy)")
        return True

    # ── 2. Check CUDA availability ──
    try:
        import torch
    except ImportError:
        logger.debug("PyTorch not available")
        return False

    if not torch.cuda.is_available():
        logger.debug("CUDA not available")
        return False

    # ── 3. Fast path: AMD architecture check ──
    arch = _get_amd_arch()
    if arch is not None:
        if arch in _AMD_HAS_FUSED:
            logger.info("AMD %s has fused attention — skipping patch", arch)
            return False
        else:
            logger.info("AMD %s has no fused attention — activating patch", arch)
            return True

    # ── 4. Fast path: NVIDIA compute capability check ──
    sm = _get_nvidia_sm()
    if sm is not None:
        if sm < _NVIDIA_MIN_SM_FOR_FLASH:
            logger.info("NVIDIA SM %d < %d — activating patch",
                        sm, _NVIDIA_MIN_SM_FOR_FLASH)
            return True
        else:
            # SM 80+: likely has Flash SDP, but verify at runtime
            # (could be a custom PyTorch build without flash support)
            pass

    # ── 5. Runtime test: probe SDPA backends ──
    has_efficient = _probe_efficient_sdpa()
    if has_efficient:
        logger.info("Efficient SDPA backend detected — skipping patch")
        return False
    else:
        logger.info("No efficient SDPA backend — activating patch")
        return True


def _get_amd_arch() -> Optional[str]:
    """Return the AMD GCN architecture string (e.g. 'gfx906'), or None."""
    import torch
    try:
        props = torch.cuda.get_device_properties(0)
        gcn = getattr(props, "gcnArchName", "")
        if not gcn:
            return None
        # Handle format like "gfx906:sramecc+:xnack-"
        return gcn.split(":")[0].strip()
    except Exception:
        return None


def _get_nvidia_sm() -> Optional[int]:
    """Return the NVIDIA SM version (e.g. 80 for A100), or None."""
    import torch
    try:
        props = torch.cuda.get_device_properties(0)
        # AMD GPUs have gcnArchName; NVIDIA GPUs don't
        if getattr(props, "gcnArchName", ""):
            return None
        major = props.major
        minor = props.minor
        if major == 0:
            return None  # Not a real GPU
        return major * 10 + minor
    except Exception:
        return None


def _probe_efficient_sdpa() -> bool:
    """
    Runtime probe: can PyTorch use an efficient (non-Math) SDPA backend?

    Creates tiny test tensors and attempts SDPA with the Math backend disabled.
    If any efficient backend (Flash, Memory-Efficient, CuDNN) succeeds, returns True.

    Cost: ~10ms (CUDA kernel launch + sync). Runs once per process.
    """
    import torch
    import torch.nn.functional as F

    try:
        # Tiny test: 1 batch, 1 head, 256 tokens, 64 dim = ~96 KB total
        Q = torch.randn(1, 1, 256, 64, device="cuda", dtype=torch.float16)
        K = torch.randn(1, 1, 256, 64, device="cuda", dtype=torch.float16)
        V = torch.randn(1, 1, 256, 64, device="cuda", dtype=torch.float16)

        # Try disabling Math backend — if an efficient backend exists, it will be used
        ctx = _sdp_context_no_math()
        if ctx is None:
            # Can't create context (very old PyTorch?) — assume no efficient backend
            return False

        with ctx:
            try:
                F.scaled_dot_product_attention(Q, K, V)
                return True
            except RuntimeError:
                # No efficient backend available for this GPU
                return False

    except torch.cuda.OutOfMemoryError:
        # GPU fully loaded — fall back to architecture-based detection
        # (should never happen with 96KB tensors, but be safe)
        logger.warning("VRAM exhausted during SDPA probe — falling back to architecture detection")
        return False

    except Exception as e:
        logger.debug("SDPA probe failed: %s — assuming no efficient backend", e)
        return False

    finally:
        try:
            del Q, K, V
            torch.cuda.empty_cache()
        except Exception:
            pass


def _sdp_context_no_math():
    """
    Create a context manager that disables the Math SDPA backend.

    Handles API differences across PyTorch versions:
      - PyTorch 2.4+:  torch.nn.attention.sdpa_kernel([FLASH, EFFICIENT, CUDNN])
      - PyTorch 2.0-2.3: torch.backends.cuda.sdp_kernel(enable_math=False)
    """
    import torch

    # PyTorch 2.4+ API
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        # CuDNN backend available in some builds
        if hasattr(SDPBackend, "CUDNN_ATTENTION"):
            backends.append(SDPBackend.CUDNN_ATTENTION)
        return sdpa_kernel(backends)
    except (ImportError, AttributeError):
        pass

    # PyTorch 2.0-2.3 API
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=True,
            enable_math=False,
        )
    except (AttributeError, TypeError):
        pass

    return None


# ───────────────────── Public helpers ─────────────────────

def get_gpu_info() -> dict:
    """
    Return a dict describing the current GPU and detection result.
    Useful for debugging and issue reports.
    """
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "patch_needed": should_activate(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["total_vram_mb"] = props.total_memory // (1024 * 1024)

        amd_arch = _get_amd_arch()
        if amd_arch:
            info["vendor"] = "AMD"
            info["arch"] = amd_arch
            info["has_fused_attn"] = amd_arch in _AMD_HAS_FUSED
        else:
            nvidia_sm = _get_nvidia_sm()
            if nvidia_sm:
                info["vendor"] = "NVIDIA"
                info["sm"] = nvidia_sm
                info["has_flash_sdp"] = nvidia_sm >= _NVIDIA_MIN_SM_FOR_FLASH
            else:
                info["vendor"] = "Unknown"

        info["efficient_sdpa_probe"] = _probe_efficient_sdpa()

    return info
