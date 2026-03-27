"""
noflash-attention -- Flash-attention-class memory efficiency for GPUs without flash attention

Two complementary optimizations for PyTorch transformers:

  1. SDPA Patch (auto-activates on import):
     Replaces torch.nn.functional.scaled_dot_product_attention with
     three-tier chunked attention. Reduces O(N^2) attention memory to O(N).

  2. FFN Chunking (opt-in via patch_ffn):
     Wraps feedforward modules to process tokens in chunks. Reduces
     O(N * D_expansion) intermediate memory. Runtime-verified for safety.

Usage:
    import noflash_attention                    # auto-patches SDPA

    handle = noflash_attention.patch_ffn(model)  # opt-in FFN chunking
    handle.remove()                              # restore originals

    noflash_attention.disable()                  # restore original SDPA
    noflash_attention.enable()                   # re-enable SDPA patch
"""

from noflash_attention.patch import (
    __version__,
    enable,
    disable,
    is_enabled,
    patched_sdpa,
    chunked_sdpa,
    chunked_sdpa_online,
    chunked_sdpa_inplace,
    FLASH_THRESHOLD,
    SOFTMAX_FTZ_THRESHOLD,
    _original_sdpa,
)

from noflash_attention.ffn import (
    patch_ffn,
    FFNPatchHandle,
    ChunkedFFNWrapper,
)

__all__ = [
    # v10 SDPA (unchanged)
    "__version__",
    "enable",
    "disable",
    "is_enabled",
    "patched_sdpa",
    "chunked_sdpa",
    "chunked_sdpa_online",
    "chunked_sdpa_inplace",
    "FLASH_THRESHOLD",
    "SOFTMAX_FTZ_THRESHOLD",
    "_original_sdpa",
    # v11 FFN chunking
    "patch_ffn",
    "FFNPatchHandle",
    "ChunkedFFNWrapper",
]
