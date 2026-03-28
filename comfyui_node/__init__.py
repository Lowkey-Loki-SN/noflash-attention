"""
ComfyUI custom node: noflash-attention

Place this folder (or symlink it) into ComfyUI/custom_nodes/.
Requires: pip install -e /path/to/noflash-attention
"""
import sys

if sys.version_info[:2] >= (3, 10):
    try:
        import noflash_attention
        import os
        os.environ.setdefault("NOFLASH_THRESHOLD", "0")
    except ImportError:
        print("[noflash-attn] Package not installed. Run: pip install -e /path/to/noflash-attention")
else:
    print("[noflash-attn] Skipping — Python < 3.12")


class NoFlashFFNChunking:
    """ComfyUI node for FFN chunking (optional memory optimization)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "num_chunks": ("INT", {"default": 0, "min": 0, "max": 32,
                                       "tooltip": "0 = adaptive based on VRAM"}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "noflash"

    def apply(self, model, num_chunks=0, enabled=True):
        if not enabled:
            return (model,)
        try:
            import noflash_attention
            m = model.clone()
            real_model = m.model
            handle = noflash_attention.patch_ffn(real_model, num_chunks=num_chunks, verbose=True)
            print(f"[noflash-ffn] Patched {handle.num_wrapped} FFN modules")
            return (m,)
        except Exception as e:
            print(f"[noflash-ffn] FFN chunking failed: {e}")
            return (model,)


# Backward compat: old class name still works
GFX906FFNChunking = NoFlashFFNChunking

NODE_CLASS_MAPPINGS = {
    "NoFlashFFNChunking": NoFlashFFNChunking,
    "GFX906FFNChunking": GFX906FFNChunking,  # backward compat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoFlashFFNChunking": "NoFlash FFN Chunking",
    "GFX906FFNChunking": "GFX906 FFN Chunking (legacy)",
}
