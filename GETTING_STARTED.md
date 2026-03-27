# Getting Started with chunked-attn-gfx906

> Memory optimizer for AMD MI50/MI60 (gfx906). Makes impossible workloads possible.

## What this does

MI50 has **no fused attention** in any AMD library (CK, AOTriton, Flash Attention ROCm all require newer GPUs). PyTorch falls back to Math SDPA, which materializes the full N x N attention matrix. For video generation (seq > 10K tokens), this instantly exceeds 32 GB VRAM.

This package replaces Math SDPA with O(N) memory chunked attention. No custom ISA, no Triton, no MFMA — just HIP + Tensile.

| Without this package | With this package |
|---|---|
| Wan 2.2 5B 720p: **OOM** (needs 511 GB) | **21 GB** — completes in 1:20 |
| LTX-2.3 22B 720p: **OOM** | **28 GB** — completes in 17:08 |
| 1536x1536 images: 30.8 GB, 158s | **16.4 GB, 113s** (47% less VRAM, 29% faster) |

## Prerequisites

- AMD MI50 or MI60 (gfx906)
- ROCm 6.x or 7.x
- PyTorch 2.x with ROCm support
- Python 3.10+

## Installation

```bash
# Clone the package
git clone https://github.com/YOUR_USERNAME/chunked-attn-gfx906.git
cd chunked-attn-gfx906

# Install (editable mode recommended during development)
pip install -e .

# Verify
python -c "import chunked_attn_gfx906; print('OK')"
# Should print: [gfx906-attn] v0.10.0 — SDPA patched for gfx906 (three-tier chunked attention)
```

---

## Quick Start

### The one-liner (attention optimization)

```python
import chunked_attn_gfx906  # That's it. Attention is now patched.
```

This single import replaces `torch.nn.functional.scaled_dot_product_attention` globally. Every model that calls SDPA — diffusion models, LLMs, vision transformers, anything — automatically uses chunked attention instead of the OOM-prone Math SDPA.

No code changes to your model. No wrapper classes. No config files.

### Adding FFN chunking (optional, for extreme VRAM pressure)

```python
import chunked_attn_gfx906
from chunked_attn_gfx906 import patch_ffn

model = load_your_model()
handle = patch_ffn(model)  # Wraps eligible FFN modules
# ... run inference as normal ...
handle.remove()  # Restore originals (optional — auto-cleans on model GC)
```

FFN chunking is opt-in and runtime-verified. It proves `f(concat(a,b)) == concat(f(a), f(b))` on the actual first input before enabling chunking for any module. If verification fails, that module passes through unchanged — zero risk.

---

## Integration Guides

### PyTorch (any project)

Add the import at the top of your script, before any model loading:

```python
import chunked_attn_gfx906  # must be before model forward passes
import torch
from my_project import MyModel

model = MyModel().cuda().half()
x = torch.randn(1, 1024, 512, device="cuda", dtype=torch.float16)
output = model(x)  # attention calls automatically use chunked SDPA
```

**That's literally it.** The patch intercepts all `F.scaled_dot_product_attention` calls. Your model code stays untouched.

### HuggingFace Transformers

```python
import chunked_attn_gfx906  # before loading the model

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    torch_dtype=torch.float16,
    device_map="cuda",
)

# All attention layers now use chunked SDPA automatically
output = model.generate(tokenizer("Hello", return_tensors="pt").input_ids.cuda(), max_new_tokens=100)
```

Works with any HuggingFace model that uses `F.scaled_dot_product_attention` internally — LLaMA, Mistral, Qwen, Phi, StableLM, Gemma, etc.

### Diffusers

```python
import chunked_attn_gfx906

from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe("a photo of an astronaut riding a horse").images[0]
```

Works with any diffusers pipeline: Stable Diffusion, SDXL, FLUX, Wan, Hunyuan, CogVideo, etc.

### ComfyUI

**Step 1: Install the package**

```bash
# From your ComfyUI Python environment
pip install -e /path/to/chunked-attn-gfx906
```

**Step 2: Install the custom node**

Copy or symlink the ComfyUI node into `custom_nodes/`:

```bash
# Option A: symlink (recommended for development)
ln -s /path/to/chunked-attn-gfx906/comfyui_node ComfyUI/custom_nodes/gfx906_flash_attn

# Option B: copy
cp -r /path/to/chunked-attn-gfx906/comfyui_node ComfyUI/custom_nodes/gfx906_flash_attn
```

**Step 3: Restart ComfyUI**

```bash
python main.py --force-fp16
```

The attention patch activates automatically when ComfyUI loads the custom node. You'll see this in the console:

```
[gfx906-attn] v0.10.0 — SDPA patched for gfx906 (three-tier chunked attention)
```

**Step 4 (optional): Add FFN Chunking node**

In the ComfyUI graph editor, add the `gfx906 FFN Chunking (v11)` node:

1. Right-click canvas > Add Node > gfx906 > gfx906 FFN Chunking (v11)
2. Connect your MODEL output to the node's model input
3. Set `num_chunks` to 0 (adaptive) or a specific value
4. Connect the node's model output to your sampler

### PyTorch Training / Fine-tuning

The attention patch supports autograd (Tier 1 and Tier 2). Tier 3 (in-place) is inference-only and is never selected when `requires_grad=True`.

```python
import chunked_attn_gfx906
import torch
import torch.nn as nn

class MyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512))

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]  # uses chunked SDPA
        x = x + self.ffn(x)
        return x

model = MyTransformer().cuda().half()
optimizer = torch.optim.Adam(model.parameters())

x = torch.randn(4, 2048, 512, device="cuda", dtype=torch.float16)
loss = model(x).sum()
loss.backward()  # gradients flow through chunked attention
optimizer.step()
```

### ONNX / TorchScript Export

The patch operates at the Python level (`F.scaled_dot_product_attention`). When exporting to ONNX or TorchScript, the export traces through the patched function. If this causes issues, disable before export:

```python
import chunked_attn_gfx906

# ... your training/inference code ...

# Before export: restore original SDPA
chunked_attn_gfx906.disable()
torch.onnx.export(model, dummy_input, "model.onnx")

# Re-enable for continued inference
chunked_attn_gfx906.enable()
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GFX906_FLASH_THRESHOLD` | `0` | Minimum sequence length to trigger chunked attention. `0` = always use chunked path (recommended). |
| `GFX906_FFN_CHUNKS` | `0` | Default chunk count for FFN chunking. `0` = adaptive based on free VRAM. |
| `GFX906_FORCE_PATCH` | unset | Set to `1` to force-enable the patch on non-gfx906 GPUs (for testing). |

```bash
# Example: only chunk attention for sequences longer than 1024
GFX906_FLASH_THRESHOLD=1024 python my_script.py

# Example: force 8 FFN chunks
GFX906_FFN_CHUNKS=8 python my_script.py
```

---

## API Reference

### Attention (auto-activates on import)

```python
import chunked_attn_gfx906

# Check status
chunked_attn_gfx906.is_enabled()    # True if patch is active

# Disable/re-enable
chunked_attn_gfx906.disable()       # Restore original SDPA
chunked_attn_gfx906.enable()        # Re-enable chunked SDPA
```

The patch has three tiers that cascade on OOM:

| Tier | Method | Memory | Speed | Training |
|------|--------|--------|-------|----------|
| 1 | Standard chunked | Moderate | Fastest | Yes |
| 2 | Online softmax (K-tiled) | Low | Moderate | Yes |
| 3 | In-place softmax | Lowest | Slowest | No (inference only) |

Block sizes are auto-tuned to keep per-chunk buffers under ~1.5 GB.

### FFN Chunking (opt-in)

```python
from chunked_attn_gfx906 import patch_ffn, FFNPatchHandle

handle = patch_ffn(
    model,                  # any nn.Module
    num_chunks=0,           # 0 = adaptive (recommended)
    min_seq=256,            # minimum sequence length to trigger chunking
    verbose=True,           # print detection results
)

# Inspect
handle.num_wrapped         # number of wrapped FFN modules
handle.active              # True if patch is active

# Remove (restore originals)
handle.remove()
```

**What gets wrapped:**
- Modules with 2+ Linear layers forming an expand-contract pattern (ratio >= 1.5x)

**What gets rejected (structural pre-filter):**
- MoE modules (cross-token expert routing)
- Modules with internal normalization (LayerNorm, RMSNorm, BatchNorm)
- Attention modules (have Q/K/V projections)
- Tiny projectors (expansion ratio < 1.5x)
- Single-Linear wrappers

**What gets caught at runtime:**
- Any module where `f(concat(a,b)) != concat(f(a), f(b))` — these fail verification and permanently pass through.

---

## Troubleshooting

### "Not gfx906, patch not applied"

The patch auto-detects your GPU. If you're on gfx906 but detection fails:

```bash
GFX906_FORCE_PATCH=1 python my_script.py
```

Or check your GPU:

```bash
rocminfo | grep gfx
# Should show: gfx906
```

### OOM despite the patch

The attention kernel reduces memory dramatically, but other parts of your pipeline can still OOM:

1. **VAE decode**: Use tiled VAE decoding (tile_size=256 for Wan 2.2, SpatioTemporal tiled for LTX-2.3)
2. **Model weights**: Large models may not fit. Use quantized weights (GGUF, FP8) or CPU offloading
3. **Multiple models loaded**: Offload unused models before running (`comfy.model_management.soft_empty_cache()` in ComfyUI)

### Performance tips

- **Always use FP16**: The kernel is optimized for `v_dot2_f32_f16` (peak FP16 instruction on gfx906). Use `--force-fp16` in ComfyUI, or `.half()` / `torch_dtype=torch.float16` in PyTorch.
- **BF16 auto-converts**: If your model outputs BF16, the kernel auto-converts to FP16 (gfx906 has no BF16 hardware). No action needed, but FP16 models avoid the conversion overhead.
- **First run is slow**: ROCm's TunableOp auto-tunes GEMM kernels on first run (~180s/step). Subsequent runs use cached kernels (~10s/step). The cache persists across restarts.
- **Set perf level to auto**: `sudo sh -c "echo auto > /sys/class/drm/card0/device/power_dpm_force_performance_level"` — "high" forces max clocks and can cause KFD memory thrash.

### torch.compile compatibility

FFN chunking uses `@torch.compiler.disable` on the wrapper forward method. This means `torch.compile` will treat wrapped FFN modules as opaque calls. The attention patch works inside compiled regions (it patches at the Python dispatcher level).

---

## How it works (30-second version)

**Attention**: Instead of computing the full N x N score matrix (which is 511 GB at 720p/81 frames), we tile along the query dimension. Each tile computes a block_m x N chunk, applies softmax, multiplies by V, and writes the result. Peak memory: O(block_m x N) instead of O(N x N).

**FFN**: Feed-forward networks process each token independently (no cross-token interaction). So `FFN([token1, token2, ..., tokenN])` = `[FFN(token1), FFN(token2), ..., FFN(tokenN)]`. We split the sequence into chunks, process each chunk through the FFN, and concatenate. Peak memory drops from `O(N * expansion * hidden)` to `O(N/chunks * expansion * hidden)`.

The kernel runtime-verifies this independence property on the actual first input before enabling chunking. If a module isn't token-independent (e.g., has attention or normalization internally), it fails verification and permanently passes through unchanged.
