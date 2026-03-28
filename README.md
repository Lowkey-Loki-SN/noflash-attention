# noflash-attention

**Unlock video generation, high-res image synthesis, long-context inference, and extended training on GPUs left behind by Flash Attention.**

Without fused attention kernels, PyTorch falls back to Math SDPA — which materializes the full N x N attention matrix and OOMs on any serious workload. This package is a drop-in replacement for `F.scaled_dot_product_attention` that reduces memory from O(N^2) to O(N) using tiled computation. No fused kernels, no custom ISA, pure PyTorch.

One import. Zero code changes:

```bash
pip install noflash-attention
```

```python
import noflash_attention  # patches SDPA globally — done
```

Workloads that were impossible — 720p video generation, 1080p audio-video synthesis, 1536x1536 image generation, 16K+ context inference, longer-context fine-tuning — now run within your existing VRAM budget.

---

## Who This Is For

Flash Attention and fused SDPA kernels only work on specific hardware. Every other GPU falls back to Math SDPA, which allocates the **full N x N score matrix** — at 75K tokens (a 5-second 720p video), that's over **500 GB for a single attention layer**.

This package exists for GPUs where no efficient SDPA backend is available:

| Vendor | GPUs | The Gap |
|---|---|---|
| **AMD GCN** | Vega 56, Vega 64, **MI50**, MI60, Radeon VII | CK requires MFMA (gfx908+), AOTriton rejects at compile time |
| **AMD RDNA 1** | RX 5600 XT, RX 5700, RX 5700 XT | No fused attention in any AMD library |
| **AMD RDNA 2** | RX 6600–6900 XT | No fused attention support in CK or AOTriton |

These AMD architectures have **zero** memory-efficient attention support — not Flash Attention, not `mem_efficient`, nothing. Math SDPA is the only path. This package is the only way to run attention-heavy workloads on these GPUs.

**What about NVIDIA?** NVIDIA GPUs from Pascal (SM 60) onward have PyTorch's `mem_efficient` SDPA backend, which already provides O(N) attention memory. Pre-Ampere GPUs (SM < 80) lack Flash Attention specifically, but `mem_efficient` covers most use cases. Our auto-detection probes for efficient backends at startup and only activates where none exist. You can force-enable with `NOFLASH_FORCE_PATCH=1` if needed.

**Other hardware** (Intel Arc, Apple MPS): Set `NOFLASH_FORCE_PATCH=1` to enable.

---

## Benchmarks

All benchmarks on the following hardware:

| Component | Detail |
|---|---|
| GPU | AMD Instinct MI50 32GB (gfx906, 60 CUs, HBM2 1024 GB/s) |
| CPU | Intel Xeon Gold 6148 (single socket) |
| RAM | 128 GB DDR4 ECC |
| Software | ROCm 7.2, PyTorch 2.x (source build), Ubuntu 24.04 |
| Power limit | 150W |

### Video Generation (via ComfyUI)

| Model | Resolution | Duration | Time | Math SDPA |
|---|---|---|---|---|
| Wan 2.2 5B | 832x480 | 2.5s (41f) | **5:04** | OOM (needs 38 GB) |
| Wan 2.2 5B | 1280x720 | 5s (81f) | **1:19:39** | OOM (needs 522 GB) |
| LTX-2.3 22B | 1280x704 | 5.2s with audio | **20:18** | OOM |
| LTX-2.3 22B | 1920x1080 | 5.2s with audio | **1:03:26** | OOM |

### Image Generation

| Model | Resolution | Time | VRAM Saved | Speedup |
|---|---|---|---|---|
| Z-Image Turbo 6B | 512x512 | 22.0s | 18% | — |
| Z-Image Turbo 6B | 1024x1024 | 57.2s | 13% | 3% |
| Z-Image Turbo 6B | 1536x1536 | **112.7s** | **47%** | **29%** |

### Isolated Attention Microbenchmarks

B=1, H=16, D=64, FP16:

| Sequence Length | Math SDPA | noflash-attention | Speedup | VRAM Saved |
|---|---|---|---|---|
| 256 | 0.28 ms / 47 MB | 0.18 ms / 38 MB | 1.6x | 19% |
| 512 | 0.55 ms / 79 MB | 0.29 ms / 53 MB | 1.9x | 33% |
| 1024 | 1.83 ms / 198 MB | 0.85 ms / 106 MB | 2.2x | 46% |
| 2048 | 8.72 ms / 652 MB | 4.74 ms / 308 MB | 1.8x | 53% |
| 4096 | 28.81 ms / 2424 MB | 17.93 ms / 1096 MB | 1.6x | 55% |
| 8192 | 102.42 ms / 9424 MB | 72.75 ms / 1124 MB | 1.4x | 88% |
| 16384 | **OOM** | 1325.69 ms / 1202 MB | Only option | — |

For extreme memory pressure, Tier 2 (online softmax) provides additional savings:

| Config | Tier 1 VRAM | Tier 2 VRAM | Additional Savings |
|---|---|---|---|
| MHA seq=4096 | 1096 MB | 481 MB | 56% |
| MHA seq=8192 | 1124 MB | 514 MB | 54% |
| GQA 14q/2kv seq=4096 | 965 MB | 427 MB | 56% |

---

## How It Works

Math SDPA computes `S = Q @ K.T` as a single dense matrix — O(N^2) memory. At 75K tokens (a 5s 720p video), that's over **500 GB** for a single attention layer.

This package tiles the computation into chunks that fit in ~1 GB, with three tiers of increasing memory efficiency:

```
F.scaled_dot_product_attention(Q, K, V, ...)
        |
        v
   patched_sdpa
        |
        +-- Non-target GPU or nested tensors --> original SDPA (passthrough)
        |
        +-- BF16 --> FP16 conversion (for GPUs without BF16 hardware)
        |
        +-- seq_q <= 8 and fits in 256MB --> Math SDPA fast path
        |
        +-- Tier 1: chunked_sdpa
        |       PyTorch softmax + GEMMs — fastest, ~2x S+P memory per chunk
        |       Auto block_m sizing, OOM retry with halved blocks
        |
        +-- Tier 2: chunked_sdpa_online
        |       K-tiled streaming softmax — moderate speed, O(block_m * block_n) memory
        |       FP32 accumulation for numerical precision
        |
        +-- Tier 3: chunked_sdpa_inplace
        |       Manual FP16 softmax — slowest, absolute minimum memory
        |       Inference only (in-place ops break autograd)
        |
        +-- All tiers exhausted --> raise OutOfMemoryError
```

### SDPA Patch

We tile along the query dimension:

```python
for i in range(0, N_q, block_m):
    Q_chunk = Q[:, :, i:i+block_m, :]     # small slice of queries
    S_chunk = Q_chunk @ K.transpose(-2,-1) # block_m x N_kv (fits in ~1 GB)
    P_chunk = softmax(S_chunk)
    O_chunk = P_chunk @ V
```

Peak memory: O(block_m * N_kv) instead of O(N_q * N_kv). The `block_m` size is auto-tuned to fit in ~1.5 GB with OOM retry at half size.

Tier 2 additionally tiles along the key dimension, streaming the softmax normalization constants — achieving O(block_m * block_n) peak memory with FP32 accumulation for numerical stability.

### FFN Chunking

Beyond attention, transformer FFN layers can also be memory-hungry (e.g., 4096 -> 16384 expansion). The `patch_ffn` API wraps eligible feedforward modules to process tokens in chunks:

```python
import noflash_attention

model = load_your_model()
handle = noflash_attention.patch_ffn(model, num_chunks=0)  # 0 = adaptive
# ... run inference ...
handle.remove()  # restore originals
```

**Safety guarantee:** Every module is runtime-verified before chunking. On the first forward pass, the wrapper proves that `f(concat(a, b)) == concat(f(a), f(b))` for the actual module and input. Only verified modules are chunked. Verification failure = permanent passthrough.

Automatically rejects: attention modules, MoE layers, modules with internal normalization, tiny projectors.

**Supported:** MHA, GQA, MQA, causal masks, bool/float attention masks, 4D per-head masks, `scale` parameter. Full `F.scaled_dot_product_attention` API compatibility.

---

## Installation

### From PyPI

```bash
pip install noflash-attention
```

### From source

```bash
git clone https://github.com/Lowkey-Loki-SN/noflash-attention.git
cd noflash-attention
pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- A supported GPU (or set `NOFLASH_FORCE_PATCH=1` for any hardware)

---

## Usage

### Standalone — any PyTorch model

```python
import noflash_attention  # auto-patches SDPA on import

import torch
import torch.nn.functional as F

Q = torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.float16)
K = torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.float16)
V = torch.randn(1, 16, 4096, 64, device="cuda", dtype=torch.float16)
out = F.scaled_dot_product_attention(Q, K, V)  # tiled, O(N) memory
```

### HuggingFace Transformers

```python
import noflash_attention
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### ComfyUI

1. Install the package into your ComfyUI Python environment:
   ```bash
   pip install noflash-attention
   ```

2. Copy the ComfyUI node into your custom nodes directory:
   ```bash
   cp -r comfyui_node /path/to/ComfyUI/custom_nodes/noflash-attention
   ```

3. Start ComfyUI. The SDPA patch activates automatically on startup. You'll see:
   ```
   [noflash-attn] v1.0.1 — SDPA patched (three-tier chunked attention)
   ```

4. **FFN Chunking** (optional): Add the `NoFlash FFN Chunking` node to your workflow for additional memory savings on feedforward layers.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NOFLASH_FORCE_PATCH` | unset | Force-enable on any GPU |
| `NOFLASH_DISABLE` | unset | Disable even on supported GPUs |
| `NOFLASH_THRESHOLD` | `0` | Minimum `seq_kv` to trigger chunked path (0 = always) |
| `NOFLASH_FFN_CHUNKS` | `0` | Default FFN chunk count (0 = adaptive) |

### API

```python
import noflash_attention

noflash_attention.enable()       # re-enable after disable
noflash_attention.disable()      # restore original SDPA
noflash_attention.is_enabled()   # check patch status

# FFN chunking
handle = noflash_attention.patch_ffn(model, num_chunks=0, verbose=True)
handle.remove()                  # restore original FFN modules
handle.num_wrapped               # number of modules wrapped
```

---

## Known Limitations

- **Training**: Tier 1 supports autograd (backward pass works). Tiers 2 and 3 are inference-only (in-place ops break autograd).
- **Benchmarks**: All results are from a single AMD MI50 32GB. Performance on other supported GPUs (Vega 56/64, RX 5000/6000 series) has not been tested and will vary based on memory bandwidth, compute units, and VRAM capacity.
- **Multi-GPU**: Not tested. The patch applies per-process and should work with data parallelism, but multi-GPU attention strategies (tensor parallelism, ring attention) have not been validated.
- **torch.compile**: Not tested. Python-level chunking may not be captured by dynamo tracing.
- **CUDA graphs**: Incompatible (dynamic OOM retry and block sizing).
- **Nested tensors**: Passed through to original SDPA.
- **FP32 inputs**: Supported but not optimized. A warning suggests FP16 for 2x faster attention.

---

## Development

```bash
git clone https://github.com/Lowkey-Loki-SN/noflash-attention.git
cd noflash-attention
pip install -e .
pytest tests/
```

128 tests covering correctness, GQA/MQA, causal masks, attention masks, online softmax, FFN verification, MoE rejection, and edge cases.

---

## License

MIT License. See [LICENSE](LICENSE).
