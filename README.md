# noflash-attention

**Flash-attention-class memory efficiency for GPUs without flash attention.**

Drop-in replacement for PyTorch's `F.scaled_dot_product_attention` that reduces memory from O(N^2) to O(N) using tiled computation — no fused kernels, no custom ISA, pure PyTorch matmuls.

```bash
pip install noflash-attention
```

```python
import noflash_attention  # patches SDPA globally — done
```

Every `F.scaled_dot_product_attention` call now goes through the optimized path. No code changes needed.

---

## Who This Is For

AMD ships no fused attention for these GPUs. PyTorch falls back to Math SDPA, which materializes the **full N x N attention matrix** and OOMs on any serious workload (video gen, long-context LLMs, high-res images).

This package fills that gap:

| GPU Family | Arch | Cards | Fused Attention | This Package |
|---|---|---|---|---|
| GCN 5.0 | gfx900 | Vega 56, Vega 64 | None | Yes |
| GCN 5.1 | gfx906 | **MI50, MI60, Radeon VII** | None | Yes |
| RDNA 1 | gfx1010 | RX 5600 XT, RX 5700, RX 5700 XT | None | Yes |
| RDNA 2 | gfx1030-1032 | RX 6600–6900 XT | None | Yes |
| CDNA | gfx908+ | MI100, MI210, MI250X, MI300X | CK / FA | Not needed |
| RDNA 3+ | gfx1100+ | RX 7600–7900 XTX | CK | Not needed |

**Every fused attention implementation excludes these GPUs:**
- Composable Kernel (CK): requires MFMA instructions (gfx908+)
- AOTriton: rejects gfx900/gfx906 at compile time
- Flash Attention ROCm: requires gfx90a+
- Triton: closed gfx906 support as "not planned"
- Community forks (triton-gfx906, vllm-gfx906): both archived Feb 2026

The algorithm is pure PyTorch — it should work on any GPU where SDPA's Math backend is the only option, including Intel Arc and Apple MPS. Auto-detection currently covers AMD GCN/RDNA; set `NOFLASH_FORCE_PATCH=1` for other hardware.

---

## What It Enables

These workloads are **impossible** on MI50/Radeon VII without this package — Math SDPA OOMs immediately:

### Video Generation

| Model | Resolution | Duration | Time | VRAM | Math SDPA |
|---|---|---|---|---|---|
| **Wan 2.2 5B** | 832x480 | 2.5s (41f) | **5:04** | 17.6 GB | OOM (needs 38 GB) |
| **Wan 2.2 5B** | 1280x720 | 5s (81f) | **1:19:39** | 21.1 GB | OOM (needs 522 GB) |
| **LTX-2.3 22B** | 1280x704 | 5.2s (129f) with audio | **20:18** | ~20 GB | OOM |

### Image Generation

| Model | Resolution | Time | VRAM Saved | Speedup |
|---|---|---|---|---|
| **Z-Image Turbo 6B** | 512x512 | 22.0s | 18% | — |
| **Z-Image Turbo 6B** | 1024x1024 | 57.2s | 13% | 3% |
| **Z-Image Turbo 6B** | 1536x1536 | **112.7s** | **47%** | **29%** |

### LLM Inference

| Context | Math SDPA | noflash-attention | Speedup |
|---|---|---|---|
| 512 tokens | 3.2 ms | 2.8 ms | 1.1x |
| 2048 tokens | 12 ms | 7.4 ms | 1.6x |
| 4096 tokens | **OOM** | 14 ms | -- |
| 8192 tokens | **OOM** | 28 ms | -- |
| 16384 tokens | **OOM** | 1326 ms | -- |

> Benchmarks on AMD MI50 32GB, ROCm 7.2, PyTorch 2.12. Wan/LTX via ComfyUI, LLM via Qwen 2.5 0.5B (GQA).

---

## Why It's Fast

Math SDPA computes `S = Q @ K.T` as a single dense matrix — O(N^2) memory. At 17K tokens (a 2.5s video), that's **26 GB for one attention layer**. At 75K tokens (5s 720p video), it's **511 GB**.

This package tiles the computation into chunks that fit in ~1 GB, with three tiers of increasing memory efficiency:

```
F.scaled_dot_product_attention(Q, K, V, ...)
        |
        v
   patched_sdpa
        |
        +-- Non-target GPU or nested tensors --> original SDPA (passthrough)
        |
        +-- BF16 --> FP16 conversion (gfx906 has no BF16 hardware)
        |
        +-- seq_q <= 8 and fits in 256MB --> Math SDPA fast path
        |
        +-- Tier 1: chunked_sdpa
        |       PyTorch softmax + Tensile GEMMs — fastest, ~2x S+P memory per chunk
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

**Supported:** MHA, GQA, MQA, causal masks, bool/float attention masks, 4D per-head masks, `scale` parameter. Full `F.scaled_dot_product_attention` API compatibility.

---

## Installation

### From PyPI

```bash
pip install noflash-attention
```

### From source

```bash
git clone https://github.com/Lowkey-LokiSN/noflash-attention.git
cd noflash-attention
pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- An AMD GPU on the supported list (or set `NOFLASH_FORCE_PATCH=1`)

---

## Usage

### Standalone — any PyTorch model

```python
import noflash_attention  # auto-patches SDPA on import

import torch
import torch.nn.functional as F

# Every SDPA call now uses the optimized path
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
   [noflash-attn] v1.0.0 — SDPA patched (gfx906 detected)
   ```

4. **FFN Chunking** (optional): Add the `NoFlash FFN Chunking` node to your workflow for additional memory savings on feedforward layers.

---

## FFN Chunking

Beyond attention, transformer FFN layers can also be memory-hungry. The `patch_ffn` API wraps eligible feedforward modules to process tokens in chunks:

```python
import noflash_attention

model = load_your_model()
handle = noflash_attention.patch_ffn(model, num_chunks=0)  # 0 = adaptive
# ... run inference ...
handle.remove()  # restore originals
```

**Safety guarantee:** Every module is runtime-verified before chunking. On the first forward pass, the wrapper proves that `f(concat(a, b)) == concat(f(a), f(b))` for the actual module and input. Only verified modules are chunked. Verification failure = permanent passthrough.

**Automatically rejects:**
- Attention modules (cross-token by definition)
- MoE layers (expert routing depends on full sequence)
- Modules with internal normalization (statistics depend on batch)
- Tiny projectors (expansion ratio < 1.5x)

---

## Isolated Attention Microbenchmarks

B=1, H=16, D=64, FP16 on MI50 32GB:

| Sequence Length | Math SDPA | noflash-attention | Speedup | VRAM Saved |
|---|---|---|---|---|
| 256 | 0.28 ms / 47 MB | 0.18 ms / 38 MB | 1.6x | 19% |
| 512 | 0.55 ms / 79 MB | 0.29 ms / 53 MB | 1.9x | 33% |
| 1024 | 1.83 ms / 198 MB | 0.85 ms / 106 MB | 2.2x | 46% |
| 2048 | 8.72 ms / 652 MB | 4.74 ms / 308 MB | 1.8x | 53% |
| 4096 | 28.81 ms / 2424 MB | 17.93 ms / 1096 MB | 1.6x | 55% |
| 8192 | 102.42 ms / 9424 MB | 72.75 ms / 1124 MB | 1.4x | 88% |
| 16384 | **OOM** | 1325.69 ms / 1202 MB | Only option | -- |

At 8192 tokens: **88% less VRAM**. At 16384+: **Math SDPA cannot run at all**.

For extreme memory pressure, Tier 2 (online softmax) provides additional savings:

| Config | Tier 1 VRAM | Tier 2 VRAM | Additional Savings |
|---|---|---|---|
| MHA seq=4096 | 1096 MB | 481 MB | 56% |
| MHA seq=8192 | 1124 MB | 514 MB | 54% |
| GQA 14q/2kv seq=4096 | 965 MB | 427 MB | 56% |

---

## MI50 vs NVIDIA: Real-World Comparison

LTX-2.3 22B video generation, real user-reported times:

| GPU | VRAM | Price | Resolution | Duration | Audio | Time |
|---|---|---|---|---|---|---|
| **MI50 + noflash-attention** | **32 GB** | **~$150** | **1280x704** | **5.2s** | **Yes** | **20:18** |
| RTX 3090 | 24 GB | ~$700 | 720p | 10s | No | ~13:30 |
| RTX 4060 Ti | 16 GB | ~$400 | 768x1280 | 10s | No | 4:22 |
| RTX 4090 | 24 GB | ~$1600 | 720p | 4s | No | 0:32 |
| RTX 5090 | 32 GB | ~$2000 | 832x480 | 3.4s | Yes | 0:22 |

MI50 is the only GPU under $200 that can run a 22B model at 720p with audio. Without `noflash-attention`, it can't run this workload at all.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NOFLASH_FORCE_PATCH` | unset | Force-enable on non-AMD GPUs |
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

## How It Works

### SDPA Patch

Traditional attention computes the full `N x N` score matrix `S = Q @ K^T`, then `softmax(S) @ V`. For a 720p 5-second video (75K tokens), `S` alone requires **511 GB** in FP32.

We tile along the query dimension:

```python
for i in range(0, N_q, block_m):
    Q_chunk = Q[:, :, i:i+block_m, :]     # small slice of queries
    S_chunk = Q_chunk @ K.transpose(-2,-1) # block_m x N_kv (fits in ~1 GB)
    P_chunk = softmax(S_chunk)
    O_chunk = P_chunk @ V
```

Peak memory: O(block_m * N_kv) instead of O(N_q * N_kv). The `block_m` size is auto-tuned to fit in ~1.5 GB with OOM retry at half size.

The online softmax tier (Tier 2) additionally tiles along the key dimension, streaming the softmax normalization constants — achieving O(block_m * block_n) peak memory with FP32 accumulation for numerical stability.

### FFN Chunking

Transformer FFN layers expand the hidden dimension (e.g., 4096 -> 16384) creating large intermediates. We chunk along the token dimension:

```python
for i in range(0, seq_len, chunk_size):
    output[i:i+chunk_size] = ffn(input[i:i+chunk_size])
```

This is only safe for **token-independent** operations — where processing tokens separately produces identical results to processing them together. Safety is guaranteed by **runtime verification**: on the first forward pass, we prove `f(cat(a,b)) == cat(f(a), f(b))` for the actual module. Modules that fail verification permanently fall back to the original path.

---

## Known Limitations

- **Training**: Tier 1 supports autograd. Tiers 2 and 3 are inference-only (in-place ops break autograd).
- **torch.compile**: Not tested. Python-level chunking may not be captured by dynamo tracing.
- **CUDA graphs**: Incompatible (dynamic OOM retry and block sizing).
- **Nested tensors**: Passed through to original SDPA.
- **FP32 inputs**: Supported but not optimized. A warning suggests FP16 for 2x faster attention.

---

## Development

```bash
git clone https://github.com/Lowkey-LokiSN/noflash-attention.git
cd noflash-attention
pip install -e .
pytest tests/
```

128 tests covering correctness, GQA/MQA, causal masks, attention masks, online softmax, FFN verification, MoE rejection, and edge cases.

### Project Stats

- 28 kernel iterations across 3 algorithmic approaches
- 40+ configurations tested
- 128/128 tests passing, zero benchmark regression
- 9.3x cumulative speedup from v0.1 to v1.0

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgments

Built for the AMD MI50 community — the GPU that datacenter forgot.

If this package helps your work, consider starring the repo. It helps others with unsupported GPUs find it.
