"""v9 Feature tests — forward-compat kwargs, GQA, autograd, NaN guard,
autocast, disable/enable, regressions, stability, and real LLM.
25 tests total."""
import time
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
device, dtype = "cuda", torch.float16
TOLERANCE = 0.05


def _rel_err(out, ref):
    denom = ref.abs().mean().item()
    return (out - ref).abs().max().item() / denom if denom > 0 else (out - ref).abs().max().item()


# --- Fix 1: Forward-compatible kwargs ---
def test_unknown_kwarg():
    Q = torch.randn(1, 8, 128, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 128, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 128, 64, device=device, dtype=dtype)
    F.scaled_dot_product_attention(Q, K, V, future_param=True)


# --- Fix 2: enable_gqa validation ---
def test_gqa_enable_gqa():
    Q = torch.randn(1, 32, 256, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    F.scaled_dot_product_attention(Q, K, V, enable_gqa=True)


def test_mqa_32q_1kv():
    Q = torch.randn(1, 32, 256, 64, device=device, dtype=dtype)
    K = torch.randn(1, 1, 256, 64, device=device, dtype=dtype)
    V = torch.randn(1, 1, 256, 64, device=device, dtype=dtype)
    F.scaled_dot_product_attention(Q, K, V, enable_gqa=True)


# --- Fix 3: Non-contiguous tensor handling ---
def test_gqa_noncontiguous_q():
    Q = torch.randn(1, 32, 256, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    Q_nc = Q.permute(0, 2, 1, 3).permute(0, 2, 1, 3)
    F.scaled_dot_product_attention(Q_nc, K, V, enable_gqa=True)


# --- Fix 4: Autograd compatibility ---
def test_inference_no_grad():
    Q = torch.randn(1, 16, 512, 64, device=device, dtype=dtype)
    K = torch.randn(1, 16, 512, 64, device=device, dtype=dtype)
    V = torch.randn(1, 16, 512, 64, device=device, dtype=dtype)
    F.scaled_dot_product_attention(Q, K, V)


def test_backward_pass():
    Qg = torch.randn(1, 16, 512, 64, device=device, dtype=dtype, requires_grad=True)
    Kg = torch.randn(1, 16, 512, 64, device=device, dtype=dtype, requires_grad=True)
    Vg = torch.randn(1, 16, 512, 64, device=device, dtype=dtype, requires_grad=True)
    out = F.scaled_dot_product_attention(Qg, Kg, Vg)
    out.sum().backward()
    assert Qg.grad is not None


# --- Fix 5: NaN guard (fully masked rows) ---
def test_nan_guard_masked_row():
    Q = torch.randn(1, 4, 8, 64, device=device, dtype=dtype)
    K = torch.randn(1, 4, 8, 64, device=device, dtype=dtype)
    V = torch.randn(1, 4, 8, 64, device=device, dtype=dtype)
    mask = torch.ones(8, 8, device=device, dtype=torch.bool)
    mask[3, :] = False
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    assert not torch.isnan(out).any().item(), "NaN in masked output"


# --- Fix 6: Batch size in auto_block_m ---
def test_batch_b4():
    Q = torch.randn(4, 16, 1024, 64, device=device, dtype=dtype)
    K = torch.randn(4, 16, 1024, 64, device=device, dtype=dtype)
    V = torch.randn(4, 16, 1024, 64, device=device, dtype=dtype)
    F.scaled_dot_product_attention(Q, K, V)


# --- Fix 7: torch.autocast protection ---
def test_autocast_fp16():
    Q = torch.randn(1, 16, 512, 64, device=device, dtype=torch.float32)
    K = torch.randn(1, 16, 512, 64, device=device, dtype=torch.float32)
    V = torch.randn(1, 16, 512, 64, device=device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = F.scaled_dot_product_attention(Q, K, V)
    assert not torch.isnan(out).any().item()


def test_autocast_bf16():
    Q = torch.randn(1, 16, 512, 64, device=device, dtype=torch.float32)
    K = torch.randn(1, 16, 512, 64, device=device, dtype=torch.float32)
    V = torch.randn(1, 16, 512, 64, device=device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = F.scaled_dot_product_attention(Q, K, V)
    assert not torch.isnan(out).any().item()


# --- Fix 8: Runtime disable/enable ---
def test_disable_enable_roundtrip():
    Q = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    noflash_attention.disable()
    try:
        ref = F.scaled_dot_product_attention(Q, K, V)
    finally:
        noflash_attention.enable()
    out = F.scaled_dot_product_attention(Q, K, V)
    diff = (ref - out).abs().max().item()
    assert diff < 0.01, f"disable/enable roundtrip diff={diff:.6f}"


# --- Regressions: MHA correctness ---
@pytest.mark.parametrize("seq", [256, 1024, 4096])
def test_mha_regression(seq):
    Q = torch.randn(1, 16, seq, 64, device=device, dtype=dtype)
    K = torch.randn(1, 16, seq, 64, device=device, dtype=dtype)
    V = torch.randn(1, 16, seq, 64, device=device, dtype=dtype)
    ref = original(Q, K, V)
    out = F.scaled_dot_product_attention(Q, K, V)
    assert _rel_err(out, ref) < TOLERANCE, f"MHA seq={seq} rel_err too high"


# --- Regressions: Head dims ---
@pytest.mark.parametrize("D", [64, 80, 96, 128])
def test_head_dim_regression(D):
    Q = torch.randn(1, 8, 512, D, device=device, dtype=dtype)
    K = torch.randn(1, 8, 512, D, device=device, dtype=dtype)
    V = torch.randn(1, 8, 512, D, device=device, dtype=dtype)
    ref = original(Q, K, V)
    out = F.scaled_dot_product_attention(Q, K, V)
    assert _rel_err(out, ref) < TOLERANCE, f"D={D} rel_err too high"


# --- Regressions: Cross-attn, masks, BF16 ---
def test_cross_attention_regression():
    Q = torch.randn(1, 16, 256, 64, device=device, dtype=dtype)
    K = torch.randn(1, 16, 1024, 64, device=device, dtype=dtype)
    V = torch.randn(1, 16, 1024, 64, device=device, dtype=dtype)
    ref = original(Q, K, V)
    out = F.scaled_dot_product_attention(Q, K, V)
    assert _rel_err(out, ref) < TOLERANCE


def test_is_causal_regression():
    Q = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    F.scaled_dot_product_attention(Q, K, V, is_causal=True)


def test_bool_mask_regression():
    Q = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    K = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    V = torch.randn(1, 8, 256, 64, device=device, dtype=dtype)
    bmask = torch.ones(256, 256, device=device, dtype=torch.bool).tril()
    F.scaled_dot_product_attention(Q, K, V, attn_mask=bmask)


def test_bf16_output_dtype():
    Q = torch.randn(1, 16, 512, 64, device=device, dtype=torch.bfloat16)
    K = torch.randn(1, 16, 512, 64, device=device, dtype=torch.bfloat16)
    V = torch.randn(1, 16, 512, 64, device=device, dtype=torch.bfloat16)
    out = F.scaled_dot_product_attention(Q, K, V)
    assert out.dtype == torch.bfloat16


# --- Performance ---
def test_performance_seq4096():
    Q = torch.randn(1, 16, 4096, 64, device=device, dtype=dtype)
    K = torch.randn(1, 16, 4096, 64, device=device, dtype=dtype)
    V = torch.randn(1, 16, 4096, 64, device=device, dtype=dtype)
    for _ in range(3):
        F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1000
    assert ms < 20.0, f"seq=4096 took {ms:.2f}ms, expected <20ms"


# --- Stability ---
def test_stability_500_calls():
    Q = torch.randn(1, 16, 2048, 64, device=device, dtype=dtype)
    K = torch.randn(1, 16, 2048, 64, device=device, dtype=dtype)
    V = torch.randn(1, 16, 2048, 64, device=device, dtype=dtype)
    for _ in range(5):
        F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    mem0 = torch.cuda.memory_allocated()
    for _ in range(500):
        out = F.scaled_dot_product_attention(Q, K, V)
        del out
    torch.cuda.synchronize()
    leak = abs(torch.cuda.memory_allocated() - mem0) / 1024 / 1024
    assert leak < 5.0, f"Memory leak: {leak:.2f} MB over 500 calls"


# --- Real LLM: Qwen 2.5 0.5B (GQA 14q/2kv) ---
def test_qwen_gqa_real_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    inp = tk("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = m.generate(**inp, max_new_tokens=15, do_sample=False)
    txt = tk.decode(o[0], skip_special_tokens=True)
    assert "Paris" in txt, f"Expected 'Paris' in output, got: {txt[:80]!r}"
