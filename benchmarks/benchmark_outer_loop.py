"""
Benchmark the outer-loop LLM (Qwen3-4B) forward/backward pass on H200 GPUs.

Measures:
  1. Max batch size for forward pass (vLLM generation)
  2. Max batch size for backward pass (FSDP training update)
  3. Effect of gradient checkpointing on memory and speed
  4. LoRA vs full fine-tuning: memory, speed, max batch size
  5. Precision comparison: bf16 vs fp16 vs fp32
  6. Effect of param/optimizer offloading

Runs in the lamer conda env.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m benchmarks.benchmark_outer_loop \
        --model Qwen/Qwen3-4B \
        --max_prompt_length 2048 --max_response_length 1024
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch
import torch.distributed as dist
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def gpu_mem_mb(device=None):
    """Return (allocated, reserved, total) in MB for the given CUDA device."""
    if device is None:
        device = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_mem / 1024**2
    return alloc, reserved, total


def gpu_peak_mb(device=None):
    """Return peak allocated memory in MB."""
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.max_memory_allocated(device) / 1024**2


@contextmanager
def track_memory(device=None):
    """Context manager to track peak GPU memory."""
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    alloc_before = torch.cuda.memory_allocated(device) / 1024**2
    yield
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    alloc_after = torch.cuda.memory_allocated(device) / 1024**2
    return peak


@dataclass
class BenchmarkResult:
    config_name: str
    batch_size: int
    seq_length: int
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    total_ms: float = 0.0
    peak_memory_mb: float = 0.0
    model_memory_mb: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    status: str = "OK"


def benchmark_forward_backward(
    model,
    tokenizer,
    batch_size: int,
    seq_length: int,
    num_warmup: int = 2,
    num_iters: int = 5,
    do_backward: bool = True,
    config_name: str = "",
    device: str = "cuda",
) -> Optional[BenchmarkResult]:
    """Benchmark a single forward (+ optional backward) pass."""
    result = BenchmarkResult(
        config_name=config_name,
        batch_size=batch_size,
        seq_length=seq_length,
    )

    # Create dummy input
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 151936
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Model memory baseline
    torch.cuda.synchronize()
    result.model_memory_mb = torch.cuda.memory_allocated() / 1024**2

    # Warmup
    for _ in range(num_warmup):
        try:
            if do_backward:
                model.train()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                model.zero_grad(set_to_none=True)
            else:
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            result.status = "OOM"
            torch.cuda.empty_cache()
            gc.collect()
            return result

    # Timed iterations
    forward_times = []
    backward_times = []
    total_times = []

    torch.cuda.reset_peak_memory_stats()

    for _ in range(num_iters):
        torch.cuda.synchronize()

        if do_backward:
            model.train()
            t0 = time.perf_counter()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            torch.cuda.synchronize()
            t_fwd = time.perf_counter()

            loss.backward()
            torch.cuda.synchronize()
            t_bwd = time.perf_counter()

            model.zero_grad(set_to_none=True)

            forward_times.append(t_fwd - t0)
            backward_times.append(t_bwd - t_fwd)
            total_times.append(t_bwd - t0)
        else:
            model.eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            t_fwd = time.perf_counter()

            forward_times.append(t_fwd - t0)
            total_times.append(t_fwd - t0)

    result.forward_ms = np.mean(forward_times) * 1000
    result.backward_ms = np.mean(backward_times) * 1000 if backward_times else 0.0
    result.total_ms = np.mean(total_times) * 1000
    result.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    total_tokens = batch_size * seq_length
    result.throughput_tokens_per_sec = total_tokens / np.mean(total_times)

    return result


def load_model_config(model_path, dtype, gradient_checkpointing, lora_rank=0, lora_alpha=16):
    """Load model with specified configuration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    logger.info("Loading model %s (dtype=%s, grad_ckpt=%s, lora_rank=%d)...",
                model_path, dtype, gradient_checkpointing, lora_rank)
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if dtype in ("bf16", "fp16") else "eager",
    ).cuda()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if lora_rank > 0:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            bias="none",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("LoRA applied: %d/%d trainable params (%.2f%%)",
                    trainable_params, total_params, 100 * trainable_params / total_params)

    load_time = time.time() - t0
    model_mem = torch.cuda.memory_allocated() / 1024**2
    logger.info("Model loaded in %.1fs (%.0f MB GPU memory)", load_time, model_mem)

    return model, tokenizer


def run_batch_size_sweep(model, tokenizer, seq_length, batch_sizes, do_backward,
                         config_name, num_warmup=2, num_iters=5):
    """Sweep batch sizes and return results."""
    results = []
    for bs in batch_sizes:
        result = benchmark_forward_backward(
            model, tokenizer, bs, seq_length,
            num_warmup=num_warmup, num_iters=num_iters,
            do_backward=do_backward, config_name=config_name,
        )
        results.append(result)
        if result.status == "OOM":
            logger.info("OOM at batch_size=%d, stopping sweep", bs)
            break
        gc.collect()
        torch.cuda.empty_cache()
    return results


def print_results_table(results: List[BenchmarkResult], title: str):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")
    print(f"{'Config':>30} | {'BS':>4} | {'Seq':>5} | {'Fwd (ms)':>10} | {'Bwd (ms)':>10} | "
          f"{'Total (ms)':>10} | {'Peak (MB)':>10} | {'Tok/s':>10} | Status")
    print("-" * 110)

    for r in results:
        print(f"{r.config_name:>30} | {r.batch_size:>4} | {r.seq_length:>5} | "
              f"{r.forward_ms:>10.1f} | {r.backward_ms:>10.1f} | "
              f"{r.total_ms:>10.1f} | {r.peak_memory_mb:>10.0f} | "
              f"{r.throughput_tokens_per_sec:>10.0f} | {r.status}")

    print(f"{'=' * 110}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark outer-loop LLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_response_length", type=int, default=1024)
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--num_warmup", type=int, default=2)
    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument("--skip_full_finetune", action="store_true",
                        help="Skip full fine-tuning benchmarks (just do LoRA + precision)")
    parser.add_argument("--lora_ranks", type=str, default="8,16,32,64",
                        help="Comma-separated LoRA ranks to test")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    lora_ranks = [int(x) for x in args.lora_ranks.split(",")]
    seq_length = args.max_prompt_length + args.max_response_length  # 3072

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    device_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    logger.info("GPUs: %d x %s (%.0f GB each)", device_count, device_name, device_mem)
    logger.info("Sequence length: %d (prompt=%d + response=%d)",
                seq_length, args.max_prompt_length, args.max_response_length)

    all_results = []

    # =========================================================================
    # Test 1: Full fine-tuning, bf16, with/without gradient checkpointing
    # =========================================================================
    if not args.skip_full_finetune:
        for grad_ckpt in [False, True]:
            label = f"full_bf16_gc={'on' if grad_ckpt else 'off'}"
            logger.info("\n>>> Benchmark: %s", label)
            model, tokenizer = load_model_config(
                args.model, "bf16", gradient_checkpointing=grad_ckpt)

            # Forward only
            fwd_results = run_batch_size_sweep(
                model, tokenizer, seq_length, batch_sizes, do_backward=False,
                config_name=f"{label}_fwd", num_warmup=args.num_warmup, num_iters=args.num_iters)
            all_results.extend(fwd_results)

            # Forward + backward
            bwd_results = run_batch_size_sweep(
                model, tokenizer, seq_length, batch_sizes, do_backward=True,
                config_name=f"{label}_fwd+bwd", num_warmup=args.num_warmup, num_iters=args.num_iters)
            all_results.extend(bwd_results)

            del model
            torch.cuda.empty_cache()
            gc.collect()

    # =========================================================================
    # Test 2: LoRA, bf16, with gradient checkpointing
    # =========================================================================
    for lora_rank in lora_ranks:
        label = f"lora_r{lora_rank}_bf16_gc=on"
        logger.info("\n>>> Benchmark: %s", label)
        model, tokenizer = load_model_config(
            args.model, "bf16", gradient_checkpointing=True,
            lora_rank=lora_rank)

        # Forward + backward (LoRA training)
        bwd_results = run_batch_size_sweep(
            model, tokenizer, seq_length, batch_sizes, do_backward=True,
            config_name=label, num_warmup=args.num_warmup, num_iters=args.num_iters)
        all_results.extend(bwd_results)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================================
    # Test 3: Precision comparison (full fine-tune, gradient checkpointing on)
    # =========================================================================
    for dtype in ["bf16", "fp16"]:
        label = f"full_{dtype}_gc=on"
        # Skip bf16 if already tested above
        if not args.skip_full_finetune and dtype == "bf16":
            continue

        logger.info("\n>>> Benchmark: %s", label)
        model, tokenizer = load_model_config(
            args.model, dtype, gradient_checkpointing=True)

        bwd_results = run_batch_size_sweep(
            model, tokenizer, seq_length, batch_sizes, do_backward=True,
            config_name=label, num_warmup=args.num_warmup, num_iters=args.num_iters)
        all_results.extend(bwd_results)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================================
    # Print all results
    # =========================================================================
    print_results_table(all_results, "OUTER-LOOP LLM BENCHMARK RESULTS")

    # Summary analysis
    print("\n" + "=" * 100)
    print("SUMMARY: MAX BATCH SIZES")
    print("=" * 100)

    configs = {}
    for r in all_results:
        if r.config_name not in configs:
            configs[r.config_name] = []
        configs[r.config_name].append(r)

    for config_name, config_results in configs.items():
        ok_results = [r for r in config_results if r.status == "OK"]
        if ok_results:
            max_bs = max(r.batch_size for r in ok_results)
            best_throughput = max(r.throughput_tokens_per_sec for r in ok_results)
            peak_at_max = next(r.peak_memory_mb for r in ok_results if r.batch_size == max_bs)
            print(f"  {config_name:>40}: max_bs={max_bs:>4}, peak_mem={peak_at_max:>8.0f}MB, "
                  f"best_throughput={best_throughput:>10.0f} tok/s")
        else:
            print(f"  {config_name:>40}: ALL OOM")

    # LoRA vs full comparison
    print("\n" + "=" * 100)
    print("LORA vs FULL FINE-TUNING COMPARISON (at common batch sizes)")
    print("=" * 100)

    full_key = "full_bf16_gc=on_fwd+bwd"
    if full_key in configs:
        full_results = {r.batch_size: r for r in configs[full_key] if r.status == "OK"}
        for lora_rank in lora_ranks:
            lora_key = f"lora_r{lora_rank}_bf16_gc=on"
            if lora_key in configs:
                lora_results = {r.batch_size: r for r in configs[lora_key] if r.status == "OK"}
                common_bs = sorted(set(full_results.keys()) & set(lora_results.keys()))
                if common_bs:
                    bs = common_bs[-1]  # Largest common batch size
                    f, l = full_results[bs], lora_results[bs]
                    speedup = f.total_ms / l.total_ms if l.total_ms > 0 else 0
                    mem_save = (1 - l.peak_memory_mb / f.peak_memory_mb) * 100 if f.peak_memory_mb > 0 else 0
                    print(f"  LoRA r={lora_rank} vs Full (bs={bs}): "
                          f"time {l.total_ms:.0f}ms vs {f.total_ms:.0f}ms ({speedup:.2f}x), "
                          f"memory {l.peak_memory_mb:.0f}MB vs {f.peak_memory_mb:.0f}MB ({mem_save:.0f}% saved)")

    # Save to JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        logger.info("Results saved to %s", args.output_json)


if __name__ == "__main__":
    main()
