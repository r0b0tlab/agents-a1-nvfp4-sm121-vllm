#!/usr/bin/env python3
"""Optimized Agents-A1 NVFP4 benchmark with concurrency sweeps.

Tests: GSM8K 50, prefill/decode throughput at c1/c2/c4/c8/c16/c32.
Captures Prometheus counters for accurate server-side tok/s.
"""
import asyncio, json, time, subprocess, csv, os, pathlib
from datetime import datetime, timezone

BASE_URL = "http://127.0.0.1:18080/v1"
MODEL = "agents-a1-nvfp4"
OUT = pathlib.Path(os.environ.get("BENCH_DIR", "."))
TELEMETRY_SAMPLES = []

import urllib.request

def get_metrics():
    try:
        m = urllib.request.urlopen("http://127.0.0.1:18080/metrics", timeout=10).read().decode()
        vals = {}
        for line in m.splitlines():
            if line.startswith("vllm:") and "{" in line and not line.startswith("#"):
                key = line.split("{")[0].split(":")[1]
                try:
                    val = float(line.split()[-1])
                    vals[key] = vals.get(key, 0) + val
                except:
                    pass
        return vals
    except:
        return {}

def gpu_telemetry():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=power.draw,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], timeout=5).decode().strip().split(", ")
        return {"power_w": float(out[0]), "util_gpu_pct": float(out[1]), "temp_c": float(out[2])}
    except:
        return {}

async def warmup():
    """Send a few requests to trigger cudagraph capture."""
    import aiohttp
    async with aiohttp.ClientSession() as sess:
        tasks = []
        for i in range(4):
            payload = {"model": MODEL, "prompt": "Write a paragraph about AI.", "max_tokens": 32, "temperature": 0}
            tasks.append(sess.post(f"{BASE_URL}/completions", json=payload))
        await asyncio.gather(*tasks)
    print("Warmup complete")

async def concurrency_sweep(concurrencies=[1,2,4,8,16,32], reps=3, max_tokens=128):
    """Measure aggregate throughput at different concurrency levels."""
    import aiohttp
    prompt = "Write a detailed technical explanation of transformer architecture, covering attention mechanisms, feed-forward networks, and layer normalization. "
    results = {}
    for c in concurrencies:
        rep_results = []
        for rep in range(reps):
            m_before = get_metrics()
            t_before = time.time()
            async with aiohttp.ClientSession() as sess:
                tasks = []
                for i in range(c):
                    payload = {
                        "model": MODEL,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0,
                        "ignore_eos": True
                    }
                    tasks.append(sess.post(f"{BASE_URL}/completions", json=payload))
                responses = await asyncio.gather(*tasks)
            t_after = time.time()
            m_after = get_metrics()
            wall = t_after - t_before
            gen_delta = m_after.get("generation_tokens_total", 0) - m_before.get("generation_tokens_total", 0)
            prompt_delta = m_after.get("prompt_tokens_total", 0) - m_before.get("prompt_tokens_total", 0)
            success = sum(1 for r in responses if r.status == 200)
            rep_results.append({
                "wall_s": wall,
                "gen_tokens": gen_delta,
                "prompt_tokens": prompt_delta,
                "success": success,
                "total": c,
                "gen_throughput": gen_delta / wall if wall > 0 else 0,
                "prompt_throughput": prompt_delta / wall if wall > 0 else 0,
            })
            TELEMETRY_SAMPLES.append(gpu_telemetry())
            print(f"  c={c} rep={rep} wall={wall:.3f}s gen={gen_delta}tok prompt={prompt_delta}tok gen_tps={gen_delta/wall:.2f} success={success}/{c}")
        # Use last 2 reps (first rep may include cold start)
        last2 = rep_results[-2:] if len(rep_results) >= 2 else rep_results
        avg_gen_tps = sum(r["gen_throughput"] for r in last2) / len(last2)
        avg_prompt_tps = sum(r["prompt_throughput"] for r in last2) / len(last2)
        avg_wall = sum(r["wall_s"] for r in last2) / len(last2)
        results[f"c{c}"] = {
            "concurrency": c,
            "reps": reps,
            "mean_gen_throughput": round(avg_gen_tps, 2),
            "mean_prompt_throughput": round(avg_prompt_tps, 2),
            "mean_wall_s": round(avg_wall, 3),
            "all_reps": rep_results,
        }
        print(f"c={c}: gen={avg_gen_tps:.2f}tok/s prefill={avg_prompt_tps:.2f}tok/s wall={avg_wall:.3f}s")
    return results

async def single_decode_latency(max_tokens=256, reps=5):
    """Measure single-request decode tok/s accurately."""
    import aiohttp
    prompt = "Explain the difference between Mixture of Experts and dense transformer models. "
    rates = []
    for i in range(reps):
        m_before = get_metrics()
        t0 = time.time()
        async with aiohttp.ClientSession() as sess:
            payload = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}
            async with sess.post(f"{BASE_URL}/completions", json=payload) as resp:
                data = await resp.json()
        t1 = time.time()
        m_after = get_metrics()
        gen_delta = m_after.get("generation_tokens_total", 0) - m_before.get("generation_tokens_total", 0)
        wall = t1 - t0
        tps = gen_delta / wall if wall > 0 else 0
        rates.append({"wall_s": round(wall,3), "gen_tokens": gen_delta, "tok_s": round(tps, 2)})
        print(f"  decode rep={i}: {gen_delta}tok in {wall:.3f}s = {tps:.2f}tok/s")
        TELEMETRY_SAMPLES.append(gpu_telemetry())
    return rates

async def single_prefill_latency(prompt_tokens_target=512, reps=5):
    """Measure prefill throughput with a large prompt and 1 output token."""
    import aiohttp
    # Build a long prompt
    prompt = ("The history of computing spans centuries. " * 200)[:prompt_tokens_target*4]
    rates = []
    for i in range(reps):
        m_before = get_metrics()
        t0 = time.time()
        async with aiohttp.ClientSession() as sess:
            payload = {"model": MODEL, "prompt": prompt, "max_tokens": 4, "temperature": 0}
            async with sess.post(f"{BASE_URL}/completions", json=payload) as resp:
                data = await resp.json()
        t1 = time.time()
        m_after = get_metrics()
        prompt_delta = m_after.get("prompt_tokens_total", 0) - m_before.get("prompt_tokens_total", 0)
        wall = t1 - t0
        tps = prompt_delta / wall if wall > 0 else 0
        rates.append({"wall_s": round(wall,3), "prompt_tokens": prompt_delta, "tok_s": round(tps, 2)})
        print(f"  prefill rep={i}: {prompt_delta}tok in {wall:.3f}s = {tps:.2f}tok/s")
        TELEMETRY_SAMPLES.append(gpu_telemetry())
    return rates

async def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"=== Agents-A1 NVFP4 Optimized Benchmark {ts} ===")
    print(f"Endpoint: {BASE_URL} Model: {MODEL}")

    await warmup()

    print("\n=== Single-request decode latency ===")
    decode = await single_decode_latency()
    avg_decode = sum(r["tok_s"] for r in decode) / len(decode)
    print(f"Mean decode: {avg_decode:.2f} tok/s")

    print("\n=== Single-request prefill latency (~512 tok prompt) ===")
    prefill = await single_prefill_latency()
    avg_prefill = sum(r["tok_s"] for r in prefill) / len(prefill)
    print(f"Mean prefill: {avg_prefill:.2f} tok/s")

    print("\n=== Concurrency sweep ===")
    sweep = await concurrency_sweep()

    # Telemetry summary
    if TELEMETRY_SAMPLES:
        import statistics
        tel = {
            "power_w": {"mean": round(statistics.mean(s["power_w"] for s in TELEMETRY_SAMPLES),2),
                         "max": round(max(s["power_w"] for s in TELEMETRY_SAMPLES),2)},
            "util_gpu_pct": {"mean": round(statistics.mean(s["util_gpu_pct"] for s in TELEMETRY_SAMPLES),2),
                              "max": round(max(s["util_gpu_pct"] for s in TELEMETRY_SAMPLES),2)},
            "temp_c": {"mean": round(statistics.mean(s["temp_c"] for s in TELEMETRY_SAMPLES),2),
                       "max": round(max(s["temp_c"] for s in TELEMETRY_SAMPLES),2)},
            "samples": len(TELEMETRY_SAMPLES),
        }
    else:
        tel = {}

    summary = {
        "timestamp": ts,
        "model": MODEL,
        "endpoint": BASE_URL,
        "config": {
            "max_num_seqs": 32,
            "enforce_eager": False,
            "speculative": "disabled",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "moe_backend": "FLASHINFER_CUTLASS",
            "gemm_kernel": "FlashInferCutlassNvFp4LinearKernel",
            "kv_cache_dtype": "fp8",
        },
        "decode_latency": decode,
        "decode_mean_tok_s": round(avg_decode, 2),
        "prefill_latency": prefill,
        "prefill_mean_tok_s": round(avg_prefill, 2),
        "concurrency_sweep": sweep,
        "telemetry": tel,
    }

    out_path = OUT / "optimized_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")

    # CSV telemetry
    if TELEMETRY_SAMPLES:
        tel_path = OUT / "telemetry_gpu_optimized.csv"
        with open(tel_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["power_w","util_gpu_pct","temp_c"])
            w.writeheader()
            for s in TELEMETRY_SAMPLES:
                w.writerow(s)
        print(f"Wrote {tel_path}")

    print("\n=== SUMMARY ===")
    print(f"Decode:  {avg_decode:.2f} tok/s (single request)")
    print(f"Prefill: {avg_prefill:.2f} tok/s (single request)")
    for k, v in sweep.items():
        print(f"{k}: gen={v['mean_gen_throughput']:.2f}tok/s prefill={v['mean_prompt_throughput']:.2f}tok/s")

if __name__ == "__main__":
    BENCH_DIR = pathlib.Path("/home/r0b0tdgx/projects/agents-a1-nvfp4-sm121-vllm/benchmarks/agents-a1-nvfp4-optimized-20260701T232200Z")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["BENCH_DIR"] = str(BENCH_DIR)
    asyncio.run(main())
