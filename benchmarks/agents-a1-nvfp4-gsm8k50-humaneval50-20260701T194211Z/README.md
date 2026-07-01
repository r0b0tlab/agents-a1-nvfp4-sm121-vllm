# Agents-A1 NVFP4 benchmark run agents-a1-nvfp4-gsm8k50-humaneval50-20260701T194211Z

Endpoint: `http://127.0.0.1:18080/v1`  
Model: `agents-a1-nvfp4`  
Effective max model length: `4096`

## Scores

| Suite | Harness | Samples | Result | Notes |
|---|---:|---:|---:|---|
| GSM8K | lm-eval `gsm8k` | 50 | strict 98.00%, flexible 98.00% | `num_concurrent=2`, thinking disabled |
| HumanEval | direct OpenAI-compatible evaluator | 50 | 48/50 (96.00%) | code extracted/evaluated locally, `num_concurrent=2` |
| HumanEval | stock lm-eval `humaneval` | 50 | 0.00% | preserved as evidence; stock stop rules truncate chat-model function output |

## Concurrency sweep

| c | success | mean wall s (last 2) | mean agg output tok/s (last 2) |
|---:|---:|---:|---:|
| 1 | 3/3 | 0.171 | 17.52 |
| 2 | 6/6 | 0.178 | 33.72 |
| 4 | 12/12 | 0.387 | 31.21 |
| 8 | 24/24 | 0.751 | 32.14 |

## Telemetry

| Metric | Avg | Max | Samples |
|---|---:|---:|---:|
| GPU power draw W | 27.88 | 36.00 | 166 |
| GPU utilization % | 70.75 | 96.00 | 166 |
| GPU temperature C | 58.83 | 65.00 | 166 |

## Key files

- `summary.json`
- `manifest.json`
- `gsm8k_50.log`
- `humaneval_50.log`
- `direct_humaneval50.log`
- `concurrency_sweep.json`
- `telemetry_gpu.csv`
- `telemetry_gpu_direct_humaneval.csv`
- `MANIFEST.sha256`
