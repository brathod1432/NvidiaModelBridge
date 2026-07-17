# Latest NVIDIA Benchmark Recommendations

Generated from local audit:

`audits/nvidia_model_benchmark_2026-07-06_01-09-54.json`

This run used `NVIDIA_BENCHMARK_TIMEOUT_SECONDS=120` and `NVIDIA_DISCOVERY_TEST_LIMIT=10`.

## Environment

- Checked variable: `NVIDIA_API_KEY`
- Key source: Windows User environment
- Masked key: `nvapi-****aLTk`
- `/v1/models`: success
- Discovered models: 121
- Priority benchmark workers: 1
- Discovery candidate workers: 5
- Total benchmark tasks: 39
- Successful benchmark tasks: 28
- Failed benchmark tasks: 11
- Priority smoke tests: 3/3 passed

## Priority Model Confirmation

All three current priority models were found in `/v1/models`, passed the sequential smoke test, and passed simple, coding, and reasoning benchmark tasks.

| Priority rank | Model | Smoke | Full benchmark | Avg latency | Score |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | pass | 3/3 | 2.20s | 0.820 |
| 2 | `deepseek-ai/deepseek-v4-pro` | pass | 3/3 | 4.09s | 0.810 |
| 3 | `qwen/qwen3.5-122b-a10b` | pass | 3/3 | 56.26s | 0.801 |

## Overall Discovered Ranking

The overall ranking includes the 3 priority models plus 10 selected non-priority candidates discovered from `/v1/models`. It is intentionally separate from the priority ranking.

| Overall rank | Model | Priority | Result | Avg latency | Score |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | `qwen/qwen3-next-80b-a3b-instruct` | no | 3/3 | 1.13s | 0.838 |
| 2 | `nvidia/nemotron-mini-4b-instruct` | no | 3/3 | 1.37s | 0.832 |
| 3 | `mistralai/mistral-nemotron` | no | 3/3 | 1.56s | 0.828 |
| 4 | `nvidia/nemotron-nano-12b-v2-vl` | no | 3/3 | 1.89s | 0.823 |
| 5 | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | yes | 3/3 | 2.20s | 0.820 |
| 6 | `openai/gpt-oss-20b` | no | 3/3 | 2.55s | 0.817 |
| 7 | `deepseek-ai/deepseek-v4-pro` | yes | 3/3 | 4.09s | 0.810 |
| 8 | `qwen/qwen3.5-122b-a10b` | yes | 3/3 | 56.26s | 0.801 |

## Recommendations

- Default model: `qwen/qwen3-next-80b-a3b-instruct`
- Reasoning model: `qwen/qwen3-next-80b-a3b-instruct`
- Coding model: `qwen/qwen3-next-80b-a3b-instruct`
- Fast fallback model: `nvidia/nemotron-mini-4b-instruct`
- Priority-list default if routing must stay inside the three priority models: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`

## Additional Good Models Found

- `qwen/qwen3-next-80b-a3b-instruct`
- `nvidia/nemotron-mini-4b-instruct`
- `mistralai/mistral-nemotron`
- `nvidia/nemotron-nano-12b-v2-vl`
- `openai/gpt-oss-20b`

## Retest Or Avoid

- Retest: `qwen/qwen3.5-397b-a17b` timed out on all three tasks in this run.
- Avoid for now: `deepseek-ai/deepseek-coder-6.7b-instruct` and `nvidia/nemotron-4-340b-instruct` were listed by `/v1/models` but returned chat-completions 404 for this account.
- Do not treat `deepseek-ai/deepseek-v4-flash` or `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` as hard avoids based on this run alone; both returned usable content but missed one local evaluator requirement.

## Investigation Note

The earlier priority-model concern was caused by benchmark harness behavior, not by the three priority models being unusable. The harness now runs priority benchmarks sequentially, retries transient SDK failures once, falls back to direct HTTP when appropriate, disables visible thinking for exact-match smoke/simple tasks, keeps thinking enabled for reasoning tasks, and keeps priority ranking separate from overall discovered ranking.
