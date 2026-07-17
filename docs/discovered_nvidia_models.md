# Discovered NVIDIA Models

Latest audit snapshot:

- Source: `GET /v1/models`
- Timestamp: `2026-07-06T01:09:54`
- Total discovered: `121`

## Tested discovered candidates

| Model ID | Status | Avg latency | Benchmark | Notes |
| --- | --- | ---: | --- | --- |
| `qwen/qwen3-next-80b-a3b-instruct` | PASS | `1.13s` | `3/3` | Best overall discovered candidate. |
| `nvidia/nemotron-mini-4b-instruct` | PASS | `1.37s` | `3/3` | Fast/light candidate. |
| `mistralai/mistral-nemotron` | PASS | `1.56s` | `3/3` | Strong general fallback. |
| `nvidia/nemotron-nano-12b-v2-vl` | PASS | `1.89s` | `3/3` | Strong multimodal-capable candidate. |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | PASS | `2.20s` | `3/3` | Official priority reasoning model. |
| `openai/gpt-oss-20b` | PASS | `2.55s` | `3/3` | Lightweight fallback candidate. |
| `deepseek-ai/deepseek-v4-pro` | PASS | `4.09s` | `3/3` | Official priority DeepSeek route. |
| `qwen/qwen3.5-122b-a10b` | PASS | `56.26s` | `3/3` | Official priority model, but slow. |
| `deepseek-ai/deepseek-v4-flash` | FAIL | `32.49s` | `2/3` | Partial candidate. |
| `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | FAIL | `2.18s` | `2/3` | Partial candidate. |
| `qwen/qwen3.5-397b-a17b` | FAIL | `240.73s` | `0/3` | Timeout-heavy; retest later with sequential mode and a longer timeout. |
| `deepseek-ai/deepseek-coder-6.7b-instruct` | FAIL | `0.22s` | `0/3` | Unavailable for the current account. |
| `nvidia/nemotron-4-340b-instruct` | FAIL | `0.22s` | `0/3` | Unavailable for the current account. |

## Partial models

- `deepseek-ai/deepseek-v4-flash`
- `nvidia/llama-3.1-nemotron-nano-vl-8b-v1`

## Avoid models

- `deepseek-ai/deepseek-coder-6.7b-instruct`
- `nvidia/nemotron-4-340b-instruct`

## Retest later

- `qwen/qwen3.5-397b-a17b`

The full discovered list remains in the latest audit JSON under `audits/`.
