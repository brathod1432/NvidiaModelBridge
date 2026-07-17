# Model Selection Notes

## Latest benchmark summary

- Discovery succeeded with `121` models returned from `GET /v1/models`.
- `3` priority models were tested.
- `10` discovered candidates were tested.
- Total tasks: `39`
- Total successes: `28`
- Total failures: `11`

## Working models

Latest best performers:

1. `qwen/qwen3-next-80b-a3b-instruct`
1. `nvidia/nemotron-mini-4b-instruct`
1. `mistralai/mistral-nemotron`
1. `nvidia/nemotron-nano-12b-v2-vl`
1. `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
1. `openai/gpt-oss-20b`
1. `deepseek-ai/deepseek-v4-pro`
1. `qwen/qwen3.5-122b-a10b`

## Failed or partial models

- `deepseek-ai/deepseek-v4-flash` - partial candidate, do not use as default yet.
- `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` - partial candidate, do not use as default yet.
- `deepseek-ai/deepseek-coder-6.7b-instruct` - unavailable for the current account.
- `nvidia/nemotron-4-340b-instruct` - unavailable for the current account.
- `qwen/qwen3.5-397b-a17b` - timed out on all tasks; retest later with sequential mode and a longer timeout.

## Router behavior

- `general` -> `qwen/qwen3-next-80b-a3b-instruct`
- `coding` -> `qwen/qwen3-next-80b-a3b-instruct`
- `reasoning` -> `qwen/qwen3-next-80b-a3b-instruct`
- `nvidia_reasoning` -> `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- `json` -> `qwen/qwen3-next-80b-a3b-instruct`
- `fast` -> `nvidia/nemotron-mini-4b-instruct`
- `fallback` -> `mistralai/mistral-nemotron`
- `deepseek` -> `deepseek-ai/deepseek-v4-pro`
- `lightweight` -> `openai/gpt-oss-20b`

If a model is omitted from `/ask`, the coordinator chooses one of the routes above. If a model is explicitly supplied, the gateway forwards to that exact model and does not silently switch models.

## Selection guidance

- Use `qwen/qwen3-next-80b-a3b-instruct` for default, reasoning, coding, and structured prompts.
- Use `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` when you want a native NVIDIA reasoning route.
- Use `nvidia/nemotron-mini-4b-instruct` for fast or lightweight tasks.
- Use `mistralai/mistral-nemotron` as the general fallback.
- Use `openai/gpt-oss-20b` as the lightweight fallback.
- Use `deepseek-ai/deepseek-v4-pro` for DeepSeek-style fallback and strong general reasoning.
- Keep `qwen/qwen3.5-122b-a10b` in the priority set even though it is slow.

## Retest candidates

- `qwen/qwen3.5-397b-a17b`
- `deepseek-ai/deepseek-v4-flash`
- `nvidia/llama-3.1-nemotron-nano-vl-8b-v1`
