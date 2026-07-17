# NVIDIA Priority Models

Current priority set:

1. `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
1. `deepseek-ai/deepseek-v4-pro`
1. `qwen/qwen3.5-122b-a10b`

| Rank | Model ID | Status | Avg latency | Benchmark | Notes |
| ---: | --- | --- | ---: | --- | --- |
| 1 | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `priority` | `2.20s` | `3/3` | Official NVIDIA-native reasoning priority model. |
| 2 | `deepseek-ai/deepseek-v4-pro` | `priority` | `4.09s` | `3/3` | Strong fallback and one of the official priority models. |
| 3 | `qwen/qwen3.5-122b-a10b` | `priority_but_slow` | `56.26s` | `3/3` | Priority coverage is intact, but it is much slower than the default route. |

## Recommendation

Keep the three models above in the priority list for validation and fallback planning, but use `qwen/qwen3-next-80b-a3b-instruct` as the default router target for general, reasoning, and coding prompts.
