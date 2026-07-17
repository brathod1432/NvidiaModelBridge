# NVIDIA UI Model References

These model IDs and settings came from NVIDIA UI snippets provided by the project owner. They are preserved here as local reference material for the benchmark registry and later gateway work.

## Curated Models

| # | Current priority | Model ID | Display name | Provider | Category | Recommended use |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | yes | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Nemotron 3 Nano Omni 30B A3B Reasoning | NVIDIA | reasoning | reasoning, thinking mode, NVIDIA-native model |
| 2 | yes | `deepseek-ai/deepseek-v4-pro` | DeepSeek V4 Pro | DeepSeek AI | reasoning/coding | reasoning, coding, high-quality answers |
| 3 | yes | `qwen/qwen3.5-122b-a10b` | Qwen 3.5 122B A10B | Qwen | general/coding/reasoning | coding, general reasoning, large model comparison |
| 4 | no | `z-ai/glm-5.2` | GLM 5.2 | Z.ai | reasoning/general | general reasoning, long response, agent testing |
| 5 | no | `moonshotai/kimi-k2.6` | Kimi K2.6 | Moonshot AI | general/reasoning | long-form reasoning, general assistant, coding comparison |
| 6 | no | `deepseek-ai/deepseek-v4-flash` | DeepSeek V4 Flash | DeepSeek AI | reasoning/fast | fast reasoning, fallback model, high-reasoning comparison |
| 7 | no | `google/gemma-4-31b-it` | Gemma 4 31B IT | Google | general/reasoning | general assistant, reasoning comparison, lightweight fallback |
| 8 | no | `openai/gpt-oss-120b` | GPT OSS 120B | OpenAI OSS | reasoning/general | fast high-quality general reasoning, default candidate |
| 9 | no | `openai/gpt-oss-20b` | GPT OSS 20B | OpenAI OSS | fast/general | fastest fallback, simple tasks, cheap/light calls |
| 10 | no | `meta/llama-3.3-70b-instruct` | Llama 3.3 70B Instruct | Meta | general | stable general assistant, summarization, safe baseline |
| 11 | no | `microsoft/phi-4-multimodal-instruct` | Phi 4 Multimodal Instruct | Microsoft | multimodal/general | multimodal testing later, simple text test now |

## Original Reference Snippets

### NVIDIA UI snippet 1

```python
from openai import OpenAI
import os
import sys

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR = "\033[0m" if _USE_COLOR else ""

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "$NVIDIA_API_KEY"
)

completion = client.chat.completions.create(
  model="z-ai/glm-5.2",
  messages=[{"role":"user","content":""}],
  temperature=1,
  top_p=1,
  max_tokens=16384,
  seed=42,
  stream=True
)
```

### NVIDIA UI snippet 2

```python
completion = client.chat.completions.create(
  model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
  messages=[{"role":"user","content":""}],
  temperature=0.6,
  top_p=0.95,
  max_tokens=65536,
  extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384},
  stream=False
)
```

### NVIDIA UI snippet 3

```python
payload = {
  "model": "moonshotai/kimi-k2.6",
  "messages": [{"role":"user","content":""}],
  "max_tokens": 16384,
  "temperature": 1.00,
  "top_p": 1.00,
  "stream": False
}
```

### NVIDIA UI snippet 4

```python
completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-v4-pro",
  messages=[{"role":"user","content":""}],
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  extra_body={"chat_template_kwargs":{"thinking":False}},
  stream=False
)
```

### NVIDIA UI snippet 5

```python
completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-v4-flash",
  messages=[{"role":"user","content":""}],
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  extra_body={"chat_template_kwargs":{"thinking":True,"reasoning_effort":"high"}},
  stream=False
)
```

### NVIDIA UI snippet 6

```python
payload = {
  "model": "google/gemma-4-31b-it",
  "messages": [{"role":"user","content":""}],
  "max_tokens": 16384,
  "temperature": 1.00,
  "top_p": 0.95,
  "stream": False,
  "chat_template_kwargs": {"enable_thinking":True}
}
```

### NVIDIA UI snippet 7

```python
payload = {
  "model": "qwen/qwen3.5-122b-a10b",
  "messages": [{"role":"user","content":""}],
  "max_tokens": 16384,
  "temperature": 0.60,
  "top_p": 0.95,
  "stream": False
}
```

### NVIDIA UI snippet 8

```python
completion = client.chat.completions.create(
  model="openai/gpt-oss-120b",
  messages=[{"role":"user","content":""}],
  temperature=1,
  top_p=1,
  max_tokens=4096,
  stream=False
)
```

### NVIDIA UI snippet 9

```python
completion = client.chat.completions.create(
  model="openai/gpt-oss-20b",
  messages=[{"role":"user","content":""}],
  temperature=1,
  top_p=1,
  max_tokens=4096,
  stream=False
)
```

### NVIDIA UI snippet 10

```python
completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[{"role":"user","content":""}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=False
)
```

### NVIDIA UI snippet 11

```python
payload = {
  "model": "microsoft/phi-4-multimodal-instruct",
  "messages": [{"role":"user","content":""}],
  "max_tokens": 512,
  "temperature": 0.10,
  "top_p": 0.70,
  "frequency_penalty": 0.00,
  "presence_penalty": 0.00,
  "stream": False
}
```

## Notes

- The project reads exactly one key variable: `NVIDIA_API_KEY`.
- On Windows, the benchmark can read `NVIDIA_API_KEY` from the Windows User environment even when the current Python process did not inherit it, then inject it into memory for that run only.
- Priority-model smoke tests should be run sequentially. This keeps transient provider errors, rate limits, or slow responses tied to the specific model being checked instead of creating avoidable parallel pressure on the same priority set.
- Full priority-model benchmark runs should use realistic per-call timeouts. Short smoke-test timeouts are useful for quick infrastructure checks, but use at least the project default and prefer 60-90 seconds for reasoning-heavy prompts before drawing conclusions.
- Some examples use the OpenAI Python SDK, while others use direct HTTP payloads.
- Some models need additional fields passed through SDK `extra_body`.
- Some HTTP examples place fields such as `chat_template_kwargs`, `frequency_penalty`, and `presence_penalty` at the top level of the JSON payload.
- Some models may expose `reasoning_content` or `reasoning` on the response message.
- Several models support or expect thinking flags such as `enable_thinking`, `thinking`, `reasoning_effort`, or `reasoning_budget`.
- The UI examples include large `max_tokens` values for real use. The benchmark intentionally overrides those with smaller task-specific limits to avoid long or expensive tests.
- `/v1/models` discovery is separate from chat compatibility. A model can appear in discovery and still fail or require different parameters for `chat.completions`.
- Rate limits, 503s, and timeouts should be documented as benchmark conditions, not model quality judgments, unless the result includes rerun notes covering timeout, worker count, and observed provider throttling.
