# Nvidia Model Bridge

Nvidia Model Bridge is a local FastAPI gateway for routing prompts to NVIDIA-hosted OpenAI-compatible models.

Current benchmark snapshot:
- `NVIDIA_API_KEY` was detected in the Windows User environment and injected into the Python process for the audit run.
- `/v1/models` succeeded with status `200`.
- NVIDIA returned `121` discovered models.
- `3` priority models were tested.
- `10` discovered candidate models were tested.
- Total tasks: `39`
- Total successes: `28`
- Total failures: `11`

Recommended routing models:
- default/general: `qwen/qwen3-next-80b-a3b-instruct`
- reasoning: `qwen/qwen3-next-80b-a3b-instruct`
- coding: `qwen/qwen3-next-80b-a3b-instruct`
- nvidia reasoning: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- fast: `nvidia/nemotron-mini-4b-instruct`
- general fallback: `mistralai/mistral-nemotron`
- lightweight fallback: `openai/gpt-oss-20b`
- deepseek fallback: `deepseek-ai/deepseek-v4-pro`

Priority models:
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- `deepseek-ai/deepseek-v4-pro`
- `qwen/qwen3.5-122b-a10b`

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare `.env`

Use the local helper to copy `NVIDIA_API_KEY` from the Windows User environment or the current Python process:

```bash
python scripts/prepare_env.py
```

The script keeps `.env` local, masks the key in output, and refuses to overwrite an existing key unless `--force` is passed.

## Run the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

or:

```bash
python main.py
```

## Endpoints

- `GET /health`
- `GET /models/priority`
- `GET /models/recommended`
- `GET /models/avoid`
- `POST /ask`
- `POST /v1/chat/completions`

`/ask` uses coordinator routing when `model` is omitted. `/v1/chat/completions` requires an explicit model and mirrors OpenAI-compatible behavior.

## Validation

```bash
python scripts/check_env.py
python scripts/test_nvidia_models.py
curl http://localhost:8000/health
curl http://localhost:8000/models/recommended
curl http://localhost:8000/models/priority
```

PowerShell examples:

```powershell
Invoke-RestMethod `
  -Uri http://localhost:8000/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt":"Write a Python function to reverse a string.","task_type":"coding"}'
```

```powershell
Invoke-RestMethod `
  -Uri http://localhost:8000/v1/chat/completions `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"model":"qwen/qwen3-next-80b-a3b-instruct","messages":[{"role":"user","content":"Say hello from Nvidia Model Bridge."}],"temperature":0,"max_tokens":80,"stream":false}'
```

## Docs

- `docs/model_selection_notes.md`
- `docs/nvidia_priority_models.md`
- `docs/discovered_nvidia_models.md`
- `docs/service_api.md`
- `docs/docker_setup.md`
- `docs/env_setup.md`
