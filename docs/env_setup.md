# Environment Setup

The project uses only one NVIDIA credential variable:

```text
NVIDIA_API_KEY
```

Do not hardcode the key. Do not print the full key. Keep `.env` local and gitignored.

## Prepare `.env`

Populate a local `.env` file from the Windows User environment or current Python process:

```bash
python scripts/prepare_env.py
```

If `.env` already contains `NVIDIA_API_KEY`, the script leaves it alone unless `--force` is passed.

## Example `.env`

```env
NVIDIA_API_KEY=...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL_BRIDGE_HOST=0.0.0.0
NVIDIA_MODEL_BRIDGE_PORT=8000
NVIDIA_DEFAULT_TASK_TYPE=general
NVIDIA_SERVICE_TIMEOUT_SECONDS=90
```

## Local development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_env.py
```

## Notes

- `.env` is ignored by git.
- `.env.example` is safe to commit.
- The benchmark scripts still use `NVIDIA_API_KEY` only.
