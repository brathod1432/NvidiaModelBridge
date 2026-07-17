"""Prepare a local .env file from the Windows User environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
ENV_FILE_LINES = [
    "NVIDIA_API_KEY={key}",
    "NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1",
    "NVIDIA_MODEL_BRIDGE_HOST=0.0.0.0",
    "NVIDIA_MODEL_BRIDGE_PORT=8000",
    "NVIDIA_DEFAULT_TASK_TYPE=general",
    "NVIDIA_SERVICE_TIMEOUT_SECONDS=90",
]
INVALID_PLACEHOLDERS = {
    "",
    "$nvidia_api_key",
    "${nvidia_api_key}",
    "your_api_key_here",
    "your_key_here",
    "nvapi-your-key-here",
    "changeme",
    "none",
    "null",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_env_is_gitignored()
    key = get_windows_user_key() or get_process_key()
    cleaned_key = clean_key(key)
    found = bool(cleaned_key)

    created = False
    updated = False
    if found:
        created, updated = write_env_file(cleaned_key, force=args.force)

    print(f"NVIDIA_API_KEY found: {'yes' if found else 'no'}")
    print(f"Masked key: {mask_key(cleaned_key)}")
    print(f".env created: {'yes' if created else 'no'}")
    print(f".env updated: {'yes' if updated else 'no'}")


def get_windows_user_key() -> str | None:
    try:
        import winreg
    except Exception:
        return None

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "NVIDIA_API_KEY")
    except Exception:
        return None
    return str(value) if value is not None else None


def get_process_key() -> str | None:
    return os.getenv("NVIDIA_API_KEY")


def clean_key(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'")
    if cleaned.lower() in INVALID_PLACEHOLDERS:
        return None
    return cleaned


def mask_key(key: str | None) -> str:
    if not key:
        return ""
    if key.startswith("nvapi-") and len(key) > 10:
        return f"nvapi-****{key[-4:]}"
    if len(key) <= 8:
        return "****"
    return f"****{key[-4:]}"


def write_env_file(key: str, force: bool) -> tuple[bool, bool]:
    existing = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    existing_map = parse_env_lines(existing)

    if ENV_PATH.exists() and existing_map.get("NVIDIA_API_KEY") and not force:
        return False, False

    desired = {
        "NVIDIA_API_KEY": key,
        "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
        "NVIDIA_MODEL_BRIDGE_HOST": "0.0.0.0",
        "NVIDIA_MODEL_BRIDGE_PORT": "8000",
        "NVIDIA_DEFAULT_TASK_TYPE": "general",
        "NVIDIA_SERVICE_TIMEOUT_SECONDS": "90",
    }
    existing_map.update(desired)
    rendered = "\n".join(f"{name}={value}" for name, value in desired.items()) + "\n"
    ENV_PATH.write_text(rendered, encoding="utf-8")
    return not bool(existing), True


def parse_env_lines(lines: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        parsed[name.strip()] = value.strip()
    return parsed


def ensure_env_is_gitignored() -> None:
    gitignore = PROJECT_ROOT / ".gitignore"
    if not gitignore.exists():
        raise RuntimeError(".gitignore is missing; cannot confirm .env is ignored.")
    contents = gitignore.read_text(encoding="utf-8").splitlines()
    if not any(line.strip() in {".env", ".env.*"} for line in contents):
        raise RuntimeError(".env is not gitignored. Add it to .gitignore before using this script.")


if __name__ == "__main__":
    main()
