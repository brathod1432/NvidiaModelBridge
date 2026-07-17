"""Check NVIDIA Model Bridge environment visibility."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_environment_diagnostics, get_nvidia_api_key  # noqa: E402


def main() -> None:
    diagnostics = get_environment_diagnostics()
    key = get_nvidia_api_key()

    print("NVIDIA Model Bridge Environment Check")
    print()
    print("Python executable:")
    print(diagnostics["python_executable"])
    print()
    print("Working directory:")
    print(diagnostics["working_directory"])
    print()
    print("Project root:")
    print(diagnostics["project_root"])
    print()
    print("Loaded .env files:")
    loaded = diagnostics["loaded_env_files"]
    if loaded:
        for path in loaded:
            print(path)
    else:
        print("None")
    print()
    print("Checked variable:")
    print("NVIDIA_API_KEY")
    print()
    print("Result:")
    print("FOUND" if key else "MISSING")
    print()
    print("Selected source:")
    print(diagnostics["api_key_source"])
    print()
    print("NVIDIA_API_KEY found:")
    print(str(bool(key)).lower())
    print()
    print("Masked key:")
    print(diagnostics["masked_api_key"])
    print()
    print("Injected for this run:")
    print(diagnostics["injected_for_run"])
    if diagnostics["injection_note"]:
        print(diagnostics["injection_note"])
    print()
    if diagnostics["key_shape_warning"]:
        print("Key shape warning:")
        print(diagnostics["key_shape_warning"])
        print()
    print("Source details:")
    for source in diagnostics["sources"]:
        print(
            f"- {source['source']}: found={source['found']}, "
            f"masked={source['masked']}, length={source['length']}, "
            f"starts_with_nvapi={source['starts_with_nvapi']}, "
            f"whitespace={source['had_leading_or_trailing_whitespace']}, "
            f"newline={source['has_newline']}"
        )
    print()
    print("NVIDIA_API_KEY present:", bool(os.getenv("NVIDIA_API_KEY")))


if __name__ == "__main__":
    main()
