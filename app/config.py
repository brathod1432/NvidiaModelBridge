"""Configuration helpers for the NVIDIA Model Bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
try:
    import winreg
except ImportError:  # pragma: no cover - Windows-only lookup.
    winreg = None

from dotenv import dotenv_values, load_dotenv


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
CHECKED_API_KEY_VARIABLE = "NVIDIA_API_KEY"
DEFAULT_MODEL_BRIDGE_HOST = "0.0.0.0"
DEFAULT_MODEL_BRIDGE_PORT = 8000
DEFAULT_DEFAULT_TASK_TYPE = "general"
DEFAULT_SERVICE_TIMEOUT_SECONDS = 90
INVALID_API_KEY_PLACEHOLDERS = {
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


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_environment() -> list[str]:
    """Load the project-root .env without overriding process environment."""

    loaded: list[str] = []
    env_path = get_project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        loaded.append(str(env_path))
    return loaded


def _load_env_value(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def resolve_nvidia_api_key(inject: bool = True) -> dict[str, object]:
    """Resolve NVIDIA_API_KEY from process, .env, Windows User, or Machine."""

    env_path = get_project_root() / ".env"
    loaded_env_files: list[str] = []
    process_before = os.getenv("NVIDIA_API_KEY")
    env_file_value: str | None = None
    if env_path.exists():
        env_file_value = dotenv_values(env_path).get("NVIDIA_API_KEY")
        load_dotenv(env_path, override=False)
        loaded_env_files.append(str(env_path))

    sources = [
        ("Python process", process_before),
        ("Project .env", env_file_value),
        ("Windows User environment", _read_windows_environment("User")),
        ("Windows Machine environment", _read_windows_environment("Machine")),
    ]

    selected_key: str | None = None
    selected_source = "missing"
    for source_name, raw_value in sources:
        cleaned = _clean_api_key(raw_value)
        if cleaned:
            selected_key = cleaned
            selected_source = source_name
            break

    injected_for_run = False
    injection_note = ""
    if inject and selected_key and not _clean_api_key(os.getenv("NVIDIA_API_KEY")):
        os.environ["NVIDIA_API_KEY"] = selected_key
        injected_for_run = True
        injection_note = (
            f"NVIDIA_API_KEY found in {selected_source} and injected into "
            "current Python process for this run."
        )

    safe_sources = [
        {
            "source": source_name,
            "found": _clean_api_key(raw_value) is not None,
            "masked": mask_api_key(_clean_api_key(raw_value)),
            "length": len(_clean_api_key(raw_value) or ""),
            "starts_with_nvapi": bool(
                (_clean_api_key(raw_value) or "").startswith("nvapi-")
            ),
            "had_leading_or_trailing_whitespace": bool(
                raw_value and raw_value != raw_value.strip()
            ),
            "has_newline": bool(raw_value and ("\n" in raw_value or "\r" in raw_value)),
        }
        for source_name, raw_value in sources
    ]
    return {
        "api_key": selected_key,
        "api_key_source": selected_source,
        "loaded_env_files": loaded_env_files,
        "injected_for_run": injected_for_run,
        "injection_note": injection_note,
        "sources": safe_sources,
    }


def get_nvidia_api_key() -> str | None:
    """Return the NVIDIA API key from NVIDIA_API_KEY only."""

    return resolve_nvidia_api_key()["api_key"] or None


def get_api_key() -> str:
    """Backward-compatible alias that still reads NVIDIA_API_KEY only."""

    return get_nvidia_api_key() or ""


def get_api_key_source() -> str:
    """Return the single supported environment variable or missing."""

    return str(resolve_nvidia_api_key()["api_key_source"])


def mask_api_key(key: str | None) -> str:
    """Mask an API key for console and audit output."""

    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    if key.startswith("nvapi-"):
        return "nvapi-****" + key[-4:]
    return "****" + key[-4:]


def get_key_shape_warning(key: str | None) -> str:
    if not key:
        return ""
    if not key.startswith("nvapi-"):
        return "NVIDIA_API_KEY does not start with expected nvapi- prefix."
    return ""


def env_file_has_variable(path: Path, variable_name: str) -> bool:
    if not path.exists():
        return False
    prefix = f"{variable_name}="
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(prefix):
            return True
    return False


def project_root_gitignored(variable_name: str = ".env") -> bool:
    gitignore = get_project_root() / ".gitignore"
    if not gitignore.exists():
        return False
    patterns = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
    if variable_name == ".env":
        return any(line == ".env" or line == ".env.*" for line in patterns)
    return variable_name in patterns


def get_environment_diagnostics() -> dict[str, object]:
    process_value_present = bool(os.getenv("NVIDIA_API_KEY"))
    resolution = resolve_nvidia_api_key()
    key = resolution["api_key"]
    return {
        "python_executable": os.sys.executable,
        "working_directory": str(Path.cwd()),
        "project_root": str(get_project_root()),
        "loaded_env_files": resolution["loaded_env_files"],
        "checked_variable": CHECKED_API_KEY_VARIABLE,
        "process_variable_present_before_dotenv": process_value_present,
        "api_key_found": key is not None,
        "api_key_source": resolution["api_key_source"],
        "masked_api_key": mask_api_key(key),
        "key_shape_warning": get_key_shape_warning(key),
        "injected_for_run": resolution["injected_for_run"],
        "injection_note": resolution["injection_note"],
        "sources": resolution["sources"],
    }


def _clean_api_key(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    if value.lower() in INVALID_API_KEY_PLACEHOLDERS:
        return None
    return value


def _read_windows_environment(target: str) -> str | None:
    if winreg is None:
        return None
    if target == "User":
        root = winreg.HKEY_CURRENT_USER
        path = "Environment"
    elif target == "Machine":
        root = winreg.HKEY_LOCAL_MACHINE
        path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    else:
        return None
    try:
        with winreg.OpenKey(root, path) as key:
            value, _ = winreg.QueryValueEx(key, "NVIDIA_API_KEY")
    except OSError:
        return None
    return str(value) if value is not None else None


@dataclass(frozen=True)
class NvidiaSettings:
    api_key: str
    api_key_source: str
    checked_variable: str = CHECKED_API_KEY_VARIABLE
    loaded_env_files: tuple[str, ...] = ()
    project_root: str = ""
    working_directory: str = ""
    key_shape_warning: str = ""
    api_key_injected_for_run: bool = False
    api_key_injection_note: str = ""
    environment_sources: tuple[dict[str, object], ...] = ()
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: int = 45
    model_limit: int = 0
    enable_discovery: bool = True
    discovery_test_limit: int = 10
    test_streaming: bool = False
    max_workers: int = 5

    @classmethod
    def load(cls) -> "NvidiaSettings":
        resolution = resolve_nvidia_api_key()
        api_key = str(resolution["api_key"] or "")
        return cls(
            api_key=api_key,
            api_key_source=str(resolution["api_key_source"]),
            checked_variable=CHECKED_API_KEY_VARIABLE,
            loaded_env_files=tuple(str(path) for path in resolution["loaded_env_files"]),
            project_root=str(get_project_root()),
            working_directory=str(Path.cwd()),
            key_shape_warning=get_key_shape_warning(api_key),
            api_key_injected_for_run=bool(resolution["injected_for_run"]),
            api_key_injection_note=str(resolution["injection_note"]),
            environment_sources=tuple(resolution["sources"]),
            base_url=os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            timeout_seconds=_parse_int(
                os.getenv("NVIDIA_BENCHMARK_TIMEOUT_SECONDS"), 45
            ),
            model_limit=max(
                0, _parse_int(os.getenv("NVIDIA_BENCHMARK_MODEL_LIMIT"), 0)
            ),
            enable_discovery=_parse_bool(os.getenv("NVIDIA_ENABLE_DISCOVERY"), True),
            discovery_test_limit=max(
                0, _parse_int(os.getenv("NVIDIA_DISCOVERY_TEST_LIMIT"), 10)
            ),
            test_streaming=_parse_bool(os.getenv("NVIDIA_TEST_STREAMING"), False),
            max_workers=max(
                1, _parse_int(os.getenv("NVIDIA_BENCHMARK_MAX_WORKERS"), 5)
            ),
        )

    @property
    def masked_api_key(self) -> str:
        return mask_api_key(self.api_key)


@dataclass(frozen=True)
class BridgeSettings:
    """Runtime settings for the FastAPI gateway."""

    api_key: str
    api_key_source: str
    base_url: str = DEFAULT_BASE_URL
    host: str = DEFAULT_MODEL_BRIDGE_HOST
    port: int = DEFAULT_MODEL_BRIDGE_PORT
    default_task_type: str = DEFAULT_DEFAULT_TASK_TYPE
    timeout_seconds: int = DEFAULT_SERVICE_TIMEOUT_SECONDS
    project_root: str = ""
    loaded_env_files: tuple[str, ...] = ()
    api_key_injected_for_run: bool = False
    api_key_injection_note: str = ""

    @classmethod
    def load(cls) -> "BridgeSettings":
        resolution = resolve_nvidia_api_key()
        api_key = str(resolution["api_key"] or "")
        return cls(
            api_key=api_key,
            api_key_source=str(resolution["api_key_source"]),
            base_url=_load_env_value("NVIDIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            host=_load_env_value("NVIDIA_MODEL_BRIDGE_HOST", DEFAULT_MODEL_BRIDGE_HOST),
            port=_parse_int(
                os.getenv("NVIDIA_MODEL_BRIDGE_PORT"), DEFAULT_MODEL_BRIDGE_PORT
            ),
            default_task_type=_load_env_value(
                "NVIDIA_DEFAULT_TASK_TYPE", DEFAULT_DEFAULT_TASK_TYPE
            ),
            timeout_seconds=max(
                1,
                _parse_int(
                    os.getenv("NVIDIA_SERVICE_TIMEOUT_SECONDS"),
                    DEFAULT_SERVICE_TIMEOUT_SECONDS,
                ),
            ),
            project_root=str(get_project_root()),
            loaded_env_files=tuple(str(path) for path in resolution["loaded_env_files"]),
            api_key_injected_for_run=bool(resolution["injected_for_run"]),
            api_key_injection_note=str(resolution["injection_note"]),
        )

    @property
    def masked_api_key(self) -> str:
        return mask_api_key(self.api_key)

    @property
    def api_key_found(self) -> bool:
        return bool(self.api_key)
