"""Configuration loading: config.yaml provides defaults, .env overrides."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment-driven settings. See .env.example."""

    model_config = SettingsConfigDict(env_prefix="SWAR_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "auto"          # auto | cuda | cpu
    spoof_model: str = "MelodyMachine/Deepfake-audio-detection-V2"
    model_dir: Path = ROOT / "data" / "models"
    force_fallback: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config() -> dict[str, Any]:
    """Parsed config.yaml. Cached; restart the server after editing it."""
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_device() -> str:
    """Return the torch device string, honouring SWAR_DEVICE=auto.

    An explicit SWAR_DEVICE=cuda on a box without CUDA raises rather than
    quietly falling back: discovering on video that the "GPU" demo is running
    on CPU at 4 s/window is worse than failing at startup.
    """
    import torch

    requested = get_settings().device.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SWAR_DEVICE=cuda but torch reports no CUDA device")
    return requested
