"""
config.py
=========
Centralized environment configuration using Pydantic v2 settings.

All values have safe local-dev defaults so the app runs out-of-the-box
with `uvicorn backend.main:app` and no .env file present.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "CaneTrash-AI"
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)
    api_prefix: str = "/api/v1"
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Database ---
    # Defaults to a local SQLite file so no external DB is required for dev/CI.
    database_url: str = Field(default="sqlite+aiosqlite:///./canetrash.db")

    # --- Hardware / Simulation toggles ---
    use_hardware_simulators: bool = Field(default=True)
    weighbridge_port: str = Field(default="/dev/ttyUSB0")
    weighbridge_baudrate: int = Field(default=9600)
    weighbridge_modbus_unit_id: int = Field(default=1)

    # --- AI Core ---
    segmentation_weights_path: str | None = Field(default=None)
    segmentation_confidence_threshold: float = Field(default=0.35)
    lidar_voxel_size_m: float = Field(default=0.05)
    force_synthetic_ai: bool = Field(default=True)  # phase-1 safe default

    # --- SLM Agent ---
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:3b")
    vllm_base_url: str | None = Field(default=None)

    # --- Mill business rules ---
    default_base_price_per_ton: float = Field(default=850_000.0)

    # --- Pipeline ---
    pipeline_scan_duration_s: float = Field(default=15.0)

    # --- Slip printer ---
    slip_printer_device: str = Field(default="mock://thermal_printer_01")


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor -- import and call get_settings() everywhere."""
    return Settings()
