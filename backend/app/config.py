"""Runtime configuration from environment variables (and an optional .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines; never overrides real env vars."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    use_anthropic_direct: bool = _bool("USE_ANTHROPIC_DIRECT", True)
    claude_model: str = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
    claude_effort: str = os.environ.get("CLAUDE_EFFORT", "high")
    claude_fallbacks: bool = _bool("CLAUDE_FALLBACKS", True)
    ai_max_tool_rounds: int = int(os.environ.get("AI_MAX_TOOL_ROUNDS", "8"))
    ai_rate_limit_per_min: int = int(os.environ.get("AI_RATE_LIMIT_PER_MIN", "12"))
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()
    )

    @property
    def ai_configured(self) -> bool:
        return self.use_anthropic_direct and bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )


settings = Settings()
