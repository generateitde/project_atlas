from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    ok: bool
    obs_delta: Dict[str, Any]
    events: list[Dict[str, Any]]
    error: Optional[str] = None
