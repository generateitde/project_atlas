from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Event:
    type: str
    payload: Dict[str, Any]
