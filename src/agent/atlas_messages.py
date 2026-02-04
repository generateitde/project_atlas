from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AtlasMessage:
    msg_type: str
    text: str
