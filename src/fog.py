from __future__ import annotations

from dataclasses import dataclass

from colour import Color

__all__ = ["FogLinear", "FogExp2", "Fog"]


@dataclass
class FogLinear:
    color: Color
    near: float = 1.0
    far: float = 100.0


@dataclass
class FogExp2:
    color: Color
    density: float = 0.00025


Fog = FogLinear | FogExp2 | None
