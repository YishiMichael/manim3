from __future__ import annotations

from utils.arrays import Mat3


class Texture:
    def __init__(self, url: str):
        self.url: str = url
        self.matrix: Mat3 = Mat3()
