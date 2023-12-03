from __future__ import annotations


import itertools
import time
from typing import (
    Iterator,
    Self
)

from .toplevel import Toplevel
from .toplevel_resource import ToplevelResource


class Timer(ToplevelResource):
    __slots__ = (
        "_start_timestamp",
        "_current_timestamp",
        "_fps_update_timestamp",
        "_recorded_fps",
        "_next_fps"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        timestamp = time.perf_counter()
        self._start_timestamp: float = timestamp
        self._current_timestamp: float = timestamp
        self._fps_update_timestamp: float = timestamp
        self._recorded_fps: int = 0
        self._next_fps: int = 0

    def __contextmanager__(
        self: Self
    ) -> Iterator[None]:
        Toplevel._timer = self
        yield
        Toplevel._timer = None

    def frame_clock(
        self: Self
    ) -> Iterator[float]:
        spf = 1.0 / Toplevel._get_config().fps
        # Integer-based counter for higher precision.
        for frame_index in itertools.count():
            timestamp = time.perf_counter()
            self._current_timestamp = timestamp
            if timestamp - self._fps_update_timestamp >= 1.0:
                self._fps_update_timestamp = timestamp
                self._recorded_fps = self._next_fps
                self._next_fps = 0
            self._next_fps += 1
            yield frame_index * spf
            if Toplevel._get_renderer()._livestreamer.is_livestreaming and (sleep_time := spf - (
                time.perf_counter() - timestamp
            )) > 0.0:
                time.sleep(sleep_time)
