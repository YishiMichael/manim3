from __future__ import annotations


import collections
import itertools
from typing import (
    Iterator,
    Self
)

import rich.box
import rich.live
import rich.table

from .toplevel import Toplevel
from .toplevel_resource import ToplevelResource


class Logger(ToplevelResource):
    __slots__ = ("_log_messages",)

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._log_messages: collections.deque[str] = collections.deque(maxlen=10)

    def __contextmanager__(
        self: Self
    ) -> Iterator[None]:
        Toplevel._logger = self
        with rich.live.Live(vertical_overflow="crop", get_renderable=self._get_table):
            yield
        Toplevel._logger = None

    def _get_table(
        self: Self
    ) -> rich.table.Table:
        table = rich.table.Table(
            rich.table.Column(header="Log Messages", no_wrap=True, overflow="crop", width=80),
            rich.table.Column(header="Status", no_wrap=True, overflow="crop", width=40),
            caption=f"manim3 v{__import__("manim3").__version__}",
            caption_justify="left",
            box=rich.box.ASCII
        )
        table.add_row(self._get_log_table(), self._get_status_table())
        return table

    def _get_log_table(
        self: Self
    ) -> rich.table.Table:
        log_table = rich.table.Table.grid(
            rich.table.Column(no_wrap=True, overflow="crop")
        )
        assert (maxlen := self._log_messages.maxlen) is not None
        for log_message in itertools.chain(
            self._log_messages,
            itertools.repeat("", maxlen - len(self._log_messages))
        ):
            log_table.add_row(log_message)
        return log_table

    def _get_status_table(
        self: Self
    ) -> rich.table.Table:

        def format_duration(
            duration: float
        ) -> str:
            minutes, seconds = divmod(int(duration), 60)
            return f"{minutes}:{seconds:02}"

        timer = Toplevel._get_timer()
        renderer = Toplevel._renderer
        scene = Toplevel._scene
        status_dict = {
            "Run Time": format_duration(timer._current_timestamp - timer._start_timestamp),
            "FPS": f"{timer._recorded_fps}",
            "Livestream": "-" if renderer is None else "[green]On" if renderer._livestreamer.is_livestreaming else "[red]Off",
            "Recording": "-" if renderer is None else "[green]On" if renderer._video_recorder.is_recording else "[red]Off",
            "Scene Name": "-" if scene is None else type(scene).__name__,
            "Scene Time": "-" if scene is None else format_duration(scene._scene_time)
        }

        status_table = rich.table.Table.grid(
            rich.table.Column(no_wrap=True, overflow="crop"),
            rich.table.Column(no_wrap=True, overflow="crop"),
            padding=(0, 2)
        )
        for status_key, status_value in status_dict.items():
            status_table.add_row(status_key, status_value)
        return status_table

    def log(
        self: Self,
        message: str
    ) -> None:
        self._log_messages.append(message)
