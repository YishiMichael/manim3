from __future__ import annotations


import datetime
import itertools
import time
from typing import (
    Iterator,
    Self
)

import rich.box
import rich.live
import rich.table
import rich.text

from .toplevel import Toplevel
from .toplevel_resource import ToplevelResource


class FPSCounter:
    __slots__ = (
        "_last_frames_count_update_timestamp",
        "_last_frames_count",
        "_frames_count"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._last_frames_count_update_timestamp: float = time.perf_counter()
        self._last_frames_count: int = 0
        self._frames_count: int = 0

    def increment_frame(
        self: Self
    ) -> None:
        if (timestamp := time.perf_counter()) - self._last_frames_count_update_timestamp >= 1.0:
            self._last_frames_count_update_timestamp = timestamp
            self._last_frames_count = self._frames_count
            self._frames_count = 0
        self._frames_count += 1


class Logger(ToplevelResource):
    __slots__ = (
        "_start_timestamp",
        "_fps_counter",
        "_livestream",
        "_recordings_count",
        "_scene_name",
        "_scene_timer",
        "_log_messages",
        "_log_messages_capacity"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        #self._console: rich.console.Console = rich.console.Console()
        self._start_timestamp: float = time.perf_counter()
        self._fps_counter: FPSCounter = FPSCounter()
        #self._last_frames_count_update_timestamp: float | None = None
        #self._last_frames_count: int | None = None
        #self._frames_count: int | None = None
        self._livestream: bool | None = None
        self._recordings_count: int | None = None
        self._scene_name: str | None = None
        self._scene_timer: float | None = None
        self._log_messages_capacity: int = 10
        self._log_messages: list[str] = []

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
        log_table = rich.table.Table.grid()
        for log_message in itertools.chain(
            self._log_messages,
            itertools.repeat("", self._log_messages_capacity - len(self._log_messages))
        ):
            log_table.add_row(
                rich.text.Text(log_message, no_wrap=True, overflow="crop")
            )

        info_table = rich.table.Table.grid(padding=(0, 2))
        for info_key, info_value in {
            "Timer": f"{datetime.timedelta(seconds=int(time.perf_counter() - self._start_timestamp))}",
            "FPS": f"{self._fps_counter._last_frames_count}",
            "Livestream": "-" if self._livestream is None else "[green]On" if self._livestream else "[red]Off",
            "Recording": "-" if self._recordings_count is None else (
                f"[green]On ({self._recordings_count})" if self._recordings_count else "[red]Off"
            ),
            "Scene Name": "-" if self._scene_name is None else self._scene_name,
            "Scene Timer": "-" if self._scene_timer is None else f"{datetime.timedelta(seconds=int(self._scene_timer))}"
        }.items():
            info_table.add_row(
                rich.text.Text(info_key, no_wrap=True, overflow="crop"),  # TODO: bold
                rich.text.Text(info_value, no_wrap=True, overflow="crop")
            )

        table = rich.table.Table(
            rich.table.Column(header="Log Messages", no_wrap=True, overflow="crop", width=80),
            rich.table.Column(header="Info", no_wrap=True, overflow="crop", width=40),
            caption=f"manim3 [green]v{__import__("manim3").__version__}",
            caption_justify="left",
            box=rich.box.ASCII
        )
        table.add_row(log_table, info_table)
        return table

    def log(
        self: Self,
        message: str
    ) -> None:
        self._log_messages.append(message)
        while len(self._log_messages) > self._log_messages_capacity:
            self._log_messages.pop(0)
