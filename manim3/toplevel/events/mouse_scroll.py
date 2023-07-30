from dataclasses import dataclass

from .event import Event


@dataclass(
    frozen=True,
    slots=True
)
class MouseScroll(Event):
    x: int | None
    y: int | None
    scroll_x: float | None
    scroll_y: float | None

    def _capture(
        self,
        event: Event
    ) -> bool:
        return isinstance(event, MouseScroll)
