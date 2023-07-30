from dataclasses import dataclass

from .event import Event


@dataclass(
    frozen=True,
    slots=True
)
class MouseMotion(Event):
    x: int | None = None
    y: int | None = None
    dx: int | None = None
    dy: int | None = None

    def _capture(
        self,
        event: Event
    ) -> bool:
        return isinstance(event, MouseMotion)
