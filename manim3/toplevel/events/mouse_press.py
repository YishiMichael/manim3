from dataclasses import dataclass

from .event import Event


@dataclass(
    frozen=True,
    slots=True
)
class MousePress(Event):
    buttons: int | None = None
    modifiers: int | None = None
    x: int | None = None
    y: int | None = None

    def _capture(
        self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, MousePress)
            and self._match(self.buttons, event.buttons, masked=True)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )
