from dataclasses import dataclass

from .event import Event


@dataclass(
    frozen=True,
    slots=True
)
class KeyRelease(Event):
    symbol: int | None = None
    modifiers: int | None = None

    def _capture(
        self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, KeyRelease)
            and self._match(self.symbol, event.symbol, masked=False)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )