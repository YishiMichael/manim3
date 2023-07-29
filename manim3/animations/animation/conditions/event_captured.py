from ....toplevel.events.event import Event
from ....toplevel.events.key_press import KeyPress
from ....toplevel.events.key_release import KeyRelease
from ....toplevel.events.mouse_drag import MouseDrag
from ....toplevel.events.mouse_press import MousePress
from ....toplevel.events.mouse_release import MouseRelease
from ....toplevel.toplevel import Toplevel
from .condition import Condition


class EventCaptured(Condition):
    __slots__ = (
        "_event_cls",
        "_symbol",
        "_buttons",
        "_modifiers"
    )

    def __init__(
        self,
        event_cls: type[Event],
        *,
        symbol: int | None = None,
        buttons: int | None = None,
        modifiers: int | None = None
    ) -> None:
        super().__init__()
        self._event_cls: type[Event] = event_cls
        self._symbol: int | None = symbol
        self._buttons: int | None = buttons
        self._modifiers: int | None = modifiers

    def _capture(
        self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, self._event_cls)
            and (
                (symbol := self._symbol) is None
                or not isinstance(event, KeyPress | KeyRelease)
                or symbol == event.symbol
            )
            and (
                (buttons := self._buttons) is None
                or not isinstance(event, MouseDrag | MousePress | MouseRelease)
                or buttons & event.buttons == event.buttons
            )
            and (
                (modifiers := self._modifiers) is None
                or not isinstance(event, KeyPress | KeyRelease | MouseDrag | MousePress | MouseRelease)
                or modifiers & event.modifiers == event.modifiers
            )
        )

    def judge(self):
        event_queue = Toplevel.event_queue
        for event in event_queue:
            if self._capture(event):
                event_queue.remove(event)
                Toplevel._event = event
                return True
        return False
