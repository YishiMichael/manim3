from ....toplevel.events.event import Event
from ....toplevel.toplevel import Toplevel
from .condition import Condition


class EventCaptured(Condition):
    __slots__ = ("_event",)

    def __init__(
        self,
        event: Event
    ) -> None:
        super().__init__()
        self._event: Event = event

    def judge(self) -> bool:
        targeting_event = self._event
        event_queue = Toplevel.event_queue
        for event in event_queue:
            if targeting_event._capture(event):
                event_queue.remove(event)
                Toplevel._event = event
                return True
        return False
