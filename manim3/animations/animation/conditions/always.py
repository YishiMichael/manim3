from .condition import Condition


class Always(Condition):
    __slots__ = ()

    def _judge(self) -> bool:
        return True
