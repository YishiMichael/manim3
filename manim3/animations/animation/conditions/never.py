from .condition import Condition


class Never(Condition):
    __slots__ = ()

    def _judge(self) -> bool:
        return False
