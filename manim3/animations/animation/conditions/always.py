from .condition import Condition


class Always(Condition):
    __slots__ = ()

    def judge(self) -> bool:
        return True
