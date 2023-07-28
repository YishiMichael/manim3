from .condition import Condition


class Never(Condition):
    __slots__ = ()

    def judge(self) -> bool:
        return False
