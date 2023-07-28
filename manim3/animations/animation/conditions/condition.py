from abc import (
    ABC,
    abstractmethod
)


class Condition(ABC):
    __slots__ = ()

    @abstractmethod
    def judge(self) -> bool:
        pass
