from abc import (
    ABC,
    abstractmethod
)


class Condition(ABC):
    __slots__ = ()

    @abstractmethod
    def _judge(self) -> bool:
        pass
