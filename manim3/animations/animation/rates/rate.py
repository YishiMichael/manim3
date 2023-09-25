from abc import (
    ABC,
    abstractmethod
)


class Rate(ABC):
    __slots__ = ()

    @abstractmethod
    def at(
        self,
        t: float
    ) -> float:
        pass
