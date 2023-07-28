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
        # Defined on `[0, 1] -> [0, 1]`, increasing.
        pass
