from abc import (
    ABC,
    abstractmethod
)
from dataclasses import dataclass


@dataclass(
    frozen=True,
    slots=True
)
class Event(ABC):
    @abstractmethod
    def _capture(
        self,
        event: "Event"
    ) -> bool:
        pass

    @classmethod
    def _match(
        cls,
        required_value: int | None,
        value: int | None,
        *,
        masked: bool
    ) -> bool:
        return (
            required_value is None
            or value is None
            or required_value == (value & required_value if masked else value)
        )
