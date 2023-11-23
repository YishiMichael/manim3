from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from contextlib import (
    AbstractContextManager,
    contextmanager
)
from types import TracebackType
from typing import (
    ClassVar,
    Iterator,
    Self
)


class ToplevelResource(ABC):
    __slots__ = ()

    __context_manager: ClassVar[AbstractContextManager[None] | None] = None

    @abstractmethod
    def __contextmanager__(
        self: Self
    ) -> Iterator[None]:
        pass

    def __enter__(
        self: Self
    ) -> None:
        cls = type(self)
        assert cls.__context_manager is None
        context_manager = contextmanager(self.__contextmanager__)()
        cls.__context_manager = context_manager
        return context_manager.__enter__()

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None
    ) -> bool | None:
        cls = type(self)
        context_manager = cls.__context_manager
        assert context_manager is not None
        cls.__context_manager = None
        return context_manager.__exit__(exc_type, exc_value, exc_traceback)
