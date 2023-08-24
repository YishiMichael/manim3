from dataclasses import dataclass
from typing import (
    Callable,
    ClassVar,
    Generic,
    TypeVar
)

from ...lazy.lazy import (
    LazyCollectionConverter,
    LazyContainer,
    LazyObject,
    LazyVariableDescriptor
)
from .operation_handlers.concatenate_handler import ConcatenateHandler
from .operation_handlers.interpolate_handler import InterpolateHandler
from .operation_handlers.split_handler import SplitHandler


_ContainerT = TypeVar("_ContainerT", bound="LazyContainer")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_DataT = TypeVar("_DataT")
_DataRawT = TypeVar("_DataRawT")


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class OperationInfo(Generic[_InstanceT, _ContainerT, _DataT, _DataRawT]):
    descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
    split_handler_cls: type[SplitHandler[_ContainerT]] | None
    interpolate_handler_cls: type[InterpolateHandler[_ContainerT]] | None
    concatenate_handler_cls: type[ConcatenateHandler[_ContainerT]] | None


class StyleMeta:
    __slots__ = ()

    _operation_infos: ClassVar[list[OperationInfo]] = []

    def __new__(cls):
        raise TypeError

    @classmethod
    def register(
        cls,
        *,
        split_operation: type[SplitHandler] | None | None = None,
        interpolate_operation: type[InterpolateHandler] | None = None,
        concatenate_operation: type[ConcatenateHandler] | None = None
    ) -> Callable[
        [LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]],
        LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
    ]:

        def callback(
            descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
        ) -> LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]:
            assert not isinstance(descriptor.converter, LazyCollectionConverter)
            cls._operation_infos.append(OperationInfo(
                descriptor=descriptor,
                split_handler_cls=split_operation,
                interpolate_handler_cls=interpolate_operation,
                concatenate_handler_cls=concatenate_operation
            ))
            return descriptor

        return callback
