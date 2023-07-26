#from dataclasses import dataclass
#from typing import (
#    Callable,
#    ClassVar,
#    Generic,
#    TypeVar
#)

#from ....lazy.lazy import (
#    LazyCollectionConverter,
#    LazyContainer,
#    LazyObject,
#    LazyVariableDescriptor
#)
#from .concatenate_handler import ConcatenateHandler
#from .interpolate_handler import InterpolateHandler
#from .partial_handler import PartialHandler


#_ContainerT = TypeVar("_ContainerT", bound="LazyContainer")
#_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
#_DataT = TypeVar("_DataT")
#_DataRawT = TypeVar("_DataRawT")


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class OperationInfo(Generic[_InstanceT, _ContainerT, _DataT, _DataRawT]):
#    descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
#    partial_handler_cls: type[PartialHandler[_ContainerT]] | None
#    interpolate_handler_cls: type[InterpolateHandler[_ContainerT]] | None
#    concatenate_handler_cls: type[ConcatenateHandler[_ContainerT]] | None


#class MobjectOperation:
#    __slots__ = ()

#    _operation_infos: ClassVar[list[OperationInfo]] = []

#    def __new__(cls):
#        raise TypeError

#    @classmethod
#    def register(
#        cls,
#        *,
#        partial: type[PartialHandler] | None | None = None,
#        interpolate: type[InterpolateHandler] | None = None,
#        concatenate: type[ConcatenateHandler] | None = None
#    ) -> Callable[
#        [LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]],
#        LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
#    ]:

#        def callback(
#            descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
#        ) -> LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]:
#            assert not isinstance(descriptor.converter, LazyCollectionConverter)
#            cls._operation_infos.append(OperationInfo(
#                descriptor=descriptor,
#                partial_handler_cls=partial,
#                interpolate_handler_cls=interpolate,
#                concatenate_handler_cls=concatenate
#            ))
#            return descriptor

#        return callback
