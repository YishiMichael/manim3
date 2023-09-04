#from dataclasses import dataclass
#from typing import (
#    Callable,
#    ClassVar,
#    Generic,
#    TypeVar
#)

#from ...lazy.lazy import LazyDescriptor
#from .operation_handlers.concatenate_handler import ConcatenateHandler
#from .operation_handlers.interpolate_handler import InterpolateHandler
#from .operation_handlers.split_handler import SplitHandler


#_ElementT = TypeVar("_ElementT")


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class OperationInfo(Generic[_DataT, _ElementT]):
#    descriptor: LazyDescriptor[_DataT, _ElementT]
#    split_handler_cls: type[SplitHandler[_DataT]] | None
#    concatenate_handler_cls: type[ConcatenateHandler[_DataT]] | None
#    interpolate_handler_cls: type[InterpolateHandler[_DataT]] | None


#class StyleMeta:
#    __slots__ = ()

#    _operation_infos: ClassVar[list[OperationInfo]] = []

#    def __new__(cls):
#        raise TypeError

#    @classmethod
#    def register(
#        cls,
#        *,
#        converter: Callable[[object], object] | None = None
#        #split_operation: type[SplitHandler[_DataT]] | None | None = None,
#        #concatenate_operation: type[ConcatenateHandler[_DataT]] | None = None,
#        #interpolate_operation: type[InterpolateHandler[_DataT]] | None = None
#    ) -> Callable[
#        [LazyDescriptor[_ElementT, _ElementT]],
#        LazyDescriptor[_ElementT, _ElementT]
#    ]:

#        def identity_converter(
#            obj: object
#        ) -> object:
#            return object

#        if converter is None:
#            converter = identity_converter

#        def callback(
#            descriptor: LazyDescriptor[_ElementT, _ElementT]
#        ) -> LazyDescriptor[_ElementT, _ElementT]:
#            assert descriptor._is_variable
#            assert not descriptor._is_multiple
#            #assert not isinstance(descriptor.converter, LazyCollectionConverter)
#            cls._operation_infos.append(OperationInfo(
#                descriptor=descriptor,
#                split_handler_cls=split_operation,
#                concatenate_handler_cls=concatenate_operation,
#                interpolate_handler_cls=interpolate_operation
#            ))
#            return descriptor

#        return callback
