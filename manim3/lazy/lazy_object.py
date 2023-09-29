#import copy
import inspect
from abc import ABC
from typing import (
    Any,
    #Callable,
    ClassVar,
    TypeVar
)

from .lazy_descriptor import LazyDescriptor
from .lazy_slot import LazySlot


#_LazyObjectT = TypeVar("_LazyObjectT", bound="LazyObject")


class LazyObject(ABC):
    __slots__ = (
        "_lazy_slots",
        "_is_frozen"
    )

    _default_constructor_kwargs: ClassVar[dict] = {}

    #_special_slot_copiers: ClassVar[dict[str, Callable | None]] = {
    #    "_lazy_slots": None,
    #    "_is_frozen": None
    #}

    _lazy_descriptors: ClassVar[dict[str, LazyDescriptor]] = {}
    #_slot_copiers: ClassVar[dict[str, Callable]] = {}

    def __init_subclass__(cls) -> None:

        def extract_tuple_annotation(
            annotation: Any
        ) -> Any:
            assert annotation.__origin__ is tuple
            element_annotation, ellipsis = annotation.__args__
            assert ellipsis is ...
            return element_annotation

        def get_descriptor_chain(
            name_chain: tuple[str, ...],
            root_class: type[LazyObject],
            parameter: inspect.Parameter
        ) -> tuple[LazyDescriptor, ...]:
            descriptor_chain: list[LazyDescriptor] = []
            element_type = root_class
            collection_level = 0
            for descriptor_name in name_chain:
                assert issubclass(element_type, LazyObject)
                descriptor = element_type._lazy_descriptors[descriptor_name]
                descriptor_chain.append(descriptor)
                element_type = descriptor._element_type
                if descriptor._is_multiple:
                    collection_level += 1
            assert descriptor_chain[-1]._freeze

            assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            assert parameter.default is inspect.Parameter.empty
            expected_element_annotation = descriptor_chain[-1]._element_type_annotation
            element_annotation = parameter.annotation
            for _ in range(collection_level):
                element_annotation = extract_tuple_annotation(element_annotation)
            assert (
                element_annotation == expected_element_annotation
                or isinstance(expected_element_annotation, TypeVar)
            )
            return tuple(descriptor_chain)

        super().__init_subclass__()
        base_cls = cls.__base__
        assert issubclass(base_cls, LazyObject)
        new_descriptors = {
            name: descriptor
            for name, descriptor in cls.__dict__.items()
            if isinstance(descriptor, LazyDescriptor)
        }
        cls._lazy_descriptors = base_cls._lazy_descriptors | new_descriptors

        #cls._slot_copiers = {
        #    slot_name: slot_copier
        #    for base_cls in reversed(cls.__mro__)
        #    if issubclass(base_cls, LazyObject)
        #    for slot_name in base_cls.__slots__
        #    if (slot_copier := base_cls._special_slot_copiers.get(slot_name, copy.copy)) is not None
        #}

        for name, descriptor in new_descriptors.items():
            signature = inspect.signature(descriptor._method, locals={cls.__name__: cls}, eval_str=True)

            return_annotation = signature.return_annotation
            if descriptor._is_multiple:
                element_type_annotation = extract_tuple_annotation(return_annotation)
            else:
                element_type_annotation = return_annotation
            descriptor._element_type_annotation = element_type_annotation

            try:
                element_type = element_type_annotation.__origin__
            except AttributeError:
                element_type = element_type_annotation
            descriptor._element_type = element_type

            descriptor._descriptor_chains = tuple(
                get_descriptor_chain(
                    name_chain=tuple(f"_{name_segment}_" for name_segment in parameter_name.split("__")),
                    root_class=cls,
                    parameter=parameter
                )
                for parameter_name, parameter in signature.parameters.items()
            )
            if descriptor._is_variable:
                assert not descriptor._descriptor_chains

            overridden_descriptor = base_cls._lazy_descriptors.get(name)
            if overridden_descriptor is not None:
                descriptor._element_registration = overridden_descriptor._element_registration
                assert descriptor._can_override(overridden_descriptor)

            if isinstance(element_type, LazyObject):
                descriptor._freezer = LazyObject._freeze

    def __init__(self) -> None:
        super().__init__()
        self._lazy_slots: dict[str, LazySlot] = {}
        self._is_frozen: bool = False
        for descriptor in type(self)._lazy_descriptors.values():
            descriptor._init(self)

    #@classmethod
    #def _copy_lazy_content(
    #    cls,
    #    dst_object: _LazyObjectT,
    #    src_object: _LazyObjectT
    #) -> None:
    #    for descriptor in cls._lazy_descriptors.values():
    #        if descriptor._is_variable:
    #            descriptor._set_elements(dst_object, descriptor._get_elements(src_object))

    def _copy(self):
        cls = type(self)
        result = cls(**cls._default_constructor_kwargs)
        for descriptor in cls._lazy_descriptors.values():
            if descriptor._is_variable:
                descriptor._set_elements(result, descriptor._get_elements(self))
        #result._lazy_slots = {}
        #result._is_frozen = False
        #for descriptor in cls._lazy_descriptors.values():
        #    #descriptor._init(result)
        #    if descriptor._is_variable:
        #        descriptor._set_elements(result, descriptor._get_elements(self))
        #for slot_name, slot_copier in cls._slot_copiers.items():
        #    result.__setattr__(slot_name, slot_copier(self.__getattribute__(slot_name)))
        return result

    def _freeze(self) -> None:
        if self._is_frozen:
            return
        self._is_frozen = True
        for descriptor in type(self)._lazy_descriptors.values():
            descriptor._get_slot(self)._is_writable = False
            for element in descriptor._get_elements(self):
                if isinstance(element, LazyObject):
                    element._freeze()
