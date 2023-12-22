from __future__ import annotations


import copy
import functools
import inspect
from abc import ABC
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Hashable,
    Never,
    Self,
    TypeAliasType,
    TypeVar
)

import attrs
import numpy as np

from .lazy_descriptor import LazyDescriptor
from .lazy_slot import LazySlot


@attrs.frozen(kw_only=True)
class AnnotationRecord:
    annotation: Any
    annotation_cls: type
    type_params_cls: type

    def specialize(
        self: Self,
        annotation_record: AnnotationRecord
    ) -> AnnotationRecord:
        assert self.type_params_cls is annotation_record.annotation_cls
        type_params = self.type_params_cls.__type_params__
        specializations = getattr(annotation_record.annotation, "__args__", ())
        assert len(type_params) == len(specializations)
        annotation = self.annotation
        if annotation in type_params:
            new_annotation = specializations[type_params.index(annotation)]
        elif (parameters := getattr(annotation, "__parameters__", ())):
            new_annotation = annotation[*(
                specializations[type_params.index(type_param)]
                for type_param in parameters
            )]
        else:
            new_annotation = annotation
        return AnnotationRecord(
            annotation=new_annotation,
            annotation_cls=self.annotation_cls,
            type_params_cls=annotation_record.type_params_cls
        )

    @classmethod
    def from_override(
        cls: type[Self],
        override_cls: type,
        base_cls: type
    ) -> AnnotationRecord:
        return AnnotationRecord(
            annotation=base_cls if not base_cls.__type_params__ else override_cls.__orig_bases__[0],
            annotation_cls=base_cls,
            type_params_cls=override_cls
        )

    @classmethod
    def from_root(
        cls: type[Self],
        root_cls: type
    ) -> AnnotationRecord:
        return AnnotationRecord(
            annotation=root_cls if not root_cls.__type_params__ else root_cls[*root_cls.__type_params__],
            annotation_cls=root_cls,
            type_params_cls=root_cls
        )

    @classmethod
    def get_cls_from_annotation(
        cls: type[Self],
        annotation: Any
    ) -> type:
        if isinstance(annotation, TypeVar):
            annotation = annotation.__bound__
        elif isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
        annotation = getattr(annotation, "__origin__", annotation)
        return annotation if isinstance(annotation, type) else object

    @classmethod
    def extract_element_annotation(
        cls: type[Self],
        annotation: Any,
        plural: bool
    ) -> Any:
        if not plural:
            return annotation
        assert annotation.__origin__ is tuple
        element_annotation, ellipsis = annotation.__args__
        assert ellipsis is ...
        return element_annotation


class LazyObject(ABC):
    __slots__ = ("_lazy_slots",)

    _lazy_descriptors: tuple[LazyDescriptor, ...] = ()
    _annotated_lazy_descriptors: dict[str, tuple[AnnotationRecord, LazyDescriptor]] = {}
    _lazy_slots_cls: ClassVar[type] = object
    _slot_names: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(
        cls: type[Self]
    ) -> None:
        super().__init_subclass__()

        annotated_lazy_descriptors = {
            name: (annotation_record.specialize(AnnotationRecord.from_override(cls, base)), descriptor)
            for base in reversed(cls.__bases__)
            if issubclass(base, LazyObject)
            for name, (annotation_record, descriptor) in base._annotated_lazy_descriptors.items()
        }
        cls._annotated_lazy_descriptors = annotated_lazy_descriptors

        root_annotation_record = AnnotationRecord.from_root(cls)
        for name, descriptor in cls.__dict__.items():
            if not isinstance(descriptor, LazyDescriptor):
                continue

            signature = inspect.signature(
                descriptor._method,
                locals={
                    cls.__name__: cls,
                    **{
                        str(type_param): type_param
                        for type_param in cls.__type_params__
                    }
                },
                eval_str=True
            )
            element_annotation = AnnotationRecord.extract_element_annotation(
                annotation=signature.return_annotation,
                plural=descriptor._plural
            )
            element_annotation_cls = AnnotationRecord.get_cls_from_annotation(element_annotation)

            assert name.startswith("_") and name.endswith("_") and "__" not in name
            descriptor._name = name
            parameter_name_chains = tuple(
                tuple(f"_{name_body}_" for name_body in parameter_name.split("__"))
                for parameter_name in signature.parameters
            )
            descriptor._parameter_name_chains = parameter_name_chains
            assert descriptor._is_property or not parameter_name_chains

            descriptor._decomposer = Implementations.decomposers.fetch(descriptor._plural)
            descriptor._composer = Implementations.composers.fetch(descriptor._plural)
            descriptor._hasher = (
                Implementations.hashers.fetch(element_annotation_cls if descriptor._freeze else object)
            )

            if (typed_overridden_descriptor := annotated_lazy_descriptors.get(name)) is not None:
                annotation_record, overridden_descriptor = typed_overridden_descriptor
                assert overridden_descriptor._plural is descriptor._plural
                assert not overridden_descriptor._freeze or descriptor._freeze
                assert annotation_record.annotation_cls is element_annotation_cls
                assert annotation_record.annotation == element_annotation

            annotated_lazy_descriptors[name] = (AnnotationRecord(
                annotation=element_annotation,
                annotation_cls=element_annotation_cls,
                type_params_cls=cls
            ), descriptor)

            for parameter_name_chain, parameter in zip(
                parameter_name_chains, signature.parameters.values(), strict=True
            ):
                last_descriptor_freeze: bool = False
                annotation_record = root_annotation_record
                provided_annotation = parameter.annotation
                for name_segment in parameter_name_chain:
                    assert issubclass(element_cls := annotation_record.annotation_cls, LazyObject)
                    parameter_annotation_record, parameter_descriptor = element_cls._annotated_lazy_descriptors[name_segment]
                    last_descriptor_freeze = parameter_descriptor._freeze
                    annotation_record = parameter_annotation_record.specialize(annotation_record)
                    provided_annotation = AnnotationRecord.extract_element_annotation(
                        annotation=provided_annotation,
                        plural=parameter_descriptor._plural
                    )
                assert last_descriptor_freeze
                assert annotation_record.annotation == provided_annotation
                assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                assert parameter.default is inspect.Parameter.empty

        cls._lazy_descriptors = tuple(descriptor for _, descriptor in annotated_lazy_descriptors.values())
        cls._lazy_slots_cls = attrs.make_class(
            name=f"__{cls.__name__}",
            attrs={
                name: attrs.field(factory=functools.partial(LazySlot, descriptor))
                for name, (_, descriptor) in annotated_lazy_descriptors.items()
            },
            slots=True,
            frozen=True
        )
        cls._slot_names = tuple(
            slot_name
            for base in reversed(cls.__mro__)
            if issubclass(base, LazyObject)
            for slot_name in base.__slots__
            if not slot_name.startswith("__")
            and slot_name != "_lazy_slots"
        )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._lazy_slots: object = type(self)._lazy_slots_cls()

    def __copy__(
        self: Self
    ) -> Self:
        return self.copy()

    def _get_lazy_slot(
        self: Self,
        name: str
    ) -> LazySlot:
        return self._lazy_slots.__getattribute__(name)

    def copy(
        self: Self
    ) -> Self:
        cls = type(self)
        result = cls.__new__(cls)
        for slot_name in cls._slot_names:
            result.__setattr__(slot_name, copy.copy(self.__getattribute__(slot_name)))
        result._lazy_slots = cls._lazy_slots_cls()
        for descriptor in cls._lazy_descriptors:
            if descriptor._is_property:
                continue
            elements = descriptor.get_elements(self)
            if descriptor._deepcopy:
                elements = tuple(copy.copy(element) for element in elements)
            descriptor.set_elements(result, elements)
        return result


class ImplementationRegistry[K: Hashable, V]:
    __slots__ = (
        "_registry",
        "_match_key"
    )

    def __init__(
        self: Self,
        match_key: Callable[[K, K], bool]
    ) -> None:
        super().__init__()
        self._registry: dict[K, V] = {}
        self._match_key: Callable[[K, K], bool] = match_key

    def register(
        self: Self,
        key: K
    ) -> Callable[[V], V]:

        def result(
            value: V
        ) -> V:
            self._registry[key] = value
            return value

        return result

    def fetch(
        self: Self,
        key: K
    ) -> V:
        for registered_key, value in self._registry.items():
            if self._match_key(key, registered_key):
                return value
        raise NotImplementedError(key)


class Implementations:
    __slots__ = ()

    decomposers: ClassVar[ImplementationRegistry[bool, Callable[[Any], tuple[Any, ...]]]] = ImplementationRegistry(bool.__eq__)
    composers: ClassVar[ImplementationRegistry[bool, Callable[[tuple[Any, ...]], Any]]] = ImplementationRegistry(bool.__eq__)
    hashers: ClassVar[ImplementationRegistry[type, Callable[[Any], Hashable]]] = ImplementationRegistry(issubclass)

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @decomposers.register(False)
    @staticmethod
    def _[T](
        data: T
    ) -> tuple[T, ...]:
        return (data,)

    @decomposers.register(True)
    @staticmethod
    def _[T](
        data: tuple[T, ...]
    ) -> tuple[T, ...]:
        return data

    @composers.register(False)
    @staticmethod
    def _[T](
        elements: tuple[T, ...]
    ) -> T:
        (element,) = elements
        return element

    @composers.register(True)
    @staticmethod
    def _[T](
        elements: tuple[T, ...]
    ) -> tuple[T, ...]:
        return elements

    @hashers.register(int)
    @hashers.register(float)
    @hashers.register(str)
    @hashers.register(bytes)
    # Note, for tuples, we require every field should be hashable.
    # Consider creating a data class if it is not the case.
    @hashers.register(tuple)
    @hashers.register(Enum)
    @staticmethod
    def _(
        element: Hashable
    ) -> Hashable:
        return element

    @hashers.register(np.ndarray)
    @staticmethod
    def _(
        element: np.ndarray
    ) -> Hashable:
        return (element.shape, element.dtype, element.tobytes())

    @hashers.register(object)
    @staticmethod
    def _(
        element: object
    ) -> Hashable:
        # We are safe to use `id` here since the memoization is a weak-value-dictionary.
        # The reallocation of an old id indicates that the object has been garbage collected,
        # and its corresponding item in the memoization should have already been removed.
        return id(element)
