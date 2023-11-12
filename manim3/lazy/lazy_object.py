from __future__ import annotations


import copy
import functools
import inspect
import operator
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
class TypeHint:
    type_params_cls: type
    annotation_cls: type
    annotation: Any

    def specialize(
        self: Self,
        type_hint: TypeHint
    ) -> TypeHint:
        assert self.type_params_cls is type_hint.annotation_cls
        type_params = self.type_params_cls.__type_params__
        specializations = getattr(type_hint.annotation, "__args__", ())
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
        return TypeHint(
            type_params_cls=type_hint.type_params_cls,
            annotation_cls=self.annotation_cls,
            annotation=new_annotation
        )

    @classmethod
    def from_override(
        cls: type[Self],
        override_cls: type,
        base_cls: type
    ) -> Self:
        return cls(
            type_params_cls=override_cls,
            annotation_cls=base_cls,
            annotation=base_cls if not base_cls.__type_params__ else override_cls.__orig_bases__[0]
        )

    @classmethod
    def from_root(
        cls: type[Self],
        root_cls: type
    ) -> Self:
        return cls(
            type_params_cls=root_cls,
            annotation_cls=root_cls,
            annotation=root_cls if not root_cls.__type_params__ else root_cls[*root_cls.__type_params__]
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
    __slots__ = (
        "_lazy_slots",
        "_is_frozen"
    )

    _special_slot_copiers: ClassVar[dict[str, Callable]] = {
        "_lazy_slots": lambda o: type(o)(),
        "_is_frozen": lambda o: False
    }

    _lazy_descriptors: tuple[LazyDescriptor, ...] = ()
    _hinted_lazy_descriptors: dict[str, tuple[TypeHint, LazyDescriptor]] = {}
    _lazy_slots_cls: ClassVar[type] = object
    _slot_copiers: ClassVar[dict[str, Callable]] = {}

    def __init_subclass__(
        cls: type[Self]
    ) -> None:
        super().__init_subclass__()

        hinted_lazy_descriptors = {
            name: (type_hint.specialize(TypeHint.from_override(cls, base)), descriptor)
            for base in reversed(cls.__bases__)
            if issubclass(base, LazyObject)
            for name, (type_hint, descriptor) in base._hinted_lazy_descriptors.items()
        }
        cls._hinted_lazy_descriptors = hinted_lazy_descriptors

        root_type_hint = TypeHint.from_root(cls)
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
            element_annotation = TypeHint.extract_element_annotation(
                annotation=signature.return_annotation,
                plural=descriptor._plural
            )
            element_annotation_cls = TypeHint.get_cls_from_annotation(element_annotation)

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
            descriptor._freezer = (
                Implementations.freezers.fetch(element_annotation_cls)
                if descriptor._freeze else Implementations.empty_freezer
            )
            descriptor._copier = (
                Implementations.copiers.fetch(element_annotation_cls)
                if descriptor._deepcopy else Implementations.empty_copier
            )

            if (typed_overridden_descriptor := hinted_lazy_descriptors.get(name)) is not None:
                type_hint, overridden_descriptor = typed_overridden_descriptor
                assert overridden_descriptor._plural is descriptor._plural
                assert not overridden_descriptor._freeze or descriptor._freeze
                assert type_hint.annotation_cls is element_annotation_cls
                assert type_hint.annotation == element_annotation

            hinted_lazy_descriptors[name] = (TypeHint(
                type_params_cls=cls,
                annotation_cls=element_annotation_cls,
                annotation=element_annotation
            ), descriptor)

            for parameter_name_chain, parameter in zip(
                parameter_name_chains, signature.parameters.values(), strict=True
            ):
                last_descriptor_freeze: bool = False
                type_hint = root_type_hint
                provided_annotation = parameter.annotation
                for name_segment in parameter_name_chain:
                    assert issubclass(element_cls := type_hint.annotation_cls, LazyObject)
                    parameter_type_hint, parameter_descriptor = element_cls._hinted_lazy_descriptors[name_segment]
                    last_descriptor_freeze = parameter_descriptor._freeze
                    type_hint = parameter_type_hint.specialize(type_hint)
                    provided_annotation = TypeHint.extract_element_annotation(
                        annotation=provided_annotation,
                        plural=parameter_descriptor._plural
                    )
                assert last_descriptor_freeze
                assert type_hint.annotation == provided_annotation
                assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                assert parameter.default is inspect.Parameter.empty

        cls._lazy_descriptors = tuple(descriptor for _, descriptor in hinted_lazy_descriptors.values())
        cls._lazy_slots_cls = attrs.make_class(
            name=f"__{cls.__name__}",
            attrs={
                name: attrs.field(factory=functools.partial(LazySlot, descriptor))
                for name, (_, descriptor) in hinted_lazy_descriptors.items()
            },
            slots=True,
            frozen=True
        )
        cls._slot_copiers = {
            slot_name: base._special_slot_copiers.get(slot_name, copy.copy)
            for base in reversed(cls.__mro__)
            if issubclass(base, LazyObject)
            for slot_name in base.__slots__
            if not slot_name.startswith("__")
        }

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._lazy_slots: object = type(self)._lazy_slots_cls()
        self._is_frozen: bool = False

    def _get_lazy_slot(
        self: Self,
        name: str
    ) -> LazySlot:
        return self._lazy_slots.__getattribute__(name)

    def _freeze(
        self: Self
    ) -> None:
        if self._is_frozen:
            return
        self._is_frozen = True
        for descriptor in type(self)._lazy_descriptors:
            descriptor.get_slot(self).disable_writability()
            for element in descriptor.get_elements(self):
                descriptor._freezer(element)

    def copy(
        self: Self
    ) -> Self:
        cls = type(self)
        result = cls.__new__(cls)
        for slot_name, slot_copier in cls._slot_copiers.items():
            result.__setattr__(slot_name, slot_copier(self.__getattribute__(slot_name)))
        for descriptor in cls._lazy_descriptors:
            if descriptor._is_property:
                continue
            descriptor.set_elements(result, tuple(
                descriptor._copier(element)
                for element in descriptor.get_elements(self)
            ))
        return result


class Registration[K: Hashable, V]:
    __slots__ = (
        "_registration",
        "_match_key"
    )

    def __init__(
        self: Self,
        match_key: Callable[[K, K], bool]
    ) -> None:
        super().__init__()
        self._registration: dict[K, V] = {}
        self._match_key: Callable[[K, K], bool] = match_key

    def register(
        self: Self,
        key: K
    ) -> Callable[[V], V]:

        def result(
            value: V
        ) -> V:
            self._registration[key] = value
            return value

        return result

    def fetch(
        self: Self,
        key: K
    ) -> V:
        for registered_key, value in self._registration.items():
            if self._match_key(key, registered_key):
                return value
        raise KeyError(key)


class Implementations:
    __slots__ = ()

    decomposers: ClassVar[Registration[bool, Callable[[Any], tuple[Any, ...]]]] = Registration(operator.is_)
    composers: ClassVar[Registration[bool, Callable[[tuple[Any, ...]], Any]]] = Registration(operator.is_)
    hashers: ClassVar[Registration[type, Callable[[Any], Hashable]]] = Registration(issubclass)
    freezers: ClassVar[Registration[type, Callable[[Any], None]]] = Registration(issubclass)
    copiers: ClassVar[Registration[type, Callable[[Any], Any]]] = Registration(issubclass)

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
    @hashers.register(str)
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
        return id(element)

    @freezers.register(LazyObject)
    @staticmethod
    def _(
        element: LazyObject
    ) -> None:
        element._freeze()

    @freezers.register(object)
    @staticmethod
    def empty_freezer(
        element: object
    ) -> None:
        pass

    @copiers.register(LazyObject)
    @staticmethod
    def _(
        element: LazyObject
    ) -> LazyObject:
        return element.copy()

    @copiers.register(object)
    @staticmethod
    def _(
        element: object
    ) -> object:
        return copy.copy(element)

    @staticmethod
    def empty_copier(
        element: object
    ) -> object:
        return element
