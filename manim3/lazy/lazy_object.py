from __future__ import annotations


import copy
import functools
import inspect
from abc import ABC
from typing import (
    Any,
    Callable,
    ClassVar,
    Self,
    TypeAliasType,
    TypeVar
)

import attrs

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
        return getattr(annotation, "__origin__", annotation)

    @classmethod
    def extract_element_annotation(
        cls: type[Self],
        annotation: Any,
        is_plural: bool
    ) -> Any:
        if not is_plural:
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
        "_lazy_slots": lambda o: type(o)._lazy_slots_cls(),
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
        base = cls.__base__
        assert issubclass(base, LazyObject)

        override_type_hint = TypeHint.from_override(cls, base)
        hinted_lazy_descriptors = {
            name: (type_hint.specialize(override_type_hint), descriptor)
            for name, (type_hint, descriptor) in base._hinted_lazy_descriptors.items()
        }
        cls._hinted_lazy_descriptors = hinted_lazy_descriptors

        root_type_hint = TypeHint.from_root(cls)
        for descriptor_name, descriptor in cls.__dict__.items():
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
                is_plural=descriptor._is_plural
            )
            element_annotation_cls = TypeHint.get_cls_from_annotation(element_annotation)
            if (typed_overridden_descriptor := hinted_lazy_descriptors.get(descriptor_name)) is not None:
                type_hint, overridden_descriptor = typed_overridden_descriptor
                assert overridden_descriptor._is_plural is descriptor._is_plural
                assert overridden_descriptor._hasher is descriptor._hasher
                assert not overridden_descriptor._freeze or descriptor._freeze
                assert type_hint.annotation_cls is element_annotation_cls  # TODO: subclass check
                assert type_hint.annotation == element_annotation

            hinted_lazy_descriptors[descriptor_name] = (TypeHint(
                type_params_cls=cls,
                annotation_cls=element_annotation_cls,
                annotation=element_annotation
            ), descriptor)

            for parameter_name_chain, (parameter_name, parameter) in zip(
                descriptor._parameter_name_chains, signature.parameters.items(), strict=True
            ):
                assert "".join(parameter_name_chain) == f"_{parameter_name}_"
                last_descriptor_freeze: bool = False
                type_hint = root_type_hint
                provided_annotation = parameter.annotation
                for name in parameter_name_chain:
                    assert issubclass(element_cls := type_hint.annotation_cls, LazyObject)
                    parameter_type_hint, parameter_descriptor = element_cls._hinted_lazy_descriptors[name]
                    last_descriptor_freeze = parameter_descriptor._freeze
                    type_hint = parameter_type_hint.specialize(type_hint)
                    provided_annotation = TypeHint.extract_element_annotation(
                        annotation=provided_annotation,
                        is_plural=parameter_descriptor._is_plural
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

    def _get_slot(
        self: Self,
        name: str
    ) -> LazySlot:
        return self._lazy_slots.__getattribute__(name)

    def _copy_lazy_content(
        self: Self,
        src_object: Self
    ) -> None:
        for descriptor in type(self)._lazy_descriptors:
            if descriptor._is_variable:
                descriptor.set_elements(self, descriptor.get_elements(src_object))

    def copy(
        self: Self
    ) -> Self:
        cls = type(self)
        result = cls.__new__(cls)
        for slot_name, slot_copier in cls._slot_copiers.items():
            result.__setattr__(slot_name, slot_copier(self.__getattribute__(slot_name)))
        result._copy_lazy_content(self)
        return result
