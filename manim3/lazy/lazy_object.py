from __future__ import annotations


import copy
import inspect
from abc import ABC
from typing import (
    Any,
    Callable,
    #Callable,
    ClassVar,
    Self
)

from .lazy_descriptor import LazyDescriptor
from .lazy_slot import LazySlot


class LazyObject(ABC):
    __slots__ = (
        "_lazy_slots",
        "_is_frozen"
    )

    #_default_constructor_kwargs: ClassVar[dict] = {}

    _special_slot_copiers: ClassVar[dict[str, Callable]] = {
        "_lazy_slots": lambda o: {},
        "_is_frozen": lambda o: False
    }

    #_lazy_descriptor_info_dict: dict[str, tuple[LazyDescriptor, Any]] = {}
    _lazy_descriptors: ClassVar[dict[str, LazyDescriptor]] = {}
    _element_annotations: ClassVar[dict[str, Any]] = {}
    _slot_copiers: ClassVar[dict[str, Callable]] = {}

    def __init_subclass__(
        cls: type[Self]
    ) -> None:

        def extract_element_annotation(
            annotation: Any,
            is_plural: bool
        ) -> Any:
            if not is_plural:
                return annotation
            assert annotation.__origin__ is tuple
            element_annotation, ellipsis = annotation.__args__
            assert ellipsis is ...
            return element_annotation

        def get_lazy_cls_from_annotation(
            annotation: Any
        ) -> type[LazyObject] | None:
            return (
                cls
                if isinstance(cls := getattr(annotation, "__origin__", annotation), type)
                and issubclass(cls, LazyObject)
                else None
            )

        #def get_descriptor_by_name(
        #    lazy_cls: type[LazyObject],
        #    descriptor_name: str
        #) -> LazyDescriptor | None:
        #    return {
        #        descriptor._name: descriptor
        #        for descriptor in lazy_cls._lazy_descriptors
        #    }.get(descriptor_name)

        def get_descriptor_info_chain(
            root_cls: Any,
            name_chain: tuple[str, ...],
            parameter: inspect.Parameter
        ) -> tuple[tuple[str, bool], ...]:
            descriptor_info_chain: list[tuple[str, bool]] = []
            last_descriptor: LazyDescriptor | None = None
            element_annotation = (
                root_cls
                if not (type_params := root_cls.__type_params__)
                else root_cls[*type_params]
            )
            provided_annotation = parameter.annotation
            for descriptor_name in name_chain:
                assert (element_cls := get_lazy_cls_from_annotation(element_annotation)) is not None
                #print(root_annotation, element_cls, element_cls._lazy_descriptor_element_annotation_dict)
                last_descriptor = element_cls._lazy_descriptors[descriptor_name]
                element_annotation = element_cls._element_annotations[descriptor_name]
                #element_annotation = element_cls._lazy_descriptors[descriptor]
                is_plural = last_descriptor._is_plural
                provided_annotation = extract_element_annotation(
                    annotation=provided_annotation,
                    is_plural=is_plural
                )
                descriptor_info_chain.append((descriptor_name, is_plural))
            assert last_descriptor is not None and last_descriptor._freeze
            assert element_annotation == provided_annotation
            assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            assert parameter.default is inspect.Parameter.empty
            return tuple(descriptor_info_chain)

            #descriptor_chain: list[LazyDescriptor] = []
            #element_type = root_class
            #collection_level = 0
            #for descriptor_name in name_chain:
            #    assert element_type is not None and issubclass(element_type, LazyObject)
            #    descriptor = element_type._lazy_descriptors[descriptor_name]
            #    descriptor_chain.append(descriptor)
            #    element_type = descriptor._element_type
            #    if descriptor._is_plural:
            #        collection_level += 1
            #assert descriptor_chain[-1]._freeze

            #assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            #assert parameter.default is inspect.Parameter.empty
            #expected_element_annotation = descriptor_chain[-1]._element_type_annotation
            #element_annotation = parameter.annotation
            #for _ in range(collection_level):
            #    element_annotation = extract_tuple_annotation(element_annotation)
            #assert (
            #    element_annotation == expected_element_annotation
            #    or isinstance(expected_element_annotation, TypeVar)
            #)
            #return tuple(descriptor_chain)

        super().__init_subclass__()
        base_cls = cls.__base__
        assert issubclass(base_cls, LazyObject)

        base_type_params = base_cls.__type_params__
        type_param_specializations: tuple = (
            ()
            if (orig_bases := getattr(cls, "__orig_bases__", None)) is None
            else getattr(orig_bases[0], "__args__", ())
        )

        #descriptor_dict = {
        #    descriptor._name: descriptor
        #    for descriptor in base_cls._lazy_descriptors
        #}

        #new_descriptors = {
        #    name: descriptor
        #    for name, descriptor in cls.__dict__.items()
        #    if isinstance(descriptor, LazyDescriptor)
        #}
        #new_descriptor_info_dict = {
        #    name: (descriptor, inspect.signature(
        #        descriptor._method,
        #        locals={
        #            cls.__name__: cls,
        #            **{
        #                str(type_param): type_param
        #                for type_param in cls.__type_params__
        #            }
        #        },
        #        eval_str=True
        #    ))
        #    for name, descriptor in cls.__dict__.items()
        #    if isinstance(descriptor, LazyDescriptor)
        #}
        #new_lazy_descriptor_element_annotation_dict = 
        #for descriptor, element_annotation in new_lazy_descriptor_element_annotation_dict.values():
        #    if issubclass(get_lazy_cls_from_annotation(element_annotation), LazyObject):
        #        descriptor._freezer = LazyObject._freeze

        #overridden_descriptor_info_dict = {
        #    name: (descriptor, (
        #        type_param_specializations[base_type_params.index(element_annotation)]
        #        if element_annotation in base_type_params
        #        else element_annotation
        #        if not (parameters := getattr(element_annotation, "__parameters__", ()))
        #        else element_annotation[*(
        #            type_param_specializations[base_type_params.index(type_param)]
        #            for type_param in parameters
        #        )]
        #    ))
        #    for name, (descriptor, element_annotation) in base_cls._lazy_descriptor_info_dict.items()
        #}
        #descriptor_info_dict = {
        #    name: (descriptor, extract_element_annotation(
        #        annotation=signature.return_annotation,
        #        is_plural=descriptor._is_plural
        #    ))
        #    for name, (descriptor, signature) in new_descriptor_info_dict.items()
        #}
        #cls._lazy_descriptor_info_dict = overridden_descriptor_info_dict | descriptor_info_dict
        #cls._lazy_descriptors = tuple(
        #    descriptor
        #    for descriptor, _ in cls._lazy_descriptor_info_dict.values()
        #)
        cls._lazy_descriptors = base_cls._lazy_descriptors.copy()
        cls._element_annotations = {
            name: (
                type_param_specializations[base_type_params.index(element_annotation)]
                if element_annotation in base_type_params
                else element_annotation
                if not (parameters := getattr(element_annotation, "__parameters__", ()))
                else element_annotation[*(
                    type_param_specializations[base_type_params.index(type_param)]
                    for type_param in parameters
                )]
            )
            for name, element_annotation in base_cls._element_annotations.items()
        }
        cls._slot_copiers = {
            slot_name: base_cls._special_slot_copiers.get(slot_name, copy.copy)
            for base_cls in reversed(cls.__mro__)
            if issubclass(base_cls, LazyObject)
            for slot_name in base_cls.__slots__
            if not slot_name.startswith("__")
        }

        for name, descriptor in cls.__dict__.items():
            if not isinstance(descriptor, LazyDescriptor):
                continue

            assert name.startswith("_") and name.endswith("_") and "__" not in name
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
            #parameter_names = descriptor._method.__code__.co_varnames
            #tuple(
            #    tuple(f"_{name_body}_" for name_body in varname.split("__"))
            #    for varname in method.__code__.co_varnames
            #)
            #parameter_annotation_dict = parameter_annotation_dicts[name]
            #return_annotation = parameter_annotation_dict.pop("return")

            parameters = signature.parameters
            if descriptor._is_variable:
                assert not parameters
            descriptor._descriptor_info_chains = tuple(
                get_descriptor_info_chain(
                    root_cls=cls,
                    name_chain=tuple(f"_{name_body}_" for name_body in parameter_name.split("__")),
                    parameter=parameter
                )
                for parameter_name, parameter in parameters.items()
            )

            #signature = inspect.signature(
            #    descriptor._method,
            #    locals={
            #        cls.__name__: cls,
            #        **{
            #            str(type_param): type_param
            #            for type_param in cls.__type_params__
            #        }
            #    },
            #    eval_str=True
            #)

            #return_annotation = signature.return_annotation
            #if descriptor._is_plural:
            #    element_type_annotation = extract_tuple_annotation(return_annotation)
            #else:
            #    element_type_annotation = return_annotation
            #descriptor._element_type_annotation = element_type_annotation

            #if isinstance(element_type_annotation, TypeVar | TypeAliasType):
            #    element_type = None
            #else:
            #    try:
            #        element_type = element_type_annotation.__origin__
            #    except AttributeError:
            #        element_type = element_type_annotation
            #descriptor._element_type = element_type

            #descriptor._descriptor_chains = tuple(
            #    get_descriptor_chain(
            #        name_chain=tuple(f"_{name_body}_" for name_body in parameter_name.split("__")),
            #        root_class=cls,
            #        parameter=parameter
            #    )
            #    for parameter_name, parameter in signature.parameters.items()
            #)

            element_annotation = extract_element_annotation(
                annotation=signature.return_annotation,
                is_plural=descriptor._is_plural
            )
            if (overridden_element_annotation := cls._element_annotations.get(name)) is not None:
                overridden_descriptor = base_cls._lazy_descriptors[name]
                assert descriptor._is_plural is overridden_descriptor._is_plural
                assert element_annotation == overridden_element_annotation
            else:
                cls._element_annotations[name] = element_annotation
            cls._lazy_descriptors[name] = descriptor

            #_, element_annotation = descriptor_info_dict[name]
            if (element_cls := get_lazy_cls_from_annotation(element_annotation)) is not None:
                descriptor._freezer = element_cls._freeze

            #if (overridden_descriptor_info := overridden_descriptor_info_dict.get(name)) is not None:
            #    overridden_descriptor, overridden_element_annotation = overridden_descriptor_info

            #if overridden_descriptor is not None:
            #    #descriptor._element_registration = overridden_descriptor._element_registration
            #    assert descriptor._can_override(overridden_descriptor)

            #overridden_descriptor = base_cls._lazy_descriptors.get(name)
            #if overridden_descriptor is not None:
            #    #descriptor._element_registration = overridden_descriptor._element_registration
            #    assert descriptor._can_override(overridden_descriptor)

            #if isinstance(element_type, LazyObject):
            #    descriptor._freezer = LazyObject._freeze

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._lazy_slots: dict[str, LazySlot] = {}
        self._is_frozen: bool = False
        for descriptor in type(self)._lazy_descriptors.values():
            descriptor._init(self)

    @classmethod
    def _freeze(
        cls: type[Self],
        element: Any
    ) -> None:
        if not isinstance(element, LazyObject):
            return
        if element._is_frozen:
            return
        element._is_frozen = True
        for descriptor in type(element)._lazy_descriptors.values():
            descriptor._get_slot(element)._is_writable = False
            for child_element in descriptor._get_elements(element):
                cls._freeze(child_element)

    def _copy_lazy_content(
        self: Self,
        src_object: Self
    ) -> None:
        for descriptor in type(self)._lazy_descriptors.values():
            if descriptor._is_variable:
                descriptor._set_elements(self, descriptor._get_elements(src_object))

    def copy(
        self: Self
    ) -> Self:
        cls = type(self)
        result = cls.__new__(cls)
        for slot_name, slot_copier in cls._slot_copiers.items():
            result.__setattr__(slot_name, slot_copier(self.__getattribute__(slot_name)))
        for descriptor in cls._lazy_descriptors.values():
            descriptor._init(result)
        result._copy_lazy_content(self)
        #for descriptor in cls._lazy_descriptors.values():
        #    if descriptor._is_variable:
        #        descriptor._set_elements(result, descriptor._get_elements(self))
        #result._lazy_slots = {}
        #result._is_frozen = False
        #for descriptor in cls._lazy_descriptors.values():
        #    #descriptor._init(result)
        #    if descriptor._is_variable:
        #        descriptor._set_elements(result, descriptor._get_elements(self))
        #for slot_name, slot_copier in cls._slot_copiers.items():
        #    result.__setattr__(slot_name, slot_copier(self.__getattribute__(slot_name)))
        return result
