from __future__ import annotations


import copy
import inspect
from abc import ABC
from typing import (
    Any,
    Callable,
    #Callable,
    ClassVar,
    Self,
    TypeVar
)

import attrs

from .lazy_descriptor import LazyDescriptor
from .lazy_overriding import LazyOverriding
from .lazy_routine import (
    LazyPluralRoutine,
    LazyRoutine,
    LazySingularRoutine
)
from .lazy_slot import (
    LazyFrozenSlot,
    LazySlot
)


@attrs.frozen(kw_only=True)
class TypeHint:
    type_params_cls: type[LazyObject]
    annotation_cls: type[LazyObject] | None
    annotation: Any

    def specialize(
        self: Self,
        type_hint: TypeHint
        #annotation: Any,
        #type_params_cls: type,
        #specializations: tuple[Any, ...]
    ) -> TypeHint:
        assert self.type_params_cls is type_hint.annotation_cls
        type_params = self.type_params_cls.__type_params__
        specializations = getattr(type_hint.annotation, "__args__", ())
        assert len(type_params) == len(specializations)
        annotation = self.annotation
        if annotation in type_params:
            new_annotation = specializations[type_params.index(annotation)]
            #return specializations[type_params.index(annotation)]
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


class LazyObject(ABC):
    __slots__ = (
        "_lazy_slots",
        "_is_frozen"
    )

    #_default_constructor_kwargs: ClassVar[dict] = {}

    _special_slot_copiers: ClassVar[dict[str, Callable]] = {
        "_lazy_slots": lambda o: type(o)._lazy_slots_cls(),
        "_is_frozen": lambda o: False
    }

    #_lazy_descriptor_info_dict: dict[str, tuple[LazyDescriptor, Any]] = {}
    #_lazy_descriptors: ClassVar[dict[str, LazyDescriptor]] = {}
    #_element_annotations: ClassVar[dict[str, Any]] = {}
    _lazy_routines: ClassVar[tuple[LazyRoutine, ...]] = ()
    _lazy_routine_dict: ClassVar[dict[str, LazyRoutine]] = {}
    _lazy_type_hint_dict: ClassVar[dict[str, TypeHint]] = {}
    _lazy_slots_cls: ClassVar[type] = object
    _slot_copiers: ClassVar[dict[str, Callable]] = {}

    def __init_subclass__(
        cls: type[Self]
    ) -> None:

        def get_lazy_cls_from_annotation(
            annotation: Any
        ) -> type[LazyObject] | None:
            if isinstance(annotation, TypeVar):
                annotation = annotation.__bound__
            if isinstance(cls := getattr(annotation, "__origin__", annotation), type):
                if issubclass(cls, LazyObject):
                    return cls
            return None
            #return (
            #    cls
            #    if isinstance(cls := getattr(annotation, "__origin__", annotation), type)
            #    and issubclass(cls, LazyObject)
            #    else get_lazy_cls_from_annotation(annotation.__bound__)
            #    if isinstance(annotation, TypeVar)
            #    else None
            #)

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

        #def get_descriptor_by_name(
        #    lazy_cls: type[LazyObject],
        #    descriptor_name: str
        #) -> LazyDescriptor | None:
        #    return {
        #        descriptor._name: descriptor
        #        for descriptor in lazy_cls._lazy_descriptors
        #    }.get(descriptor_name)

        def get_parameter_overriding_chain(
            #root_cls: Any,
            #root_lazy_routines: dict[str, LazyRoutine],
            root_type_hint: TypeHint,
            name_chain: tuple[str, ...],
            parameter: inspect.Parameter
        ) -> tuple[LazyOverriding, ...]:
            parameter_overriding_chain: list[LazyOverriding] = []
            #element_cls: type[LazyObject] | None = root_cls
            #element_annotation = (
            #    root_cls
            #    if not (type_params := root_cls.__type_params__)
            #    else root_cls[*type_params]
            #)
            last_routine_freeze: bool = False
            type_hint = root_type_hint
            provided_annotation = parameter.annotation
            for name in name_chain:
                #assert element_cls is not None
                #type_hint = (root_lazy_routines if element_cls is root_cls else {
                #    routine._overriding._name: routine
                #    for routine in element_cls._lazy_routines
                #})[name]._overriding._element_type_hint_dict[]
                #assert (element_cls := get_lazy_cls_from_annotation(element_annotation)) is not None
                #element_annotation = specialize_type_params(
                #    annotation=,
                #    type_params_cls=root_cls,
                #    specializations=getattr(element_annotation, "__parameters__", ())
                #)
                #print(root_annotation, element_cls, element_cls._lazy_descriptor_element_annotation_dict)
                assert (element_cls := type_hint.annotation_cls) is not None
                routine = element_cls._lazy_routine_dict[name]
                last_routine_freeze = routine._freeze
                type_hint = element_cls._lazy_type_hint_dict[name].specialize(type_hint)
                #last_descriptor = element_cls._lazy_descriptors[descriptor_name]
                #element_annotation = element_cls._element_annotations[descriptor_name]
                #element_annotation = element_cls._lazy_descriptors[descriptor]
                #is_plural = last_descriptor._is_plural
                overriding = routine._overriding
                provided_annotation = extract_element_annotation(
                    annotation=provided_annotation,
                    is_plural=overriding._is_plural
                )
                parameter_overriding_chain.append(overriding)
            #print(cls, name_chain, dict(last_overriding._routines), type_hint.type_params_cls)
            assert last_routine_freeze
            assert type_hint.annotation == provided_annotation
            assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            assert parameter.default is inspect.Parameter.empty
            return tuple(parameter_overriding_chain)

        #def get_descriptor_info_chain(
        #    root_cls: Any,
        #    name_chain: tuple[str, ...],
        #    parameter: inspect.Parameter
        #) -> tuple[tuple[str, bool], ...]:
        #    descriptor_info_chain: list[tuple[str, bool]] = []
        #    last_descriptor: LazyDescriptor | None = None
        #    element_annotation = (
        #        root_cls
        #        if not (type_params := root_cls.__type_params__)
        #        else root_cls[*type_params]
        #    )
        #    provided_annotation = parameter.annotation
        #    for descriptor_name in name_chain:
        #        assert (element_cls := get_lazy_cls_from_annotation(element_annotation)) is not None
        #        #print(root_annotation, element_cls, element_cls._lazy_descriptor_element_annotation_dict)
        #        last_descriptor = element_cls._lazy_descriptors[descriptor_name]
        #        element_annotation = element_cls._element_annotations[descriptor_name]
        #        #element_annotation = element_cls._lazy_descriptors[descriptor]
        #        is_plural = last_descriptor._is_plural
        #        provided_annotation = extract_element_annotation(
        #            annotation=provided_annotation,
        #            is_plural=is_plural
        #        )
        #        descriptor_info_chain.append((descriptor_name, is_plural))
        #    assert last_descriptor is not None and last_descriptor._freeze
        #    assert element_annotation == provided_annotation
        #    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        #    assert parameter.default is inspect.Parameter.empty
        #    return tuple(descriptor_info_chain)

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
        base = cls.__base__
        assert issubclass(base, LazyObject)

        #base_type_params = base.__type_params__
        #type_param_specializations: tuple = (
        #    ()
        #    if (orig_bases := getattr(cls, "__orig_bases__", None)) is None
        #    or (args := getattr(orig_bases[0], "__args__", None)) is None
        #    else args
        #)
        #assert len(base_type_params) == len(type_param_specializations)

        #descriptor_dict = {
        #    descriptor._name: descriptor
        #    for descriptor in base._lazy_descriptors
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
        #    for name, (descriptor, element_annotation) in base._lazy_descriptor_info_dict.items()
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

        #base_overridings = {
        #    routine._overriding._name: routine._overriding
        #    for routine in base._lazy_routines
        #}
        routine_dict: dict[str, LazyRoutine] = {}
        #lazy_overridings: dict[str, LazyOverriding] = {}
        type_hint_dict: dict[str, TypeHint] = {}
        cls._lazy_routine_dict = routine_dict
        cls._lazy_type_hint_dict = type_hint_dict

        override_type_hint = TypeHint(
            type_params_cls=cls,
            annotation_cls=base,
            annotation=base if not base.__type_params__ else cls.__orig_bases__[0]
        )
        #base_element_type_hint_dict = base._lazy_type_hint_dict
        for name, routine in base._lazy_routine_dict.items():
            routine._overriding._routines[cls] = routine
            routine_dict[name] = routine
            type_hint_dict[name] = base._lazy_type_hint_dict[name].specialize(override_type_hint)
            #lazy_overridings[name] = overriding
            #overriding = routine._overriding
            #base_element_type_hint = base._lazy_type_hint_dict[name]
            #element_type_hint = base._lazy_type_hint_dict[name].specialize(override_type_hint)
            #element_annotation = specialize_type_params(
            #    annotation=base_element_type_hint.annotation,
            #    type_params_cls=base,
            #    specializations=type_param_specializations
            #)
            #element_annotation = (
            #    type_param_specializations[base_type_params.index(base_element_annotation)]
            #    if base_element_annotation in base_type_params
            #    else base_element_annotation
            #    if not (parameters := getattr(base_element_annotation, "__parameters__", ()))
            #    else base_element_annotation[*(
            #        type_param_specializations[base_type_params.index(type_param)]
            #        for type_param in parameters
            #    )]
            #)
            #base_element_lazy_cls = base_element_type_hint.lazy_cls
            #element_lazy_cls = get_lazy_cls_from_annotation(element_annotation)
            #overriding._element_type_hint_dict[cls] = TypeHint(
            #    type_params_cls=cls,
            #    annotation_cls=element_annotation,
            #    annotation=
            #)
            #overriding._element_annotations[cls] = element_annotation
            #overriding._element_lazy_cls[cls] = element_lazy_cls

        root_type_hint = TypeHint(
            type_params_cls=cls,
            annotation_cls=cls,
            annotation=cls if not cls.__type_params__ else cls[*cls.__type_params__]
        )
        for name, descriptor in cls.__dict__.items():
            if not isinstance(descriptor, LazyDescriptor):
                continue

            assert name.startswith("_") and name.endswith("_") and "__" not in name
            info = descriptor._info
            signature = inspect.signature(
                info.method,
                locals={
                    cls.__name__: cls,
                    **{
                        str(type_param): type_param
                        for type_param in cls.__type_params__
                    }
                },
                eval_str=True
            )
            element_annotation = extract_element_annotation(
                annotation=signature.return_annotation,
                is_plural=info.is_plural
            )
            element_lazy_cls = get_lazy_cls_from_annotation(element_annotation)
            if (overridden_routine := routine_dict.get(name)) is not None:
                assert not overridden_routine._freeze or info.freeze
                overriding = overridden_routine._overriding
                assert overriding._name == name
                assert overriding._is_plural == info.is_plural
                assert overriding._hasher is info.hasher
                type_hint = type_hint_dict[name]
                assert type_hint.annotation == element_annotation
                assert type_hint.annotation_cls is element_lazy_cls  # TODO: subclass check
                #base_element_lazy_cls = overriding._element_lazy_cls[base]
                #assert (
                #    base_element_lazy_cls is None and element_lazy_cls is None
                #    or base_element_lazy_cls is not None and element_lazy_cls is not None
                #    and issubclass(element_lazy_cls, base_element_lazy_cls)
                #)
            else:
                overriding = LazyOverriding(
                    name=name,
                    is_plural=info.is_plural,
                    hasher=info.hasher,
                    freezer=cls._lazy_freezer if element_lazy_cls is not None else cls._empty_freezer
                )
                type_hint = TypeHint(
                    type_params_cls=cls,
                    annotation_cls=element_lazy_cls,
                    annotation=element_annotation
                )
                #lazy_overridings[name] = overriding
                type_hint_dict[name] = type_hint
                #overriding._element_annotations[cls] = element_annotation
                #overriding._element_lazy_cls[cls] = element_lazy_cls

            parameter_overriding_chains = tuple(
                get_parameter_overriding_chain(
                    #root_cls=cls,
                    #root_lazy_routines=lazy_routines,
                    root_type_hint=root_type_hint,
                    name_chain=tuple(f"_{name_body}_" for name_body in parameter_name.split("__")),
                    parameter=parameter
                )
                for parameter_name, parameter in signature.parameters.items()
            )
            assert not info.is_variable or not parameter_overriding_chains
            routine_cls: type[LazyRoutine] = LazyPluralRoutine if info.is_plural else LazySingularRoutine
            routine = routine_cls(
                overriding=overriding,
                method=info.method,
                parameter_overriding_chains=parameter_overriding_chains,
                is_variable=info.is_variable,
                freeze=info.freeze,
                cache_capacity=info.cache_capacity
            )
            overriding._routines[cls] = routine
            routine_dict[name] = routine
            descriptor._routine = routine

        cls._lazy_routines = tuple(routine_dict.values())
        cls._lazy_slots_cls = attrs.make_class(
            name=f"__{cls.__name__}",
            attrs={
                name: attrs.field(factory=LazySlot if routine._is_variable else LazyFrozenSlot)
                for name, routine in routine_dict.items()
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


        #cls._lazy_descriptors = base._lazy_descriptors.copy()
        #cls._element_annotations = {
        #    name: (
        #        type_param_specializations[base_type_params.index(element_annotation)]
        #        if element_annotation in base_type_params
        #        else element_annotation
        #        if not (parameters := getattr(element_annotation, "__parameters__", ()))
        #        else element_annotation[*(
        #            type_param_specializations[base_type_params.index(type_param)]
        #            for type_param in parameters
        #        )]
        #    )
        #    for name, element_annotation in base._element_annotations.items()
        #}

        #for name, descriptor in cls.__dict__.items():
        #    if not isinstance(descriptor, LazyDescriptor):
        #        continue

        #    assert name.startswith("_") and name.endswith("_") and "__" not in name
        #    signature = inspect.signature(
        #        descriptor._method,
        #        locals={
        #            cls.__name__: cls,
        #            **{
        #                str(type_param): type_param
        #                for type_param in cls.__type_params__
        #            }
        #        },
        #        eval_str=True
        #    )
        #    #parameter_names = descriptor._method.__code__.co_varnames
        #    #tuple(
        #    #    tuple(f"_{name_body}_" for name_body in varname.split("__"))
        #    #    for varname in method.__code__.co_varnames
        #    #)
        #    #parameter_annotation_dict = parameter_annotation_dicts[name]
        #    #return_annotation = parameter_annotation_dict.pop("return")

        #    parameters = signature.parameters
        #    if descriptor._is_variable:
        #        assert not parameters
        #    descriptor._descriptor_info_chains = tuple(
        #        get_descriptor_info_chain(
        #            root_cls=cls,
        #            name_chain=tuple(f"_{name_body}_" for name_body in parameter_name.split("__")),
        #            parameter=parameter
        #        )
        #        for parameter_name, parameter in parameters.items()
        #    )

        #    #signature = inspect.signature(
        #    #    descriptor._method,
        #    #    locals={
        #    #        cls.__name__: cls,
        #    #        **{
        #    #            str(type_param): type_param
        #    #            for type_param in cls.__type_params__
        #    #        }
        #    #    },
        #    #    eval_str=True
        #    #)

        #    #return_annotation = signature.return_annotation
        #    #if descriptor._is_plural:
        #    #    element_type_annotation = extract_tuple_annotation(return_annotation)
        #    #else:
        #    #    element_type_annotation = return_annotation
        #    #descriptor._element_type_annotation = element_type_annotation

        #    #if isinstance(element_type_annotation, TypeVar | TypeAliasType):
        #    #    element_type = None
        #    #else:
        #    #    try:
        #    #        element_type = element_type_annotation.__origin__
        #    #    except AttributeError:
        #    #        element_type = element_type_annotation
        #    #descriptor._element_type = element_type

        #    #descriptor._descriptor_chains = tuple(
        #    #    get_descriptor_chain(
        #    #        name_chain=tuple(f"_{name_body}_" for name_body in parameter_name.split("__")),
        #    #        root_class=cls,
        #    #        parameter=parameter
        #    #    )
        #    #    for parameter_name, parameter in signature.parameters.items()
        #    #)

        #    element_annotation = extract_element_annotation(
        #        annotation=signature.return_annotation,
        #        is_plural=descriptor._is_plural
        #    )
        #    if (overridden_element_annotation := cls._element_annotations.get(name)) is not None:
        #        overridden_descriptor = base._lazy_descriptors[name]
        #        assert descriptor._is_plural is overridden_descriptor._is_plural
        #        assert element_annotation == overridden_element_annotation
        #    else:
        #        cls._element_annotations[name] = element_annotation
        #    cls._lazy_descriptors[name] = descriptor

        #    #_, element_annotation = descriptor_info_dict[name]
        #    if (element_cls := get_lazy_cls_from_annotation(element_annotation)) is not None:
        #        descriptor._freezer = element_cls._freeze

        #    #if (overridden_descriptor_info := overridden_descriptor_info_dict.get(name)) is not None:
        #    #    overridden_descriptor, overridden_element_annotation = overridden_descriptor_info

        #    #if overridden_descriptor is not None:
        #    #    #descriptor._element_registration = overridden_descriptor._element_registration
        #    #    assert descriptor._can_override(overridden_descriptor)

        #    #overridden_descriptor = base._lazy_descriptors.get(name)
        #    #if overridden_descriptor is not None:
        #    #    #descriptor._element_registration = overridden_descriptor._element_registration
        #    #    assert descriptor._can_override(overridden_descriptor)

        #    #if isinstance(element_type, LazyObject):
        #    #    descriptor._freezer = LazyObject._freeze

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._lazy_slots: tuple = type(self)._lazy_slots_cls()
        self._is_frozen: bool = False
        #for routine in type(self)._lazy_routines:
        #    routine.init(self)

    def _get_slot(
        self: Self,
        name: str
    ) -> LazySlot:
        return self._lazy_slots.__getattribute__(name)

    @classmethod
    def _empty_freezer(
        cls: type[Self],
        element: Any
    ) -> None:
        return

    @classmethod
    def _lazy_freezer(
        cls: type[Self],
        element: LazyObject
    ) -> None:
        #if not isinstance(element, LazyObject):
        #    return
        if element._is_frozen:
            return
        element._is_frozen = True
        for routine in type(element)._lazy_routines:
            overriding = routine._overriding
            overriding.get_slot(element)._is_writable = False
            if type(element)._lazy_type_hint_dict[overriding._name].annotation_cls is not None:
                for child_element in routine.get_elements(element):
                    cls._lazy_freezer(child_element)

    def _copy_lazy_content(
        self: Self,
        src_object: Self
    ) -> None:
        for routine in type(self)._lazy_routines:
            if routine._is_variable:
                routine.set_elements(self, routine.get_elements(src_object))

    def copy(
        self: Self
    ) -> Self:
        cls = type(self)
        result = cls.__new__(cls)
        for slot_name, slot_copier in cls._slot_copiers.items():
            result.__setattr__(slot_name, slot_copier(self.__getattribute__(slot_name)))
        #for routine in cls._lazy_routines:
        #    routine.init(result)
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
