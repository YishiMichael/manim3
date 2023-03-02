# TODO: The document is outdated.

"""
This module implements a basic class with lazy properties.

The functionality of `LazyBase` is to save resource as much as it can.
On one hand, all data of `lazy_object` and `lazy_property` are shared among
instances. On the other hand, these data will be restocked (recursively) if
they themselves are instances of `LazyBase`. One may also define custom
restockers for individual data.

Every child class of `LazyBase` shall be declared with an empty `__slots__`,
and all methods shall be sorted in the following way:
- magic methods
- lazy_variable_shared
- lazy_object
- lazy_property
- lazy_slot
- private class methods
- private methods
- public methods

All methods decorated by any of `lazy_object`, `lazy_variable_shared`,
`lazy_property` and `lazy_slot` should be static methods, and so are their
restockers. Type annotation is strictly applied to reduce chances of running
into unexpected behaviors.

Methods decorated by `lazy_object` should be named with underscores appeared
on both sides, i.e. `_data_`. Each should not take any argument and return
the *initial* value for this data. `NotImplemented` may be an alternative for
the value returned, as long as the data is initialized in `__new__` method.
In principle, the data can be of any type including mutable ones, but one must
keep in mind that data *cannot be mutated* as they are shared. The only way to
change the value is to reset the data via `__set__`, and the new value shall
be wrapped up with `LazyWrapper`. This makes it possible to manually share data
which is not the initial value. Note, the `__get__` method will return the
unwrapped data. One shall use `instance.__class__._data_._get_data(instance)`
to obtain the wrapped data if one wishes to share it with other instances.

Methods decorated by `lazy_variable_shared` are pretty much similar to ones
decorated by `lazy_object`, except that an argument `hasher` should be
additionally passed to the decorator. Data handled in these methods are
expected to be light-weighted and have much duplicated usage so that caching
can take effect. Data wrapping is not necessary when calling `__set__`.

Methods decorated by `lazy_property` should be named with the same style of
`lazy_object`. They should take *at least one* argument, and all names of
arguments should be matched with any `lazy_object` or other `lazy_property`
where underscores on edges are eliminated. Data is immutable, and calling
`__set__` method will trigger an exception. As the name `lazy` suggests, if
any correlated `lazy_object` is altered, as long as the calculation is never
done before, the recalculation will be executed when one calls `__get__`.

Methods decorated by `lazy_slot` should be named with an underscore inserted
at front, i.e. `_data`. They behave like a normal attribute of the class.
Again, each should not take any argument and return the *initial* value for
this data, with `NotImplemented` as an alternative if the data is set in
`__new__`. Data can be freely mutated because they are no longer shared
(as long as one does not do something like `b._data = a._data`, or calls the
`_copy` method). Data wrapping is not necessary when calling `__set__`.
"""


__all__ = [
    "LazyCollection",
    "LazyCollectionDescriptor",
    "LazyEntity",
    "LazyObject",
    "LazyObjectDescriptor",
    "LazyPropertyDescriptor",
]


from abc import (
    ABC,
    abstractmethod
)
#from functools import wraps
#from abc import (
#    ABC,
#    abstractmethod
#)
#from dataclasses import (
#    field,
#    make_dataclass
#)
import re
#from types import GenericAlias
#from types import MappingProxyType
#from types import GenericAlias
from typing import (
    Any,
    Callable,
    ClassVar,
    #Concatenate,
    Generator,
    Generic,
    Iterator,
    #Iterator,
    #Never,
    #ParamSpec,
    TypeVar,
    #Union,
    overload
)

#from ordered_set import OrderedSet

from ..lazy.dag import DAGNode


#_DAGNodeT = TypeVar("_DAGNodeT", bound="DAGNode")
#_KeyT = TypeVar("_KeyT", bound=Hashable)
#_LazyBaseT = TypeVar("_LazyBaseT", bound="LazyBase")
_LazyEntityT = TypeVar("_LazyEntityT", bound="LazyEntity")
_LazyObjectT = TypeVar("_LazyObjectT", bound="LazyObject")
#_ParameterElementsT = TypeVar("_ParameterElementsT", bound=Hashable)
#_ObjT = TypeVar("_ObjT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
#_ParameterSpec = ParamSpec("_ParameterSpec")
_LazyDescriptorT = TypeVar("_LazyDescriptorT", bound="LazyDescriptor")
#_Annotation = Any


#class LazyObjectNode(DAGNode):
#    __slots__ = ("_lazy_object", "_expired")
#
#    def __init__(self, *, lazy_object: "LazyObject" = NotImplemented, expired: bool = True):
#        super().__init__()
#        self._lazy_object: LazyObject = lazy_object
#        self._expired: bool = expired


#class LazyParameterNode(DAGNode):
#    __slots__ = ("_parameter",)
#
#    def __init__(self, instance: "LazyInstance") -> None:
#        super().__init__()
#        self._parameter: LazyInstance = instance


class LazyNode(DAGNode):
    __slots__ = ("_ref",)

    def __init__(
        self,
        instance: "LazyBase"
    ) -> None:
        super().__init__()
        self._ref: LazyBase = instance


class LazyBase(ABC):
    __slots__ = (
        "_dependency_node",
        "_parameter_node"
        #"_readonly",
        #"_restock_callbacks"
    )

    _VACANT_INSTANCES: "ClassVar[list[LazyBase]]"
    #_VARIABLE_DESCRS: "ClassVar[list[LazyObjectDescriptor]]"
    _dependency_node: LazyNode
    _parameter_node: LazyNode

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._VACANT_INSTANCES = []
        #return super().__init_subclass__()

    def __new__(
        cls,
        *args,
        **kwargs
    ):
        if (instances := cls._VACANT_INSTANCES):
            instance = instances.pop()
            assert isinstance(instance, cls)
        else:
            instance = super().__new__(cls)
            instance._init_nodes()
        return instance

    def _init_nodes(self):
        self._dependency_node = LazyNode(self)
        self._parameter_node = LazyNode(self)

    #def __init__(self) -> None:
    #    super().__init__()
    #    #self._readonly: bool = False
    #    #self._restock_callbacks: list[Callable[[Any], None]] | None = []  # TODO: typing

    def _iter_dependency_children(self) -> "Generator[LazyBase, None, None]":
        for child in self._dependency_node._children:
            yield child._ref

    def _iter_dependency_parents(self) -> "Generator[LazyBase, None, None]":
        for parent in self._dependency_node._parents:
            yield parent._ref

    def _iter_dependency_descendants(self) -> "Generator[LazyBase, None, None]":
        for descendant in self._dependency_node._iter_descendants():
            yield descendant._ref

    def _iter_dependency_ancestors(self) -> "Generator[LazyBase, None, None]":
        for ancestor in self._dependency_node._iter_ancestors():
            yield ancestor._ref

    def _bind_dependency_children(
        self,
        *instances: "LazyBase"
    ):
        self._dependency_node._bind_children(*(
            instance._dependency_node
            for instance in instances
        ))

    def _unbind_dependency_children(
        self,
        *instances: "LazyBase"
    ):
        self._dependency_node._unbind_children(*(
            instance._dependency_node
            for instance in instances
        ))

    def _iter_parameter_children(self) -> "Generator[LazyBase, None, None]":
        for child in self._parameter_node._children:
            yield child._ref

    def _iter_parameter_parents(self) -> "Generator[LazyBase, None, None]":
        for parent in self._parameter_node._parents:
            yield parent._ref

    def _iter_parameter_descendants(self) -> "Generator[LazyBase, None, None]":
        for descendant in self._parameter_node._iter_descendants():
            yield descendant._ref

    def _iter_parameter_ancestors(self) -> "Generator[LazyBase, None, None]":
        for ancestor in self._parameter_node._iter_ancestors():
            yield ancestor._ref

    def _bind_parameter_children(
        self,
        *instances: "LazyBase"
    ):
        self._parameter_node._bind_children(*(
            instance._parameter_node
            for instance in instances
        ))

    def _unbind_parameter_children(
        self,
        *instances: "LazyBase"
    ):
        self._parameter_node._unbind_children(*(
            instance._parameter_node
            for instance in instances
        ))


class LazyEntity(LazyBase):
    __slots__ = ()

    def _is_readonly(self) -> bool:
        return any(
            isinstance(instance, LazyProperty)
            for instance in self._iter_dependency_ancestors()
        )

    def _expire_properties(self) -> None:
        for expired in self._iter_parameter_ancestors():
            #if isinstance(expired, LazyParameter):
            #    #print(expired)
            #    expired._unbind_parameter_children(*expired._iter_parameter_children())
            #    expired._set(None)
            if not isinstance(expired, LazyProperty):
                continue
            expired._unbind_parameter_children(*expired._iter_parameter_children())
            expired._unbind_dependency_children(*expired._iter_dependency_children())
            expired._set(None)


class LazyObject(LazyEntity):
    __slots__ = ()

    _LAZY_DESCRIPTORS: "ClassVar[dict[str, LazyDescriptor]]"
    _ALL_SLOTS: "ClassVar[tuple[str, ...]]"
    #_OBJECT_DESCRIPTORS: "ClassVar[list[LazyObjectDescriptor]]"
    #_COLLECTION_DESCRIPTORS: "ClassVar[list[LazyCollectionDescriptor]]"
    #_PARAMETER_DESCRIPTORS: "ClassVar[list[LazyParameterDescriptor]]"
    #_PROPERTY_DESCRIPTORS: "ClassVar[list[LazyPropertyDescriptor]]"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        #attrs: dict[str, Any] = {
        #    name: attr
        #    for parent_cls in reversed(cls.__mro__)
        #    for name, attr in parent_cls.__dict__.items()
        #}
        descriptors: dict[str, LazyDescriptor] = {
            name: attr
            for parent_cls in reversed(cls.__mro__)
            for name, attr in parent_cls.__dict__.items()
            if isinstance(attr, LazyDescriptor)
        }
        #object_descriptors = {
        #    name: attr
        #    for name, attr in attrs.items()
        #    if isinstance(attr, LazyObjectDescriptor)
        #}
        #collection_descriptors = {
        #    name: attr
        #    for name, attr in attrs.items()
        #    if isinstance(attr, LazyCollectionDescriptor)
        #}
        #property_descriptors = {
        #    name: attr
        #    for name, attr in attrs.items()
        #    if isinstance(attr, LazyPropertyDescriptor)
        #}
        #descriptors: dict[str, LazyObjectDescriptor | LazyCollectionDescriptor | LazyPropertyDescriptor] = {
        #    **object_descriptors,
        #    **collection_descriptors,
        #    **property_descriptors
        #}
        #assert all(
        #    re.fullmatch(r"_\w+_", name)
        #    for name in descriptors
        #)


        for name, descriptor in descriptors.items():
            if name not in cls.__dict__:
                continue
            assert isinstance(descriptor, LazyObjectDescriptor | LazyCollectionDescriptor | LazyPropertyDescriptor)
            assert re.fullmatch(r"_\w+_", name)

        cls._LAZY_DESCRIPTORS = descriptors
        cls._ALL_SLOTS = tuple(set(
            slot
            for parent_cls in reversed(cls.__mro__)
            if issubclass(parent_cls, LazyObject)
            for slot in parent_cls.__slots__
        ))

            #if descriptor.object_type is _SelfPlaceholder:
            #    descriptor.object_type = cls

            #is_collection = isinstance(descriptor, LazyCollectionDescriptor)
            #descriptor.is_collection = is_collection

            #return_annotation = inspect.signature(descriptor.method).return_annotation
            #if isinstance(return_annotation, type) and not isinstance(return_annotation, GenericAlias):
            #    if issubclass(return_annotation, LazyObject):
            #        descriptor.object_type = return_annotation
            #elif hasattr(return_annotation, "__origin__"):
            #    if return_annotation.__origin__ is LazyCollection:
            #        descriptor.object_type = return_annotation.__args__[0]
            #elif isinstance(return_annotation, str):
            #    if return_annotation == cls.__name__:
            #        descriptor.object_type = cls
            #    elif return_annotation == f"LazyCollection[{cls.__name__}]":
            #        descriptor.object_type = cls

        #for descriptor in descriptors.values():
        #    if not isinstance(descriptor, LazyPropertyDescriptor):
        #        continue

        #    parameter_descriptor_chains: list[tuple[LazyDescriptor, ...]] = []
        #    for name in descriptor.parameter_names:
        #        descriptor_chain: list[LazyDescriptor] = []
        #        object_cls: type[LazyObject] = cls
        #        for name_segment in re.findall(r"_\w+?_(?=_|$)", name):
        #            descriptor_segment = object_cls._LAZY_DESCRIPTORS[name_segment]
        #            descriptor_chain.append(descriptor_segment)
        #            object_cls = descriptor_segment.object_type

        #    #for name in tuple(inspect.signature(descriptor.method).parameters)[1:]:  # remove cls
        #    #    descriptor_chain: list[LazyDescriptor] = []
        #    #    requires_unwrapping = False
        #    #    if not re.fullmatch(r"_\w+_", name):
        #    #        requires_unwrapping = True
        #    #        name = f"_{name}_"

        #    #    object_cls: type[LazyObject] = cls
        #    #    #print(name)
        #    #    for name_segment in re.findall(r"_\w+?_(?=_|$)", name):
        #    #        descriptor_segment = object_cls._LAZY_DESCRIPTORS[name_segment]
        #    #        descriptor_chain.append(descriptor_segment)
        #    #        object_cls = descriptor_segment.object_type
        #    #        #return_annotation = descriptor_segment.return_annotation
        #    #        ##print(return_annotation)
        #    #        #if isinstance(return_annotation, GenericAlias):
        #    #        #    object_cls = return_annotation.__args__[0]
        #    #        #else:
        #    #        #    object_cls = return_annotation

        #    #    parameter_items.append((tuple(descriptor_chain), requires_unwrapping))
        #        parameter_descriptor_chains.append(tuple(descriptor_chain))
        #    descriptor.parameter_items[cls] = tuple(parameter_descriptor_chains)




        #for object_descriptor in object_descriptors.values():
        #    return_annotation = inspect.signature(object_descriptor.method)
        #    if isinstance(return_annotation, type):
        #        object_type = return_annotation
        #    elif isinstance(return_annotation, str) and return_annotation == cls.__name__:
        #        object_type = cls
        #    else:
        #        raise TypeError
        #    object_descriptor.object_type = object_type

        #for collection_descriptor in collection_descriptors.values():
        #    return_annotation = inspect.signature(collection_descriptor.method)
        #    if isinstance(return_annotation, GenericAlias):
        #        object_type = return_annotation.__origin__
        #    elif isinstance(return_annotation, str) and return_annotation == f"LazyCollection[{cls.__name__}]":
        #        object_type = cls
        #    else:
        #        raise TypeError
        #    collection_descriptor.object_type = object_type



        #def construct_obj_from_descriptor_chain(
        #    descriptor_chain: tuple[LazyObjectDescriptor | LazyCollectionDescriptor | LazyPropertyDescriptor, ...],
        #    obj: Any
        #) -> Any:
        #    for descriptor in descriptor_chain:
        #        if isinstance(descriptor, LazyCollectionDescriptor):
        #            obj = apply_deepest(lambda instance: tuple(descriptor.__get__(instance)._node_children), obj)
        #        else:
        #            obj = apply_deepest(lambda instance: descriptor.__get__(instance), obj)
        #    return obj


        #descrs: dict[str, LazyDescriptor] = {
        #    name: attr
        #    for name, attr in attrs.items()
        #    if isinstance(attr, LazyDescriptor)
        #}
        #cls._LAZY_DESCRIPTORS = list(descrs.values())

        #parameter_descriptors: dict[str, LazyParameterDescriptor] = {}

        #for property_descriptor in property_descriptors.values():
        #    property_parameter_descriptors: list[LazyParameterDescriptor] = []
        #    for parameter_name, requires_unwrapping in zip(
        #        tuple(inspect.signature(property_descriptor.method).parameters)[1:],
        #        property_descriptor.requires_unwrapping_tuple,
        #        strict=True
        #    ):
        #        if (parameter_descriptor := parameter_descriptors.get(parameter_name)) is None:
        #            name = f"_{parameter_name}_" if requires_unwrapping else parameter_name
        #            parameter_descriptor = LazyParameterDescriptor(tuple(
        #                descriptors[name_piece]
        #                for name_piece in re.findall(r"_\w+?_(?=_|$)", name)
        #            ))
        #            parameter_descriptors[parameter_name] = parameter_descriptor
        #        property_parameter_descriptors.append(parameter_descriptor)
        #    property_descriptor.parameter_descriptors[cls] = tuple(property_parameter_descriptors)


            #method = property_descriptor.method
            #parameter_items: list[tuple[tuple[Union[
            #    LazyObjectDescriptor[LazyObject, Any],
            #    LazyCollectionDescriptor[LazyEntity, Any],
            #    LazyPropertyDescriptor[LazyEntity, Any]
            #], ...], bool]] = []
            #for parameter_name in inspect.signature(property_descriptor.method).parameters:
            #    is_lazy_value = False
            #    if not re.fullmatch(r"_\w+_", parameter_name):
            #        is_lazy_value = True
            #        parameter_name = f"_{parameter_name}_"
            #    descriptor_chain = tuple(
            #        descriptors[name]
            #        for name in re.findall(r"_\w+?_(?=_|$)", parameter_name)
            #    )
            #    parameter_items.append((descriptor_chain, is_lazy_value))
            #property_descriptor.[cls] = tuple(
            #    (tuple(
            #        descriptors[name_piece]
            #        for name_piece in re.findall(r"_\w+?_(?=_|$)", name)
            #    ), is_lazy_value)
            #    for name, is_lazy_value in (
            #        (parameter_name, False) if re.fullmatch(r"_\w+_", parameter_name) else (f"_{parameter_name}_", True)
            #        for parameter_name in inspect.signature(property_descriptor.method).parameters
            #    )
            #)

            #descr._setup_callables(descrs)

        #cls.__dict__ = MappingProxyType({**cls.__dict__, **descriptors})
        #cls._OBJECT_DESCRIPTORS = list(object_descriptors.values())
        #cls._COLLECTION_DESCRIPTORS = list(collection_descriptors.values())
        #cls._PARAMETER_DESCRIPTORS = list(parameter_descriptors.values())
        #cls._PROPERTY_DESCRIPTORS = list(property_descriptors.values())

    def __init__(self) -> None:
        super().__init__()
        #self._restock_callbacks: list[Callable[[Any], None]] | None = []
        cls = self.__class__
        for descriptor in cls._LAZY_DESCRIPTORS.values():
            descriptor.initialize(self)
        #for object_descriptor in cls._OBJECT_DESCRIPTORS:
        #    object_descriptor.initialize(self)
        #for collection_descriptor in cls._COLLECTION_DESCRIPTORS:
        #    collection_descriptor.initialize(self)
        ##for parameter_descriptor in cls._PARAMETER_DESCRIPTORS:
        ##    parameter_descriptor.initialize(self)
        #for property_descriptor in cls._PROPERTY_DESCRIPTORS:
        #    property_descriptor.initialize(self)
            
        #for descr in self.__class__._LAZY_DESCRIPTORS:
        #    if isinstance(descr, LazyObjectDescriptor):
        #        if (default_object := descr._default_object) is None:
        #            default_object = descr.method()
        #            default_object._restock_callbacks = None  # Never restock
        #            descr._default_object = default_object
        #        descr.initialize(self, default_object)
        #        children.append(default_object)
        #    elif isinstance(descr, LazyCollectionDescriptor):
        #        default_collection = descr.method()
        #        descr.initialize(self, default_collection)
        #        children.append(default_collection)
        #self._bind_dependency_children(*children)

    #def __del__(self) -> None:
    #    self._restock_node()

    def _unbind_dependency_children(
        self,
        *instances: "LazyBase"
    ):
        super()._unbind_dependency_children(*instances)
        for instance in instances:
            if not isinstance(instance, LazyObject):
                continue
            if instance._iter_dependency_parents():
                continue
            for obj in instance._iter_dependency_descendants():
                if not isinstance(obj, LazyObject):
                    continue
                obj._restock_node()

    def _copy(self: _LazyObjectT) -> _LazyObjectT:
        cls = self.__class__
        result = cls.__new__(cls)
        result._init_nodes()
        for descriptor in cls._LAZY_DESCRIPTORS.values():
            descriptor.copy_initialize(result, self)
        for slot_name in cls._ALL_SLOTS:
            result.__setattr__(slot_name, self.__getattribute__(slot_name))
        #for object_descriptor in cls._OBJECT_DESCRIPTORS:
        #    object_descriptor.copy_initialize(result, self)
        #for collection_descriptor in cls._COLLECTION_DESCRIPTORS:
        #    collection_descriptor.copy_initialize(result, self)
        ##for parameter_descriptor in cls._PARAMETER_DESCRIPTORS:
        ##    parameter_descriptor.copy_initialize(result, self)
        #for property_descriptor in cls._PROPERTY_DESCRIPTORS:
        #    property_descriptor.copy_initialize(result, self)
        return result

    def _restock_node(self) -> None:
        # TODO: check refcnt
        #for instance in self._iter_dependency_descendants():
        #    if not isinstance(instance, LazyObject):
        #        continue
            #if (callbacks := instance._restock_callbacks) is None:
            #    continue
            #for callback in callbacks:
            #    callback(instance)
            #callbacks.clear()
        for descriptor in self.__class__._LAZY_DESCRIPTORS.values():
            descriptor.restock(self)
        self.__class__._VACANT_INSTANCES.append(self)

    #def _at_restock(
    #    self,
    #    callback: Callable[[Any], None]
    #) -> None:
    #    if (callbacks := self._restock_callbacks) is not None:
    #        callbacks.append(callback)


class LazyCollection(Generic[_LazyObjectT], LazyEntity):
    __slots__ = ("_elements",)

    def __init__(
        self,
        *elements: _LazyObjectT
    ) -> None:
        super().__init__()
        self._elements: list[_LazyObjectT] = []
        self.add(*elements)

    def __iter__(self) -> Iterator[_LazyObjectT]:
        return self._elements.__iter__()

    def __len__(self) -> int:
        return self._elements.__len__()

    @overload
    def __getitem__(
        self,
        index: int
    ) -> _LazyObjectT:
        ...

    @overload
    def __getitem__(
        self,
        index: slice
    ) -> list[_LazyObjectT]:
        ...

    def __getitem__(
        self,
        index: int | slice
    ) -> _LazyObjectT | list[_LazyObjectT]:
        #if isinstance(index, int):
        #    return self._reference_node._children.__getitem__(index)._instance
        return self._elements.__getitem__(index)

    def add(
        self,
        *elements: _LazyObjectT
    ):
        assert not self._is_readonly()
        if not elements:
            return self
        for entity in self._iter_dependency_ancestors():
            assert isinstance(entity, LazyEntity)
            entity._expire_properties()
        #self._expire_properties()
        self._elements.extend(elements)
        self._bind_dependency_children(*elements)
        return self

    def remove(
        self,
        *elements: _LazyObjectT
    ):
        assert not self._is_readonly()
        if not elements:
            return self
        for entity in self._iter_dependency_ancestors():
            assert isinstance(entity, LazyEntity)
            entity._expire_properties()
        #self._expire_properties()
        for element in elements:
            self._elements.remove(element)
        self._unbind_dependency_children(*elements)
        return self


#class LazyParameter(Generic[_ParameterElementsT], LazyBase):
#    __slots__ = ("_wrapped_elements",)

#    def __init__(self) -> None:
#        super().__init__()
#        self._wrapped_elements: LazyWrapper[_ParameterElementsT] | None = None

#    def _get(self) -> "LazyWrapper[_ParameterElementsT] | None":
#        return self._wrapped_elements

#    def _set(
#        self,
#        wrapped_elements: "LazyWrapper[_ParameterElementsT] | None"
#    ) -> None:
#        self._wrapped_elements = wrapped_elements

    #def _bind_entities(self, *entities: LazyEntity):
    #    self._bind_parameter_children(*entities)
    #    return self

    #def _bind_properties(self, *properties: "LazyProperty"):
    #    self._bind_parameter_children(*properties)
    #    return self


class LazyProperty(Generic[_LazyEntityT], LazyBase):
    #__slots__ = ()
    __slots__ = ("_entity",)

    def __init__(self) -> None:
        super().__init__()
        self._entity: _LazyEntityT | None = None

    def _get(self) -> _LazyEntityT | None:
        return self._entity
        #try:
        #    return next(self._iter_reference_children())
        #except StopIteration:
        #    return None

    def _set(
        self,
        entity: _LazyEntityT | None
    ) -> None:
        self._entity = entity
        #old_entity = self._entity
        #if old_entity is entity:
        #    return
        #self._entity = entity
        #if old_entity is not None:
        #    self._unbind_dependency_children(old_entity)
        #if entity is not None:
        #    self._bind_dependency_children(entity)

    #def _bind_parameters(self, *parameters: LazyParameter):
    #    self._bind_parameter_children(*parameters)
    #    return self


#class LazyDescriptor(Generic[_LazyInstanceT, _ObjT]):
#    __slots__ = (
#        "name",
#        "values_dict"
#    )

#    def __init__(self, name: str) -> None:
#        self.name: str = name
#        self.values_dict: dict[_ObjT, _LazyInstanceT] = {}

#    @overload
#    def __get__(
#        self,
#        obj: None,
#        owner: type[_ObjT] | None = None
#    ): ...

#    @overload
#    def __get__(
#        self,
#        obj: _ObjT,
#        owner: type[_ObjT] | None = None
#    ) -> _LazyInstanceT: ...

#    def __get__(
#        self,
#        obj: _ObjT | None,
#        owner: type[_ObjT] | None = None
#    )  | _LazyInstanceT:
#        if obj is None:
#            return self
#        if (value := self.get(obj)) is None:
#            value = self.missing(obj)
#            self.values_dict[obj] = value
#        return value

#    def __set__(
#        self,
#        obj: _ObjT,
#        value: _LazyInstanceT
#    ) -> None:
#        self.values_dict[obj] = value

#    def initialize(
#        self,
#        obj: _ObjT,
#        value: _LazyInstanceT
#    ) -> None:
#        assert obj not in self.values_dict
#        self.values_dict[obj] = value

#    def pop(
#        self,
#        obj: _ObjT
#    ) -> _LazyInstanceT:
#        return self.values_dict.pop(obj)

#    def get(
#        self,
#        obj: _ObjT
#    ) -> _LazyInstanceT | None:
#        return self.values_dict.get(obj)

#    def missing(
#        self,
#        obj: _ObjT
#    ) -> _LazyInstanceT:
#        raise KeyError


class LazyDescriptor(Generic[_InstanceT, _LazyEntityT], ABC):
    #__slots__ = (
    #    "object_type",
    #    "is_collection"
    #)

    #def __init__(
    #    self,
    #    object_type: type[LazyObject],
    #    is_collection: bool
    #):
    #    super().__init__()
    #    self.object_type: type[LazyObject] = object_type
    #    self.is_collection: bool = is_collection

    @overload
    def __get__(
        self: _LazyDescriptorT,
        instance: None,
        owner: type[_InstanceT] | None = None
    ) -> _LazyDescriptorT: ...  # TODO: typing

    @overload
    def __get__(
        self,
        instance: _InstanceT,
        owner: type[_InstanceT] | None = None
    ) -> _LazyEntityT: ...

    def __get__(
        self: _LazyDescriptorT,
        instance: _InstanceT | None,
        owner: type[_InstanceT] | None = None
    ) -> _LazyDescriptorT | _LazyEntityT:
        if instance is None:
            return self
        return self.instance_get(instance)

    @abstractmethod
    def instance_get(
        self,
        instance: _InstanceT
    ) -> _LazyEntityT:
        pass

    @abstractmethod
    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        pass

    @abstractmethod
    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        pass

    @abstractmethod
    def restock(
        self,
        instance: _InstanceT
    ) -> None:
        pass


class LazyObjectDescriptor(LazyDescriptor[_InstanceT, _LazyObjectT]):
    __slots__ = (
        #"object_type",
        "method",
        "instance_to_object_dict",
        "_default_object"
    )

    def __init__(
        self,
        #object_type: type[_LazyObjectT],
        method: Callable[[type[_InstanceT]], _LazyObjectT]
    ) -> None:
        super().__init__()
        #super().__init__(method.__name__)
        #self.object_type: type = NotImplemented
        self.method: Callable[[type[_InstanceT]], _LazyObjectT] = method
        self.instance_to_object_dict: dict[_InstanceT, _LazyObjectT] = {}
        self._default_object: _LazyObjectT | None = None

    #@overload
    #def __get__(
    #    self,
    #    instance: None,
    #    owner: type[_InstanceT] | None = None
    #) -> "LazyObjectDescriptor[_InstanceT, _LazyObjectT]": ...  # TODO: typing

    #@overload
    #def __get__(
    #    self,
    #    instance: _InstanceT,
    #    owner: type[_InstanceT] | None = None
    #) -> _LazyObjectT: ...

    #def __get__(
    #    self,
    #    instance: _InstanceT | None,
    #    owner: type[_InstanceT] | None = None
    #) -> "LazyObjectDescriptor[_InstanceT, _LazyObjectT] | _LazyObjectT":
    #    if instance is None:
    #        return self
    #    return self.instance_to_object_dict[instance]

    def __set__(
        self,
        instance: _InstanceT,
        new_object: _LazyObjectT
    ) -> None:
        assert not instance._is_readonly()
        if (old_object := self.instance_to_object_dict[instance]) is not NotImplemented:
            if old_object is new_object:
                return
            #old_object._expire_properties()
            #instance._expire_properties(old_object)
            for entity in old_object._iter_dependency_descendants():
                assert isinstance(entity, LazyEntity)
                entity._expire_properties()
            instance._unbind_dependency_children(old_object)
        for entity in instance._iter_dependency_ancestors():
            assert isinstance(entity, LazyEntity)
            entity._expire_properties()
        self.instance_to_object_dict[instance] = new_object
        if new_object is not NotImplemented:
            instance._bind_dependency_children(new_object)

    def instance_get(
        self,
        instance: _InstanceT
    ) -> _LazyObjectT:
        return self.get_object(instance)

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        if (default_object := self._default_object) is None:
            default_object = self.method(type(instance))
            #if default_object is NotImplemented:  # TODO
            #    default_object = LazyObjectNotImplemented()
            #if default_object is NotImplemented:
            #    default_object = LazyWrapper(NotImplemented)
            #if default_object is not NotImplemented:
            #    default_object._restock_callbacks = None  # Never restock
            self._default_object = default_object
        self.instance_to_object_dict[instance] = default_object
        if default_object is not NotImplemented:
            instance._bind_dependency_children(default_object)

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        self.initialize(dst)
        if self.instance_to_object_dict[dst] is not NotImplemented:
            dst._unbind_dependency_children(self.instance_to_object_dict[dst])
        if self.instance_to_object_dict[src] is not NotImplemented:
            dst._bind_dependency_children(self.instance_to_object_dict[src])
        self.instance_to_object_dict[dst] = self.instance_to_object_dict[src]

    def restock(
        self,
        instance: _InstanceT
    ) -> None:
        self.instance_to_object_dict.pop(instance)

    def get_object(
        self,
        instance: _InstanceT
    ) -> _LazyObjectT:
        return self.instance_to_object_dict[instance]


class LazyCollectionDescriptor(Generic[_InstanceT, _LazyObjectT], LazyDescriptor[_InstanceT, LazyCollection[_LazyObjectT]]):
    __slots__ = (
        #"object_type",
        "method",
        "instance_to_collection_dict"
    )

    def __init__(
        self,
        #object_type: type[_LazyObjectT],
        method: Callable[[type[_InstanceT]], LazyCollection[_LazyObjectT]]
    ) -> None:
        super().__init__()
        #self.is_collection = True
        #super().__init__(method.__name__)
        #self.object_type: type = NotImplemented
        self.method: Callable[[type[_InstanceT]], LazyCollection[_LazyObjectT]] = method
        self.instance_to_collection_dict: dict[_InstanceT, LazyCollection[_LazyObjectT]] = {}

    #@overload
    #def __get__(
    #    self,
    #    instance: None,
    #    owner: type[_InstanceT] | None = None
    #) -> "LazyCollectionDescriptor[_InstanceT, _LazyEntityT]": ...

    #@overload
    #def __get__(
    #    self,
    #    instance: _InstanceT,
    #    owner: type[_InstanceT] | None = None
    #) -> LazyCollection[_LazyEntityT]: ...

    #def __get__(
    #    self,
    #    instance: _InstanceT | None,
    #    owner: type[_InstanceT] | None = None
    #) -> "LazyCollectionDescriptor[_InstanceT, _LazyEntityT] | LazyCollection[_LazyEntityT]":
    #    if instance is None:
    #        return self
    #    return self.instance_to_collection_dict[instance]

    def __set__(
        self,
        instance: _InstanceT,
        new_collection: LazyCollection[_LazyObjectT]
    ) -> None:
        assert not instance._is_readonly()
        for entity in instance._iter_dependency_ancestors():
            assert isinstance(entity, LazyEntity)
            entity._expire_properties()
        self.instance_to_collection_dict[instance] = new_collection
        instance._bind_dependency_children(new_collection)

    def instance_get(
        self,
        instance: _InstanceT
    ) -> LazyCollection[_LazyObjectT]:
        return self.get_collection(instance)

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        default_object = self.method(type(instance))
        self.instance_to_collection_dict[instance] = default_object
        instance._bind_dependency_children(default_object)

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        self.initialize(dst)
        self.instance_to_collection_dict[dst].add(*self.instance_to_collection_dict[src])

    def restock(
        self,
        instance: _InstanceT
    ) -> None:
        self.instance_to_collection_dict.pop(instance)

    def get_collection(
        self,
        instance: _InstanceT
    ) -> LazyCollection[_LazyObjectT]:
        return self.instance_to_collection_dict[instance]

    #def __set__(
    #    self,
    #    instance: _InstanceT,
    #    value: Never
    #) -> None:
    #    raise RuntimeError("Attempting to set a collection object directly")


#class LazyParameterDescriptor(Generic[_InstanceT, _ParameterElementsT]):
#    __slots__ = (
#        "descriptor_chain",
#        "is_lazy_value",
#        "instance_to_parameter_dict"
#    )

#    def __init__(
#        self,
#        descriptor_chain: """tuple[Union[
#            LazyObjectDescriptor[_InstanceT, LazyObject],
#            LazyCollectionDescriptor[_InstanceT, LazyEntity],
#            LazyPropertyDescriptor[_InstanceT, _ParameterSpec, LazyEntity]
#        ], ...]"""
#    ) -> None:
#        self.descriptor_chain: tuple[Union[
#            LazyObjectDescriptor[_InstanceT, LazyObject],
#            LazyCollectionDescriptor[_InstanceT, LazyEntity],
#            LazyPropertyDescriptor[_InstanceT, _ParameterSpec, LazyEntity]
#        ], ...] = descriptor_chain
#        #self.is_lazy_value: bool = NotImplemented
#        #self.get_parameter_from_instance: Callable[[_InstanceT], _ParameterElementsT] = NotImplemented
#        self.instance_to_parameter_dict: dict[_InstanceT, LazyParameter[LazyWrapper[_ParameterElementsT]]] = {}

#    @overload
#    def __get__(
#        self,
#        instance: None,
#        owner: type[_InstanceT] | None = None
#    ) -> "LazyParameterDescriptor[_InstanceT, _ParameterElementsT]": ...

#    @overload
#    def __get__(
#        self,
#        instance: _InstanceT,
#        owner: type[_InstanceT] | None = None
#    ) -> "LazyWrapper[_ParameterElementsT]": ...

#    def __get__(
#        self,
#        instance: _InstanceT | None,
#        owner: type[_InstanceT] | None = None
#    ) -> "LazyParameterDescriptor[_InstanceT, _ParameterElementsT] | LazyWrapper[_ParameterElementsT]":
#        if instance is None:
#            return self

#        def apply_deepest(
#            callback: Callable[[Any], Any],
#            obj: Any
#        ) -> Any:
#            if not isinstance(obj, tuple):
#                return callback(obj)
#            return tuple(
#                apply_deepest(callback, child_obj)
#                for child_obj in obj
#            )

#        def yield_deepest(
#            obj: Any
#        ) -> Generator[Any, None, None]:
#            if not isinstance(obj, tuple):
#                yield obj
#            else:
#                for child_obj in obj:
#                    yield from yield_deepest(child_obj)

#        parameter = self.instance_to_parameter_dict[instance]
#        if (wrapped_elements := parameter._get()) is None:
#            elements = instance
#            binding_completed = False
#            for descriptor in self.descriptor_chain:
#                if isinstance(descriptor, LazyCollectionDescriptor):
#                    elements = apply_deepest(lambda obj: tuple(descriptor.__get__(obj)), elements)
#                else:
#                    elements = apply_deepest(lambda obj: descriptor.__get__(obj), elements)
#                    if isinstance(descriptor, LazyPropertyDescriptor):
#                        parameter._bind_parameter_children(*yield_deepest(elements))
#                        binding_completed = True
#            if not binding_completed:
#                parameter._bind_parameter_children(*yield_deepest(elements))
#            #if self.is_lazy_value:
#            #    elements = apply_deepest(lambda obj: obj.value, elements)
#            wrapped_elements = LazyWrapper(elements)
#            parameter._set(wrapped_elements)
#        return wrapped_elements

#    def initialize(
#        self,
#        instance: _InstanceT
#    ) -> None:
#        self.instance_to_parameter_dict[instance] = LazyParameter()

#    def copy_initialize(
#        self,
#        dst: _InstanceT,
#        src: _InstanceT
#    ) -> None:
#        self.initialize(dst)
#        self.instance_to_parameter_dict[dst]._set(self.instance_to_parameter_dict[src]._get())


class LazyPropertyDescriptor(LazyDescriptor[_InstanceT, _LazyEntityT]):
    __slots__ = (
        #"object_type",
        "method",
        "parameter_chains",
        #"parameter_descriptors",
        #"requires_unwrapping_tuple",
        #"parameter_items",
        "instance_to_property_dict",
        "parameters_to_entity_dict"
        #"entity_to_parameters_composition_dict"
    )

    def __init__(
        self,
        #object_type: type[LazyObject],
        #is_collection: bool,
        method: Callable[..., _LazyEntityT],
        parameter_chains: tuple[tuple[str, ...], ...]
    ) -> None:
        super().__init__()
        #self.object_type: type = NotImplemented
        self.method: Callable[..., _LazyEntityT] = method
        self.parameter_chains: tuple[tuple[str, ...], ...] = parameter_chains
        #self.parameter_items: dict[type, tuple[tuple[tuple[Union[
        #    LazyObjectDescriptor[LazyObject, _InstanceT],
        #    LazyCollectionDescriptor[LazyEntity, _InstanceT],
        #    LazyPropertyDescriptor[LazyEntity, _InstanceT]
        #], ...], bool], ...]] = {}
        #self.require_unwrapping: bool = NotImplemented
        #self.parameter_descriptors: dict[type, tuple[LazyParameterDescriptor, ...]] = {}
        #self.requires_unwrapping_tuple: tuple[bool, ...] = tuple(
        #    re.fullmatch(r"_\w+_", parameter_name) is None
        #    for parameter_name in tuple(inspect.signature(method).parameters)[1:]  # TODO
        #)
        #self.parameter_items: dict[type, tuple[tuple[tuple[LazyDescriptor, ...], bool], ...]] = {}
        #self.parameter_items: dict[type, tuple[tuple[LazyDescriptor, ...], ...]] = {}
        #self.get_entity_from_parameters: Callable[[tuple], _LazyEntityT] = NotImplemented
        #self.parameters: tuple[str, ...] = parameter_tuple
        self.instance_to_property_dict: dict[_InstanceT, LazyProperty[_LazyEntityT]] = {}
        self.parameters_to_entity_dict: dict[tuple, _LazyEntityT] = {}
        #self.entity_to_parameters_composition_dict: dict[_LazyEntityT, list[tuple]] = {}
        #self.property_to_parameters_dict: dict[_DAGNodeT, tuple] = {}
        #self.instance_to_property_record_dict: dict[_InstanceT, LazyPropertyRecord[_DAGNodeT]] = {}
        #self.instance_to_variable_tuple_dict: dict[_InstanceT, tuple[_LazyObjectT, ...]] = {}
        #self.variable_tuple_to_instances_dict: dict[tuple[_LazyObjectT, ...], list[_InstanceT]] = {}
        #self.variable_tuple_to_property_dict: dict[tuple[_LazyObjectT, ...], _LazyObjectT] = {}

    #@overload
    #def __get__(
    #    self,
    #    instance: None,
    #    owner: type[_InstanceT] | None = None
    #) -> "LazyPropertyDescriptor[_InstanceT, _ParameterSpec, _LazyEntityT]": ...

    #@overload
    #def __get__(
    #    self,
    #    instance: _InstanceT,
    #    owner: type[_InstanceT] | None = None
    #) -> _LazyEntityT: ...

    #def __get__(
    #    self,
    #    instance: _InstanceT | None,
    #    owner: type[_InstanceT] | None = None
    #) -> "LazyPropertyDescriptor[_InstanceT, _ParameterSpec, _LazyEntityT] | _LazyEntityT":
    #    if instance is None:
    #        return self

    def __set__(
        self,
        instance: _InstanceT,
        value: Any
    ) -> None:
        raise ValueError("Attempting to set a readonly property")

    def instance_get(
        self,
        instance: _InstanceT
    ) -> _LazyEntityT:
        # TODO
        #def cleanup_method(
        #    entity: _LazyEntityT
        #) -> None:
        #    #parameters = self.property_to_parameters_dict.pop(prop)
        #    self.entity_to_parameters_composition_dict.pop(entity)

        #def yield_deepest(
        #    parameter_obj: Any
        #) -> Generator[Any, None, None]:
        #    occurred: set[Any] = set()

        #    def yield_deepest_atom(
        #        obj: Any
        #    ) -> Generator[Any, None, None]:
        #        if obj in occurred:
        #            return
        #        if not isinstance(obj, tuple):
        #            yield obj
        #        else:
        #            for child_obj in obj:
        #                yield from yield_deepest_atom(child_obj)

        #    yield from yield_deepest_atom(parameter_obj)

        prop = self.get_property(instance)
        #print()

        #def get_parameter(
        #    descriptor_chain: tuple[Union[
        #        LazyObjectDescriptor[LazyObject, _InstanceT],
        #        LazyCollectionDescriptor[LazyEntity, _InstanceT],
        #        LazyPropertyDescriptor[LazyEntity, _InstanceT]
        #    ], ...]
        #) -> Any:
        #    parameter = instance
        #    requires_parameter_binding = True
        #    for descriptor in descriptor_chain:
        #        if requires_parameter_binding:
        #            print(list(prop._iter_parameter_children()))
        #            prop._bind_parameter_children(*yield_deepest(parameter))
        #        if isinstance(descriptor, LazyCollectionDescriptor):
        #            parameter = apply_deepest(lambda obj: tuple(descriptor.__get__(obj)), parameter)
        #        else:
        #            parameter = apply_deepest(lambda obj: descriptor.__get__(obj), parameter)
        #            if isinstance(descriptor, LazyPropertyDescriptor):
        #                requires_parameter_binding = False
        #    #if require_unwrapping:
        #    #    parameter = apply_deepest(lambda obj: obj.value, parameter)
        #    return parameter

        def construct_parameter_item(
            obj: LazyObject,
            descriptor_name: str
        ) -> LazyObject | tuple[LazyObject, ...]:
            descriptor = type(obj)._LAZY_DESCRIPTORS[descriptor_name]
            if isinstance(descriptor, LazyCollectionDescriptor):
                return tuple(descriptor.__get__(obj))
            if isinstance(descriptor, LazyPropertyDescriptor):
                prop._bind_parameter_children(descriptor.get_property(obj))
            return descriptor.__get__(obj)

        if (entity := prop._get()) is None:
            #parameter_items = self.parameter_items[type(instance)]
            parameter_list = []
            for descriptor_name_chain in self.parameter_chains:
                parameter = instance
                #binding_completed = False
                for descriptor_name in descriptor_name_chain:
                    parameter = self.apply_deepest(lambda obj: construct_parameter_item(obj, descriptor_name), parameter)
                    #descriptor = type(instance)._LAZY_DESCRIPTORS[descriptor_name]
                    #if isinstance(descriptor, LazyPropertyDescriptor):
                    #    prop._bind_parameter_children(*self.yield_deepest(self.apply_deepest(lambda obj: descriptor.get_property(obj), parameter)))
                    #    #binding_completed = True
                    #if isinstance(descriptor, LazyCollectionDescriptor):
                    #    parameter = self.apply_deepest(lambda obj: tuple(descriptor.__get__(obj)), parameter)
                    #else:
                    #    parameter = self.apply_deepest(lambda obj: descriptor.__get__(obj), parameter)
                #if not binding_completed:
                    #print(list(self.yield_deepest(parameter)))
                prop._bind_parameter_children(*self.yield_deepest(parameter))
                #if requires_unwrapping:
                #    parameter = apply_deepest(lambda obj: obj.value, parameter)
                parameter_list.append(parameter)
                #wrapped_elements = LazyWrapper(elements)
                #prop._set(wrapped_elements)


            #parameters = tuple(
            #    parameter_descriptor.__get__(instance)
            #    for parameter_descriptor in self.parameter_descriptors[type(instance)]
            #)
            #prop._bind_parameter_children(*parameters)
            #parameter_items = self.parameter_items[type(instance)]
            #parameter_list: list[Any] = []
            #for descriptor_chain, _ in parameter_items:
            #    parameter = instance
            #    requires_parameter_binding = True
            #    print(descriptor_chain)
            #    for descriptor in descriptor_chain:
            #        if requires_parameter_binding:
            #            print(list(prop._iter_parameter_children()))
            #            print(list(yield_deepest(parameter)))
            #            prop._bind_parameter_children(*yield_deepest(parameter))
            #        if isinstance(descriptor, LazyCollectionDescriptor):
            #            parameter = apply_deepest(lambda obj: tuple(descriptor.__get__(obj)), parameter)
            #        else:
            #            parameter = apply_deepest(lambda obj: descriptor.__get__(obj), parameter)
            #            if isinstance(descriptor, LazyPropertyDescriptor):
            #                requires_parameter_binding = False
            #    parameter_list.append(parameter)
            ##descriptor_chain_tuple = tuple(
            ##    descriptor_chain for descriptor_chain, _ in parameter_items
            ##)
            parameters = tuple(parameter_list)
            if (entity := self.parameters_to_entity_dict.get(parameters)) is None:
                #entity = self.method(type(instance), *(
                #    parameter if not requires_unwrapping else apply_deepest(lambda obj: obj.value, parameter)
                #    for parameter, (_, requires_unwrapping) in zip(parameters, parameter_items, strict=True)
                #))
                entity = self.method(type(instance), *parameters)
                self.parameters_to_entity_dict[parameters] = entity
                #self.entity_to_parameters_composition_dict.setdefault(entity, []).append(parameters)
                #entity._at_restock(cleanup_method)  # TODO
            #entity = self.get_entity_from_parameters()
            prop._bind_dependency_children(entity)
            prop._set(entity)
        return entity

        #def flatten_deepest(obj: tuple | LazyProperty[_LazyEntityT]) -> Generator[LazyProperty[_LazyEntityT], None, None]:
        #    if not isinstance(obj, tuple):
        #        yield obj
        #    else:
        #        for child_obj in obj:
        #            yield from flatten_deepest(child_obj)


        #prop = self.instance_to_property_dict[instance]
        #if (entity := prop._get()) is None:
        #    parameters = self.get_parameters_from_instance(instance)
        #    if (entity := self.parameters_to_entity_bidict.get(parameters)) is None:
        #        entity = self.get_entity_from_parameters(parameters)
        #        entity._bind_children(*flatten_deepest(parameters))
        #        self.parameters_to_entity_bidict[parameters] = entity
        #        #self.property_to_parameters_dict[prop] = parameters
        #        entity._at_restock(restock_method)

        #if (prop := self.values_dict.get(instance)) is None:
        #    parameters = self.get_parameters_from_instance(instance)
        #    if (prop := self.parameters_property_bidict.get(parameters)) is None:
        #        prop = self.get_property_from_parameters(parameters)
        #        prop._bind_children(*flatten_deepest(parameters))
        #        self.parameters_property_bidict[parameters] = prop
        #        #self.property_to_parameters_dict[prop] = parameters
        #        prop._at_restock(restock_method)
        #    self.values_dict[instance] = prop

        #return self.values_dict[instance]._get()

    #    record = super().__get__(instance)
    #    #record = self.instance_to_property_record_dict[instance]
    #    if (prop := record._slot) is None:
    #        parameters = self.get_parameters_from_instance(instance)
    #        if (prop := self.parameters_property_bidict.get(parameters)) is None:
    #            record.bind(*flatten_deepest(parameters))
    #            prop = self.get_property_from_parameters(parameters)
    #            self.parameters_property_bidict[parameters] = prop
    #            #self.property_to_parameters_dict[prop] = parameters
    #            prop._at_restock(restock_method)

    #        record._slot = prop
    #        #record._expired = False
    #    #if (prop := self.instance_to_property_record_dict.get(instance)) is None:
    #    #    prop = self.instance_method(instance)
    #    #    self.instance_to_property_dict[instance] = prop
    #    
    #    return prop
    #    #if (variable_tuple := self.instance_to_variable_tuple_dict.get(instance)) is None:
    #    #    variable_tuple = tuple(
    #    #        variable_descr.__get__(instance)
    #    #        for variable_descr in instance.__class__._PROPERTY_DESCR_TO_VARIABLE_DESCRS[self]
    #    #    )
    #    #    self.instance_to_variable_tuple_dict[instance] = variable_tuple
    #    #self.variable_tuple_to_instances_dict.setdefault(variable_tuple, []).append(instance)
    #    #if (result := self.variable_tuple_to_property_dict.get(variable_tuple)) is None:
    #    #    result = self.method(*(
    #    #        param_descr.__get__(instance)
    #    #        for param_descr in instance.__class__._PROPERTY_DESCR_TO_PARAMETER_DESCRS[self]
    #    #    ))
    #    #    self.variable_tuple_to_property_dict[variable_tuple] = result
    #    #return result

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        #prop = LazyProperty()
        #prop._bind_parameter_children(*(
        #    parameter_descriptor.__get__(instance)
        #    for parameter_descriptor in self.parameter_descriptors[type(instance)]
        #))
        self.instance_to_property_dict[instance] = LazyProperty()

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        self.initialize(dst)
        self.instance_to_property_dict[dst]._set(self.instance_to_property_dict[src]._get())

    def restock(
        self,
        instance: _InstanceT
    ) -> None:
        self.instance_to_property_dict.pop(instance)

    def get_property(
        self,
        instance: _InstanceT
    ) -> LazyProperty[_LazyEntityT]:
        return self.instance_to_property_dict[instance]

    @classmethod
    def apply_deepest(
        cls,
        callback: Callable[[Any], Any],
        obj: Any
    ) -> Any:
        if not isinstance(obj, tuple):
            return callback(obj)
        return tuple(
            cls.apply_deepest(callback, child_obj)
            for child_obj in obj
        )

    @classmethod
    def yield_deepest(
        cls,
        parameter_obj: Any
    ) -> Generator[Any, None, None]:
        occurred: set[Any] = set()

        def yield_deepest_atom(
            obj: Any
        ) -> Generator[Any, None, None]:
            if not isinstance(obj, tuple):
                if obj in occurred:
                    return
                yield obj
                occurred.add(obj)
            else:
                for child_obj in obj:
                    yield from yield_deepest_atom(child_obj)

        yield from yield_deepest_atom(parameter_obj)

    #def handle_new_property(
    #    self,
    #    entity: _LazyEntityT
    #) -> _LazyEntityT:
    #    return entity

    #def missing(
    #    self,
    #    instance: _InstanceT
    #) -> _DAGNodeT:
    #    def flatten_deepest(obj: tuple | DAGNode) -> Generator[DAGNode, None, None]:
    #        if not isinstance(obj, tuple):
    #            yield obj
    #        else:
    #            for child_obj in obj:
    #                yield from flatten_deepest(child_obj)

    #    def restock_method(prop: _DAGNodeT) -> None:
    #        #parameters = self.property_to_parameters_dict.pop(prop)
    #        self.parameters_property_bidict.inverse.pop(prop)

    #    parameters = self.get_parameters_from_instance(instance)
    #    if (prop := self.parameters_property_bidict.get(parameters)) is None:
    #        prop = self.get_property_from_parameters(parameters)
    #        prop._bind_children(*flatten_deepest(parameters))
    #        self.parameters_property_bidict[parameters] = prop
    #        #self.property_to_parameters_dict[prop] = parameters
    #        prop._at_restock(restock_method)
    #    return prop

    #def __set__(
    #    self,
    #    instance: _InstanceT,
    #    value: Never
    #) -> None:
    #    raise RuntimeError("Attempting to set a readonly lazy property")

    #def _setup_callables(self, descrs: dict[str, LazyDescriptor]) -> None:
    #    #parameter_names = list(inspect.signature(self.method).parameters)
    #    parameter_items: list[tuple[tuple[LazyDescriptor, ...], bool]] = [
    #        (tuple(
    #            descrs[name] for name in re.findall(r"_\w+?_(?=_|$)", name)
    #        ), is_lazy_value)
    #        for name, is_lazy_value in (
    #            (name, False) if re.fullmatch(r"_\w+_", name) else (f"_{name}_", True)
    #            for name in inspect.signature(self.method).parameters
    #        )
    #    ]

    #    def apply_deepest(
    #        callback: Callable[[DAGNode], DAGNode],
    #        obj: tuple | DAGNode
    #    ) -> tuple | DAGNode:
    #        if not isinstance(obj, tuple):
    #            return callback(obj)
    #        return tuple(
    #            apply_deepest(callback, child_obj)
    #            for child_obj in obj
    #        )

    #    def construct_obj_from_descr_chain(
    #        descr_chain: tuple[LazyDescriptor, ...],
    #        obj: tuple | DAGNode
    #    ) -> tuple | DAGNode:
    #        for descr in descr_chain:
    #            if isinstance(descr, LazyObjectDescriptor | LazyPropertyDescriptor):
    #                obj = apply_deepest(lambda instance: descr.__get__(instance), obj)
    #            elif isinstance(descr, LazyCollectionDescriptor):
    #                obj = apply_deepest(lambda instance: tuple(descr.__get__(instance)._node_children), obj)
    #            else:
    #                raise TypeError
    #        return obj
    #        #if not descr_chain:
    #        #    return obj
    #        #descr = descr_chain[0]
    #        #rest_chain = descr_chain[1:]
    #        #if isinstance(descr, LazyObjectDescriptor | LazyPropertyDescriptor):
    #        #    return construct_obj_from_descr_chain(
    #        #        rest_chain,
    #        #        apply_deepest(lambda instance: descr.__get__(instance), obj)
    #        #    )
    #        #if isinstance(descr, LazyCollectionDescriptor):
    #        #    return construct_obj_from_descr_chain(
    #        #        rest_chain,
    #        #        apply_deepest(lambda instance: tuple(descr.__get__(instance)._node_children), obj)
    #        #    )
    #        #raise TypeError

    #    def get_parameters_from_instance(instance: _InstanceT) -> tuple:
    #        return tuple(
    #            construct_obj_from_descr_chain(
    #                descr_chain, instance
    #            )
    #            for descr_chain, _ in parameter_items
    #        )

    #    def get_property_from_parameters(parameters: tuple) -> _DAGNodeT:
    #        return self.method(*(
    #            apply_deepest(lambda lazy_value: lazy_value.value, parameter) if is_lazy_value else parameter
    #            for parameter, (_, is_lazy_value) in zip(parameters, parameter_items, strict=True)
    #        ))

    #    self.get_parameters_from_instance = get_parameters_from_instance
    #    self.get_property_from_parameters = get_property_from_parameters


#class LazyObjectNotImplemented(LazyObject):
#    pass
