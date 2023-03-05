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
    "LazyCollectionPropertyDescriptor",
    "LazyCollectionVariableDescriptor",
    "LazyObject",
    "LazyObjectPropertyDescriptor",
    "LazyObjectVariableDescriptor"
]


from abc import (
    ABC,
    abstractmethod
)
import copy
import re
from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    Generic,
    Iterator,
    TypeVar,
    Union,
    overload
)

from ..lazy.dag import DAGNode


_LazyEntityT = TypeVar("_LazyEntityT", bound="LazyEntity")
_LazyObjectT = TypeVar("_LazyObjectT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_ElementT = TypeVar("_ElementT", bound="LazyObject")
_LazyDescriptorT = TypeVar("_LazyDescriptorT", bound="LazyDescriptor")


class LazyDependencyNode(DAGNode):
    __slots__ = ("_ref",)

    def __init__(
        self,
        instance: "LazyBase"
    ) -> None:
        super().__init__()
        self._ref: LazyBase = instance


class LazyBase(ABC):
    __slots__ = ("_dependency_node",)

    #_dependency_node: LazyDependencyNode
    #_parameter_node: LazyNode

    #def __init_subclass__(cls) -> None:
    #    super().__init_subclass__()

    def __init__(self) -> None:
        super().__init__()
        self._dependency_node: LazyDependencyNode = LazyDependencyNode(self)

    #def __new__(
    #    cls,
    #    *args,
    #    **kwargs
    #):
    #    instance = super().__new__(cls)
    #    instance._dependency_node = LazyDependencyNode(instance)
    #    return instance

    #def _init_dependency_node(self):
    #    self._dependency_node = LazyDependencyNode(self)
    #    #self._parameter_node = LazyNode(self)

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

    def _bind_dependency(
        self,
        instance: "LazyBase"
    ) -> None:
        self._dependency_node._bind(instance._dependency_node)

    def _unbind_dependency(
        self,
        instance: "LazyBase"
    ) -> None:
        self._dependency_node._unbind(instance._dependency_node)

    def _clear_dependency(self) -> None:
        self._dependency_node._clear()

    #def _iter_parameter_children(self) -> "Generator[LazyBase, None, None]":
    #    for child in self._parameter_node._children:
    #        yield child._ref

    #def _iter_parameter_parents(self) -> "Generator[LazyBase, None, None]":
    #    for parent in self._parameter_node._parents:
    #        yield parent._ref

    #def _iter_parameter_descendants(self) -> "Generator[LazyBase, None, None]":
    #    for descendant in self._parameter_node._iter_descendants():
    #        yield descendant._ref

    #def _iter_parameter_ancestors(self) -> "Generator[LazyBase, None, None]":
    #    for ancestor in self._parameter_node._iter_ancestors():
    #        yield ancestor._ref

    #def _bind_parameter(
    #    self,
    #    instance: "LazyBase"
    #) -> None:
    #    self._parameter_node._bind(instance._parameter_node)

    #def _unbind_parameter(
    #    self,
    #    instance: "LazyBase"
    #) -> None:
    #    self._parameter_node._unbind(instance._parameter_node)

    #def _clear_parameter(self) -> None:
    #    self._parameter_node._clear()


class LazyEntity(LazyBase):
    __slots__ = ("_linked_properties",)

    def __init__(self) -> None:
        super().__init__()
        self._linked_properties: list[LazyProperty] = []

    def _is_readonly(self) -> bool:
        return any(
            isinstance(instance, LazyProperty)
            for instance in self._iter_dependency_ancestors()
        )

    def _expire_properties(self) -> None:
        #expired_properties = [
        #    expired_property
        #    for expired_property in self._iter_parameter_ancestors()
        #    if isinstance(expired_property, LazyProperty)
        #]
        for expired_property in self._linked_properties:
            if (entity := expired_property._get()) is None:
                continue
            expired_property._clear_dependency()
            expired_property._clear_parameter_children()
            expired_property._set(None)
            entity._restock_descendants_if_no_dependency_parents()
        #self._linked_properties.clear()

    def _restock_descendants_if_no_dependency_parents(self) -> None:
        if self._iter_dependency_parents():
            return
        for entity in self._iter_dependency_descendants():
            assert isinstance(entity, LazyEntity)
            if isinstance(entity, LazyObject):
                entity._restock()


class LazyObject(LazyEntity):
    __slots__ = ()

    _VACANT_INSTANCES: "ClassVar[list[LazyObject]]"
    _LAZY_DESCRIPTORS: "ClassVar[tuple[LazyDescriptor, ...]]"
    _LAZY_DESCRIPTOR_OVERLOADINGS: "ClassVar[dict[str, LazyDescriptorOverloading]]"
    #_LAZY_OBJECT_DESCRIPTOR_GROUPS: "ClassVar[dict[str, LazyObjectDescriptorOverloading]]"
    #_LAZY_COLLECTION_DESCRIPTOR_GROUPS: "ClassVar[dict[str, LazyCollectionDescriptorOverloading]]"
    _ALL_SLOTS: "ClassVar[tuple[str, ...]]"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        base_cls = cls.__base__
        assert issubclass(base_cls, LazyObject)
        #descriptors: list[LazyDescriptor] = []
        if base_cls is not LazyObject:
            descriptor_overloadings = base_cls._LAZY_DESCRIPTOR_OVERLOADINGS.copy()
        else:
            descriptor_overloadings = {}
        #collection_descriptor_overloadings = base_cls._LAZY_COLLECTION_DESCRIPTOR_GROUPS.copy()
        for name, attr in cls.__dict__.items():
            if not isinstance(attr, LazyDescriptor):
                continue
            assert re.fullmatch(r"_\w+_", name)
            if attr.element_type is NotImplemented:
                attr.element_type = cls
            if (descriptor_overloading := descriptor_overloadings.get(name)) is not None:
                if isinstance(descriptor_overloading, LazyObjectDescriptorOverloading):
                    assert isinstance(attr, LazyObjectVariableDescriptor | LazyObjectPropertyDescriptor)
                    descriptor_overloading.set_descriptor(cls, attr)
                elif isinstance(descriptor_overloading, LazyCollectionDescriptorOverloading):
                    assert isinstance(attr, LazyCollectionVariableDescriptor | LazyCollectionPropertyDescriptor)
                    descriptor_overloading.set_descriptor(cls, attr)
                else:
                    raise TypeError
            else:
                if isinstance(attr, LazyObjectVariableDescriptor | LazyObjectPropertyDescriptor):
                    descriptor_overloading = LazyObjectDescriptorOverloading(cls, attr)
                elif isinstance(attr, LazyCollectionVariableDescriptor | LazyCollectionPropertyDescriptor):
                    descriptor_overloading = LazyCollectionDescriptorOverloading(cls, attr)
                else:
                    raise TypeError
                descriptor_overloadings[name] = descriptor_overloading

        #if base_cls is not LazyObject:
        for descriptor_overloading in descriptor_overloadings.values():
            if descriptor_overloading.type_implemented(cls):
                continue
            descriptor_overloading.set_descriptor(cls, descriptor_overloading.get_descriptor(base_cls))

        cls._VACANT_INSTANCES = []
        cls._LAZY_DESCRIPTORS = tuple(
            descriptor_overloading.get_descriptor(cls)
            for descriptor_overloading in descriptor_overloadings.values()
        )
        cls._LAZY_DESCRIPTOR_OVERLOADINGS = descriptor_overloadings
        cls._ALL_SLOTS = tuple(set(
            slot
            for parent_cls in reversed(cls.__mro__)
            if issubclass(parent_cls, LazyEntity)
            for slot in parent_cls.__slots__
        ))

        for attr in cls.__dict__.values():
            if not isinstance(attr, LazyPropertyDescriptor):
                continue
            #descriptor_overloading_chains = tuple(
            #    tuple(
            #        descriptor_overloadings[parameter_name]
            #        for parameter_name in parameter_name_chain
            #    )
            #    for parameter_name_chain in attr.parameter_name_chains
            #)
            descriptor_overloading_chains: list[tuple[LazyDescriptorOverloading, ...]] = []
            for parameter_name_chain in attr.parameter_name_chains:
                element_type = cls
                descriptor_overloading_chain: list[LazyDescriptorOverloading] = []
                for parameter_name in parameter_name_chain:
                    descriptor_overloading = element_type._LAZY_DESCRIPTOR_OVERLOADINGS[parameter_name]
                    element_type = descriptor_overloading.element_type
                    descriptor_overloading_chain.append(descriptor_overloading)
                descriptor_overloading_chains.append(tuple(descriptor_overloading_chain))
            attr.descriptor_overloading_chains = tuple(descriptor_overloading_chains)
            #attr.parameter_depths = tuple(
            #    sum(
            #        isinstance(descriptor_overloading, LazyCollectionDescriptorOverloading)
            #        for descriptor_overloading in descriptor_overloading_chain
            #    )
            #    for descriptor_overloading_chain in descriptor_overloading_chains
            #)


        #descriptors = {
        #    name: attr
        #    for parent_cls in reversed(cls.__mro__)
        #    for name, attr in parent_cls.__dict__.items()
        #    if isinstance(attr, LazyDescriptor)
        #}
        #assert all(
        #    re.fullmatch(r"_\w+_", name)
        #    for name in descriptors
        #)

        #for name, descriptor in descriptors.items():
        #    if name not in cls.__dict__:
        #        continue
        #    #assert isinstance(descriptor, LazyObjectDescriptor | LazyObjectCollectionDescriptor | LazyPropertyDescriptor)
        #    assert re.fullmatch(r"_\w+_", name)

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
        #instance._dependency_node = LazyDependencyNode(instance)
        return instance

    def __init__(self) -> None:
        super().__init__()
        cls = self.__class__
        for descriptor in cls._LAZY_DESCRIPTORS:
            descriptor.initialize(self)

    #def __del__(self) -> None:
    #    self._restock()

    def _copy(self: _LazyObjectT) -> _LazyObjectT:
        cls = self.__class__
        result = cls.__new__(cls)
        result._dependency_node = LazyDependencyNode(result)
        #result._init_dependency_node()
        for slot_name in cls._ALL_SLOTS:
            result.__setattr__(slot_name, copy.copy(self.__getattribute__(slot_name)))
        for descriptor in cls._LAZY_DESCRIPTORS:
            descriptor.copy_initialize(result, self)
        return result

    def _restock(self) -> None:
        # TODO: check refcnt
        # TODO: Never restock the default object
        cls = self.__class__
        for descriptor in cls._LAZY_DESCRIPTORS:
            descriptor.restock(self)
        cls._VACANT_INSTANCES.append(self)


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
        return self._elements.__getitem__(index)

    def _copy(self) -> "LazyCollection[_LazyObjectT]":
        return LazyCollection(*self._elements)

    def add(
        self,
        *elements: _LazyObjectT
    ):
        assert not self._is_readonly()
        if not elements:
            return self
        #for entity in self._iter_dependency_ancestors():
        #    assert isinstance(entity, LazyEntity)
        #    entity._expire_properties()
        self._expire_properties()
        for element in elements:
            if element in self._elements:
                continue
            self._elements.append(element)
            self._bind_dependency(element)
        return self

    def remove(
        self,
        *elements: _LazyObjectT
    ):
        assert not self._is_readonly()
        if not elements:
            return self
        #for entity in self._iter_dependency_ancestors():
        #    assert isinstance(entity, LazyEntity)
        #    entity._expire_properties()
        self._expire_properties()
        for element in elements:
            if element not in self._elements:
                continue
            self._elements.remove(element)
            self._unbind_dependency(element)
            element._restock_descendants_if_no_dependency_parents()
        return self


class LazyProperty(Generic[_LazyEntityT], LazyBase):
    __slots__ = (
        "_entity",
        "_parameter_children"
    )

    def __init__(self) -> None:
        super().__init__()
        self._entity: _LazyEntityT | None = None
        self._parameter_children: set[LazyEntity] = set()

    def _get(self) -> _LazyEntityT | None:
        return self._entity

    def _set(
        self,
        entity: _LazyEntityT | None
    ) -> None:
        self._entity = entity

    def _get_parameter_children(self) -> set[LazyEntity]:
        return self._parameter_children

    def _bind_parameter_children(
        self,
        parameter_children: set[LazyEntity]
    ) -> None:
        for parameter_child in parameter_children:
            parameter_child._linked_properties.append(self)
        self._parameter_children.update(parameter_children)

    def _clear_parameter_children(self) -> None:
        for parameter_child in self._parameter_children:
            parameter_child._linked_properties.remove(self)
        self._parameter_children.clear()


class LazyDescriptor(Generic[_InstanceT, _LazyEntityT, _ElementT], ABC):
    __slots__ = ("element_type",)

    def __init__(
        self,
        element_type: type[_ElementT]
    ) -> None:
        super().__init__()
        self.element_type: type[_ElementT] = element_type

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


class LazyVariableDescriptor(LazyDescriptor[_InstanceT, _LazyEntityT, _ElementT]):
    __slots__ = (
        "method",
        "instance_to_entity_dict"
    )

    def __init__(
        self,
        element_type: type[_ElementT],
        method: Callable[[type[_InstanceT]], _LazyEntityT]
    ) -> None:
        super().__init__(
            element_type=element_type
        )
        self.method: Callable[[type[_InstanceT]], _LazyEntityT] = method
        self.instance_to_entity_dict: dict[_InstanceT, _LazyEntityT] = {}

    def __set__(
        self,
        instance: _InstanceT,
        new_entity: _LazyEntityT
    ) -> None:
        assert not instance._is_readonly()
        #for entity in instance._iter_dependency_ancestors():
        #    assert isinstance(entity, LazyEntity)
        #    entity._expire_properties()
        #self.instance_to_entity_dict[instance] = new_entity
        #instance._bind_dependency(new_entity)
        if (old_entity := self.get_entity(instance)) is not NotImplemented:
            if old_entity is new_entity:
                return
            old_entity._expire_properties()
            instance._unbind_dependency(old_entity)
            old_entity._restock_descendants_if_no_dependency_parents()
        self.set_entity(instance, new_entity)
        if new_entity is not NotImplemented:
            instance._bind_dependency(new_entity)

    def instance_get(
        self,
        instance: _InstanceT
    ) -> _LazyEntityT:
        return self.get_entity(instance)

    def restock(
        self,
        instance: _InstanceT
    ) -> None:
        self.instance_to_entity_dict.pop(instance)

    def get_entity(
        self,
        instance: _InstanceT
    ) -> _LazyEntityT:
        return self.instance_to_entity_dict[instance]

    def set_entity(
        self,
        instance: _InstanceT,
        entity: _LazyEntityT
    ) -> None:
        self.instance_to_entity_dict[instance] = entity


class LazyObjectVariableDescriptor(LazyVariableDescriptor[_InstanceT, _LazyObjectT, _LazyObjectT]):
    __slots__ = ("_default_object",)

    def __init__(
        self,
        element_type: type[_LazyObjectT],
        method: Callable[[type[_InstanceT]], _LazyObjectT]
    ) -> None:
        super().__init__(
            element_type=element_type,
            method=method
        )
        self._default_object: _LazyObjectT | None = None

    #def __set__(
    #    self,
    #    instance: _InstanceT,
    #    new_object: _LazyObjectT
    #) -> None:
    #    assert not instance._is_readonly()
    #    if (old_object := self.instance_to_object_dict[instance]) is not NotImplemented:
    #        if old_object is new_object:
    #            return
    #        #for entity in old_object._iter_dependency_descendants():
    #        #    assert isinstance(entity, LazyEntity)
    #        #    entity._expire_properties()
    #        old_object._expire_properties()
    #        instance._unbind_dependency(old_object)
    #        old_object._restock_descendants_if_no_dependency_parents()
    #    #instance._expire_properties()
    #    #for entity in instance._iter_dependency_ancestors():
    #    #    assert isinstance(entity, LazyEntity)
    #    #    entity._expire_properties()
    #    self.instance_to_object_dict[instance] = new_object
    #    if new_object is not NotImplemented:
    #        instance._bind_dependency(new_object)

    #def instance_get(
    #    self,
    #    instance: _InstanceT
    #) -> _LazyObjectT:
    #    return self.get_object(instance)

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        if (default_object := self._default_object) is None:
            default_object = self.method(type(instance))
            self._default_object = default_object
        self.set_entity(instance, default_object)
        if default_object is not NotImplemented:
            instance._bind_dependency(default_object)

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        self.initialize(dst)
        if (dst_object := self.get_entity(dst)) is not NotImplemented:
            dst._unbind_dependency(dst_object)
            dst_object._restock_descendants_if_no_dependency_parents()
        if (src_object := self.get_entity(src)) is not NotImplemented:
            dst._bind_dependency(src_object)
        self.set_entity(dst, src_object)

    #def restock(
    #    self,
    #    instance: _InstanceT
    #) -> None:
    #    self.instance_to_object_dict.pop(instance)

    #def get_object(
    #    self,
    #    instance: _InstanceT
    #) -> _LazyObjectT:
    #    return self.instance_to_object_dict[instance]


class LazyCollectionVariableDescriptor(LazyVariableDescriptor[_InstanceT, LazyCollection[_LazyObjectT], _LazyObjectT]):
    __slots__ = ()

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        default_object = self.method(type(instance))
        self.set_entity(instance, default_object)
        instance._bind_dependency(default_object)

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        self.initialize(dst)
        self.get_entity(dst).add(*self.get_entity(src))


class LazyPropertyDescriptor(LazyDescriptor[_InstanceT, _LazyEntityT, _ElementT]):
    __slots__ = (
        "method",
        "parameter_name_chains",
        "parameter_preapplied_methods",
        "descriptor_overloading_chains",
        "instance_to_property_dict",
        "parameters_to_entity_dict"
    )

    def __init__(
        self,
        element_type: type[_ElementT],
        method: Callable[..., _LazyEntityT],
        parameter_name_chains: tuple[tuple[str, ...], ...],
        parameter_preapplied_methods: tuple[Callable[[Any], Any] | None, ...]
    ) -> None:
        super().__init__(
            element_type=element_type
        )
        self.method: Callable[..., _LazyEntityT] = method
        self.parameter_name_chains: tuple[tuple[str, ...], ...] = parameter_name_chains
        self.parameter_preapplied_methods: tuple[Callable[[Any], Any] | None, ...] = parameter_preapplied_methods
        self.descriptor_overloading_chains: tuple[tuple[LazyDescriptorOverloading, ...], ...] = NotImplemented
        self.instance_to_property_dict: dict[_InstanceT, LazyProperty[_LazyEntityT]] = {}
        self.parameters_to_entity_dict: dict[tuple, _LazyEntityT] = {}

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
        prop = self.get_property(instance)
        if (entity := prop._get()) is None:
            parameter_items = tuple(
                self.construct_parameter_item(descriptor_overloading_chain, instance)
                for descriptor_overloading_chain in self.descriptor_overloading_chains
            )
            prop._bind_parameter_children(set(
                parameter_child
                for _, _, parameter_children in parameter_items
                for parameter_child in parameter_children
            ))
            parameters = tuple(
                parameter
                for parameter, _, _ in parameter_items
            )
            if (entity := self.parameters_to_entity_dict.get(parameters)) is None:
                entity = self.method(type(instance), *(
                    parameter
                    if preapplied_method is None
                    else self.apply_at_depth(preapplied_method, parameter, depth)
                    for (parameter, depth, _), preapplied_method in zip(
                        parameter_items, self.parameter_preapplied_methods, strict=True
                    )
                ))
                self.parameters_to_entity_dict[parameters] = entity
            prop._bind_dependency(entity)
            prop._set(entity)
        return entity

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        self.instance_to_property_dict[instance] = LazyProperty()

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        self.initialize(dst)
        #self.instance_to_property_dict[dst]._set(
        #    self.instance_to_property_dict[src]._get()
        #)
        #self.instance_to_property_dict[dst]._bind_parameter_children(
        #    self.instance_to_property_dict[src]._get_parameter_children()
        #)

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
    def construct_parameter_item(
        cls,
        descriptor_overloading_chain: "tuple[LazyDescriptorOverloading, ...]",
        instance: _InstanceT
    ) -> tuple[Any, int, set[LazyEntity]]:
        parameter_children: set[LazyEntity] = set()

        def construct_from_collection(
            item: tuple[LazyObject, bool],
            descriptor_overloading: LazyCollectionDescriptorOverloading
        ) -> tuple[tuple[LazyObject, bool], ...]:
            obj, binding_completed = item
            descriptor = descriptor_overloading.get_descriptor(type(obj))
            collection = descriptor.instance_get(obj)
            if not binding_completed:
                parameter_children.add(collection)
                if isinstance(descriptor, LazyCollectionPropertyDescriptor):
                    parameter_children.update(descriptor.get_property(obj)._get_parameter_children())
                    binding_completed = True
                else:
                    parameter_children.update(collection)
            return tuple(
                (element, binding_completed)
                for element in collection
            )

        def construct_from_collection_tail(
            item: tuple[LazyObject, bool],
            descriptor_overloading: LazyCollectionDescriptorOverloading
        ) -> tuple[LazyObject, ...]:
            obj, binding_completed = item
            descriptor = descriptor_overloading.get_descriptor(type(obj))
            collection = descriptor.instance_get(obj)
            if not binding_completed:
                parameter_children.add(collection)
                if isinstance(descriptor, LazyCollectionPropertyDescriptor):
                    parameter_children.update(descriptor.get_property(obj)._get_parameter_children())
                else:
                    parameter_children.update(*(
                        element._iter_dependency_descendants()
                        for element in collection
                    ))
            return tuple(collection)

        def construct_from_object(
            item: tuple[LazyObject, bool],
            descriptor_overloading: LazyObjectDescriptorOverloading
        ) -> tuple[LazyObject, bool]:
            obj, binding_completed = item
            descriptor = descriptor_overloading.get_descriptor(type(obj))
            element = descriptor.instance_get(obj)
            if not binding_completed:
                if isinstance(descriptor, LazyObjectPropertyDescriptor):
                    parameter_children.update(descriptor.get_property(obj)._get_parameter_children())
                    binding_completed = True
                else:
                    parameter_children.add(element)
            return (element, binding_completed)

        def construct_from_object_tail(
            item: tuple[LazyObject, bool],
            descriptor_overloading: LazyObjectDescriptorOverloading
        ) -> LazyObject:
            obj, binding_completed = item
            descriptor = descriptor_overloading.get_descriptor(type(obj))
            element = descriptor.instance_get(obj)
            if not binding_completed:
                if isinstance(descriptor, LazyObjectPropertyDescriptor):
                    parameter_children.update(descriptor.get_property(obj)._get_parameter_children())
                else:
                    parameter_children.update(
                        element._iter_dependency_descendants()
                    )
            return element

        parameter_item = (instance, False)
        depth = 0
        for descriptor_overloading in descriptor_overloading_chain[:-1]:
            #is_chain_tail = index == len(descriptor_overloading_chain) - 1
            if isinstance(descriptor_overloading, LazyCollectionDescriptorOverloading):
                parameter_item = cls.apply_at_depth(
                    lambda item: construct_from_collection(
                        item, descriptor_overloading
                    ),
                    parameter_item,
                    depth
                )
                depth += 1
            elif isinstance(descriptor_overloading, LazyObjectDescriptorOverloading):
                parameter_item = cls.apply_at_depth(
                    lambda item: construct_from_object(
                        item, descriptor_overloading
                    ),
                    parameter_item,
                    depth
                )
            else:
                raise TypeError
        descriptor_overloading = descriptor_overloading_chain[-1]
        if isinstance(descriptor_overloading, LazyCollectionDescriptorOverloading):
            parameter = cls.apply_at_depth(
                lambda item: construct_from_collection_tail(
                    item, descriptor_overloading
                ),
                parameter_item,
                depth
            )
            depth += 1
        elif isinstance(descriptor_overloading, LazyObjectDescriptorOverloading):
            parameter = cls.apply_at_depth(
                lambda item: construct_from_object_tail(
                    item, descriptor_overloading
                ),
                parameter_item,
                depth
            )
        else:
            raise TypeError
        return parameter, depth, parameter_children

    @classmethod
    def apply_at_depth(
        cls,
        callback: Callable[[Any], Any],
        obj: Any,
        depth: int
    ) -> Any:
        if not depth:
            return callback(obj)
        return tuple(
            cls.apply_at_depth(callback, child_obj, depth - 1)
            for child_obj in obj
        )


class LazyObjectPropertyDescriptor(LazyPropertyDescriptor[_InstanceT, _LazyObjectT, _LazyObjectT]):
    __slots__ = ()


class LazyCollectionPropertyDescriptor(LazyPropertyDescriptor[_InstanceT, LazyCollection[_LazyObjectT], _LazyObjectT]):
    __slots__ = ()


class LazyDescriptorOverloading(Generic[_InstanceT, _LazyObjectT, _LazyDescriptorT], ABC):
    __slots__ = (
        "element_type",
        "descriptors"
    )

    def __init__(
        self,
        instance_type: type[_InstanceT],
        descriptor: _LazyDescriptorT
    ) -> None:
        super().__init__()
        self.element_type: type[_LazyObjectT] = descriptor.element_type
        self.descriptors: dict[type[_InstanceT], _LazyDescriptorT] = {instance_type: descriptor}

    def type_implemented(
        self,
        instance_type: type[_InstanceT]
    ) -> bool:
        return instance_type in self.descriptors

    def get_descriptor(
        self,
        instance_type: type[_InstanceT]
    ) -> _LazyDescriptorT:
        return self.descriptors[instance_type]

    def set_descriptor(
        self,
        instance_type: type[_InstanceT],
        descriptor: _LazyDescriptorT
    ) -> None:
        assert issubclass(descriptor.element_type, self.element_type)
        self.descriptors[instance_type] = descriptor


class LazyObjectDescriptorOverloading(LazyDescriptorOverloading[_InstanceT, _LazyObjectT, Union[
    LazyObjectVariableDescriptor[_InstanceT, _LazyObjectT],
    LazyObjectPropertyDescriptor[_InstanceT, _LazyObjectT]
]]):
    __slots__ = ()


class LazyCollectionDescriptorOverloading(LazyDescriptorOverloading[_InstanceT, _LazyObjectT, Union[
    LazyCollectionVariableDescriptor[_InstanceT, _LazyObjectT],
    LazyCollectionPropertyDescriptor[_InstanceT, _LazyObjectT]
]]):
    __slots__ = ()
