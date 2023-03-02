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
    overload
)

from ..lazy.dag import DAGNode


_LazyEntityT = TypeVar("_LazyEntityT", bound="LazyEntity")
_LazyObjectT = TypeVar("_LazyObjectT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_LazyDescriptorT = TypeVar("_LazyDescriptorT", bound="LazyDescriptor")


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
    )

    _VACANT_INSTANCES: "ClassVar[list[LazyBase]]"
    _dependency_node: LazyNode
    _parameter_node: LazyNode

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._VACANT_INSTANCES = []

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

    def _bind_parameter(
        self,
        instance: "LazyBase"
    ) -> None:
        self._parameter_node._bind(instance._parameter_node)

    def _unbind_parameter(
        self,
        instance: "LazyBase"
    ) -> None:
        self._parameter_node._unbind(instance._parameter_node)

    def _clear_parameter(self) -> None:
        self._parameter_node._clear()


class LazyEntity(LazyBase):
    __slots__ = ()

    def _is_readonly(self) -> bool:
        return any(
            isinstance(instance, LazyProperty)
            for instance in self._iter_dependency_ancestors()
        )

    def _expire_properties(self) -> None:
        expired_properties = [
            expired_prop
            for expired_prop in self._iter_parameter_ancestors()
            if isinstance(expired_prop, LazyProperty)
        ]
        for expired_prop in expired_properties:
            if (entity := expired_prop._get()) is not None:
                expired_prop._clear_dependency()
                expired_prop._clear_parameter()
                entity._restock_descendants_if_no_dependency_parents()
            expired_prop._set(None)

    def _restock_descendants_if_no_dependency_parents(self):
        if self._iter_dependency_parents():
            return
        for obj in self._iter_dependency_descendants():
            if not isinstance(obj, LazyObject):
                continue
            obj._restock()


class LazyObject(LazyEntity):
    __slots__ = ()

    _LAZY_DESCRIPTORS: "ClassVar[dict[str, LazyDescriptor]]"
    _ALL_SLOTS: "ClassVar[tuple[str, ...]]"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        descriptors: dict[str, LazyDescriptor] = {
            name: attr
            for parent_cls in reversed(cls.__mro__)
            for name, attr in parent_cls.__dict__.items()
            if isinstance(attr, LazyDescriptor)
        }

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

    def __init__(self) -> None:
        super().__init__()
        cls = self.__class__
        for descriptor in cls._LAZY_DESCRIPTORS.values():
            descriptor.initialize(self)

    #def __del__(self) -> None:
    #    self._restock()

    def _copy(self: _LazyObjectT) -> _LazyObjectT:
        cls = self.__class__
        result = cls.__new__(cls)
        result._init_nodes()
        for descriptor in cls._LAZY_DESCRIPTORS.values():
            descriptor.copy_initialize(result, self)
        for slot_name in cls._ALL_SLOTS:
            result.__setattr__(slot_name, copy.copy(self.__getattribute__(slot_name)))
        return result

    def _restock(self) -> None:
        # TODO: check refcnt
        # TODO: Never restock the default object
        for descriptor in self.__class__._LAZY_DESCRIPTORS.values():
            descriptor.restock(self)
        self.__class__._VACANT_INSTANCES.append(self)


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
        for entity in self._iter_dependency_ancestors():
            assert isinstance(entity, LazyEntity)
            entity._expire_properties()
        #self._expire_properties()
        for element in elements:
            if element not in self._elements:
                continue
            self._elements.remove(element)
            self._unbind_dependency(element)
            element._restock_descendants_if_no_dependency_parents()
        return self


class LazyProperty(Generic[_LazyEntityT], LazyBase):
    __slots__ = ("_entity",)

    def __init__(self) -> None:
        super().__init__()
        self._entity: _LazyEntityT | None = None

    def _get(self) -> _LazyEntityT | None:
        return self._entity

    def _set(
        self,
        entity: _LazyEntityT | None
    ) -> None:
        self._entity = entity


class LazyDescriptor(Generic[_InstanceT, _LazyEntityT], ABC):
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
        "method",
        "instance_to_object_dict",
        "_default_object"
    )

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], _LazyObjectT]
    ) -> None:
        super().__init__()
        self.method: Callable[[type[_InstanceT]], _LazyObjectT] = method
        self.instance_to_object_dict: dict[_InstanceT, _LazyObjectT] = {}
        self._default_object: _LazyObjectT | None = None

    def __set__(
        self,
        instance: _InstanceT,
        new_object: _LazyObjectT
    ) -> None:
        assert not instance._is_readonly()
        if (old_object := self.instance_to_object_dict[instance]) is not NotImplemented:
            if old_object is new_object:
                return
            for entity in old_object._iter_dependency_descendants():
                assert isinstance(entity, LazyEntity)
                entity._expire_properties()
            instance._unbind_dependency(old_object)
            old_object._restock_descendants_if_no_dependency_parents()
        for entity in instance._iter_dependency_ancestors():
            assert isinstance(entity, LazyEntity)
            entity._expire_properties()
        self.instance_to_object_dict[instance] = new_object
        if new_object is not NotImplemented:
            instance._bind_dependency(new_object)

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
            self._default_object = default_object
        self.instance_to_object_dict[instance] = default_object
        if default_object is not NotImplemented:
            instance._bind_dependency(default_object)

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        self.initialize(dst)
        if (dst_object := self.instance_to_object_dict[dst]) is not NotImplemented:
            dst._unbind_dependency(dst_object)
            dst_object._restock_descendants_if_no_dependency_parents()
        if (src_object := self.instance_to_object_dict[src]) is not NotImplemented:
            dst._bind_dependency(src_object)
        self.instance_to_object_dict[dst] = src_object

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
        "method",
        "instance_to_collection_dict"
    )

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], LazyCollection[_LazyObjectT]]
    ) -> None:
        super().__init__()
        self.method: Callable[[type[_InstanceT]], LazyCollection[_LazyObjectT]] = method
        self.instance_to_collection_dict: dict[_InstanceT, LazyCollection[_LazyObjectT]] = {}

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
        instance._bind_dependency(new_collection)

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
        instance._bind_dependency(default_object)

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


class LazyPropertyDescriptor(LazyDescriptor[_InstanceT, _LazyEntityT]):
    __slots__ = (
        "method",
        "parameter_chains",
        "instance_to_property_dict",
        "parameters_to_entity_dict"
    )

    def __init__(
        self,
        method: Callable[..., _LazyEntityT],
        parameter_chains: tuple[tuple[str, ...], ...]
    ) -> None:
        super().__init__()
        self.method: Callable[..., _LazyEntityT] = method
        self.parameter_chains: tuple[tuple[str, ...], ...] = parameter_chains
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

        def construct_parameter_item(
            obj: LazyObject,
            descriptor_name: str
        ) -> LazyObject | tuple[LazyObject, ...]:
            descriptor = type(obj)._LAZY_DESCRIPTORS[descriptor_name]
            if isinstance(descriptor, LazyCollectionDescriptor):
                return tuple(descriptor.__get__(obj))
            return descriptor.__get__(obj)

        prop = self.get_property(instance)
        if (entity := prop._get()) is None:
            parameter_list = []
            for descriptor_name_chain in self.parameter_chains:
                parameter = instance
                for descriptor_name in descriptor_name_chain:
                    parameter = self.apply_deepest(lambda obj: construct_parameter_item(obj, descriptor_name), parameter)
                parameter_list.append(parameter)

            parameters = tuple(parameter_list)
            for parameter_child in self.yield_deepest(parameters):
                prop._bind_parameter(parameter_child)
            if (entity := self.parameters_to_entity_dict.get(parameters)) is None:
                entity = self.method(type(instance), *parameters)
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
