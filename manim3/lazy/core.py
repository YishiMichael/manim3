"""
This module implements the functionality of lazy evaluation.

Every child class of `LazyData` shall define `__slots__`, and all methods
shall basically be sorted in the following way:
- magic methods
- lazy variables
- lazy properties
- private class methods
- private methods
- public methods

All methods decorated by any decorator provided by `Lazy` should be class
methods and be named with underscores appeared on both sides, i.e. `_data_`.
Successive underscores shall not occur, due to the name convension handled by
lazy properties.

## Lazy Variable

Lazy variables are what users can modify freely from the outer scope. They are
dependent variables of lazy properties. Once modified, the related lazy
properties will be expired, and will be recomputed when fetching.

Methods decorated by `Lazy.variable` should not take any argument except for
`cls` and return the *initial* value for this data. `NotImplemented` may be
an alternative for the value returned, as long as one ensures the data is
initialized when computation relating to it is performed.

+------------+----------------+----------------+--------------------+
| LazyMode   | method return  | __get__ return | __set__ type       |
+------------+----------------+----------------+--------------------+
| OBJECT     | LazyObject     | LazyObject     | LazyObject         |
| COLLECTION | LazyCollection | LazyCollection | LazyCollection     |
| UNWRAPPED  | T              | LazyWrapper[T] | T | LazyWrapper[T] |
| SHARED     | T (Hashable)   | LazyWrapper[T] | T                  |
+------------+----------------+----------------+--------------------+

The `__get__` method always returns an instance of `LazyEntity`, which is the
base class of both `LazyObject` and `LazyCollection`. `LazyCollection`, as its
name suggests, is just a dynamic collection of `LazyObject`s, and provides
`add`, `remove` as its public interface. `LazyWrapper` is derived from
`LazyObject`, which is just responsible for bringing a value into the lazy
scope, and the value is obtained via the readonly `value` property. One may
picture a lazy object as a tree (it's a DAG really), where `LazyWrapper`s sit
on all leaves.

Lazy variables are of course mutable. All can be mutated via `__set__` method.
Among all cases above, a common value will be shared among instances, except
for providing `T` type in `UNWRAPPED` mode, in which case a new `LazyWrapper`
object will be instanced and assigned specially to the instance. Additionally,
`LazyCollection`s can be mutated via `add` and `remove`.

The `LazyObject._copy` method will make all its children `LazyObject`s shared,
and construct new `LazyCollection` holding the same references, just like a
shallow copy.

## Lazy Property

Lazy properties depend on lazy variables and therefore cannot be modified.
Their values are computed only when expired, otherwise the cached value is
directly returned for usage.

Methods decorated by `Lazy.property` defines how lazy properties are related
to their dependent variables.

+------------+----------------+----------------+
| LazyMode   | method return  | __get__ return |
+------------+----------------+----------------+
| OBJECT     | LazyObject     | LazyObject     |
| COLLECTION | LazyCollection | LazyCollection |
| UNWRAPPED  | T              | LazyWrapper[T] |
| SHARED     | T (Hashable)   | LazyWrapper[T] |
+------------+----------------+----------------+

The return type of `__get__` is basically the same as that of lazy variables.
Values will also be shared if the leaf nodes of objects of parameters (which
forms a complicated structure of `LazyWrapper`s) match completely.

The parameters can be lazy variables, or other lazy properties (as long as
cyclic dependency doesn't exist), or a mixed collection of those, as long as
the types are consistent. The name of a parameter needs to indicate how it is
constructed through some specific patterns. Below are some examples.

Suppose `_o_` is a descriptor returning `LazyObject` when calling `__get__`,
and `_w_`, `_c_` for `LazyWrapper`, `LazyCollection`, respectively.
+--------------+--------------------------------------------------------+
| param name   | what is fed into the param                             |
+--------------+--------------------------------------------------------+
| _o_          | inst._o_                                               |
| w            | inst._w_.value                                         |
| _o__o_       | inst._o_._o_                                           |
| o__w         | inst._o_._w_.value                                     |
| _c_          | [e for e in inst._c_]                                  |
| _c__o_       | [e._o_ for e in inst._c_]                              |
| c__w         | [e._w_.value for e in inst._c_]                        |
| _c__c__o_    | [[ee._o_ for ee in e._c_] for e in inst._c_]           |
| _c__o__c__o_ | [[ee._o_ for ee in e._o_._c_] for e in inst._c_]       |
| c__o__c__w   | [[ee._w_.value for ee in e._o_._c_] for e in inst._c_] |
+--------------+--------------------------------------------------------+

As a conclusion, if there are `n` collection descriptor in the name chain, the
parameter will be fed with an `n`-fold list. If the underscores on ends are
missing, it's assumed the last descriptor will return `LazyWrapper` and values
are pulled out from the wrappers.

Lazy properties are immutable. This also applies to children of a `LazyObject`
and elements of `LazyCollection`. That is, one cannot set the value of
`inst._w_` even when it's a lazy variable, given that `inst` itself is the
calculation result of some property.
"""


__all__ = [
    "LazyCollection",
    "LazyCollectionPropertyDescriptor",
    "LazyCollectionVariableDescriptor",
    "LazyObject",
    "LazyObjectPropertyDescriptor",
    "LazyObjectVariableDescriptor",
    "LazyWrapper"
]


from abc import (
    ABC,
    abstractmethod
)
import copy
import itertools as it
import re
from typing import (
    Any,
    Callable,
    ClassVar,
    Concatenate,
    Generator,
    Generic,
    Hashable,
    Iterator,
    ParamSpec,
    TypeVar,
    Union,
    final,
    overload
)

from ..lazy.dag import DAGNode


_T = TypeVar("_T")
_InputT = TypeVar("_InputT")
_TreeNodeContentT = TypeVar("_TreeNodeContentT", bound=Hashable)
_LazyDataT = TypeVar("_LazyDataT", bound="LazyData")
_LazyEntityT = TypeVar("_LazyEntityT", bound="LazyEntity")
_LazyObjectT = TypeVar("_LazyObjectT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_ElementT = TypeVar("_ElementT", bound="LazyObject")
_LazyDescriptorT = TypeVar("_LazyDescriptorT", bound="LazyDescriptor")
_PropertyParameters = ParamSpec("_PropertyParameters")


@final
class LazyDependencyNode(DAGNode):
    __slots__ = ("_ref",)

    def __init__(
        self,
        instance: "LazyData"
    ) -> None:
        super().__init__()
        self._ref: LazyData = instance


@final
class TreeNode(Generic[_TreeNodeContentT], ABC):
    __slots__ = (
        "_children",
        "_content"
    )

    def __init__(
        self,
        content: _TreeNodeContentT
    ) -> None:
        super().__init__()
        self._children: tuple[TreeNode[_TreeNodeContentT], ...] | None = None
        self._content: _TreeNodeContentT = content

    def _bind_only(
        self,
        *nodes: "TreeNode[_TreeNodeContentT]"
    ) -> None:
        self._children = nodes

    def _apply_deepest(
        self,
        callback: "Callable[[_TreeNodeContentT], TreeNode[_TreeNodeContentT]]"
    ) -> "TreeNode[_TreeNodeContentT]":
        if self._children is None:
            return callback(self._content)
        result = TreeNode(self._content)
        result._bind_only(*(
            child._apply_deepest(callback)
            for child in self._children
        ))
        return result

    def _construct_tuple(self) -> Hashable:
        if self._children is None:
            return self._content
        return tuple(
            child._construct_tuple()
            for child in self._children
        )

    def _construct_list(
        self,
        callback: Callable[[_TreeNodeContentT], Any] | None = None
    ) -> Any:
        if self._children is None:
            if callback is None:
                return self._content
            return callback(self._content)
        return [
            child._construct_list(callback=callback)
            for child in self._children
        ]

    def _flatten(self) -> list[_TreeNodeContentT]:
        if self._children is None:
            return [self._content]
        return list(it.chain(*(
            child._flatten()
            for child in self._children
        )))


class LazyData(ABC):
    __slots__ = ("_dependency_node",)

    def __init__(self) -> None:
        super().__init__()
        self._dependency_node: LazyDependencyNode = LazyDependencyNode(self)

    def _iter_dependency_children(self) -> "Generator[LazyData, None, None]":
        for child in self._dependency_node._children:
            yield child._ref

    def _iter_dependency_parents(self) -> "Generator[LazyData, None, None]":
        for parent in self._dependency_node._parents:
            yield parent._ref

    def _iter_dependency_descendants(self) -> "Generator[LazyData, None, None]":
        for descendant in self._dependency_node._iter_descendants():
            yield descendant._ref

    def _iter_dependency_ancestors(self) -> "Generator[LazyData, None, None]":
        for ancestor in self._dependency_node._iter_ancestors():
            yield ancestor._ref

    def _bind_dependency(
        self,
        instance: "LazyData"
    ) -> None:
        self._dependency_node._bind(instance._dependency_node)

    def _unbind_dependency(
        self,
        instance: "LazyData"
    ) -> None:
        self._dependency_node._unbind(instance._dependency_node)

    def _clear_dependency(self) -> None:
        self._dependency_node._clear()

    @abstractmethod
    def _clear_ref(self) -> None:
        pass


class LazyEntity(LazyData):
    __slots__ = (
        "_linked_properties",
        "_always_alive"
    )

    def __init__(self) -> None:
        super().__init__()
        self._linked_properties: list[LazyProperty] = []
        self._always_alive: bool = False

    def _get_linked_properties(self) -> "list[LazyProperty]":
        return self._linked_properties

    def _bind_linked_property(
        self,
        linked_property: "LazyProperty"
    ) -> None:
        self._linked_properties.append(linked_property)

    def _unbind_linked_property(
        self,
        linked_property: "LazyProperty"
    ) -> None:
        self._linked_properties.remove(linked_property)

    def _is_readonly(self) -> bool:
        return any(
            isinstance(data, LazyProperty)
            for data in self._iter_dependency_ancestors()
        )

    @classmethod
    def _expire_properties(
        cls,
        *expired_properties: "LazyProperty"
    ) -> None:
        expired_entities: list[LazyEntity] = []
        for expired_property in expired_properties:
            if (entity := expired_property._get()) is None:
                continue
            expired_property._unbind_dependency(entity)
            expired_property._set(None)
            expired_property._clear_linked_entities()
            expired_entities.append(entity)
        for entity in dict.fromkeys(expired_entities):
            entity._clear_descendants_ref_if_no_dependency_parents()

    def _clear_descendants_ref_if_no_dependency_parents(self) -> None:
        stack: list[LazyData] = [self]
        while stack:
            data = stack.pop(0)
            if list(data._iter_dependency_parents()):
                continue
            if isinstance(data, LazyEntity) and data._always_alive:
                continue
            stack.extend(
                child for child in data._iter_dependency_children()
                if child not in stack
            )
            data._clear_dependency()
            data._clear_ref()


class LazyObject(LazyEntity):
    __slots__ = ()

    _LAZY_VARIABLE_DESCRIPTORS: "ClassVar[tuple[LazyVariableDescriptor, ...]]"
    _LAZY_PROPERTY_DESCRIPTORS: "ClassVar[tuple[LazyPropertyDescriptor, ...]]"
    _LAZY_DESCRIPTOR_OVERLOADINGS: "ClassVar[dict[str, LazyDescriptorOverloading]]"
    _ALL_SLOTS: "ClassVar[tuple[str, ...]]"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        base_cls = cls.__base__
        assert issubclass(base_cls, LazyObject)
        if base_cls is not LazyObject:
            descriptor_overloadings = base_cls._LAZY_DESCRIPTOR_OVERLOADINGS.copy()
        else:
            descriptor_overloadings = {}

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

        for descriptor_overloading in descriptor_overloadings.values():
            if descriptor_overloading.type_implemented(cls):
                continue
            descriptor_overloading.set_descriptor(cls, descriptor_overloading.get_descriptor(base_cls))

        # Ensure property descriptors come after variable descriptors.
        # This is a requirement for the functionality of `copy_initialize`.
        descriptors = [
            descriptor_overloading.get_descriptor(cls)
            for descriptor_overloading in descriptor_overloadings.values()
        ]
        cls._LAZY_VARIABLE_DESCRIPTORS = tuple(
            descriptor for descriptor in descriptors
            if isinstance(descriptor, LazyVariableDescriptor)
        )
        cls._LAZY_PROPERTY_DESCRIPTORS = tuple(
            descriptor for descriptor in descriptors
            if isinstance(descriptor, LazyPropertyDescriptor)
        )
        #cls._LAZY_DESCRIPTORS = tuple(sorted(
        #    (
        #        descriptor_overloading.get_descriptor(cls)
        #        for descriptor_overloading in descriptor_overloadings.values()
        #    ),
        #    key=lambda descriptor: isinstance(descriptor, LazyPropertyDescriptor)
        #))
        cls._LAZY_DESCRIPTOR_OVERLOADINGS = descriptor_overloadings
        # Use dict.fromkeys to preserve order (by first occurrance)
        cls._ALL_SLOTS = tuple(dict.fromkeys(
            slot
            for parent_cls in reversed(cls.__mro__)
            if issubclass(parent_cls, LazyEntity)
            for slot in parent_cls.__slots__
        ))

        for attr in cls.__dict__.values():
            if not isinstance(attr, LazyPropertyDescriptor):
                continue
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

    def __init__(self) -> None:
        super().__init__()
        #self._always_alive: bool = False
        cls = self.__class__
        for descriptor in cls._LAZY_VARIABLE_DESCRIPTORS:
            descriptor.initialize(self)
        for descriptor in cls._LAZY_PROPERTY_DESCRIPTORS:
            descriptor.initialize(self)

    def _copy(self: _LazyObjectT) -> _LazyObjectT:
        cls = self.__class__
        result = cls.__new__(cls)
        result._dependency_node = LazyDependencyNode(result)
        for slot_name in cls._ALL_SLOTS:
            result.__setattr__(slot_name, copy.copy(self.__getattribute__(slot_name)))
        for descriptor in cls._LAZY_VARIABLE_DESCRIPTORS:
            descriptor.copy_initialize(result, self)
        for descriptor in cls._LAZY_PROPERTY_DESCRIPTORS:
            descriptor.copy_initialize(result, self)
        return result

    def _clear_ref(self) -> None:
        # TODO: check refcnt
        cls = self.__class__
        for descriptor in cls._LAZY_VARIABLE_DESCRIPTORS:
            descriptor.clear_ref(self)
        for descriptor in cls._LAZY_PROPERTY_DESCRIPTORS:
            descriptor.clear_ref(self)
        #import sys
        #print(sys.getrefcount(self), type(self))


@final
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
    ) -> _LazyObjectT: ...

    @overload
    def __getitem__(
        self,
        index: slice
    ) -> list[_LazyObjectT]: ...

    def __getitem__(
        self,
        index: int | slice
    ) -> _LazyObjectT | list[_LazyObjectT]:
        return self._elements.__getitem__(index)

    def _clear_ref(self) -> None:
        self._elements.clear()

    def add(
        self,
        *elements: _LazyObjectT
    ):
        assert not self._is_readonly()
        if not elements:
            return self
        self._expire_properties(*self._get_linked_properties())
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
        self._expire_properties(*self._get_linked_properties())
        for element in elements:
            if element not in self._elements:
                continue
            self._elements.remove(element)
            self._unbind_dependency(element)
            element._clear_descendants_ref_if_no_dependency_parents()
        return self


@final
class LazyProperty(Generic[_LazyEntityT], LazyData):
    __slots__ = (
        "_entity",
        "_linked_entities"
    )

    def __init__(self) -> None:
        super().__init__()
        self._entity: _LazyEntityT | None = None
        self._linked_entities: list[LazyEntity] = []

    def _get(self) -> _LazyEntityT | None:
        return self._entity

    def _set(
        self,
        entity: _LazyEntityT | None
    ) -> None:
        self._entity = entity

    def _get_linked_entities(self) -> list[LazyEntity]:
        return self._linked_entities

    def _bind_linked_entities(
        self,
        *linked_entities: LazyEntity
    ) -> None:
        for linked_entity in linked_entities:
            linked_entity._bind_linked_property(self)
        self._linked_entities.extend(linked_entities)

    def _clear_linked_entities(self) -> None:
        for linked_entity in self._linked_entities:
            linked_entity._unbind_linked_property(self)
        self._linked_entities.clear()

    def _clear_ref(self) -> None:
        self._set(None)
        self._clear_linked_entities()


class LazyDescriptor(Generic[_InstanceT, _LazyEntityT, _ElementT, _LazyDataT, _InputT], ABC):
    __slots__ = (
        "element_type",
        "instance_to_data_dict"
    )

    def __init__(
        self,
        element_type: type[_ElementT]
    ) -> None:
        super().__init__()
        self.element_type: type[_ElementT] = element_type
        self.instance_to_data_dict: dict[_InstanceT, _LazyDataT] = {}

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
        return self.get_impl(instance)

    def __set__(
        self,
        instance: _InstanceT,
        new_input: _InputT
    ) -> None:
        self.set_impl(instance, new_input)

    @abstractmethod
    def get_impl(
        self,
        instance: _InstanceT
    ) -> _LazyEntityT:
        pass

    @abstractmethod
    def set_impl(
        self,
        instance: _InstanceT,
        new_input: _InputT
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

    def clear_ref(
        self,
        instance: _InstanceT
    ) -> None:
        self.instance_to_data_dict.pop(instance)

    def get_description(
        self,
        instance: _InstanceT
    ) -> _LazyDataT:
        return self.instance_to_data_dict[instance]

    def set_description(
        self,
        instance: _InstanceT,
        lazy_data: _LazyDataT
    ) -> None:
        self.instance_to_data_dict[instance] = lazy_data


class LazyVariableDescriptor(LazyDescriptor[_InstanceT, _LazyEntityT, _ElementT, _LazyEntityT, _InputT]):
    __slots__ = ("method",)

    def __init__(
        self,
        element_type: type[_ElementT],
        method: Callable[[type[_InstanceT]], _LazyEntityT]
    ) -> None:
        super().__init__(
            element_type=element_type
        )
        self.method: Callable[[type[_InstanceT]], _LazyEntityT] = method

    def get_impl(
        self,
        instance: _InstanceT
    ) -> _LazyEntityT:
        return self.get_description(instance)

    def set_impl(
        self,
        instance: _InstanceT,
        new_input: _InputT
    ) -> None:
        assert not instance._is_readonly()
        new_entity = self.convert_input(new_input)
        old_entity = self.get_description(instance)
        if old_entity is new_entity:
            return
        if old_entity is not NotImplemented:
            old_entity._expire_properties(*(
                prop
                for prop in old_entity._get_linked_properties()
                if prop in instance._iter_dependency_children()
            ))
            instance._unbind_dependency(old_entity)
            old_entity._clear_descendants_ref_if_no_dependency_parents()
        if new_entity is not NotImplemented:
            instance._bind_dependency(new_entity)
        self.set_description(instance, new_entity)

    @abstractmethod
    def convert_input(
        self,
        new_input: _InputT
    ) -> _LazyEntityT:
        pass


class LazyObjectVariableDescriptor(LazyVariableDescriptor[_InstanceT, _LazyObjectT, _LazyObjectT, _InputT]):
    __slots__ = ("default_object",)

    def __init__(
        self,
        element_type: type[_LazyObjectT],
        method: Callable[[type[_InstanceT]], _LazyObjectT]
    ) -> None:
        self.default_object: _LazyObjectT | None = None
        super().__init__(
            element_type=element_type,
            method=method
        )

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        if (default_object := self.default_object) is None:
            default_object = self.method(type(instance))
            if default_object is not NotImplemented:
                for default_object_descendant in default_object._iter_dependency_descendants():
                    if isinstance(default_object_descendant, LazyEntity):
                        default_object_descendant._always_alive = True
            self.default_object = default_object
        #default_object = self.method(type(instance))
        if default_object is not NotImplemented:
            instance._bind_dependency(default_object)
        self.set_description(instance, default_object)

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        if (src_object := self.get_description(src)) is not NotImplemented:
            dst._bind_dependency(src_object)
        self.set_description(dst, src_object)


class LazyCollectionVariableDescriptor(LazyVariableDescriptor[_InstanceT, LazyCollection[_LazyObjectT], _LazyObjectT, _InputT]):
    __slots__ = ()

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        default_collection = self.method(type(instance))
        instance._bind_dependency(default_collection)
        self.set_description(instance, default_collection)

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        dst_collection = LazyCollection(*self.get_description(src))
        dst._bind_dependency(dst_collection)
        self.set_description(dst, dst_collection)


class LazyPropertyDescriptor(LazyDescriptor[_InstanceT, _LazyEntityT, _ElementT, LazyProperty[_LazyEntityT], Any]):
    __slots__ = (
        "method",
        "release_method",
        "parameter_name_chains",
        "requires_unwrapping_tuple",
        "descriptor_overloading_chains",
        "key_to_entity_dict",
        "instance_to_key_dict",
        "key_to_instances_dict"
    )

    def __init__(
        self,
        element_type: type[_ElementT],
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _LazyEntityT],
        parameter_name_chains: tuple[tuple[str, ...], ...],
        requires_unwrapping_tuple: tuple[bool, ...]
    ) -> None:
        super().__init__(
            element_type=element_type
        )
        self.method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _LazyEntityT] = method
        self.release_method: Callable[[type[_InstanceT], _LazyEntityT], None] | None = None
        self.parameter_name_chains: tuple[tuple[str, ...], ...] = parameter_name_chains
        self.requires_unwrapping_tuple: tuple[bool, ...] = requires_unwrapping_tuple
        self.descriptor_overloading_chains: tuple[tuple[LazyDescriptorOverloading, ...], ...] = NotImplemented
        self.key_to_entity_dict: dict[tuple[Hashable, ...], _LazyEntityT] = {}
        self.instance_to_key_dict: dict[_InstanceT, tuple[Hashable, ...]] = {}
        self.key_to_instances_dict: dict[tuple[Hashable, ...], list[_InstanceT]] = {}

    def get_impl(
        self,
        instance: _InstanceT
    ) -> _LazyEntityT:

        def expand_dependencies(
            obj: LazyObject
        ) -> TreeNode[LazyObject]:
            result = TreeNode(obj)
            if isinstance(obj, LazyWrapper):
                return result
            children_nodes: list[TreeNode[LazyObject]] = []
            for child in obj._iter_dependency_children():
                if isinstance(child, LazyObject):
                    children_nodes.append(expand_dependencies(child))
                elif isinstance(child, LazyCollection):
                    collection_node: TreeNode[LazyObject] = TreeNode(NotImplemented)
                    collection_node._bind_only(*(
                        expand_dependencies(element) for element in child
                    ))
                    children_nodes.append(collection_node)
            result._bind_only(*children_nodes)
            return result

        def value_getter(
            obj: LazyObject
        ) -> Any:
            assert isinstance(obj, LazyWrapper)
            return obj.value

        prop = self.get_description(instance)
        if (entity := prop._get()) is None:
            parameter_items = tuple(
                self.construct_parameter_item(descriptor_overloading_chain, instance)
                for descriptor_overloading_chain in self.descriptor_overloading_chains
            )
            prop._bind_linked_entities(*dict.fromkeys(it.chain(*(
                linked_entities
                for _, linked_entities in parameter_items
            ))))
            parameter_trees = tuple(
                parameter_tree
                for parameter_tree, _ in parameter_items
            )
            key = tuple(
                parameter_tree._apply_deepest(expand_dependencies)._construct_tuple()
                for parameter_tree in parameter_trees
            )
            self.instance_to_key_dict[instance] = key
            self.key_to_instances_dict.setdefault(key, []).append(instance)
            if (entity := self.key_to_entity_dict.get(key)) is None:
                parameters = tuple(
                    parameter_tree._construct_list(callback=value_getter if requires_unwrapping else None)
                    for parameter_tree, requires_unwrapping in zip(
                        parameter_trees, self.requires_unwrapping_tuple, strict=True
                    )
                )
                entity = self.method(type(instance), *parameters)
                self.key_to_entity_dict[key] = entity
            prop._bind_dependency(entity)
            prop._set(entity)
        return entity

    def set_impl(
        self,
        instance: _InstanceT,
        new_input: Any
    ) -> None:
        raise ValueError("Attempting to set a readonly property")

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        default_property = LazyProperty()
        instance._bind_dependency(default_property)
        self.set_description(instance, default_property)

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        dst_property = LazyProperty()
        src_property = self.get_description(src)

        src_entity = src_property._get()
        dst_property._set(src_entity)
        if src_entity is not None:
            dst_property._bind_dependency(src_entity)

            dst_collection_children = [
                entity for entity in dst._iter_dependency_children()
                if isinstance(entity, LazyCollection)
            ]
            src_collection_children = [
                entity for entity in src._iter_dependency_children()
                if isinstance(entity, LazyCollection)
            ]
            dst_property._bind_linked_entities(*(
                linked_entity if linked_entity not in src_collection_children
                else dst_collection_children[src_collection_children.index(linked_entity)]
                for linked_entity in src_property._get_linked_entities()
            ))

        dst._bind_dependency(dst_property)
        self.set_description(dst, dst_property)

    def clear_ref(
        self,
        instance: _InstanceT
    ) -> None:
        super().clear_ref(instance)
        if (key := self.instance_to_key_dict.pop(instance, None)) is None:
            return
        instances = self.key_to_instances_dict[key]
        instances.remove(instance)
        if instances:
            return
        self.key_to_instances_dict.pop(key)
        entity = self.key_to_entity_dict.pop(key)
        if (release_method := self.release_method) is not None:
            release_method(type(instance), entity)

    @classmethod
    def construct_parameter_item(
        cls,
        descriptor_overloading_chain: "tuple[LazyDescriptorOverloading, ...]",
        instance: _InstanceT
    ) -> tuple[TreeNode[LazyObject], list[LazyEntity]]:
        parameter_tree: TreeNode[LazyObject] = TreeNode(instance)
        linked_objects_tree: TreeNode[LazyObject] = TreeNode(instance)
        linked_collections: list[LazyCollection] = []
        linked_properties: list[LazyProperty] = []

        def construct_parameter(
            descriptor_overloading: LazyDescriptorOverloading
        ) -> Callable[[LazyObject], TreeNode[LazyObject]]:

            def callback(
                obj: LazyObject
            ) -> TreeNode[LazyObject]:
                if isinstance(descriptor_overloading, LazyObjectDescriptorOverloading):
                    descriptor = descriptor_overloading.get_descriptor(type(obj))
                    result: TreeNode[LazyObject] = TreeNode(descriptor.get_impl(obj))
                elif isinstance(descriptor_overloading, LazyCollectionDescriptorOverloading):
                    descriptor = descriptor_overloading.get_descriptor(type(obj))
                    result: TreeNode[LazyObject] = TreeNode(obj)
                    result._bind_only(*(TreeNode(element) for element in descriptor.get_impl(obj)))
                else:
                    raise TypeError
                return result

            return callback

        def construct_linked_objects(
            descriptor_overloading: LazyDescriptorOverloading
        ) -> Callable[[LazyObject], TreeNode[LazyObject]]:

            def callback(
                obj: LazyObject
            ) -> TreeNode[LazyObject]:
                result: TreeNode[LazyObject] = TreeNode(obj)
                if isinstance(descriptor_overloading, LazyObjectDescriptorOverloading):
                    descriptor = descriptor_overloading.get_descriptor(type(obj))
                    if isinstance(descriptor, LazyObjectVariableDescriptor):
                        result._bind_only(TreeNode(descriptor.get_impl(obj)))
                    else:
                        result._bind_only()
                        linked_properties.append(descriptor.get_description(obj))
                elif isinstance(descriptor_overloading, LazyCollectionDescriptorOverloading):
                    descriptor = descriptor_overloading.get_descriptor(type(obj))
                    if isinstance(descriptor, LazyCollectionVariableDescriptor):
                        result._bind_only(*(TreeNode(element) for element in descriptor.get_impl(obj)))
                        linked_collections.append(descriptor.get_description(obj))
                    else:
                        result._bind_only()
                        linked_properties.append(descriptor.get_description(obj))
                else:
                    raise TypeError
                return result

            return callback

        for descriptor_overloading in descriptor_overloading_chain:
            parameter_tree = parameter_tree._apply_deepest(construct_parameter(descriptor_overloading))
            linked_objects_tree = linked_objects_tree._apply_deepest(construct_linked_objects(descriptor_overloading))

        linked_entities: list[LazyEntity] = []
        linked_entities.extend(linked_objects_tree._flatten())
        linked_entities.extend(linked_collections)
        linked_entities.extend(it.chain(*(
            linked_property._get_linked_entities()
            for linked_property in linked_properties
        )))
        return parameter_tree, linked_entities


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


@final
class LazyObjectDescriptorOverloading(LazyDescriptorOverloading[_InstanceT, _LazyObjectT, Union[
    LazyObjectVariableDescriptor[_InstanceT, _LazyObjectT, _InputT],
    LazyObjectPropertyDescriptor[_InstanceT, _LazyObjectT]
]]):
    __slots__ = ()


@final
class LazyCollectionDescriptorOverloading(LazyDescriptorOverloading[_InstanceT, _LazyObjectT, Union[
    LazyCollectionVariableDescriptor[_InstanceT, _LazyObjectT, _InputT],
    LazyCollectionPropertyDescriptor[_InstanceT, _LazyObjectT]
]]):
    __slots__ = ()


@final
class LazyWrapper(Generic[_T], LazyObject):
    __slots__ = ("__value",)

    def __init__(
        self,
        value: _T
    ) -> None:
        super().__init__()
        self.__value: _T = value

    @property
    def value(self) -> _T:
        return self.__value
