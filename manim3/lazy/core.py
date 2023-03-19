# TODO: The documentation is outdated.

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
    "LazyDynamicPropertyDescriptor",
    "LazyDynamicVariableDescriptor",
    "LazyObject",
    "LazyUnitaryPropertyDescriptor",
    "LazyUnitaryVariableDescriptor",
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
    Iterable,
    Iterator,
    ParamSpec,
    TypeVar,
    Union,
    overload
)

#from ..lazy.dag import DAGNode


_T = TypeVar("_T")
_TreeNodeContentT = TypeVar("_TreeNodeContentT", bound=Hashable)
_ElementT = TypeVar("_ElementT", bound="LazyObject")
_ContainerT = TypeVar("_ContainerT", bound="LazyContainer")
_SlotT = TypeVar("_SlotT", bound="LazySlot")
#_VariableSlotT = TypeVar("_VariableSlotT", bound="LazyVariableSlot")
#_PropertySlotT = TypeVar("_PropertySlotT", bound="LazyPropertySlot")
#_LazyDataT = TypeVar("_LazyDataT", bound="LazyData")
#_LazyEntityT = TypeVar("_LazyEntityT", bound="LazyEntity")
_LazyObjectT = TypeVar("_LazyObjectT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
#_DescriptorGetT = TypeVar("_DescriptorGetT", bound="LazyEntity")
_DescriptorGetT = TypeVar("_DescriptorGetT")
_DescriptorSetT = TypeVar("_DescriptorSetT")
_LazyDescriptorT = TypeVar("_LazyDescriptorT", bound="LazyDescriptor")
_PropertyParameters = ParamSpec("_PropertyParameters")


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

    def bind_only(
        self,
        *nodes: "TreeNode[_TreeNodeContentT]"
    ) -> None:
        self._children = nodes

    def apply_deepest(
        self,
        callback: "Callable[[_TreeNodeContentT], TreeNode[_TreeNodeContentT]]"
    ) -> "TreeNode[_TreeNodeContentT]":
        if (children := self._children) is None:
            return callback(self._content)
        result = TreeNode(self._content)
        result.bind_only(*(
            child.apply_deepest(callback)
            for child in children
        ))
        return result

    def transform(
        self,
        callback: Callable[[_TreeNodeContentT], _T]
    ) -> "TreeNode[_T]":
        result = TreeNode(callback(self._content))
        if (children := self._children) is not None:
            result.bind_only(*(
                child.transform(callback)
                for child in children
            ))
        return result


#@final
#class LazyDependencyNode(DAGNode):
#    __slots__ = ("_ref",)
#
#    def __init__(
#        self,
#        instance: "LazySlot"
#    ) -> None:
#        super().__init__()
#        self._ref: LazySlot = instance


#class LazySlot(Generic[_ElementT], ABC):
#    __slots__ = ("_dependency_node",)

#    def __init__(self) -> None:
#        super().__init__()
#        self._dependency_node: LazyDependencyNode = LazyDependencyNode(self)

#    def _iter_dependency_children(self) -> "Generator[LazySlot, None, None]":
#        for child in self._dependency_node._children:
#            yield child._ref

#    def _iter_dependency_parents(self) -> "Generator[LazySlot, None, None]":
#        for parent in self._dependency_node._parents:
#            yield parent._ref

#    def _iter_dependency_descendants(self) -> "Generator[LazySlot, None, None]":
#        for descendant in self._dependency_node._iter_descendants():
#            yield descendant._ref

#    def _iter_dependency_ancestors(self) -> "Generator[LazySlot, None, None]":
#        for ancestor in self._dependency_node._iter_ancestors():
#            yield ancestor._ref

#    def _bind_dependency(
#        self,
#        instance: "LazySlot"
#    ) -> None:
#        self._dependency_node._bind(instance._dependency_node)

#    def _unbind_dependency(
#        self,
#        instance: "LazySlot"
#    ) -> None:
#        self._dependency_node._unbind(instance._dependency_node)

#    def _clear_dependency(self) -> None:
#        self._dependency_node._clear()


class LazyContainer(Generic[_InstanceT, _ElementT], ABC):
    __slots__ = ("_container_backref",)

    def __init__(self) -> None:
        super().__init__()
        self._container_backref: LazySlot = NotImplemented

    @abstractmethod
    def _iter_elements(self) -> Generator[_ElementT, None, None]:
        pass

    @abstractmethod
    def _copy_container(self) -> "LazyContainer[_InstanceT, _ElementT]":
        pass

    def _set_container_backref(
        self,
        container_backref: "LazySlot"
    ) -> None:
        assert self._container_backref is NotImplemented
        self._container_backref = container_backref

    #@abstractmethod
    #def _copy_from(
    #    self,
    #    src: "LazyContainer[_InstanceT, _ElementT]"
    #) -> None:
    #    pass


class LazyUnitaryContainer(LazyContainer[_InstanceT, _ElementT]):
    __slots__ = ("_element",)

    def __init__(
        self,
        element: _ElementT
    ) -> None:
        super().__init__()
        element._element_backrefs.append(self)
        self._element = element

    def _iter_elements(self) -> Generator[_ElementT, None, None]:
        yield self._element

    def _copy_container(self) -> "LazyUnitaryContainer[_InstanceT, _ElementT]":
        return LazyUnitaryContainer(
            element=self._element
        )

    #def _copy_from(
    #    self,
    #    src: "LazyUnitaryContainer[_InstanceT, _ElementT]"
    #) -> None:
    #    self._element._element_backrefs.remove(self)
    #    src._element._element_backrefs.append(self)
    #    self._element = src._element

    #def _get_container_repr(self) -> _ElementT:
    #    return self._element


class LazyDynamicContainer(LazyContainer[_InstanceT, _ElementT]):
    __slots__ = ("_elements",)

    def __init__(
        self,
        elements: Iterable[_ElementT]
    ) -> None:
        super().__init__()
        elements_list = list(elements)
        self._elements: list[_ElementT] = elements_list
        for element in elements_list:
            element._element_backrefs.append(self)

    def __iter__(self) -> Iterator[_ElementT]:
        return self._elements.__iter__()

    def __len__(self) -> int:
        return self._elements.__len__()

    @overload
    def __getitem__(
        self,
        index: int
    ) -> _ElementT: ...

    @overload
    def __getitem__(
        self,
        index: slice
    ) -> list[_ElementT]: ...

    def __getitem__(
        self,
        index: int | slice
    ) -> _ElementT | list[_ElementT]:
        return self._elements.__getitem__(index)

    def _iter_elements(self) -> Generator[_ElementT, None, None]:
        yield from self._elements

    def _copy_container(self) -> "LazyDynamicContainer[_InstanceT, _ElementT]":
        return LazyDynamicContainer(
            elements=self._elements
        )

    #def _get_container_repr(self) -> "LazyDynamicContainer[_InstanceT, _ElementT]":
    #    return self

    def add(
        self,
        *elements: _ElementT
    ):
        slot = self._container_backref
        assert not slot.is_readonly()
        if not elements:
            return self
        for element in elements:
            if element in self._elements:
                continue
            self._elements.append(element)
            element._element_backrefs.append(self)
        assert isinstance(slot, LazyVariableSlot)
        slot.expire_property_slots()
        return self

    def remove(
        self,
        *elements: _ElementT
    ):
        slot = self._container_backref
        assert not slot.is_readonly()
        if not elements:
            return self
        for element in elements:
            if element not in self._elements:
                continue
            self._elements.remove(element)
            element._element_backrefs.remove(self)
            element._kill_if_no_element_backrefs()
        assert isinstance(slot, LazyVariableSlot)
        slot.expire_property_slots()
        return self


class LazySlot(Generic[_InstanceT, _ContainerT], ABC):
    __slots__ = ("_descriptor_backref",)

    def __init__(
        self,
        instance: _InstanceT
    ) -> None:
        super().__init__()
        self._descriptor_backref = instance

    def yield_ancestor_slots(self) -> "Generator[LazySlot, None, None]":
        yield self
        for container in self._descriptor_backref._element_backrefs:
            yield from container._container_backref.yield_ancestor_slots()

    #@abstractmethod
    #def get_container_if_exists(self) -> _ContainerT | None:
    #    pass

    def is_readonly(self) -> bool:
        #if isinstance(self, LazyPropertySlot):
        #    return False
        return any(
            isinstance(slot, LazyPropertySlot)
            for slot in self.yield_ancestor_slots()
        )


class LazyVariableSlot(LazySlot[_InstanceT, _ContainerT]):
    __slots__ = (
        "_container",
        "_linked_property_slots"
    )

    def __init__(
        self,
        instance: _InstanceT,
        container: _ContainerT
    ) -> None:
        super().__init__(
            instance=instance
        )
        container._set_container_backref(self)
        self._container: _ContainerT = container
        self._linked_property_slots: list[LazyPropertySlot] = []

    #def get_container_if_exists(self) -> _ContainerT | None:
    #    return self._container

    def get_variable_container(self) -> _ContainerT:
        return self._container

    def set_variable_container(
        self,
        new_container: _ContainerT
    ) -> None:
        assert not self.is_readonly()
        old_container = self.get_variable_container()
        old_elements = tuple(old_container._iter_elements())
        new_elements = tuple(new_container._iter_elements())
        if old_elements == new_elements:
            return
        for old_element in old_elements:
            old_element._element_backrefs.remove(old_container)
            old_element._kill_if_no_element_backrefs()
        new_container._set_container_backref(self)
        self._container = new_container
        #for new_element in new_elements:
        #    new_element._element_backrefs.append(self)
        self.expire_property_slots()

    def expire_property_slots(self) -> None:
        #expired_objects: list[LazyObject] = []
        for expired_property_slot in self._linked_property_slots:
            expired_property_slot.expire()
            #expired_property._linked_variable_slots.clear()
            #if isinstance(expired_property, LazyPropertyUnitarySlot):
            #    if (element := expired_property._element) is not None:
            #        expired_objects.append(element)
            #        expired_property._element = None
            #elif isinstance(expired_property, LazyPropertyDynamicSlot):
            #    if (elements := expired_property._elements) is not None:
            #        expired_objects.extend(elements)
            #        expired_property._elements = None
            #if (entity := expired_property._get()) is None:
            #    continue
            #expired_property._unbind_dependency(entity)
            #expired_property._descriptor.clear_ref(self)
            #expired_property._clear_ref()
            #expired_objects.append(entity)
        self._linked_property_slots.clear()
        #for obj in expired_objects:
        #    obj._clear_descendants_ref_if_no_dependency_parents()

    def yield_descendant_variable_slots(self) -> "Generator[LazyVariableSlot, None, None]":
        yield self
        for element in self.get_variable_container()._iter_elements():
            for slot in element._iter_variable_slots():
                yield from slot.yield_descendant_variable_slots()

    #def _clear_descendants_ref_if_no_dependency_parents(self) -> None:
    #    stack: list[LazyData] = [self]
    #    while stack:
    #        data = stack.pop(0)
    #        if list(data._iter_dependency_parents()):
    #            continue
    #        if data._always_alive:
    #            continue
    #        stack.extend(
    #            child for child in data._iter_dependency_children()
    #            if child not in stack
    #        )
    #        data._clear_dependency()
    #        data._clear_ref()


#class LazyUnitaryVariableSlot(LazyVariableSlot[_InstanceT, _ElementT]):
#    __slots__ = ("_element",)

#    def __init__(
#        self,
#        instance: _InstanceT,
#        element: _ElementT
#    ) -> None:
#        super().__init__(
#            instance=instance
#        )
#        self._element: _ElementT = element

#    def _get(self) -> _ElementT:
#        return self._element

#    def _set(
#        self,
#        new_element: _ElementT
#    ) -> None:
#        assert not self.is_readonly()
#        old_element = self._element
#        if old_element is new_element:
#            return
#        old_element._element_backrefs.remove(self)
#        old_element._kill_if_no_element_backrefs()
#        self._element = new_element
#        new_element._element_backrefs.append(self)
#        self._expire_properties()


#class LazyDynamicVariableSlot(LazyVariableSlot[_InstanceT, _ElementT]):
#    __slots__ = ("_collection",)

#    def __init__(
#        self,
#        instance: _InstanceT,
#        elements: Iterable[_ElementT]
#    ) -> None:
#        super().__init__(
#            instance=instance
#        )
#        self._collection: LazyCollection[_InstanceT, _ElementT] = LazyCollection(
#            slot=self,
#            elements=elements
#        )

#    def _get(self) -> "LazyCollection[_InstanceT, _ElementT]":
#        return self._collection

#    def _set(
#        self,
#        new_collection: "LazyCollection[_InstanceT, _ElementT]"
#    ) -> None:
#        assert not self.is_readonly()
#        old_collection = self._collection
#        if old_collection is new_collection:
#            return
#        for old_element in old_collection._elements:
#            old_element._element_backrefs.remove(self)
#            old_element._kill_if_no_element_backrefs()
#        self._collection = new_collection
#        for new_element in new_collection._elements:
#            new_element._element_backrefs.append(self)
#        self._expire_properties()


class LazyPropertySlot(LazySlot[_InstanceT, _ContainerT]):
    __slots__ = (
        "_container",
        "_linked_variable_slots"
    )

    def __init__(
        self,
        instance: _InstanceT
    ) -> None:
        super().__init__(
            instance=instance
        )
        self._container: _ContainerT | None = None
        self._linked_variable_slots: list[LazyVariableSlot] = []

    #def get_container_if_exists(self) -> _ContainerT | None:
    #    return self._container

    def get_property_container(self) -> _ContainerT | None:
        return self._container

    def set_property_container(
        self,
        new_container: _ContainerT
    ) -> None:
        assert self._container is None
        new_container._set_container_backref(self)
        self._container = new_container

    def expire(self) -> None:
        if (container := self._container) is None:
            return
        for element in container._iter_elements():
            element._element_backrefs.remove(container)
            element._kill_if_no_element_backrefs()
        self._container = None
        self._linked_variable_slots.clear()

    def bind_linked_variable_slots(
        self,
        *linked_variable_slots: LazyVariableSlot
    ) -> None:
        for linked_variable_slot in linked_variable_slots:
            linked_variable_slot._linked_property_slots.append(self)
        self._linked_variable_slots.extend(linked_variable_slots)


#class LazyUnitaryPropertySlot(LazyPropertySlot[_InstanceT, _ElementT]):
#    __slots__ = ("_element",)

#    def __init__(
#        self,
#        instance: _InstanceT
#    ) -> None:
#        super().__init__(
#            instance=instance
#        )
#        self._element: _ElementT | None = None

#    def _get(self) -> _ElementT | None:
#        return self._element

#    def _set(
#        self,
#        new_element: _ElementT
#    ) -> None:
#        self._element = new_element

#    def _expire(self) -> None:
#        if (element := self._element) is None:
#            return
#        self._linked_variable_slots.clear()
#        element._element_backrefs.remove(self)
#        element._kill_if_no_element_backrefs()
#        self._element = None


#class LazyDynamicPropertySlot(LazyPropertySlot[_InstanceT, _ElementT]):
#    __slots__ = ("_collection",)

#    def __init__(
#        self,
#        instance: _InstanceT
#    ) -> None:
#        super().__init__(
#            instance=instance
#        )
#        self._collection: LazyCollection[_InstanceT, _ElementT] | None = None

#    def _get(self) -> "LazyCollection[_InstanceT, _ElementT] | None":
#        return self._collection

#    def _set(
#        self,
#        new_collection: "LazyCollection[_InstanceT, _ElementT]"
#    ) -> None:
#        self._collection = new_collection

#    def _expire(self) -> None:
#        if (collection := self._collection) is None:
#            return
#        self._linked_variable_slots.clear()
#        for element in collection._elements:
#            element._element_backrefs.remove(self)
#            element._kill_if_no_element_backrefs()
#        self._collection = None


class LazyObject(ABC):
    __slots__ = (
        "_element_backrefs",
        "_always_alive",
        "_is_dead"
    )

    _LAZY_DESCRIPTORS: "ClassVar[tuple[LazyDescriptor, ...]]"
    _LAZY_VARIABLE_DESCRIPTORS: "ClassVar[tuple[LazyVariableDescriptor, ...]]"
    #_LAZY_PROPERTY_DESCRIPTORS: "ClassVar[tuple[LazyPropertyDescriptor, ...]]"
    _LAZY_DESCRIPTOR_OVERLOADING_DICT: "ClassVar[dict[str, LazyDescriptorOverloading]]"
    _PY_SLOTS: "ClassVar[tuple[str, ...]]"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        base_cls = cls.__base__
        assert issubclass(base_cls, LazyObject)
        if base_cls is not LazyObject:
            descriptor_overloading_dict = base_cls._LAZY_DESCRIPTOR_OVERLOADING_DICT.copy()
        else:
            descriptor_overloading_dict = {}

        overridden_descriptor_overloadings: list[LazyDescriptorOverloading] = []
        for name, attr in cls.__dict__.items():
            if not isinstance(attr, LazyDescriptor):
                continue
            #attr.name = name
            assert re.fullmatch(r"_\w+_", name)
            if attr.element_type is NotImplemented:
                attr.element_type = cls
            if (descriptor_overloading := descriptor_overloading_dict.get(name)) is not None:
                if isinstance(descriptor_overloading, LazyUnitaryDescriptorOverloading):
                    assert isinstance(attr, LazyUnitaryVariableDescriptor | LazyUnitaryPropertyDescriptor)
                    descriptor_overloading.set_descriptor(cls, attr)
                elif isinstance(descriptor_overloading, LazyDynamicDescriptorOverloading):
                    assert isinstance(attr, LazyDynamicVariableDescriptor | LazyDynamicPropertyDescriptor)
                    descriptor_overloading.set_descriptor(cls, attr)
                else:
                    raise TypeError
            else:
                if isinstance(attr, LazyUnitaryVariableDescriptor | LazyUnitaryPropertyDescriptor):
                    descriptor_overloading = LazyUnitaryDescriptorOverloading(cls, attr)
                elif isinstance(attr, LazyDynamicVariableDescriptor | LazyDynamicPropertyDescriptor):
                    descriptor_overloading = LazyDynamicDescriptorOverloading(cls, attr)
                else:
                    raise TypeError
                descriptor_overloading_dict[name] = descriptor_overloading
            overridden_descriptor_overloadings.append(descriptor_overloading)

        #lazy_variable_descriptors: list[LazyVariableDescriptor] = []
        #lazy_property_descriptors: list[LazyPropertyDescriptor] = []
        lazy_descriptors: list[LazyDescriptor] = []
        for descriptor_overloading in descriptor_overloading_dict.values():
            if descriptor_overloading not in overridden_descriptor_overloadings:
                descriptor_overloading.set_descriptor(cls, descriptor_overloading.get_descriptor(base_cls))
            lazy_descriptors.append(descriptor_overloading.get_descriptor(cls))
            #if isinstance(descriptor, LazyVariableDescriptor):
            #    lazy_variable_descriptors.append(descriptor)
            #elif isinstance(descriptor, LazyPropertyDescriptor):
            #    lazy_property_descriptors.append(descriptor)
            #else:
            #    raise TypeError

        #descriptors = [
        #    descriptor_overloading.get_descriptor(cls)
        #    for descriptor_overloading in descriptor_overloadings.values()
        #]
        cls._LAZY_DESCRIPTORS = tuple(lazy_descriptors)
        cls._LAZY_VARIABLE_DESCRIPTORS = tuple(
            descriptor for descriptor in lazy_descriptors
            if isinstance(descriptor, LazyVariableDescriptor)
        )
        #cls._LAZY_VARIABLE_DESCRIPTORS = tuple(lazy_variable_descriptors)
        #cls._LAZY_PROPERTY_DESCRIPTORS = tuple(lazy_property_descriptors)
        #cls._LAZY_DESCRIPTORS = tuple(sorted(
        #    (
        #        descriptor_overloading.get_descriptor(cls)
        #        for descriptor_overloading in descriptor_overloadings.values()
        #    ),
        #    key=lambda descriptor: isinstance(descriptor, LazyPropertyDescriptor)
        #))
        cls._LAZY_DESCRIPTOR_OVERLOADING_DICT = descriptor_overloading_dict
        # Use dict.fromkeys to preserve order (by first occurrance)
        cls._PY_SLOTS = tuple(dict.fromkeys(
            slot
            for parent_cls in reversed(cls.__mro__)
            if issubclass(parent_cls, LazyObject)
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
                    descriptor_overloading = element_type._LAZY_DESCRIPTOR_OVERLOADING_DICT[parameter_name]
                    element_type = descriptor_overloading.element_type
                    descriptor_overloading_chain.append(descriptor_overloading)
                descriptor_overloading_chains.append(tuple(descriptor_overloading_chain))
            attr.descriptor_overloading_chains = tuple(descriptor_overloading_chains)

    def __init__(self) -> None:
        super().__init__()
        #self._always_alive: bool = False
        cls = self.__class__
        for descriptor in cls._LAZY_DESCRIPTORS:
            descriptor.initialize(self)
        #for descriptor in cls._LAZY_PROPERTY_DESCRIPTORS:
        #    descriptor.initialize(self)
        self._element_backrefs: list[LazyContainer] = []
        self._always_alive: bool = False
        self._is_dead: bool = False

    def _copy(self: _LazyObjectT) -> _LazyObjectT:
        cls = self.__class__
        result = cls.__new__(cls)
        for slot_name in cls._PY_SLOTS:
            result.__setattr__(slot_name, copy.copy(self.__getattribute__(slot_name)))
        #result._dependency_node = LazyDependencyNode(result)
        for descriptor in cls._LAZY_DESCRIPTORS:
            descriptor.copy_initialize(result, self)
        #for descriptor in cls._LAZY_PROPERTY_DESCRIPTORS:
        #    descriptor.copy_initialize(result, self)
        return result

    #def _clear_ref(self) -> None:
    #    # TODO: check refcnt
    #    cls = self.__class__
    #    for descriptor in cls._LAZY_DESCRIPTORS:
    #        descriptor.clear_ref(self)
    #    #for descriptor in cls._LAZY_PROPERTY_DESCRIPTORS:
    #    #    descriptor.clear_ref(self)
    #    #import sys
    #    #print(sys.getrefcount(self), type(self))

    def _iter_variable_slots(self) -> Generator[LazyVariableSlot, None, None]:
        for descriptor in self.__class__._LAZY_DESCRIPTORS:
            if not isinstance(descriptor, LazyVariableDescriptor):
                continue
            yield descriptor.get_slot(self)

    def _make_always_alive(self) -> None:
        self._always_alive = True
        for variable_slot in self._iter_variable_slots():
            for element in variable_slot.get_variable_container()._iter_elements():
                element._make_always_alive()

    def _kill_if_no_element_backrefs(self) -> None:
        if self._is_dead:
            return
        if self._always_alive:
            return
        if self._element_backrefs:
            return
        self._is_dead = True
        # TODO: check refcnt
        cls = self.__class__
        for descriptor in cls._LAZY_DESCRIPTORS:
            descriptor.clear_ref(self)









#class LazyEntity(ABC):
#    __slots__ = (
#        "_dependency_node",
#        "_is_property",
#        "_linked_extities",
#        "_always_alive"
#        #"_is_expired"
#    )

#    def __init__(self) -> None:
#        super().__init__()
#        self._dependency_node: LazyDependencyNode = LazyDependencyNode(self)
#        self._is_property: bool = False
#        self._linked_entities: list[LazyEntity] = []
#        self._always_alive: bool = False
#        #self._is_expired: bool = False

#    def _iter_dependency_children(self) -> "Generator[LazyEntity, None, None]":
#        for child in self._dependency_node._children:
#            yield child._ref

#    def _iter_dependency_parents(self) -> "Generator[LazyEntity, None, None]":
#        for parent in self._dependency_node._parents:
#            yield parent._ref

#    def _iter_dependency_descendants(self) -> "Generator[LazyEntity, None, None]":
#        for descendant in self._dependency_node._iter_descendants():
#            yield descendant._ref

#    def _iter_dependency_ancestors(self) -> "Generator[LazyEntity, None, None]":
#        for ancestor in self._dependency_node._iter_ancestors():
#            yield ancestor._ref

#    def _bind_dependency(
#        self,
#        instance: "LazyEntity"
#    ) -> None:
#        self._dependency_node._bind(instance._dependency_node)

#    def _unbind_dependency(
#        self,
#        instance: "LazyEntity"
#    ) -> None:
#        self._dependency_node._unbind(instance._dependency_node)

#    def _clear_dependency(self) -> None:
#        self._dependency_node._clear()

#    @abstractmethod
#    def _clear_ref(self) -> None:
#        pass

#    def _setup_property_flag(self) -> None:
#        self._is_property = True

#    def is_readonly(self) -> bool:
#        return any(
#            entity._is_property
#            for entity in self._iter_dependency_ancestors()
#        )

#    def _get_linked_entities(self) -> "list[LazyEntity]":
#        return self._linked_entities

#    def _bind_linked_entities(
#        self,
#        *linked_entities: "LazyEntity"
#    ) -> None:
#        for linked_entity in linked_entities:
#            linked_entity._linked_entities.append(self)
#            self._linked_entities.append(linked_entity)

#    def _unbind_linked_entities(
#        self,
#        *linked_entities: "LazyEntity"
#    ) -> None:
#        for linked_entity in linked_entities:
#            linked_entity._linked_entities.remove(self)
#            self._linked_entities.remove(linked_entity)

#    def _clear_linked_entities(self) -> None:
#        self._unbind_linked_entities(*self._get_linked_entities())

#    def _make_always_alive(self) -> None:
#        for entity in self._iter_dependency_descendants():
#            entity._always_alive = True


#class LazyEntity(LazyData):
#    __slots__ = ("_linked_properties",)

#    def __init__(self) -> None:
#        super().__init__()
#        self._linked_properties: list[LazyProperty] = []
#        #self._always_alive: bool = False

#    def _get_linked_properties(self) -> "list[LazyProperty]":
#        return self._linked_properties

#    def _bind_linked_property(
#        self,
#        linked_property: "LazyProperty"
#    ) -> None:
#        self._linked_properties.append(linked_property)

#    def _unbind_linked_property(
#        self,
#        linked_property: "LazyProperty"
#    ) -> None:
#        self._linked_properties.remove(linked_property)

#    def is_readonly(self) -> bool:
#        return any(
#            isinstance(data, LazyProperty)
#            for data in self._iter_dependency_ancestors()
#        )

#    def _expire_properties(
#        self,
#        *expired_properties: "LazyProperty"
#    ) -> None:
#        expired_entities: list[LazyEntity] = []
#        for expired_property in expired_properties:
#            if (entity := expired_property._get()) is None:
#                continue
#            expired_property._unbind_dependency(entity)
#            expired_property._descriptor.clear_ref(self)
#            expired_property._clear_ref()
#            expired_entities.append(entity)
#        for entity in dict.fromkeys(expired_entities):
#            entity._clear_descendants_ref_if_no_dependency_parents()

#    def _clear_descendants_ref_if_no_dependency_parents(self) -> None:
#        stack: list[LazyData] = [self]
#        while stack:
#            data = stack.pop(0)
#            if list(data._iter_dependency_parents()):
#                continue
#            if data._always_alive:
#                continue
#            stack.extend(
#                child for child in data._iter_dependency_children()
#                if child not in stack
#            )
#            print("Cleared data: ", data)
#            data._clear_dependency()
#            data._clear_ref()
#            data._is_expired = True


#class LazyObject(LazyEntity):
#    __slots__ = ()

#    _LAZY_VARIABLE_DESCRIPTORS: "ClassVar[tuple[LazyVariableDescriptor, ...]]"
#    _LAZY_PROPERTY_DESCRIPTORS: "ClassVar[tuple[LazyPropertyDescriptor, ...]]"
#    _LAZY_DESCRIPTOR_OVERLOADING_DICT: "ClassVar[dict[str, LazyDescriptorOverloading]]"
#    _PY_SLOTS: "ClassVar[tuple[str, ...]]"

#    def __init_subclass__(cls) -> None:
#        super().__init_subclass__()

#        base_cls = cls.__base__
#        assert issubclass(base_cls, LazyObject)
#        if base_cls is not LazyObject:
#            descriptor_overloadings = base_cls._LAZY_DESCRIPTOR_OVERLOADING_DICT.copy()
#        else:
#            descriptor_overloadings = {}

#        overridden_descriptor_overloadings: list[LazyDescriptorOverloading] = []
#        for name, attr in cls.__dict__.items():
#            if not isinstance(attr, LazyDescriptor):
#                continue
#            attr.name = name
#            assert re.fullmatch(r"_\w+_", name)
#            if attr.element_type is NotImplemented:
#                attr.element_type = cls
#            if (descriptor_overloading := descriptor_overloadings.get(name)) is not None:
#                if isinstance(descriptor_overloading, LazyUnitaryDescriptorOverloading):
#                    assert isinstance(attr, LazyObjectVariableDescriptor | LazyObjectPropertyDescriptor)
#                    descriptor_overloading.set_descriptor(cls, attr)
#                elif isinstance(descriptor_overloading, LazyDynamicDescriptorOverloading):
#                    assert isinstance(attr, LazyCollectionVariableDescriptor | LazyCollectionPropertyDescriptor)
#                    descriptor_overloading.set_descriptor(cls, attr)
#                else:
#                    raise TypeError
#            else:
#                if isinstance(attr, LazyObjectVariableDescriptor | LazyObjectPropertyDescriptor):
#                    descriptor_overloading = LazyUnitaryDescriptorOverloading(cls, attr)
#                elif isinstance(attr, LazyCollectionVariableDescriptor | LazyCollectionPropertyDescriptor):
#                    descriptor_overloading = LazyDynamicDescriptorOverloading(cls, attr)
#                else:
#                    raise TypeError
#                descriptor_overloadings[name] = descriptor_overloading
#            overridden_descriptor_overloadings.append(descriptor_overloading)

#        lazy_variable_descriptors: list[LazyVariableDescriptor] = []
#        lazy_property_descriptors: list[LazyPropertyDescriptor] = []
#        for descriptor_overloading in descriptor_overloadings.values():
#            if descriptor_overloading not in overridden_descriptor_overloadings:
#                descriptor_overloading.set_descriptor(cls, descriptor_overloading.get_descriptor(base_cls))
#            descriptor = descriptor_overloading.get_descriptor(cls)
#            if isinstance(descriptor, LazyVariableDescriptor):
#                lazy_variable_descriptors.append(descriptor)
#            elif isinstance(descriptor, LazyPropertyDescriptor):
#                lazy_property_descriptors.append(descriptor)
#            else:
#                raise TypeError

#        #descriptors = [
#        #    descriptor_overloading.get_descriptor(cls)
#        #    for descriptor_overloading in descriptor_overloadings.values()
#        #]
#        cls._LAZY_VARIABLE_DESCRIPTORS = tuple(lazy_variable_descriptors)
#        cls._LAZY_PROPERTY_DESCRIPTORS = tuple(lazy_property_descriptors)
#        #cls._LAZY_DESCRIPTORS = tuple(sorted(
#        #    (
#        #        descriptor_overloading.get_descriptor(cls)
#        #        for descriptor_overloading in descriptor_overloadings.values()
#        #    ),
#        #    key=lambda descriptor: isinstance(descriptor, LazyPropertyDescriptor)
#        #))
#        cls._LAZY_DESCRIPTOR_OVERLOADING_DICT = descriptor_overloadings
#        # Use dict.fromkeys to preserve order (by first occurrance)
#        cls._PY_SLOTS = tuple(dict.fromkeys(
#            slot
#            for parent_cls in reversed(cls.__mro__)
#            if issubclass(parent_cls, LazyEntity)
#            for slot in parent_cls.__slots__
#        ))

#        for attr in cls.__dict__.values():
#            if not isinstance(attr, LazyPropertyDescriptor):
#                continue
#            descriptor_overloading_chains: list[tuple[LazyDescriptorOverloading, ...]] = []
#            for parameter_name_chain in attr.parameter_name_chains:
#                element_type = cls
#                descriptor_overloading_chain: list[LazyDescriptorOverloading] = []
#                for parameter_name in parameter_name_chain:
#                    descriptor_overloading = element_type._LAZY_DESCRIPTOR_OVERLOADING_DICT[parameter_name]
#                    element_type = descriptor_overloading.element_type
#                    descriptor_overloading_chain.append(descriptor_overloading)
#                descriptor_overloading_chains.append(tuple(descriptor_overloading_chain))
#            attr.descriptor_overloading_chains = tuple(descriptor_overloading_chains)

#    def __init__(self) -> None:
#        super().__init__()
#        #self._always_alive: bool = False
#        cls = self.__class__
#        for descriptor in cls._LAZY_VARIABLE_DESCRIPTORS:
#            descriptor.initialize(self)
#        #for descriptor in cls._LAZY_PROPERTY_DESCRIPTORS:
#        #    descriptor.initialize(self)

#    def _copy(self: _LazyObjectT) -> _LazyObjectT:
#        cls = self.__class__
#        result = cls.__new__(cls)
#        for slot_name in cls._PY_SLOTS:
#            result.__setattr__(slot_name, copy.copy(self.__getattribute__(slot_name)))
#        result._dependency_node = LazyDependencyNode(result)
#        for descriptor in cls._LAZY_VARIABLE_DESCRIPTORS:
#            descriptor.copy_initialize(result, self)
#        #for descriptor in cls._LAZY_PROPERTY_DESCRIPTORS:
#        #    descriptor.copy_initialize(result, self)
#        return result

#    def _clear_ref(self) -> None:
#        # TODO: check refcnt
#        cls = self.__class__
#        for descriptor in cls._LAZY_VARIABLE_DESCRIPTORS:
#            descriptor.clear_ref(self)
#        for descriptor in cls._LAZY_PROPERTY_DESCRIPTORS:
#            descriptor.clear_ref(self)
#        #import sys
#        #print(sys.getrefcount(self), type(self))






#@final
#class LazyProperty(Generic[_LazyEntityT], LazyData):
#    __slots__ = (
#        "_descriptor",
#        "_entity",
#        "_linked_entities"
#    )
#
#    def __init__(
#        self,
#        descriptor: "LazyPropertyDescriptor"
#    ) -> None:
#        super().__init__()
#        self._descriptor: LazyPropertyDescriptor = descriptor
#        self._entity: _LazyEntityT | None = None
#        self._linked_entities: list[LazyEntity] = []
#
#    def _get(self) -> _LazyEntityT | None:
#        return self._entity
#
#    def _set(
#        self,
#        entity: _LazyEntityT | None
#    ) -> None:
#        self._entity = entity
#
#    def _get_linked_entities(self) -> list[LazyEntity]:
#        return self._linked_entities
#
#    def _bind_linked_entities(
#        self,
#        *linked_entities: LazyEntity
#    ) -> None:
#        for linked_entity in linked_entities:
#            linked_entity._bind_linked_property(self)
#        self._linked_entities.extend(linked_entities)
#
#    def _clear_linked_entities(self) -> None:
#        for linked_entity in self._linked_entities:
#            linked_entity._unbind_linked_property(self)
#        self._linked_entities.clear()
#
#    def _clear_ref(self) -> None:
#        self._set(None)
#        self._clear_linked_entities()


class LazyDescriptor(Generic[_InstanceT, _SlotT, _ContainerT, _ElementT, _DescriptorGetT, _DescriptorSetT], ABC):
    __slots__ = (
        #"name",
        "element_type",
        "instance_to_slot_dict"
    )

    def __init__(
        self,
        element_type: type[_ElementT]
    ) -> None:
        super().__init__()
        #self.name: str = NotImplemented
        self.element_type: type[_ElementT] = element_type
        self.instance_to_slot_dict: dict[_InstanceT, _SlotT] = {}

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
    ) -> _DescriptorGetT: ...

    def __get__(
        self: _LazyDescriptorT,
        instance: _InstanceT | None,
        owner: type[_InstanceT] | None = None
    ) -> _LazyDescriptorT | _DescriptorGetT:
        if instance is None:
            return self
        return self.convert_get(self.get_impl(instance))

    def __set__(
        self,
        instance: _InstanceT,
        new_value: _DescriptorSetT
    ) -> None:
        self.set_impl(instance, self.convert_set(new_value))

    @abstractmethod
    def get_impl(
        self,
        instance: _InstanceT
    ) -> _ContainerT:
        pass

    @abstractmethod
    def set_impl(
        self,
        instance: _InstanceT,
        new_container: _ContainerT
    ) -> None:
        pass

    @abstractmethod
    def convert_get(
        self,
        container: _ContainerT
    ) -> _DescriptorGetT:
        pass

    @abstractmethod
    def convert_set(
        self,
        new_value: _DescriptorSetT
    ) -> _ContainerT:
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
        self.instance_to_slot_dict.pop(instance)

    def get_slot(
        self,
        instance: _InstanceT
    ) -> _SlotT:
        return self.instance_to_slot_dict[instance]

    def set_slot(
        self,
        instance: _InstanceT,
        slot: _SlotT
    ) -> None:
        self.instance_to_slot_dict[instance] = slot


class LazyVariableDescriptor(LazyDescriptor[
    _InstanceT, LazyVariableSlot[_InstanceT, _ContainerT], _ContainerT, _ElementT, _DescriptorGetT, _DescriptorSetT
]):
    __slots__ = (
        "method",
        "default_container"
    )

    def __init__(
        self,
        element_type: type[_ElementT],
        method: Callable[[type[_InstanceT]], _DescriptorSetT]
    ) -> None:
        super().__init__(
            element_type=element_type
        )
        self.method: Callable[[type[_InstanceT]], _DescriptorSetT] = method
        self.default_container: _ContainerT | None = None

    def get_impl(
        self,
        instance: _InstanceT
    ) -> _ContainerT:
        return self.get_slot(instance).get_variable_container()

    def set_impl(
        self,
        instance: _InstanceT,
        new_container: _ContainerT
    ) -> None:
        slot = self.get_slot(instance)
        slot.set_variable_container(new_container)
        #self.get_slot(instance).set_container(self.convert_set(new_value))
        #assert not slot.is_readonly()
        #old_object = slot._element
        #if old_object is new_object:
        #    return

        #slot._expire_properties()
        ##instance._expire_properties(*(
        ##    prop
        ##    for prop in old_object._get_linked_properties()
        ##    if prop in instance._iter_dependency_children()
        ##))
        #instance._unbind_dependency(old_object)
        #old_object._clear_descendants_ref_if_no_dependency_parents()
        ##if new_object is not NotImplemented:
        #instance._bind_dependency(new_object)
        #self.set_description(instance, new_object)

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        if (default_container := self.default_container) is None:
            default_container = self.convert_set(self.method(type(instance)))
            for default_object in default_container._iter_elements():
                default_object._make_always_alive()
            #if default_object is not NotImplemented:
            #    for default_object_descendant in default_object._iter_dependency_descendants():
            #        if isinstance(default_object_descendant, LazyEntity):
            #            default_object_descendant._always_alive = True
            self.default_container = default_container
        self.set_slot(instance, LazyVariableSlot(
            instance=instance,
            container=default_container._copy_container()
        ))
        #default_object = self.method(type(instance))
        #if default_object is not NotImplemented:
        #instance._bind_dependency(default_object)
        #self.set_description(instance, default_object)

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        #if (src_object := self.get_description(src)) is not NotImplemented:
        #dst._bind_dependency(src_object)
        #self.set_description(dst, src_object)
        self.set_slot(dst, LazyVariableSlot(
            instance=dst,
            container=self.get_slot(src).get_variable_container()._copy_container()
        ))

    #@abstractmethod
    #def get_default_container(
    #    self,
    #    instance_type: type[_InstanceT]
    #) -> _ContainerT:
    #    pass


class LazyUnitaryVariableDescriptor(LazyVariableDescriptor[
    _InstanceT, LazyUnitaryContainer[_InstanceT, _ElementT], _ElementT, _ElementT, _DescriptorSetT
]):
    #__slots__ = ("method",)

    #def __init__(
    #    self,
    #    element_type: type[_ElementT],
    #    method: Callable[[type[_InstanceT]], _DescriptorSetT]
    #) -> None:
    #    super().__init__(
    #        element_type=element_type
    #    )
    #    self.method: Callable[[type[_InstanceT]], _DescriptorSetT] = method

    def convert_get(
        self,
        container: LazyUnitaryContainer[_InstanceT, _ElementT]
    ) -> _ElementT:
        return container._element

    #def get_default_container(
    #    self,
    #    instance_type: type[_InstanceT]
    #) -> LazyUnitaryContainer[_InstanceT, _ElementT]:
    #    return self.convert_set(self.method(instance_type))

    #def initialize(
    #    self,
    #    instance: _InstanceT
    #) -> None:
    #    if (default_object := self.default_object) is None:
    #        default_object = self.method(type(instance))
    #        default_object._make_always_alive()
    #        #if default_object is not NotImplemented:
    #        #    for default_object_descendant in default_object._iter_dependency_descendants():
    #        #        if isinstance(default_object_descendant, LazyEntity):
    #        #            default_object_descendant._always_alive = True
    #        self.default_object = default_object
    #    self.set_slot(instance, LazyUnitaryVariableSlot(
    #        instance=instance,
    #        element=default_object
    #    ))
    #    #default_object = self.method(type(instance))
    #    #if default_object is not NotImplemented:
    #    #instance._bind_dependency(default_object)
    #    #self.set_description(instance, default_object)

    #def copy_initialize(
    #    self,
    #    dst: _InstanceT,
    #    src: _InstanceT
    #) -> None:
    #    #if (src_object := self.get_description(src)) is not NotImplemented:
    #    dst._bind_dependency(src_object)
    #    self.set_description(dst, src_object)


class LazyDynamicVariableDescriptor(LazyVariableDescriptor[
    _InstanceT, LazyDynamicContainer[_InstanceT, _ElementT], _ElementT, LazyDynamicContainer[_InstanceT, _ElementT], _DescriptorSetT
]):
    #__slots__ = ("method",)

    #def __init__(
    #    self,
    #    element_type: type[_ElementT],
    #    method: Callable[[type[_InstanceT]], _DescriptorSetT]
    #) -> None:
    #    super().__init__(
    #        element_type=element_type
    #    )
    #    self.method: Callable[[type[_InstanceT]], _DescriptorSetT] = method
    #    #self.default_objects: tuple[_ElementT, ...] | None = None

    def convert_get(
        self,
        container: LazyDynamicContainer[_InstanceT, _ElementT]
    ) -> LazyDynamicContainer[_InstanceT, _ElementT]:
        return container

    #def get_default_container(
    #    self,
    #    instance_type: type[_InstanceT]
    #) -> LazyDynamicContainer[_InstanceT, _ElementT]:
    #    return self.convert_set(self.method(instance_type))

    #def get_impl(
    #    self,
    #    instance: _InstanceT
    #) -> LazyCollection[_InstanceT, _ElementT]:
    #    return self.get_slot(instance)._get()

    #def set_impl(
    #    self,
    #    instance: _InstanceT,
    #    new_value: Iterable[_ElementT]
    #) -> None:
    #    slot = self.get_slot(instance)
    #    slot._set(LazyCollection(
    #        slot=slot,
    #        elements=new_value
    #    ))
        #assert not slot.is_readonly()
        #old_object = slot._element
        #if old_object is new_object:
        #    return

        #slot._expire_properties()
        ##instance._expire_properties(*(
        ##    prop
        ##    for prop in old_object._get_linked_properties()
        ##    if prop in instance._iter_dependency_children()
        ##))
        #instance._unbind_dependency(old_object)
        #old_object._clear_descendants_ref_if_no_dependency_parents()
        ##if new_object is not NotImplemented:
        #instance._bind_dependency(new_object)
        #self.set_description(instance, new_object)

    #def initialize(
    #    self,
    #    instance: _InstanceT
    #) -> None:
    #    if (default_objects := self.default_objects) is None:
    #        default_objects = tuple(self.method(type(instance)))
    #        for default_object in default_objects:
    #            default_object._make_always_alive()
    #        #if default_object is not NotImplemented:
    #        #    for default_object_descendant in default_object._iter_dependency_descendants():
    #        #        if isinstance(default_object_descendant, LazyEntity):
    #        #            default_object_descendant._always_alive = True
    #        self.default_objects = default_objects
    #    self.instance_to_slot_dict[instance] = LazyDynamicVariableSlot(
    #        instance=instance,
    #        elements=default_objects
    #    )
    #    #default_object = self.method(type(instance))
    #    #if default_object is not NotImplemented:
    #    #instance._bind_dependency(default_object)
    #    #self.set_description(instance, default_object)

    #def copy_initialize(
    #    self,
    #    dst: _InstanceT,
    #    src: _InstanceT
    #) -> None:
    #    #if (src_object := self.get_description(src)) is not NotImplemented:
    #    #dst._bind_dependency(src_object)
    #    #self.set_description(dst, src_object)
    #    self.instance_to_slot_dict[dst] = LazyDynamicVariableSlot(
    #        instance=dst,
    #        elements=self.instance_to_slot_dict[src]._get()
    #    )


#class LazyCollectionVariableDescriptor(LazyVariableDescriptor[_InstanceT, LazyCollection[_LazyObjectT], _LazyObjectT, _DescriptorSetT]):
#    __slots__ = ()
#
#    def initialize(
#        self,
#        instance: _InstanceT
#    ) -> None:
#        default_collection = self.method(type(instance))
#        instance._bind_dependency(default_collection)
#        self.set_description(instance, default_collection)
#
#    def copy_initialize(
#        self,
#        dst: _InstanceT,
#        src: _InstanceT
#    ) -> None:
#        dst_collection = LazyCollection(*self.get_description(src))
#        dst._bind_dependency(dst_collection)
#        self.set_description(dst, dst_collection)


class LazyPropertyDescriptor(LazyDescriptor[
    _InstanceT, LazyPropertySlot[_InstanceT, _ContainerT], _ContainerT, _ElementT, _DescriptorGetT, _DescriptorSetT
]):
    __slots__ = (
        "method",
        "release_method",
        "parameter_name_chains",
        "requires_unwrapping_tuple",
        "descriptor_overloading_chains",
        "key_to_container_dict",
        "instance_to_key_dict",
        "key_to_instances_dict"
    )

    def __init__(
        self,
        element_type: type[_ElementT],
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _DescriptorSetT],
        parameter_name_chains: tuple[tuple[str, ...], ...],
        requires_unwrapping_tuple: tuple[bool, ...]
    ) -> None:
        super().__init__(
            element_type=element_type
        )
        self.method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _DescriptorSetT] = method
        self.release_method: Callable[[type[_InstanceT], _DescriptorGetT], None] | None = None
        self.parameter_name_chains: tuple[tuple[str, ...], ...] = parameter_name_chains
        self.requires_unwrapping_tuple: tuple[bool, ...] = requires_unwrapping_tuple
        self.descriptor_overloading_chains: tuple[tuple[LazyDescriptorOverloading, ...], ...] = NotImplemented
        self.key_to_container_dict: dict[tuple[Hashable, ...], _ContainerT] = {}
        self.instance_to_key_dict: dict[_InstanceT, tuple[Hashable, ...]] = {}
        self.key_to_instances_dict: dict[Hashable, list[_InstanceT]] = {}

    def get_impl(
        self,
        instance: _InstanceT
    ) -> _ContainerT:

        def construct_parameter_item(
            descriptor_overloading_chain: "tuple[LazyDescriptorOverloading, ...]",
            instance: _InstanceT
        ) -> tuple[TreeNode[LazyObject], list[LazyVariableSlot]]:
            parameter_tagged_tree: TreeNode[tuple[LazyObject, bool]] = TreeNode((instance, False))
            linked_variable_slots: list[LazyVariableSlot] = []
            #linked_slots_tree: TreeNode[LazySlot] = TreeNode(self.get_slot(instance))
            #linked_collections: list[LazyCollection] = []
            #linked_propertiy_slots: list[LazyProperty] = []

            def construct_parameter_tagged_tree(
                descriptor_overloading: LazyDescriptorOverloading,
                is_chain_tail: bool
            ) -> Callable[[tuple[LazyObject, bool]], TreeNode[tuple[LazyObject, bool]]]:

                def callback(
                    obj_tagged: tuple[LazyObject, bool]
                ) -> TreeNode[tuple[LazyObject, bool]]:
                    obj, binding_completed = obj_tagged
                    container = descriptor_overloading.get_descriptor(type(obj)).get_impl(obj)
                    if not binding_completed:
                        slot = container._container_backref
                        if isinstance(slot, LazyVariableSlot):
                            if is_chain_tail:
                                linked_variable_slots.extend(slot.yield_descendant_variable_slots())
                            else:
                                linked_variable_slots.append(slot)
                        elif isinstance(slot, LazyPropertySlot):
                            binding_completed = True
                            linked_variable_slots.extend(slot._linked_variable_slots)
                        else:
                            raise TypeError
                    if isinstance(descriptor_overloading, LazyUnitaryDescriptorOverloading):
                        assert isinstance(container, LazyUnitaryContainer)
                        result = TreeNode((container._element, binding_completed))
                    elif isinstance(descriptor_overloading, LazyDynamicDescriptorOverloading):
                        assert isinstance(container, LazyDynamicContainer)
                        result = TreeNode((obj, binding_completed))
                        result.bind_only(*(TreeNode((element, binding_completed)) for element in container._elements))
                    else:
                        raise TypeError
                    return result

                return callback

            def remove_tag(
                obj_tagged: tuple[LazyObject, bool]
            ) -> LazyObject:
                obj, _ = obj_tagged
                return obj

            #def construct_linked_slots_tree(
            #    descriptor_overloading: LazyDescriptorOverloading
            #) -> Callable[[LazySlot], TreeNode[LazySlot]]:

            #    def callback(
            #        slot: LazySlot
            #    ) -> TreeNode[LazySlot]:
            #        result: TreeNode[LazySlot] = TreeNode(slot)
            #        if isinstance(slot, LazyVariableSlot):
            #            obj = slot._descriptor_backref
            #            descriptor = descriptor_overloading.get_descriptor(type(obj))
            #            result.bind_only(
            #                TreeNode(descriptor.get_impl(obj))
            #            )
            #            if isinstance(descriptor_overloading, LazyUnitaryDescriptorOverloading):
            #                descriptor = descriptor_overloading.get_descriptor(type(obj))
            #                if isinstance(descriptor, LazyObjectVariableDescriptor):
            #                    result.bind_only(TreeNode(descriptor.get_impl(obj)))
            #                else:
            #                    result.bind_only()
            #                    linked_properties.append(descriptor.get_description(obj))
            #            elif isinstance(descriptor_overloading, LazyDynamicDescriptorOverloading):
            #                descriptor = descriptor_overloading.get_descriptor(type(obj))
            #                if isinstance(descriptor, LazyCollectionVariableDescriptor):
            #                    result.bind_only(*(TreeNode(element) for element in descriptor.get_impl(obj)))
            #                    linked_collections.append(descriptor.get_description(obj))
            #                else:
            #                    result.bind_only()
            #                    linked_properties.append(descriptor.get_description(obj))
            #        elif isinstance(slot, LazyPropertySlot):
            #            pass
            #        else:
            #            raise TypeError
            #        return result

            #    return callback

            #def flatten_tree(
            #    tree_node: TreeNode[_TreeNodeContentT]
            #) -> Generator[_TreeNodeContentT, None, None]:
            #    if tree_node._children is None:
            #        yield tree_node._content
            #    else:
            #        for child in tree_node._children:
            #            yield from flatten_tree(child)

            for reversed_index, descriptor_overloading in reversed(list(zip(it.count(), reversed(descriptor_overloading_chain)))):
                parameter_tagged_tree = parameter_tagged_tree.apply_deepest(
                    construct_parameter_tagged_tree(descriptor_overloading, not reversed_index)
                )
                #linked_slots_tree = linked_slots_tree.apply_deepest(construct_linked_slots_tree(descriptor_overloading))

            #linked_entities: list[LazyEntity] = []
            #linked_entities.extend(flatten_tree(linked_objects_tree))
            #linked_entities.extend(linked_collections)
            #linked_entities.extend(it.chain(*(
            #    linked_property._get_linked_entities()
            #    for linked_property in linked_properties
            #)))
            #if l := [data for data in flatten_tree(parameter_tree) if data._is_expired]:
            #    print("***", l)
            #assert all(not data._is_expired for data in flatten_tree(parameter_tree))
            return parameter_tagged_tree.transform(remove_tag), linked_variable_slots

        def expand_dependencies(
            obj: LazyObject
        ) -> TreeNode[LazyObject]:
            result = TreeNode(obj)
            if isinstance(obj, LazyWrapper):
                return result
            slot_nodes: list[TreeNode[LazyObject]] = []
            for variable_slot in obj._iter_variable_slots():
                slot_node: TreeNode[LazyObject] = TreeNode(NotImplemented)
                slot_node.bind_only(*(
                    expand_dependencies(element)
                    for element in variable_slot.get_variable_container()._iter_elements()
                ))
                slot_nodes.append(slot_node)
                #if isinstance(child, LazyObject):
                #    children_nodes.append(expand_dependencies(child))
                #elif isinstance(child, LazyCollection):
                #    collection_node: TreeNode[LazyObject] = TreeNode(NotImplemented)
                #    collection_node.bind_only(*(
                #        expand_dependencies(element) for element in child
                #    ))
                #    children_nodes.append(collection_node)
            result.bind_only(*slot_nodes)
            return result

        def construct_parameter_key(
            tree_node: TreeNode[_TreeNodeContentT]
        ) -> Hashable:
            if tree_node._children is None:
                return tree_node._content
            # The additional `type` is crucial for constructing the key,
            # as different classes overridding the same lazy property may
            # process the same set of parameters in different ways.
            # In other words, the `type` encodes the processing method.
            return (type(tree_node._content), *(
                construct_parameter_key(child)
                for child in tree_node._children
            ))

        def construct_parameter(
            tree_node: TreeNode[_TreeNodeContentT],
            *,
            requires_unwrapping: bool
        ) -> Any:
            if tree_node._children is None:
                content = tree_node._content
                if requires_unwrapping:
                    assert isinstance(content, LazyWrapper)
                    return content.value
                return content
            return [
                construct_parameter(child, requires_unwrapping=requires_unwrapping)
                for child in tree_node._children
            ]

        slot = self.get_slot(instance)
        if (container := slot.get_property_container()) is None:
            #if (old_key := self.instance_to_key_dict.get(instance)) is not None:
            #    self.key_to_instances_dict[old_key].remove(instance)
            parameter_items = tuple(
                construct_parameter_item(descriptor_overloading_chain, instance)
                for descriptor_overloading_chain in self.descriptor_overloading_chains
            )
            slot.bind_linked_variable_slots(*dict.fromkeys(it.chain(*(
                linked_variable_slots
                for _, linked_variable_slots in parameter_items
            ))))
            parameter_trees = tuple(
                parameter_tree
                for parameter_tree, _ in parameter_items
            )
            key = tuple(
                construct_parameter_key(parameter_tree.apply_deepest(expand_dependencies))
                for parameter_tree in parameter_trees
            )
            self.instance_to_key_dict[instance] = key
            #self.key_to_instances_dict.setdefault(key, []).append(instance)
            #print(type(instance))
            #if parameter_trees[0]._content.__class__.__name__ == "PrismoidGeometry":
            #    print(parameter_trees[0]._content._shape_)
            #print(key)
            #if self.name == "_ub_winding_sign_":
            #    #print(l)
            #    print(self.name)
            #    print(f"{key=}")
            #    print(f"{self.key_to_entity_dict=}")
            #    #print(f"{self.key_to_instances_dict=}")
            #    print("-------------------------------------------------------")
            if (cached_container := self.key_to_container_dict.get(key)) is not None:
                container = cached_container._copy_container()
            else:
                parameters = tuple(
                    construct_parameter(parameter_tree, requires_unwrapping=requires_unwrapping)
                    for parameter_tree, requires_unwrapping in zip(
                        parameter_trees, self.requires_unwrapping_tuple, strict=True
                    )
                )
                container = self.convert_set(self.method(type(instance), *parameters))
                #if self.name == "_geometry_data_":
                #    print(type(instance))
                self.key_to_container_dict[key] = container
            #prop._bind_dependency(entity)
            slot.set_property_container(container)
            if instance not in (instances := self.key_to_instances_dict.setdefault(key, [])):
                instances.append(instance)
        return container

    def set_impl(
        self,
        instance: _InstanceT,
        new_container: _ContainerT
    ) -> None:
        raise ValueError("Attempting to set a readonly property")

    #def convert_set(
    #    self,
    #    new_value: _DescriptorSetT
    #) -> _ContainerT:
    #    raise ValueError("Attempting to set a readonly property")

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        self.set_slot(instance, LazyPropertySlot(
            instance=instance
        ))
        #default_property = LazyProperty(self)
        #instance._bind_dependency(default_property)
        #self.set_description(instance, default_property)

    def copy_initialize(
        self,
        dst: _InstanceT,
        src: _InstanceT
    ) -> None:
        self.set_slot(dst, LazyPropertySlot(
            instance=dst
        ))
        #dst_property = LazyProperty(self)
        #src_property = self.get_description(src)

        #src_entity = src_property._get()
        #dst_property._set(src_entity)
        #if src_entity is not None:
        #    dst_property._bind_dependency(src_entity)

        #    dst_collection_children = [
        #        entity for entity in dst._iter_dependency_children()
        #        if isinstance(entity, LazyCollection)
        #    ]
        #    src_collection_children = [
        #        entity for entity in src._iter_dependency_children()
        #        if isinstance(entity, LazyCollection)
        #    ]
        #    dst_property._bind_linked_entities(*(
        #        linked_entity if linked_entity not in src_collection_children
        #        else dst_collection_children[src_collection_children.index(linked_entity)]
        #        for linked_entity in src_property._get_linked_entities()
        #    ))

        #dst._bind_dependency(dst_property)
        #self.set_description(dst, dst_property)

    def clear_ref(
        self,
        instance: _InstanceT
    ) -> None:
        super().clear_ref(instance)
        if (key := self.instance_to_key_dict.pop(instance, None)) is None:
            return
        #instances = self.key_to_instances_dict[key]
        #instances.remove(instance)
        #if instances:
        #    return
        #self.key_to_instances_dict.pop(key)
        #self.key_to_container_dict[key]
        instances = self.key_to_instances_dict[key]
        instances.remove(instance)
        if instances:
            return
        container = self.key_to_container_dict.pop(key)
        if (release_method := self.release_method) is None:
            return
        release_method(type(instance), self.convert_get(container))


class LazyUnitaryPropertyDescriptor(LazyPropertyDescriptor[
    _InstanceT, LazyUnitaryContainer[_InstanceT, _ElementT], _ElementT, _ElementT, _DescriptorSetT
]):
    __slots__ = ()

    def convert_get(
        self,
        container: LazyUnitaryContainer[_InstanceT, _ElementT]
    ) -> _ElementT:
        return container._element


class LazyDynamicPropertyDescriptor(LazyPropertyDescriptor[
    _InstanceT, LazyDynamicContainer[_InstanceT, _ElementT], _ElementT, LazyDynamicContainer[_InstanceT, _ElementT], _DescriptorSetT
]):
    __slots__ = ()

    def convert_get(
        self,
        container: LazyDynamicContainer[_InstanceT, _ElementT]
    ) -> LazyDynamicContainer[_InstanceT, _ElementT]:
        return container


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
        assert instance_type not in self.descriptors
        self.descriptors[instance_type] = descriptor


class LazyUnitaryDescriptorOverloading(LazyDescriptorOverloading[_InstanceT, _ElementT, Union[
    LazyUnitaryVariableDescriptor[_InstanceT, _ElementT, _DescriptorSetT],
    LazyUnitaryPropertyDescriptor[_InstanceT, _ElementT, _DescriptorSetT]
]]):
    __slots__ = ()


class LazyDynamicDescriptorOverloading(LazyDescriptorOverloading[_InstanceT, _ElementT, Union[
    LazyDynamicVariableDescriptor[_InstanceT, _ElementT, _DescriptorSetT],
    LazyDynamicPropertyDescriptor[_InstanceT, _ElementT, _DescriptorSetT]
]]):
    __slots__ = ()


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
