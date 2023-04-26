# TODO: The documentation is outdated.
"""
This module implements lazy evaluation based on weak reference. Meanwhile,
this also introduces functional programming into the project paradigm.

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

The structure of this implementation can be summarized in the following graph:

LazyObject --> LazySlot         --> LazyContainer        --> LazyObject
            |> LazyVariableSlot  |> LazyUnitaryContainer
            |> LazyPropertySlot  |> LazyDynamicContainer

# Lazy Variable

Lazy variables are what users can modify freely from the outer scope. They are
dependent variables of lazy properties. Once modified, the related lazy
properties will be expired, and will be recomputed when fetching.

Containers under `LazyVariableSlot` objects are never shared among instances,
since data mutating is modifying to affect a particular instance. Data
contained, on the other hand, can be shared freely.

Methods decorated by `Lazy.variable` should not take any argument except for
`cls` and return the *initial* value for this data.

+------------+-----------------+-----------------+-----------------------+
| LazyMode   | method return   | __get__ return  | __set__ type          |
+------------+-----------------+-----------------+-----------------------+
| OBJECT     | LazyObjectT     | LazyObjectT     | LazyObjectT           |
| UNWRAPPED  | T               | LazyWrapper[T]  | T | LazyWrapper[T]    |
| SHARED     | HT              | LazyWrapper[HT] | HT | LazyWrapper[HT]  |
| COLLECTION | LazyCollectionT | LazyCollectionT | Iterable[LazyObjectT] |
+------------+-----------------+-----------------+-----------------------+
`HT`: `HashableT`
`LazyCollectionT`: `LazyDynamicContainer[LazyObjectT]`

The `__get__` method always returns an instance of either `LazyObject` or
`LazyDynamicContainer`, the latter of which is just a dynamic collection of
`LazyObject`s, and provides `add`, `discard` as its public interface.
`LazyWrapper` is derived from `LazyObject`, which is just responsible for
bringing a value into the lazy scope, and the value is obtained via the
readonly `value` property. One may picture a lazy object as a tree (it's a
DAG really), where `LazyWrapper`s sit on all leaves.

Lazy variables are of course mutable. All can be modified via `__set__`.
Among all cases above, a common value will be shared among instances, except
for providing `T` type in `UNWRAPPED` mode, in which case a new `LazyWrapper`
object will be instanced and assigned specially to the instance. Additionally,
`LazyDynamicContainer`s can be modified via `add` and `discard`.

Notice that what `__get__` method returns is always a valid input of
`__set__`. This ensures the statement `a._data_ = b._data_` is always valid.

The `LazyObject._copy` method will only copy containers under variable slots.
This means all children `LazyObject`s will be shared, and new
`LazyDynamicContainer`s will be created, holding the same references, just
like a shallow copy.

# Lazy Property

Lazy properties depend on lazy variables and therefore cannot be modified.
Their values are computed only when expired, otherwise the cached value is
directly returned for usage.

Containers under `LazyPropertySlot` objects can be shared due to read-only.

Methods decorated by `Lazy.property` defines how lazy properties are related
to their dependent variables.

+------------+-----------------+-----------------+
| LazyMode   | method return   | __get__ return  |
+------------+-----------------+-----------------+
| OBJECT     | LazyObject      | LazyObject      |
| UNWRAPPED  | T               | LazyWrapper[T]  |
| SHARED     | HT              | LazyWrapper[HT] |
| COLLECTION | LazyCollectionT | LazyCollectionT |
+------------+-----------------+-----------------+
`HT`: `HashableT`
`LazyCollectionT`: `LazyDynamicContainer[LazyObjectT]`

The return type of `__get__` is basically the same as that of lazy variables.
Containers will be entirely shared if the leaf nodes of objects of parameters
(which forms a complicated structure of `LazyWrapper`s) match completely.

The parameters can be lazy variables, or other lazy properties (as long as
cyclic dependency doesn't exist), or a mixed collection of those, as long as
the types are consistent. The name of a parameter needs to indicate how it is
constructed through some specific patterns. Below are some examples.

Suppose `_o_` is a descriptor returning `LazyObject` when calling `__get__`,
and `_w_`, `_c_` for `LazyWrapper`, `LazyDynamicContainer`, respectively.
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

As a conclusion, if there are `n` dynamic descriptor in the name chain, the
parameter will be fed with an `n`-fold list. If the underscores on ends are
missing, it's assumed the last descriptor will return `LazyWrapper` and values
are pulled out from the wrappers.

Lazy properties are immutable. This also applies to children of a `LazyObject`
and elements of `LazyCollection`. That is, one cannot set the value of
`inst._w_` even when it's a lazy variable, given that `inst` itself is the
calculation result of some property.
"""


from abc import (
    ABC,
    abstractmethod
)
#from collections.abc import MutableSequence
import copy
#from enum import Enum
import inspect
import itertools as it
from functools import wraps
import re
import sys
from typing import (
    Any,
    Callable,
    ClassVar,
    Concatenate,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    ParamSpec,
    TypeVar,
    #Union,
    #final,
    overload
)
import weakref


_T = TypeVar("_T")
_HT = TypeVar("_HT", bound=Hashable)
_TreeNodeContentT = TypeVar("_TreeNodeContentT", bound=Hashable)
_ElementT = TypeVar("_ElementT", bound="LazyObject")
_SlotT = TypeVar("_SlotT", bound="LazySlot")
_ContainerT = TypeVar("_ContainerT", bound="LazyContainer")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_DescriptorGetT = TypeVar("_DescriptorGetT")
_DescriptorSetT = TypeVar("_DescriptorSetT")
_DescriptorT = TypeVar("_DescriptorT", bound="LazyDescriptor")
_Parameters = ParamSpec("_Parameters")
#_PropertyParameters = ParamSpec("_PropertyParameters")
_AnnotationT = Any


class TreeNode(Generic[_TreeNodeContentT]):
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
    ):
        self._children = nodes
        return self

    def apply_deepest(
        self,
        callback: "Callable[[_TreeNodeContentT], TreeNode[_TreeNodeContentT]]"
    ) -> "TreeNode[_TreeNodeContentT]":
        if (children := self._children) is None:
            return callback(self._content)
        return TreeNode(self._content).bind_only(*(
            child.apply_deepest(callback)
            for child in children
        ))

    #def transform(
    #    self,
    #    callback: Callable[[_TreeNodeContentT], _T]
    #) -> "TreeNode[_T]":
    #    result = TreeNode(callback(self._content))
    #    if (children := self._children) is not None:
    #        result.bind_only(*(
    #            child.transform(callback)
    #            for child in children
    #        ))
    #    return result


class LazyContainer(ABC, Generic[_ElementT]):
    __slots__ = (
        "__weakref__",
        "_variable_slot_backref"
    )

    def __init__(self) -> None:
        super().__init__()
        self._variable_slot_backref: weakref.ref[LazyVariableSlot] | None = None

    def _get_writable_slot_with_verification(self) -> "LazyVariableSlot":
        assert (slot_ref := self._variable_slot_backref) is not None
        assert (slot := slot_ref()) is not None
        assert slot._is_writable
        return slot

    @abstractmethod
    def _iter_elements(self) -> Iterator[_ElementT]:
        pass

    @abstractmethod
    def _write(
        self,
        src: "LazyContainer[_ElementT]"
    ) -> None:
        pass

    @abstractmethod
    def _copy_container(self) -> "LazyContainer[_ElementT]":
        pass


class LazyUnitaryContainer(LazyContainer[_ElementT]):
    __slots__ = ("_element",)

    def __init__(
        self,
        element: _ElementT
    ) -> None:
        super().__init__()
        self._element: _ElementT = element

    def _iter_elements(self) -> Iterator[_ElementT]:
        yield self._element

    def _write(
        self,
        src: "LazyUnitaryContainer[_ElementT]"
    ) -> None:
        # TODO: shall we copy expired property slots to maintain references?
        self._element = src._element

    def _copy_container(self) -> "LazyUnitaryContainer[_ElementT]":
        return LazyUnitaryContainer(
            element=self._element
        )


class LazyDynamicContainer(LazyContainer[_ElementT]):
    __slots__ = ("_elements",)

    def __init__(
        self,
        elements: Iterable[_ElementT]
    ) -> None:
        super().__init__()
        self._elements: list[_ElementT] = list(elements)

    def _iter_elements(self) -> Iterator[_ElementT]:
        yield from self._elements

    def _write(
        self,
        src: "LazyDynamicContainer[_ElementT]"
    ) -> None:
        self._elements = list(src._elements)

    def _copy_container(self) -> "LazyDynamicContainer[_ElementT]":
        return LazyDynamicContainer(
            elements=self._elements
        )

    # list methods
    # Only those considered necessary are ported.

    @staticmethod
    def force_expire(
        method: "Callable[Concatenate[LazyDynamicContainer[_ElementT], _Parameters], _T]",
    ) -> "Callable[Concatenate[LazyDynamicContainer[_ElementT], _Parameters], _T]":

        @wraps(method)
        def new_method(
            self: "LazyDynamicContainer[_ElementT]",
            *args: _Parameters.args,
            **kwargs: _Parameters.kwargs
        ) -> _T:
            slot = self._get_writable_slot_with_verification()
            slot.expire_property_slots()
            return method(self, *args, **kwargs)

        return new_method

    def __len__(self) -> int:
        return self._elements.__len__()

    def __iter__(self) -> Iterator[_ElementT]:
        return self._elements.__iter__()

    def __contains__(
        self,
        element: _ElementT
    ) -> bool:
        return self._elements.__contains__(element)

    def __reversed__(self) -> Iterator[_ElementT]:
        return self._elements.__reversed__()

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

    def index(
        self,
        value: _ElementT,
        start: int = 0,
        stop: int = sys.maxsize
    ) -> int:
        return self._elements.index(value, start, stop)

    def count(
        self,
        value: _ElementT
    ) -> int:
        return self._elements.count(value)

    @force_expire
    def insert(
        self,
        index: int,
        value: _ElementT
    ) -> None:
        return self._elements.insert(index, value)

    @force_expire
    def append(
        self,
        value: _ElementT
    ) -> None:
        return self._elements.append(value)

    @force_expire
    def extend(
        self,
        values: Iterable[_ElementT]
    ) -> None:
        return self._elements.extend(values)

    @force_expire
    def reverse(self) -> None:
        return self._elements.reverse()

    @force_expire
    def pop(
        self,
        index: int = -1
    ) -> _ElementT:
        return self._elements.pop(index)

    @force_expire
    def remove(
        self,
        value: _ElementT
    ) -> None:
        return self._elements.remove(value)

    @force_expire
    def clear(self) -> None:
        return self._elements.clear()

    # For convenience.
    @force_expire
    def eliminate(
        self,
        values: Iterable[_ElementT]
    ) -> None:
        for value in values:
            self._elements.remove(value)

    @force_expire
    def reset(
        self,
        values: Iterable[_ElementT]
    ) -> None:
        self._elements.clear()
        self._elements.extend(values)

    #def add(
    #    self,
    #    *elements: _ElementT
    #):
    #    slot = self._get_writable_slot_with_verification()
    #    if not elements:
    #        return self
    #    for element in elements:
    #        if element in self._elements:
    #            continue
    #        self._elements.append(element)
    #    slot.expire_property_slots()
    #    return self

    #def discard(
    #    self,
    #    *elements: _ElementT
    #):
    #    slot = self._get_writable_slot_with_verification()
    #    if not elements:
    #        return self
    #    for element in elements:
    #        if element not in self._elements:
    #            continue
    #        self._elements.remove(element)
    #    slot.expire_property_slots()
    #    return self


class LazySlot(ABC, Generic[_ContainerT]):
    __slots__ = ("__weakref__",)

    @abstractmethod
    def copy_from(
        self,
        src: "LazySlot[_ContainerT]"
    ) -> None:
        pass


class LazyVariableSlot(LazySlot[_ContainerT]):
    __slots__ = (
        "_container",
        "_linked_property_slots",
        "_is_writable"
    )

    def __init__(
        self,
        container: _ContainerT
    ) -> None:
        super().__init__()
        container._variable_slot_backref = weakref.ref(self)
        self._container: _ContainerT = container
        self._linked_property_slots: weakref.WeakSet[LazyPropertySlot] = weakref.WeakSet()
        self._is_writable: bool = True

    def get_variable_container(self) -> _ContainerT:
        return self._container

    def set_variable_container(
        self,
        container: _ContainerT
    ) -> None:
        self._container._write(container)

    def copy_from(
        self,
        src: "LazyVariableSlot[_ContainerT]"
    ) -> None:
        self.set_variable_container(src.get_variable_container())
        self._linked_property_slots.clear()
        self._is_writable = src._is_writable

    def expire_property_slots(self) -> None:
        for expired_property_slot in self._linked_property_slots:
            expired_property_slot.expire()
        self._linked_property_slots.clear()

    def yield_descendant_variable_slots(self) -> "Iterator[LazyVariableSlot]":
        yield self
        for element in self.get_variable_container()._iter_elements():
            for slot in element._iter_variable_slots():
                yield from slot.yield_descendant_variable_slots()

    def make_readonly(self) -> None:
        for variable_slot in self.yield_descendant_variable_slots():
            variable_slot._is_writable = False


class LazyPropertySlot(LazySlot[_ContainerT]):
    __slots__ = (
        "_container",
        "_linked_variable_slots",
        "_is_expired"
    )

    def __init__(self) -> None:
        super().__init__()
        self._container: _ContainerT | None = None
        self._linked_variable_slots: weakref.WeakSet[LazyVariableSlot] = weakref.WeakSet()
        self._is_expired: bool = True

    def get_property_container(self) -> _ContainerT | None:
        return self._container

    def set_property_container(
        self,
        container: _ContainerT | None
    ) -> None:
        if container is None:
            self._container = None
            return
        if self._container is not None:
            self._container._write(container)
        else:
            self._container = container  # TODO

    def copy_from(
        self,
        src: "LazyPropertySlot[_ContainerT]"
    ) -> None:
        # TODO: Shall we keep references to objects under property slots?
        #self.set_property_container(src.get_property_container())  # TODO
        if src._is_expired:
            self.set_property_container(src.get_property_container())
        else:
            self.set_property_container(None)
        self._linked_variable_slots.clear()
        self._is_expired = True

    def expire(self) -> None:
        self._linked_variable_slots.clear()
        self._is_expired = True

    def bind_linked_variable_slots(
        self,
        *linked_variable_slots: LazyVariableSlot
    ) -> None:
        for linked_variable_slot in linked_variable_slots:
            linked_variable_slot._linked_property_slots.add(self)
        self._linked_variable_slots.update(linked_variable_slots)


#class LazyMode(Enum):
#    INDIVIDUAL = 0
#    COLLECTION = 1
#    EXTERNAL = 2
#    #UNSHARED = 2
#    #SHARED = 3

#    @property
#    def is_unitary(self) -> bool:
#        return self != LazyMode.COLLECTION

#    @property
#    def is_internal(self) -> bool:
#        return self != LazyMode.EXTERNAL


class LazyConverter(ABC, Generic[_ContainerT, _DescriptorGetT, _DescriptorSetT]):
    __slots__ = (
        #"lazy_mode",
        #"name",
        #"instance_to_slot_dict",
    )

    #def __init__(self) -> None:
    #    super().__init__()
    #    #self.lazy_mode: LazyMode = NotImplemented
    #    #self.name: str = NotImplemented

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


class LazyDescriptor(ABC, Generic[_InstanceT, _SlotT, _ContainerT, _ElementT, _DescriptorGetT, _DescriptorSetT]):
    __slots__ = (
        "element_type",
        "return_annotation",
        "converter",
        "instance_to_slot_dict"
    )

    _converter_class: type[LazyConverter[_ContainerT, _DescriptorGetT, _DescriptorSetT]] = LazyConverter

    def __init__(self) -> None:
        super().__init__()
        self.element_type: type[_ElementT] = NotImplemented
        self.return_annotation: _AnnotationT = NotImplemented
        self.converter: LazyConverter[_ContainerT, _DescriptorGetT, _DescriptorSetT] = type(self)._converter_class()
        self.instance_to_slot_dict: weakref.WeakKeyDictionary[_InstanceT, _SlotT] = weakref.WeakKeyDictionary()

    @overload
    def __get__(
        self: _DescriptorT,
        instance: None,
        owner: type[_InstanceT] | None = None
    ) -> _DescriptorT: ...  # TODO: typing

    @overload
    def __get__(
        self,
        instance: _InstanceT,
        owner: type[_InstanceT] | None = None
    ) -> _DescriptorGetT: ...

    def __get__(
        self: _DescriptorT,
        instance: _InstanceT | None,
        owner: type[_InstanceT] | None = None
    ) -> _DescriptorT | _DescriptorGetT:
        if instance is None:
            return self
        return self.converter.convert_get(self.get_container(instance))

    def __set__(
        self,
        instance: _InstanceT,
        new_value: _DescriptorSetT
    ) -> None:
        self.set_container(instance, self.converter.convert_set(new_value))

    def __delete__(
        self,
        instance: _InstanceT
    ) -> None:
        raise TypeError("Cannot delete attributes of a lazy object")

    @abstractmethod
    def get_container(
        self,
        instance: _InstanceT
    ) -> _ContainerT:
        pass

    @abstractmethod
    def set_container(
        self,
        instance: _InstanceT,
        new_container: _ContainerT
    ) -> None:
        pass

    @abstractmethod
    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        pass

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
    _InstanceT, LazyVariableSlot[_ContainerT], _ContainerT, _ElementT, _DescriptorGetT, _DescriptorSetT
]):
    __slots__ = (
        "method",
        "default_container"
    )

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], _DescriptorSetT]
    ) -> None:
        super().__init__()
        self.method: Callable[[type[_InstanceT]], _DescriptorSetT] = method
        self.default_container: _ContainerT | None = None

    def get_container(
        self,
        instance: _InstanceT
    ) -> _ContainerT:
        return self.get_slot(instance).get_variable_container()

    def set_container(
        self,
        instance: _InstanceT,
        new_container: _ContainerT
    ) -> None:
        slot = self.get_slot(instance)
        assert slot._is_writable
        old_container = slot.get_variable_container()
        if tuple(old_container._iter_elements()) == tuple(new_container._iter_elements()):
            return
        slot.expire_property_slots()
        slot.set_variable_container(new_container)

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        if (default_container := self.default_container) is None:
            default_container = self.converter.convert_set(self.method(type(instance)))
            self.default_container = default_container
        self.set_slot(instance, LazyVariableSlot(
            container=default_container._copy_container()
        ))


# TODO: Add LazyParameter class


class LazyPropertyDescriptor(LazyDescriptor[
    _InstanceT, LazyPropertySlot[_ContainerT], _ContainerT, _ElementT, _DescriptorGetT, _DescriptorSetT
]):
    __slots__ = (
        "method",
        "finalize_method",
        "descriptor_name_chains",
        "requires_unwrapping_tuple",
        #"descriptor_overloading_chains",
        "instance_to_key_dict",
        "key_to_container_dict"
    )

    def __init__(
        self,
        #lazy_mode: LazyMode,
        method: Callable[Concatenate[type[_InstanceT], _Parameters], _DescriptorSetT]
        #parameter_name_chains: tuple[tuple[str, ...], ...],
        #requires_unwrapping_tuple: tuple[bool, ...]
    ) -> None:
        super().__init__()
        self.method: Callable[Concatenate[type[_InstanceT], _Parameters], _DescriptorSetT] = method
        self.finalize_method: Callable[[type[_InstanceT], _DescriptorGetT], None] | None = None
        self.descriptor_name_chains: tuple[tuple[str, ...], ...] = NotImplemented
        self.requires_unwrapping_tuple: tuple[bool, ...] = NotImplemented
        #self.descriptor_overloading_chains: tuple[tuple[LazyDescriptorOverloading, ...], ...] = NotImplemented
        self.instance_to_key_dict: weakref.WeakKeyDictionary[_InstanceT, tuple] = weakref.WeakKeyDictionary()
        self.key_to_container_dict: weakref.WeakValueDictionary[tuple, _ContainerT] = weakref.WeakValueDictionary()

    def get_container(
        self,
        instance: _InstanceT
    ) -> _ContainerT:

        def construct_parameter_item(
            descriptor_name_chain: tuple[str, ...],
            instance: _InstanceT
        ) -> tuple[TreeNode[LazyObject], list[LazyVariableSlot]]:

            def construct_parameter_tree(
                descriptor_name: str,
                is_chain_tail: bool,
                linked_variable_slots: list[LazyVariableSlot]
            ) -> Callable[[LazyObject], TreeNode[LazyObject]]:

                def callback(
                    obj: LazyObject
                ) -> TreeNode[LazyObject]:
                    #obj, binding_completed = obj_tagged
                    descriptor = type(obj)._lazy_descriptor_dict[descriptor_name]
                    slot = descriptor.get_slot(obj)
                    container = descriptor.get_container(obj)
                    #if not binding_completed:
                    if isinstance(slot, LazyVariableSlot):
                        if is_chain_tail:
                            linked_variable_slots.extend(slot.yield_descendant_variable_slots())
                        else:
                            linked_variable_slots.append(slot)
                    elif isinstance(slot, LazyPropertySlot):
                        #binding_completed = True
                        linked_variable_slots.extend(slot._linked_variable_slots)
                    else:
                        raise TypeError
                    if isinstance(container, LazyUnitaryContainer):
                        result = TreeNode(container._element)
                    elif isinstance(container, LazyDynamicContainer):
                        result = TreeNode(obj).bind_only(*(
                            TreeNode(element) for element in container._elements
                        ))
                    else:
                        raise TypeError
                    return result

                return callback

            #def remove_tag(
            #    obj_tagged: tuple[LazyObject, bool]
            #) -> LazyObject:
            #    obj, _ = obj_tagged
            #    return obj

            parameter_tree: TreeNode[LazyObject] = TreeNode(instance)
            linked_variable_slots: list[LazyVariableSlot] = []
            for reversed_index, descriptor_name in reversed(list(zip(it.count(), reversed(descriptor_name_chain)))):
                parameter_tree = parameter_tree.apply_deepest(
                    construct_parameter_tree(descriptor_name, not reversed_index, linked_variable_slots)
                )
            return parameter_tree, [
                linked_variable_slot for linked_variable_slot in linked_variable_slots
                if linked_variable_slot._is_writable
            ]

        def expand_dependencies(
            obj: LazyObject
        ) -> TreeNode[LazyObject]:
            result = TreeNode(obj)
            if not isinstance(obj, LazyWrapper):
                result.bind_only(*(
                    TreeNode(obj).bind_only(*(
                        expand_dependencies(element)
                        for element in variable_slot.get_variable_container()._iter_elements()
                    ))
                    for variable_slot in obj._iter_variable_slots()
                ))
            return result

        def construct_parameter_key(
            tree_node: TreeNode[_TreeNodeContentT]
        ) -> Hashable:
            if tree_node._children is None:
                assert (isinstance(content := tree_node._content, LazyWrapper))
                return content._hash_value
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
        if not slot._is_expired:
            container = slot.get_property_container()
            assert container is not None
        else:
            parameter_items = tuple(
                construct_parameter_item(descriptor_name_chain, instance)
                for descriptor_name_chain in self.descriptor_name_chains
            )
            slot.bind_linked_variable_slots(*it.chain.from_iterable(
                linked_variable_slots
                for _, linked_variable_slots in parameter_items
            ))
            parameter_trees = tuple(
                parameter_tree
                for parameter_tree, _ in parameter_items
            )
            key = tuple(
                construct_parameter_key(parameter_tree.apply_deepest(expand_dependencies))
                for parameter_tree in parameter_trees
            )
            self.instance_to_key_dict[instance] = key

            if (container := self.key_to_container_dict.get(key)) is None:
                parameters = tuple(
                    construct_parameter(parameter_tree, requires_unwrapping=requires_unwrapping)
                    for parameter_tree, requires_unwrapping in zip(
                        parameter_trees, self.requires_unwrapping_tuple, strict=True
                    )
                )
                container = self.converter.convert_set(self.method(type(instance), *parameters))
                for element in container._iter_elements():
                    for variable_slot in element._iter_variable_slots():
                        variable_slot.make_readonly()
                self.key_to_container_dict[key] = container

                if (finalize_method := self.finalize_method) is not None:
                    weakref.finalize(container, finalize_method, type(instance), self.converter.convert_get(container))

            slot.set_property_container(container)
            slot._is_expired = False
        return container

    def set_container(
        self,
        instance: _InstanceT,
        new_container: _ContainerT
    ) -> None:
        raise ValueError("Attempting to set a writable property")

    def initialize(
        self,
        instance: _InstanceT
    ) -> None:
        self.set_slot(instance, LazyPropertySlot())


#class LazyDescriptorOverloading(ABC, Generic[_InstanceT, _ElementT, _DescriptorT]):
#    __slots__ = (
#        "element_type",
#        "descriptors"
#    )

#    def __init__(
#        self,
#        instance_type: type[_InstanceT],
#        descriptor: _DescriptorT
#    ) -> None:
#        super().__init__()
#        self.element_type: type[_ElementT] = descriptor.element_type
#        self.descriptors: dict[type[_InstanceT], _DescriptorT] = {instance_type: descriptor}

#    def get_descriptor(
#        self,
#        instance_type: type[_InstanceT]
#    ) -> _DescriptorT:
#        return self.descriptors[instance_type]

#    def set_descriptor(
#        self,
#        instance_type: type[_InstanceT],
#        descriptor: _DescriptorT
#    ) -> None:
#        assert issubclass(descriptor.element_type, self.element_type)
#        assert instance_type not in self.descriptors
#        self.descriptors[instance_type] = descriptor


#@final
#class LazyUnitaryDescriptorOverloading(LazyDescriptorOverloading[_InstanceT, _ElementT, Union[
#    LazyUnitaryVariableDescriptor[_InstanceT, _ElementT, _DescriptorSetT],
#    LazyUnitaryPropertyDescriptor[_InstanceT, _ElementT, _DescriptorSetT]
#]]):
#    __slots__ = ()


#@final
#class LazyDynamicDescriptorOverloading(LazyDescriptorOverloading[_InstanceT, _ElementT, Union[
#    LazyDynamicVariableDescriptor[_InstanceT, _ElementT, _DescriptorSetT],
#    LazyDynamicPropertyDescriptor[_InstanceT, _ElementT, _DescriptorSetT]
#]]):
#    __slots__ = ()


class LazyObject(ABC):
    __slots__ = ("__weakref__",)

    #_subclasses: "ClassVar[dict[str, type[LazyObject]]]" = {}
    _lazy_descriptor_dict: ClassVar[dict[str, LazyDescriptor]] = {}
    _lazy_descriptors: ClassVar[tuple[LazyDescriptor, ...]] = ()
    #_LAZY_VARIABLE_DESCRIPTORS: ClassVar[tuple[LazyVariableDescriptor, ...]]
    #_LAZY_DESCRIPTOR_OVERLOADING_DICT: ClassVar[dict[str, LazyDescriptorOverloading]]
    _py_slots: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        def complete_descriptor_info(
            #name: str,
            descriptor: LazyVariableDescriptor | LazyPropertyDescriptor,
            method_signature: inspect.Signature,
            overridden_descriptor: LazyDescriptor | None
        ) -> None:
            #method = descriptor.method
            #assert name == method.__name__

            return_annotation: _AnnotationT = method_signature.return_annotation
            #if isinstance(return_annotation, str):
            #    if return_annotation == cls.__name__:
            #        return_annotation = cls
            #    elif return_annotation == f"list[{cls.__name__}]":
            #        return_annotation = list[cls]
            #    else:
            #        raise ValueError
            descriptor.return_annotation = return_annotation
            #lazy_mode = converter.lazy_mode
            converter_class = descriptor._converter_class
            if not issubclass(converter_class, LazyExternalConverter):
                element_type = return_annotation
                if issubclass(converter_class, LazyCollectionConverter):
                    assert element_type.__origin__ is list
                    element_type = element_type.__args__[0]
                    #assert (match_obj := re.fullmatch(r"list\[(.+)\]", element_type_str)) is not None
                    #element_type_str = match_obj.group(1)
                #element_type = cls._subclasses[element_type_str]
            else:
                element_type = LazyWrapper
            assert issubclass(element_type, LazyObject)
            descriptor.element_type = element_type

            if overridden_descriptor is not None:
                #overridden_converter = overridden_descriptor.converter
                #print(name, type(converter), type(overridden_converter))
                #assert type(converter) is type(overridden_converter)
                assert issubclass(element_type, overridden_descriptor.element_type)
                assert return_annotation == overridden_descriptor.return_annotation or \
                    issubclass(return_annotation, overridden_descriptor.return_annotation)

        def complete_property_descriptor_info(
            property_descriptor: LazyPropertyDescriptor,
            method_signature: inspect.Signature,
            root_cls: type[LazyObject]
        ) -> None:
            descriptor_name_chain_list: list[tuple[str, ...]] = []
            requires_unwrapping_list: list[bool] = []
            for name, parameter in method_signature.parameters.items():
                if name == "cls":
                    continue
                if (requires_unwrapping := re.fullmatch(r"_\w+_", name) is None):
                    name = f"_{name}_"
                descriptor_name_chain = tuple(re.findall(r"_\w+?_(?=_|$)", name))
                assert "".join(descriptor_name_chain) == name
                descriptor_name_chain_list.append(descriptor_name_chain)
                requires_unwrapping_list.append(requires_unwrapping)

                #descriptor: LazyDescriptor = NotImplemented
                element_type: type[LazyObject] = root_cls
                collection_level: int = 0
                for descriptor_name in descriptor_name_chain[:-1]:
                    descriptor = element_type._lazy_descriptor_dict[descriptor_name]
                    converter_class = descriptor._converter_class
                    assert not issubclass(converter_class, LazyExternalConverter)
                    if issubclass(converter_class, LazyCollectionConverter):
                        collection_level += 1
                    element_type = descriptor.element_type
                descriptor = element_type._lazy_descriptor_dict[descriptor_name_chain[-1]]
                if requires_unwrapping:
                    assert descriptor.element_type is LazyWrapper
                expected_annotation = descriptor.return_annotation
                for _ in range(collection_level):
                    expected_annotation = list[expected_annotation]
                assert expected_annotation == parameter.annotation

            property_descriptor.descriptor_name_chains = tuple(descriptor_name_chain_list)
            property_descriptor.requires_unwrapping_tuple = tuple(requires_unwrapping_list)

        base_classes = tuple(
            base_cls
            for base_cls in reversed(cls.__mro__)
            if issubclass(base_cls, LazyObject)
        )
        descriptor_dict = {
            name: descriptor
            for base_cls in base_classes
            for name, descriptor in base_cls._lazy_descriptor_dict.items()
        }
        new_descriptor_items = tuple(
            (name, descriptor, inspect.signature(descriptor.method, locals={cls.__name__: cls}, eval_str=True))
            for name, descriptor in cls.__dict__.items()
            if isinstance(descriptor, LazyVariableDescriptor | LazyPropertyDescriptor)
        )
        #for base_cls in reversed(cls.__mro__):
        #    #base_cls = cls.__bases__
        #    if not issubclass(base_cls, LazyObject):
        #        continue
        #    descriptor_dict.update({
        #        descriptor.name: descriptor
        #        for descriptor in base_cls._lazy_descriptors
        #    })
        #    #if base_cls is not LazyObject:
        #    #    descriptor_overloading_dict = base_cls._LAZY_DESCRIPTOR_OVERLOADING_DICT.copy()
        #    #else:
        #    #    descriptor_overloading_dict = {}

        #overridden_descriptor_overloadings: list[LazyDescriptorOverloading] = []
        for name, descriptor, method_signature in new_descriptor_items:
            #if not isinstance(attr, LazyDescriptor):
            #    continue
            #assert isinstance(attr, LazyVariableDescriptor | LazyPropertyDescriptor)
            assert re.fullmatch(r"_\w+_", name)
            complete_descriptor_info(
                descriptor=descriptor,
                method_signature=method_signature,
                overridden_descriptor=descriptor_dict.get(name)
            )
            #element_type, return_annotation = get_element_type_and_return_annotation(
            #    annotation=,
            #    lazy_mode=attr.lazy_mode
            #)
            descriptor_dict[name] = descriptor
            #if attr.element_type is NotImplemented:
            #    attr.element_type = cls
            #if (descriptor_overloading := descriptor_overloading_dict.get(name)) is not None:
            #    if isinstance(descriptor_overloading, LazyUnitaryDescriptorOverloading):
            #        assert isinstance(attr, LazyUnitaryVariableDescriptor | LazyUnitaryPropertyDescriptor)
            #        descriptor_overloading.set_descriptor(cls, attr)
            #    elif isinstance(descriptor_overloading, LazyDynamicDescriptorOverloading):
            #        assert isinstance(attr, LazyDynamicVariableDescriptor | LazyDynamicPropertyDescriptor)
            #        descriptor_overloading.set_descriptor(cls, attr)
            #    else:
            #        raise TypeError
            #else:
            #    if isinstance(attr, LazyUnitaryVariableDescriptor | LazyUnitaryPropertyDescriptor):
            #        descriptor_overloading = LazyUnitaryDescriptorOverloading(cls, attr)
            #    elif isinstance(attr, LazyDynamicVariableDescriptor | LazyDynamicPropertyDescriptor):
            #        descriptor_overloading = LazyDynamicDescriptorOverloading(cls, attr)
            #    else:
            #        raise TypeError
            #    descriptor_overloading_dict[name] = descriptor_overloading
            #overridden_descriptor_overloadings.append(descriptor_overloading)

        #lazy_descriptors: list[LazyDescriptor] = []
        #for descriptor_overloading in descriptor_overloading_dict.values():
        #    if descriptor_overloading not in overridden_descriptor_overloadings:
        #        descriptor_overloading.set_descriptor(cls, descriptor_overloading.get_descriptor(base_cls))
        #    lazy_descriptors.append(descriptor_overloading.get_descriptor(cls))

        cls._lazy_descriptor_dict = descriptor_dict
        cls._lazy_descriptors = tuple(descriptor_dict.values())
        #cls._LAZY_VARIABLE_DESCRIPTORS = tuple(
        #    descriptor for descriptor in lazy_descriptors
        #    if isinstance(descriptor, LazyVariableDescriptor)
        #)
        #cls._LAZY_DESCRIPTOR_OVERLOADING_DICT = descriptor_overloading_dict
        # Use dict.fromkeys to preserve order (by first occurrance).
        cls._py_slots = tuple(dict.fromkeys(
            slot
            for base_cls in base_classes
            for slot in base_cls.__slots__
            if slot != "__weakref__"
        ))

        for _, descriptor, method_signature in new_descriptor_items:
            if not isinstance(descriptor, LazyPropertyDescriptor):
                continue
            complete_property_descriptor_info(
                property_descriptor=descriptor,
                method_signature=method_signature,
                root_cls=cls
            )
            #descriptor_overloading_chains: list[tuple[LazyDescriptorOverloading, ...]] = []
            #for parameter_name_chain in attr.parameter_name_chains:
            #    element_type = cls
            #    descriptor_overloading_chain: list[LazyDescriptorOverloading] = []
            #    for parameter_name in parameter_name_chain:
            #        descriptor_overloading = element_type._LAZY_DESCRIPTOR_OVERLOADING_DICT[parameter_name]
            #        element_type = descriptor_overloading.element_type
            #        descriptor_overloading_chain.append(descriptor_overloading)
            #    descriptor_overloading_chains.append(tuple(descriptor_overloading_chain))
            #attr.descriptor_overloading_chains = tuple(descriptor_overloading_chains)

    def __init__(self) -> None:
        super().__init__()
        self._initialize_descriptors()

    def _initialize_descriptors(self) -> None:
        cls = type(self)
        for descriptor in cls._lazy_descriptors:
            descriptor.initialize(self)

    def _becomes(
        self: _ElementT,
        src: _ElementT
    ) -> None:
        # Two instances can "become" each other if the type of one is inherited from that of another,
        # and the inheritance does not commit any addition or overloading of lazy descriptors,
        # or addition of slots.
        assert (descriptors := type(self)._lazy_descriptors) == type(src)._lazy_descriptors
        for descriptor in descriptors:
            descriptor.get_slot(self).copy_from(descriptor.get_slot(src))
        assert (slot_names := type(self)._py_slots) == type(src)._py_slots
        for slot_name in slot_names:
            src_value = copy.copy(src.__getattribute__(slot_name))
            # TODO: This looks like a temporary patch... Is there any better practice?
            if isinstance(src_value, weakref.WeakSet):
                # Use `WeakSet.update` instead of `copy.copy` for better behavior.
                dst_value = src_value.copy()
            else:
                dst_value = copy.copy(src_value)
            self.__setattr__(slot_name, dst_value)

    def _copy(self: _ElementT) -> _ElementT:
        cls = type(self)
        result = cls.__new__(cls)
        result._initialize_descriptors()
        result._becomes(self)
        return result

    def _iter_variable_slots(self) -> Iterator[LazyVariableSlot]:
        return (
            descriptor.get_slot(self)
            for descriptor in type(self)._lazy_descriptors
            if isinstance(descriptor, LazyVariableDescriptor)
        )


class LazyWrapper(LazyObject, Generic[_T]):
    __slots__ = (
        "_value",
        "_hash_value"
    )

    _hash_counter: ClassVar[int] = 1

    def __init__(
        self,
        value: _T
    ) -> None:
        super().__init__()
        cls = type(self)
        self._value: _T = value
        self._hash_value: int = cls._hash_counter  # Unique for each instance.
        cls._hash_counter += 1

    @property
    def value(self) -> _T:
        return self._value

    def _becomes(
        self: _ElementT,
        src: _ElementT
    ) -> None:
        raise ValueError


class LazyIndividualConverter(LazyConverter[
    LazyUnitaryContainer[_ElementT], _ElementT, _ElementT
]):
    __slots__ = ()

    def convert_get(
        self,
        container: LazyUnitaryContainer[_ElementT]
    ) -> _ElementT:
        return container._element

    def convert_set(
        self,
        new_value: _ElementT
    ) -> LazyUnitaryContainer[_ElementT]:
        return LazyUnitaryContainer(
            element=new_value
        )


class LazyCollectionConverter(LazyConverter[
    LazyDynamicContainer[_ElementT], LazyDynamicContainer[_ElementT], Iterable[_ElementT]
]):
    __slots__ = ()

    def convert_get(
        self,
        container: LazyDynamicContainer[_ElementT]
    ) -> LazyDynamicContainer[_ElementT]:
        return container

    def convert_set(
        self,
        new_value: Iterable[_ElementT]
    ) -> LazyDynamicContainer[_ElementT]:
        return LazyDynamicContainer(
            elements=new_value
        )


class LazyExternalConverter(LazyConverter[
    LazyUnitaryContainer[LazyWrapper[_T]], LazyWrapper[_T], _T | LazyWrapper[_T]
]):
    __slots__ = ()

    def convert_get(
        self,
        container: LazyUnitaryContainer[_ElementT]
    ) -> _ElementT:
        return container._element

    def convert_set(
        self,
        new_value: _T | LazyWrapper[_T]
    ) -> LazyUnitaryContainer[LazyWrapper[_T]]:
        if not isinstance(new_value, LazyWrapper):
            new_value = LazyWrapper(new_value)
        return LazyUnitaryContainer(
            element=new_value
        )


class LazySharedConverter(LazyExternalConverter[_HT]):
    __slots__ = ("content_to_element_dict",)

    def __init__(self) -> None:
        super().__init__()
        self.content_to_element_dict: weakref.WeakValueDictionary[_HT, LazyWrapper[_HT]] = weakref.WeakValueDictionary()

    def convert_set(
        self,
        new_value: _HT | LazyWrapper[_HT]
    ) -> LazyUnitaryContainer[LazyWrapper[_HT]]:
        if not isinstance(new_value, LazyWrapper) and (cached_value := self.content_to_element_dict.get(new_value)) is None:
            cached_value = LazyWrapper(new_value)
            self.content_to_element_dict[new_value] = cached_value
            new_value = cached_value
        return super().convert_set(new_value)
        #if isinstance(new_value, LazyWrapper):
        #    value = new_value
        #else:
        #    
        #return LazyUnitaryContainer(
        #    element=value
        #)
