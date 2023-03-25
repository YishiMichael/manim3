"""
This module implements lazy evaluation based on weak reference.

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

+------------+-------------------+-------------------+--------------------+
| LazyMode   | method return     | __get__ return    | __set__ type       |
+------------+-------------------+-------------------+--------------------+
| OBJECT     | LazyObjectT       | LazyObjectT       | LazyObjectT        |
| UNWRAPPED  | T                 | LazyWrapper[T]    | T | LazyWrapper[T] |
| SHARED     | T (Hashable)      | LazyWrapper[T]    | T                  |
| COLLECTION | LazyCollectionT * | LazyCollectionT * | LazyCollectionT *  |
+------------+-------------------+-------------------+--------------------+
*: Abbreviation of `LazyDynamicContainer[LazyObjectT]`.

The `__get__` method always returns an instance of either `LazyObject` or
`LazyDynamicContainer`, the latter of which is just a dynamic collection of
`LazyObject`s, and provides `add`, `remove` as its public interface.
`LazyWrapper` is derived from `LazyObject`, which is just responsible for
bringing a value into the lazy scope, and the value is obtained via the
readonly `value` property. One may picture a lazy object as a tree (it's a
DAG really), where `LazyWrapper`s sit on all leaves.

Lazy variables are of course mutable. All can be modified via `__set__`.
Among all cases above, a common value will be shared among instances, except
for providing `T` type in `UNWRAPPED` mode, in which case a new `LazyWrapper`
object will be instanced and assigned specially to the instance. Additionally,
`LazyDynamicContainer`s can be modified via `add` and `remove`.

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

+------------+-------------------+-------------------+
| LazyMode   | method return     | __get__ return    |
+------------+-------------------+-------------------+
| OBJECT     | LazyObject        | LazyObject        |
| UNWRAPPED  | T                 | LazyWrapper[T]    |
| SHARED     | T (Hashable)      | LazyWrapper[T]    |
| COLLECTION | LazyCollectionT * | LazyCollectionT * |
+------------+-------------------+-------------------+
*: Abbreviation of `LazyDynamicContainer[LazyObjectT]`.

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


__all__ = [
    "LazyDynamicContainer",
    "LazyDynamicPropertyDescriptor",
    "LazyDynamicVariableDescriptor",
    "LazyObject",
    "LazyUnitaryContainer",
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
    final,
    overload
)
import weakref


_T = TypeVar("_T")
_TreeNodeContentT = TypeVar("_TreeNodeContentT", bound=Hashable)
_ElementT = TypeVar("_ElementT", bound="LazyObject")
_SlotT = TypeVar("_SlotT", bound="LazySlot")
_ContainerT = TypeVar("_ContainerT", bound="LazyContainer")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_DescriptorGetT = TypeVar("_DescriptorGetT")
_DescriptorSetT = TypeVar("_DescriptorSetT")
_DescriptorT = TypeVar("_DescriptorT", bound="LazyDescriptor")
_PropertyParameters = ParamSpec("_PropertyParameters")


@final
class TreeNode(ABC, Generic[_TreeNodeContentT]):
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
    def _iter_elements(self) -> Generator[_ElementT, None, None]:
        pass

    @abstractmethod
    def _write(
        self,
        src: "LazyContainer[_ElementT]"
    ) -> None:
        pass

    @abstractmethod
    def _copy(self) -> "LazyContainer[_ElementT]":
        pass


@final
class LazyUnitaryContainer(LazyContainer[_ElementT]):
    __slots__ = ("_element",)

    def __init__(
        self,
        element: _ElementT
    ) -> None:
        super().__init__()
        self._element: _ElementT = element

    def _iter_elements(self) -> Generator[_ElementT, None, None]:
        yield self._element

    def _write(
        self,
        src: "LazyUnitaryContainer[_ElementT]"
    ) -> None:
        # TODO: shall we copy expired property slots to maintain references?
        self._element = src._element

    def _copy(self) -> "LazyUnitaryContainer[_ElementT]":
        return LazyUnitaryContainer(
            element=self._element
        )


@final
class LazyDynamicContainer(LazyContainer[_ElementT]):
    __slots__ = ("_elements",)

    def __init__(
        self,
        elements: Iterable[_ElementT]
    ) -> None:
        super().__init__()
        self._elements: list[_ElementT] = list(elements)

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

    def _write(
        self,
        src: "LazyDynamicContainer[_ElementT]"
    ) -> None:
        self._elements = src._elements
        #print(self is src)
        #self._elements.clear()
        #self._elements.extend(src._elements)
        #dst_len = len(self)
        #src_len = len(src)
        #if dst_len < src_len:
        #    self._elements += src._elements[dst_len:]
        #elif dst_len > src_len:
        #    self._elements = self._elements[:src_len]
        #for dst_element, src_element in zip(self._elements, src._elements, strict=True):
        #    if dst_element is src_element:
        #        continue
        #    dst_element._write(src_element)

    def _copy(self) -> "LazyDynamicContainer[_ElementT]":
        return LazyDynamicContainer(
            elements=self._elements
        )

    def add(
        self,
        *elements: _ElementT
    ):
        slot = self._get_writable_slot_with_verification()
        if not elements:
            return self
        for element in elements:
            if element in self._elements:
                continue
            self._elements.append(element)
        slot.expire_property_slots()
        return self

    def remove(
        self,
        *elements: _ElementT
    ):
        slot = self._get_writable_slot_with_verification()
        if not elements:
            return self
        for element in elements:
            if element not in self._elements:
                continue
            self._elements.remove(element)
        slot.expire_property_slots()
        return self


class LazySlot(ABC, Generic[_ContainerT]):
    __slots__ = ("__weakref__",)

    @abstractmethod
    def copy_from(
        self,
        src: "LazySlot[_ContainerT]"
    ) -> None:
        pass


@final
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
        #src_instance: "LazyObject"
    ) -> None:
        self.set_variable_container(src.get_variable_container())
        #instance_property_slots = {
        #    descriptor.get_slot(src_instance): descriptor
        #    for descriptor in type(src_instance)._LAZY_DESCRIPTORS
        #    if isinstance(descriptor, LazyPropertyDescriptor)
        #}
        self._linked_property_slots.clear()
        #self._linked_property_slots.update({
        #    linked_variable_slot
        #    if (descriptor := instance_property_slots.get(linked_variable_slot)) is None
        #    else descriptor.get_slot(self)
        #    for linked_variable_slot in src._linked_property_slots
        #})
        self._is_writable = src._is_writable

    def expire_property_slots(self) -> None:
        for expired_property_slot in self._linked_property_slots:
            expired_property_slot.expire()
        self._linked_property_slots.clear()

    def yield_descendant_variable_slots(self) -> "Generator[LazyVariableSlot, None, None]":
        yield self
        for element in self.get_variable_container()._iter_elements():
            for slot in element._iter_variable_slots():
                yield from slot.yield_descendant_variable_slots()

    def make_readonly(self) -> None:
        for variable_slot in self.yield_descendant_variable_slots():
            variable_slot._is_writable = False


@final
class LazyPropertySlot(LazySlot[_ContainerT]):
    __slots__ = (
        "_container",
        "_linked_variable_slots",
        "_is_expired"
    )

    def __init__(self) -> None:
        super().__init__()
        self._container: _ContainerT = NotImplemented
        self._linked_variable_slots: weakref.WeakSet[LazyVariableSlot] = weakref.WeakSet()
        self._is_expired: bool = True

    def get_property_container(self) -> _ContainerT:
        return self._container

    def set_property_container(
        self,
        container: _ContainerT
    ) -> None:
        #assert self._is_expired
        if self._container is not NotImplemented and container is not NotImplemented:
            self._container._write(container)
        else:
            self._container = container
        #self._is_expired = container is NotImplemented

    def copy_from(
        self,
        src: "LazyPropertySlot[_ContainerT]"
        #src_instance: "LazyObject"
    ) -> None:
        self.set_property_container(src.get_property_container())
        #instance_variable_slots = {
        #    descriptor.get_slot(src_instance): descriptor
        #    for descriptor in type(src_instance)._LAZY_DESCRIPTORS
        #    if isinstance(descriptor, LazyVariableDescriptor)
        #}
        self._linked_variable_slots.clear()
        #self._linked_variable_slots.update({
        #    linked_variable_slot
        #    if (descriptor := instance_variable_slots.get(linked_variable_slot)) is None
        #    else descriptor.get_slot(self)
        #    for linked_variable_slot in src._linked_variable_slots
        #})
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


class LazyObject(ABC):
    __slots__ = ("__weakref__",)

    _LAZY_DESCRIPTORS: "ClassVar[tuple[LazyDescriptor, ...]]"
    _LAZY_VARIABLE_DESCRIPTORS: "ClassVar[tuple[LazyVariableDescriptor, ...]]"
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

        lazy_descriptors: list[LazyDescriptor] = []
        for descriptor_overloading in descriptor_overloading_dict.values():
            if descriptor_overloading not in overridden_descriptor_overloadings:
                descriptor_overloading.set_descriptor(cls, descriptor_overloading.get_descriptor(base_cls))
            lazy_descriptors.append(descriptor_overloading.get_descriptor(cls))

        cls._LAZY_DESCRIPTORS = tuple(lazy_descriptors)
        cls._LAZY_VARIABLE_DESCRIPTORS = tuple(
            descriptor for descriptor in lazy_descriptors
            if isinstance(descriptor, LazyVariableDescriptor)
        )
        cls._LAZY_DESCRIPTOR_OVERLOADING_DICT = descriptor_overloading_dict
        # Use dict.fromkeys to preserve order (by first occurrance).
        cls._PY_SLOTS = tuple(dict.fromkeys(
            slot
            for parent_cls in reversed(cls.__mro__)
            if issubclass(parent_cls, LazyObject)
            for slot in parent_cls.__slots__
            if slot != "__weakref__"
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
        self._initialize_descriptors()

    def _initialize_descriptors(self) -> None:
        cls = type(self)
        for descriptor in cls._LAZY_DESCRIPTORS:
            descriptor.initialize(self)

    #def _write(
    #    self: _ElementT,
    #    src: _ElementT
    #) -> None:
    #    cls = type(self)
    #    assert cls is type(src)
    #    for descriptor in cls._LAZY_DESCRIPTORS:
    #        descriptor.get_slot(self).copy_from(descriptor.get_slot(src))
    #    for slot_name in cls._PY_SLOTS:
    #        self.__setattr__(slot_name, copy.copy(src.__getattribute__(slot_name)))

    def _copy(self: _ElementT) -> _ElementT:
        cls = type(self)
        result = cls.__new__(cls)
        result._initialize_descriptors()
        for descriptor in cls._LAZY_DESCRIPTORS:
            descriptor.get_slot(result).copy_from(descriptor.get_slot(self))
        for slot_name in cls._PY_SLOTS:
            result.__setattr__(slot_name, copy.copy(self.__getattribute__(slot_name)))
        #for slot_name in cls._PY_SLOTS:
        #    result.__setattr__(slot_name, copy.copy(self.__getattribute__(slot_name)))
        #for descriptor in cls._LAZY_DESCRIPTORS:
        #    descriptor.copy_initialize(result, self)
        return result

    def _iter_variable_slots(self) -> Generator[LazyVariableSlot, None, None]:
        for descriptor in type(self)._LAZY_DESCRIPTORS:
            if not isinstance(descriptor, LazyVariableDescriptor):
                continue
            yield descriptor.get_slot(self)


class LazyDescriptor(ABC, Generic[_InstanceT, _SlotT, _ContainerT, _ElementT, _DescriptorGetT, _DescriptorSetT]):
    __slots__ = (
        "element_type",
        "instance_to_slot_dict"
    )

    def __init__(
        self,
        element_type: type[_ElementT]
    ) -> None:
        super().__init__()
        self.element_type: type[_ElementT] = element_type
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
        return self.convert_get(self.get_implementation(instance))

    def __set__(
        self,
        instance: _InstanceT,
        new_value: _DescriptorSetT
    ) -> None:
        self.set_implementation(instance, self.convert_set(new_value))

    def __delete__(
        self,
        instance: _InstanceT
    ) -> None:
        raise TypeError("Cannot delete attributes of a lazy object")

    @abstractmethod
    def get_implementation(
        self,
        instance: _InstanceT
    ) -> _ContainerT:
        pass

    @abstractmethod
    def set_implementation(
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

    #@abstractmethod
    #def copy_initialize(
    #    self,
    #    dst: _InstanceT,
    #    src: _InstanceT
    #) -> None:
    #    pass

    #@abstractmethod
    #def copy_from(
    #    self,
    #    dst: _InstanceT,
    #    src: _InstanceT
    #) -> None:
    #    pass

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
        element_type: type[_ElementT],
        method: Callable[[type[_InstanceT]], _DescriptorSetT]
    ) -> None:
        super().__init__(
            element_type=element_type
        )
        self.method: Callable[[type[_InstanceT]], _DescriptorSetT] = method
        self.default_container: _ContainerT | None = None

    def get_implementation(
        self,
        instance: _InstanceT
    ) -> _ContainerT:
        return self.get_slot(instance).get_variable_container()

    def set_implementation(
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
            default_container = self.convert_set(self.method(type(instance)))
            self.default_container = default_container
        self.set_slot(instance, LazyVariableSlot(
            container=default_container._copy()
        ))

    #def copy_initialize(
    #    self,
    #    dst: _InstanceT,
    #    src: _InstanceT
    #) -> None:
    #    self.set_slot(dst, LazyVariableSlot(
    #        container=self.get_slot(src).get_variable_container()._copy_container()
    #    ))

    #def copy_from(
    #    self,
    #    dst: _InstanceT,
    #    src: _InstanceT
    #) -> None:
    #    self.get_slot(dst).set_variable_container(self.get_slot(src).get_variable_container())


class LazyUnitaryVariableDescriptor(LazyVariableDescriptor[
    _InstanceT, LazyUnitaryContainer[_ElementT], _ElementT, _ElementT, _DescriptorSetT
]):
    __slots__ = ()

    def convert_get(
        self,
        container: LazyUnitaryContainer[_ElementT]
    ) -> _ElementT:
        return container._element


class LazyDynamicVariableDescriptor(LazyVariableDescriptor[
    _InstanceT, LazyDynamicContainer[_ElementT], _ElementT, LazyDynamicContainer[_ElementT], _DescriptorSetT
]):
    __slots__ = ()

    def convert_get(
        self,
        container: LazyDynamicContainer[_ElementT]
    ) -> LazyDynamicContainer[_ElementT]:
        return container


class LazyPropertyDescriptor(LazyDescriptor[
    _InstanceT, LazyPropertySlot[_ContainerT], _ContainerT, _ElementT, _DescriptorGetT, _DescriptorSetT
]):
    __slots__ = (
        "method",
        "finalize_method",
        "parameter_name_chains",
        "requires_unwrapping_tuple",
        "descriptor_overloading_chains",
        "instance_to_key_dict",
        "key_to_container_dict"
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
        self.finalize_method: Callable[[type[_InstanceT], _DescriptorGetT], None] | None = None
        self.parameter_name_chains: tuple[tuple[str, ...], ...] = parameter_name_chains
        self.requires_unwrapping_tuple: tuple[bool, ...] = requires_unwrapping_tuple
        self.descriptor_overloading_chains: tuple[tuple[LazyDescriptorOverloading, ...], ...] = NotImplemented
        self.instance_to_key_dict: weakref.WeakKeyDictionary[_InstanceT, int] = weakref.WeakKeyDictionary()
        self.key_to_container_dict: weakref.WeakValueDictionary[int, _ContainerT] = weakref.WeakValueDictionary()

    def get_implementation(
        self,
        instance: _InstanceT
    ) -> _ContainerT:

        def construct_parameter_item(
            descriptor_overloading_chain: "tuple[LazyDescriptorOverloading, ...]",
            instance: _InstanceT
        ) -> tuple[TreeNode[LazyObject], list[LazyVariableSlot]]:

            def construct_parameter_tagged_tree(
                descriptor_overloading: LazyDescriptorOverloading,
                is_chain_tail: bool,
                linked_variable_slots: list[LazyVariableSlot]
            ) -> Callable[[tuple[LazyObject, bool]], TreeNode[tuple[LazyObject, bool]]]:

                def callback(
                    obj_tagged: tuple[LazyObject, bool]
                ) -> TreeNode[tuple[LazyObject, bool]]:
                    obj, binding_completed = obj_tagged
                    descriptor = descriptor_overloading.get_descriptor(type(obj))
                    slot = descriptor.get_slot(obj)
                    container = descriptor.get_implementation(obj)
                    if not binding_completed:
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
                        result = TreeNode((obj, binding_completed)).bind_only(*(
                            TreeNode((element, binding_completed)) for element in container._elements
                        ))
                    else:
                        raise TypeError
                    return result

                return callback

            def remove_tag(
                obj_tagged: tuple[LazyObject, bool]
            ) -> LazyObject:
                obj, _ = obj_tagged
                return obj

            parameter_tagged_tree: TreeNode[tuple[LazyObject, bool]] = TreeNode((instance, False))
            linked_variable_slots: list[LazyVariableSlot] = []
            for reversed_index, descriptor_overloading in reversed(list(zip(it.count(), reversed(descriptor_overloading_chain)))):
                parameter_tagged_tree = parameter_tagged_tree.apply_deepest(
                    construct_parameter_tagged_tree(descriptor_overloading, not reversed_index, linked_variable_slots)
                )
            return parameter_tagged_tree.transform(remove_tag), linked_variable_slots

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
            #slot_nodes: list[TreeNode[LazyObject]] = []
            #for variable_slot in obj._iter_variable_slots():
            #    slot_node: TreeNode[LazyObject] = TreeNode(NotImplemented)
            #    slot_node.bind_only(*(
            #        expand_dependencies(element)
            #        for element in variable_slot.get_variable_container()._iter_elements()
            #    ))
            #    slot_nodes.append(slot_node)
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
        if not slot._is_expired:
            container = slot.get_property_container()
        else:
            parameter_items = tuple(
                construct_parameter_item(descriptor_overloading_chain, instance)
                for descriptor_overloading_chain in self.descriptor_overloading_chains
            )
            slot.bind_linked_variable_slots(*it.chain(*(
                linked_variable_slots
                for _, linked_variable_slots in parameter_items
            )))
            parameter_trees = tuple(
                parameter_tree
                for parameter_tree, _ in parameter_items
            )
            key = hash(tuple(
                construct_parameter_key(parameter_tree.apply_deepest(expand_dependencies))
                for parameter_tree in parameter_trees
            ))
            self.instance_to_key_dict[instance] = key

            if (container := self.key_to_container_dict.get(key)) is None:
                parameters = tuple(
                    construct_parameter(parameter_tree, requires_unwrapping=requires_unwrapping)
                    for parameter_tree, requires_unwrapping in zip(
                        parameter_trees, self.requires_unwrapping_tuple, strict=True
                    )
                )
                container = self.convert_set(self.method(type(instance), *parameters))
                for element in container._iter_elements():
                    for variable_slot in element._iter_variable_slots():
                        variable_slot.make_readonly()
                self.key_to_container_dict[key] = container

                if (finalize_method := self.finalize_method) is not None:
                    weakref.finalize(container, finalize_method, type(instance), self.convert_get(container))

            slot.set_property_container(container)
            slot._is_expired = False
        return container

    def set_implementation(
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

    #def copy_initialize(
    #    self,
    #    dst: _InstanceT,
    #    src: _InstanceT
    #) -> None:
    #    self.set_slot(dst, LazyPropertySlot())


class LazyUnitaryPropertyDescriptor(LazyPropertyDescriptor[
    _InstanceT, LazyUnitaryContainer[_ElementT], _ElementT, _ElementT, _DescriptorSetT
]):
    __slots__ = ()

    def convert_get(
        self,
        container: LazyUnitaryContainer[_ElementT]
    ) -> _ElementT:
        return container._element


class LazyDynamicPropertyDescriptor(LazyPropertyDescriptor[
    _InstanceT, LazyDynamicContainer[_ElementT], _ElementT, LazyDynamicContainer[_ElementT], _DescriptorSetT
]):
    __slots__ = ()

    def convert_get(
        self,
        container: LazyDynamicContainer[_ElementT]
    ) -> LazyDynamicContainer[_ElementT]:
        return container


class LazyDescriptorOverloading(ABC, Generic[_InstanceT, _ElementT, _DescriptorT]):
    __slots__ = (
        "element_type",
        "descriptors"
    )

    def __init__(
        self,
        instance_type: type[_InstanceT],
        descriptor: _DescriptorT
    ) -> None:
        super().__init__()
        self.element_type: type[_ElementT] = descriptor.element_type
        self.descriptors: dict[type[_InstanceT], _DescriptorT] = {instance_type: descriptor}

    def get_descriptor(
        self,
        instance_type: type[_InstanceT]
    ) -> _DescriptorT:
        return self.descriptors[instance_type]

    def set_descriptor(
        self,
        instance_type: type[_InstanceT],
        descriptor: _DescriptorT
    ) -> None:
        assert issubclass(descriptor.element_type, self.element_type)
        assert instance_type not in self.descriptors
        self.descriptors[instance_type] = descriptor


@final
class LazyUnitaryDescriptorOverloading(LazyDescriptorOverloading[_InstanceT, _ElementT, Union[
    LazyUnitaryVariableDescriptor[_InstanceT, _ElementT, _DescriptorSetT],
    LazyUnitaryPropertyDescriptor[_InstanceT, _ElementT, _DescriptorSetT]
]]):
    __slots__ = ()


@final
class LazyDynamicDescriptorOverloading(LazyDescriptorOverloading[_InstanceT, _ElementT, Union[
    LazyDynamicVariableDescriptor[_InstanceT, _ElementT, _DescriptorSetT],
    LazyDynamicPropertyDescriptor[_InstanceT, _ElementT, _DescriptorSetT]
]]):
    __slots__ = ()


@final
class LazyWrapper(LazyObject, Generic[_T]):
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
