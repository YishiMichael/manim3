"""
This module implements lazy evaluation based on weak reference. Meanwhile,
this also introduces functional programming into the project paradigm.

Every child class of `LazyObject` shall define `__slots__`, and all methods
shall be basically sorted in the following way:
- magic methods;
- lazy variables;
- lazy properties;
- private class methods;
- private methods;
- public methods.

All methods decorated by any decorator provided by `Lazy` should be static
methods and be named with underscores appeared on both sides, i.e. `_data_`.
Successive underscores shall not occur, due to the name convension handled by
lazy properties.

The constructor of `LazyDescriptor[_DataT, _T]` has the following notable
parameters:

- `method: Callable[..., _DataT]`
  Defines how the data is calculated through parameters. The returned data
  should be a single element, or a tuple of elements, determined by
  the `is_multiple` flag.

  When `is_variable` is true, the method should not take any parameter, and
  returns the initial value for the variable slot.

  The name of each parameter should be concatenated by names of a descriptor
  chain started from the local class, with underscores stripped. For example,
  the name `a__b__c` under class `A` fetches data through the path
  `A._a_._b_._c_`. The fetched data will be an `n`-layer tuple tree, where `n`
  is the number of descriptors with their `is_multiple` flags set to be true.

- `is_multiple: bool`
  Determines whether data contains a singular element or multiple elements.
  When true, `_DataT` is specialized as `tuple[_T]`; when false, specialized
  as `_T`.

- `is_variable: bool`
  Determines whether the descriptor behaves as a variable or a property.

  One can call `__set__` of the descriptor on some instance only when:
  - the `is_variable` is true;
  - the instance is not frozen.

- `hasher: Callable[[_T], Hashable]`
  Defines how elements are shared. Defaults to be `id`, meaning elements are
  never shared unless direct assignment is performed. Other options are also
  provided under `Lazy` namespace, named as `xxx_hasher`.

  Providing a hasher is encouraged whenever applicable, as this reduces
  redundant calculations. However, one cannot provide a hasher other than `id`
  when `frozen` is false.

- `frozen: bool`
  Determines whether data should be frozen when binding. Defaults to be true.
  Forced to be true when `is_variable` is false. When false, `hasher` is
  forced to be `id`.

  Note, freezing data does not block `__set__`. Unbinding data by reassigning
  a new one does not unfreeze the data.

  In fact, the freezing procedure can not go beyond the lazy scope. It only
  prevents users from calling `__set__` of variable descriptors on descendant
  lazy objects, but does not prevent users from modifying data that is not of
  type `LazyObject`, e.g., `np.ndarray`.

- `cache_capacity: int`
  Determines the capacity of the lru cache of parameters-data pairs generated
  from `method`. Forced to be 1 when `is_variable` is true (the parameter list
  is always empty). Defaults to be 128 when `is_variable` is false.

Descriptor overriding is allowed. The overriding descriptor should match the
overridden one in `is_multiple` and `hasher`. Furthermore, the type of element
(the specialization of type variable `_T`) should be consistent between
descriptors.
"""


import copy
import inspect
import re
import weakref
from abc import ABC
from typing import (
    Callable,
    ClassVar,
    Generic,
    Hashable,
    Iterator,
    TypeVar,
    overload
)

import numpy as np


_T = TypeVar("_T")
_KT = TypeVar("_KT", bound=Hashable)
_VT = TypeVar("_VT")
_DataT = TypeVar("_DataT")


class Registered(Generic[_T]):
    __slots__ = (
        "__weakref__",
        "_value"
    )

    def __init__(
        self,
        value: _T
    ) -> None:
        super().__init__()
        self._value: _T = value


class Registration(weakref.WeakValueDictionary[_KT, Registered[_VT]]):
    __slots__ = ()

    def register(
        self,
        key: _KT,
        value: _VT
    ) -> Registered[_VT]:
        if (registered_value := self.get(key)) is None:
            registered_value = Registered(value)
            self[key] = registered_value
        return registered_value


class Cache(weakref.WeakKeyDictionary[_KT, _VT]):
    __slots__ = ("_capacity",)

    def __init__(
        self,
        capacity: int
    ) -> None:
        super().__init__()
        self._capacity: int = capacity

    def set(
        self,
        key: _KT,
        value: _VT
    ) -> None:
        assert key not in self
        if len(self) == self._capacity:
            self.pop(next(iter(self)))
        self[key] = value


class PseudoTree:
    __slots__ = (
        "_leaf_tuple",
        "_branch_structure"
    )

    def __init__(
        self,
        leaf_tuple: tuple,
        branch_structure: tuple[tuple[int, ...], ...]
    ) -> None:
        super().__init__()
        self._leaf_tuple: tuple = leaf_tuple
        self._branch_structure: tuple[tuple[int, ...], ...] = branch_structure

    def key(self) -> Hashable:
        return (
            tuple(id(leaf) for leaf in self._leaf_tuple),
            self._branch_structure
        )

    def tree(self) -> tuple:

        def iter_chunks(
            leaf_tuple: tuple,
            branch_sizes: tuple[int, ...]
        ) -> Iterator[tuple]:
            rest_leaf_tuple = leaf_tuple
            for branch_size in branch_sizes:
                yield rest_leaf_tuple[:branch_size]
                rest_leaf_tuple = rest_leaf_tuple[branch_size:]
            assert not rest_leaf_tuple

        result = self._leaf_tuple
        for branch_sizes in reversed(self._branch_structure):
            result = tuple(iter_chunks(result, branch_sizes))
        return result


class LazySlot(Generic[_T]):
    __slots__ = (
        "__weakref__",
        "_elements",
        "_parameter_key",
        "_linked_slots",
        "_is_writable"
    )

    def __init__(self) -> None:
        super().__init__()
        self._elements: tuple[Registered[_T], ...] | None = None
        self._parameter_key: Registered[Hashable] | None = None
        self._linked_slots: weakref.WeakSet[LazySlot] = weakref.WeakSet()
        self._is_writable: bool = True

    def _get(self) -> tuple[Registered[_T], ...] | None:
        return self._elements

    def _set(
        self,
        elements: tuple[Registered[_T], ...],
        parameter_key: Registered[Hashable] | None,
        linked_slots: "set[LazySlot]"
    ) -> None:
        self._elements = elements
        self._parameter_key = parameter_key
        assert not self._linked_slots
        self._linked_slots.update(linked_slots)
        for slot in linked_slots:
            slot._linked_slots.add(self)

    def _expire(self) -> None:
        self._elements = None
        for slot in self._linked_slots:
            slot._linked_slots.remove(self)
        self._linked_slots.clear()

    def _iter_linked_slots(self) -> "Iterator[LazySlot]":
        return iter(set(self._linked_slots))


class LazyDescriptor(ABC, Generic[_DataT, _T]):
    __slots__ = (
        "_method",
        "_name",
        "_is_multiple",
        "_decomposer",
        "_composer",
        "_is_variable",
        "_hasher",
        "_frozen",
        "_cache",
        "_parameter_key_registration",
        "_element_registration",
        "_element_type",
        "_element_type_annotation",
        "_descriptor_chains"
    )

    def __init__(
        self,
        method: Callable[..., _DataT],
        is_multiple: bool,
        decomposer: Callable[[_DataT], tuple[_T, ...]],
        composer: Callable[[tuple[_T, ...]], _DataT],
        is_variable: bool,
        hasher: Callable[[_T], Hashable],
        frozen: bool,
        cache_capacity: int
    ) -> None:
        assert re.fullmatch(r"_([^_]+_)+", method.__name__)
        assert hasher is id or frozen
        super().__init__()
        self._method: Callable[..., _DataT] = method
        self._name: str = method.__name__
        self._is_multiple: bool = is_multiple
        self._decomposer: Callable[[_DataT], tuple[_T, ...]] = decomposer
        self._composer: Callable[[tuple[_T, ...]], _DataT] = composer
        self._is_variable: bool = is_variable
        self._hasher: Callable[[_T], Hashable] = hasher
        self._frozen: bool = frozen
        self._cache: Cache[Registered[Hashable], tuple[Registered[_T], ...]] = Cache(capacity=cache_capacity)
        self._parameter_key_registration: Registration[Hashable, Hashable] = Registration()
        # Shared when overridden.
        self._element_registration: Registration[Hashable, _T] = Registration()

        self._element_type: type[_T] = NotImplemented
        self._element_type_annotation: type = NotImplemented
        self._descriptor_chains: tuple[tuple[LazyDescriptor, ...], ...] = NotImplemented

    @overload
    def __get__(
        self,
        instance: None,
        owner: "type[LazyObject] | None" = None
    ) -> "LazyDescriptor": ...  # TODO: type Self

    @overload
    def __get__(
        self,
        instance: "LazyObject",
        owner: "type[LazyObject] | None" = None
    ) -> _DataT: ...

    def __get__(
        self,
        instance: "LazyObject | None",
        owner: "type[LazyObject] | None" = None
    ) -> "LazyDescriptor | _DataT":
        if instance is None:
            return self
        return self._composer(self._get_elements(instance))

    def __set__(
        self,
        instance: "LazyObject",
        data: _DataT
    ) -> None:
        return self._set_elements(instance, self._decomposer(data))

    def __delete__(
        self,
        instance: "LazyObject"
    ) -> None:
        raise TypeError("Cannot delete attributes of a lazy object")

    def _can_override(
        self,
        descriptor: "LazyDescriptor"
    ) -> bool:
        return (
            self._is_multiple is descriptor._is_multiple
            and self._hasher is descriptor._hasher
            and (
                self._element_type_annotation == descriptor._element_type_annotation
                or issubclass(self._element_type, descriptor._element_type)
            )
        )

    def _get_slot(
        self,
        instance: "LazyObject"
    ) -> LazySlot[_T]:
        return instance._lazy_slots[self._name]

    def _set_slot(
        self,
        instance: "LazyObject",
        slot: LazySlot[_T]
    ) -> None:
        instance._lazy_slots[self._name] = slot

    def _init(
        self,
        instance: "LazyObject"
    ) -> None:
        slot = LazySlot()
        slot._is_writable = self._is_variable
        self._set_slot(instance, slot)

    def _get_elements(
        self,
        instance: "LazyObject"
    ) -> tuple[_T, ...]:

        def get_leaf_items(
            leaf: LazyObject,
            descriptor_name: str
        ) -> tuple[tuple, set[LazySlot]]:
            descriptor = type(leaf)._lazy_descriptors[descriptor_name]
            slot = descriptor._get_slot(leaf)
            elements = descriptor._get_elements(leaf)
            leaf_linked_variable_slots = {slot} if descriptor._is_variable else set(slot._iter_linked_slots())
            return elements, leaf_linked_variable_slots

        def iter_descriptor_items(
            descriptor_chain: tuple[LazyDescriptor, ...],
            instance: LazyObject
        ) -> Iterator[tuple[tuple, tuple[int, ...] | None, set[LazySlot]]]:
            leaf_tuple: tuple = (instance,)
            for descriptor in descriptor_chain:
                leaf_items = tuple(
                    get_leaf_items(leaf, descriptor._name)
                    for leaf in leaf_tuple
                )
                leaf_tuple = tuple(
                    element
                    for elements, _ in leaf_items
                    for element in elements
                )
                branch_sizes = tuple(
                    len(elements)
                    for elements, _ in leaf_items
                ) if descriptor._is_multiple else None
                descriptor_linked_variable_slots = set().union(*(
                    leaf_linked_variable_slots
                    for _, leaf_linked_variable_slots in leaf_items
                ))
                yield leaf_tuple, branch_sizes, descriptor_linked_variable_slots

        def get_parameter_items(
            descriptor_chain: tuple[LazyDescriptor, ...],
            instance: LazyObject
        ) -> tuple[PseudoTree, set[LazySlot]]:
            descriptor_items = tuple(iter_descriptor_items(descriptor_chain, instance))
            leaf_tuple, _, _ = descriptor_items[-1]
            pseudo_tree = PseudoTree(
                leaf_tuple=leaf_tuple,
                branch_structure=tuple(
                    branch_sizes
                    for _, branch_sizes, _ in descriptor_items
                    if branch_sizes is not None
                )
            )
            parameter_linked_variable_slots = set().union(*(
                descriptor_linked_variable_slots
                for _, _, descriptor_linked_variable_slots in descriptor_items
            ))
            return pseudo_tree, parameter_linked_variable_slots

        def get_pseudo_trees_and_linked_variable_slots(
            descriptor_chains: tuple[tuple[LazyDescriptor, ...], ...],
            instance: LazyObject
        ) -> tuple[tuple[PseudoTree, ...], set[LazySlot]]:
            parameter_items = tuple(
                get_parameter_items(descriptor_chain, instance)
                for descriptor_chain in descriptor_chains
            )
            pseudo_trees = tuple(pseudo_tree for pseudo_tree, _ in parameter_items)
            linked_variable_slots = set().union(*(
                parameter_linked_variable_slots for _, parameter_linked_variable_slots in parameter_items
            ))
            return pseudo_trees, linked_variable_slots

        slot = self._get_slot(instance)
        if (registered_elements := slot._get()) is None:
            # If there's at least a parameter, `slot` is guaranteed to be a property slot.
            # Link it with variable slots.
            pseudo_trees, linked_variable_slots = get_pseudo_trees_and_linked_variable_slots(
                self._descriptor_chains, instance
            )
            registered_parameter_key = self._register_parameter_key(tuple(
                pseudo_tree.key()
                for pseudo_tree in pseudo_trees
            ))
            if (registered_elements := self._cache.get(registered_parameter_key)) is None:
                registered_elements = self._register_elements(self._decomposer(self._method(*(
                    pseudo_tree.tree()[0]
                    for pseudo_tree in pseudo_trees
                ))))
                self._cache.set(registered_parameter_key, registered_elements)
            slot._set(
                elements=registered_elements,
                parameter_key=registered_parameter_key,
                linked_slots=linked_variable_slots
            )
        return tuple(registered_element._value for registered_element in registered_elements)

    def _set_elements(
        self,
        instance: "LazyObject",
        elements: tuple[_T, ...]
    ) -> None:
        slot = self._get_slot(instance)
        assert slot._is_writable, "Attempting to write to a readonly slot"
        # `slot` is guaranteed to be a variable slot. Expire linked property slots.
        for expired_property_slot in slot._iter_linked_slots():
            expired_property_slot._expire()
        registered_elements = self._register_elements(elements)
        slot._set(
            elements=registered_elements,
            parameter_key=None,
            linked_slots=set()
        )

    def _register_parameter_key(
        self,
        parameter_key: Hashable
    ) -> Registered[Hashable]:
        return self._parameter_key_registration.register(parameter_key, parameter_key)

    def _register_elements(
        self,
        elements: tuple[_T, ...]
    ) -> tuple[Registered[_T], ...]:

        def freeze(
            element: object
        ) -> None:
            if not isinstance(element, LazyObject) or element._is_frozen:
                return
            element._is_frozen = True
            for descriptor in type(element)._lazy_descriptors.values():
                descriptor._get_slot(element)._is_writable = False
                for child_element in descriptor._get_elements(element):
                    freeze(child_element)

        if self._frozen:
            for element in elements:
                freeze(element)
        element_registration = self._element_registration
        hasher = self._hasher
        return tuple(
            element_registration.register(hasher(element), element)
            for element in elements
        )


class LazyObject(ABC):
    __slots__ = (
        "_lazy_slots",
        "_is_frozen"
    )
    _special_slot_copiers: ClassVar[dict[str, Callable | None]] = {
        "_lazy_slots": None,
        "_is_frozen": None
    }

    _lazy_descriptors: ClassVar[dict[str, LazyDescriptor]] = {}
    _slot_copiers: ClassVar[dict[str, Callable]] = {}

    def __init_subclass__(cls) -> None:

        def get_descriptor_chain(
            name_chain: tuple[str, ...],
            root_class: type[LazyObject],
            parameter: inspect.Parameter
        ) -> tuple[LazyDescriptor, ...]:
            descriptor_chain: list[LazyDescriptor] = []
            element_type = root_class
            collection_level = 0
            for descriptor_name in name_chain:
                assert issubclass(element_type, LazyObject)
                descriptor = element_type._lazy_descriptors[descriptor_name]
                descriptor_chain.append(descriptor)
                element_type = descriptor._element_type
                if descriptor._is_multiple:
                    collection_level += 1
            expected_annotation = descriptor_chain[-1]._element_type_annotation
            for _ in range(collection_level):
                expected_annotation = tuple[expected_annotation, ...]
            assert parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            assert parameter.annotation == expected_annotation or isinstance(expected_annotation, TypeVar)
            return tuple(descriptor_chain)

        super().__init_subclass__()
        base_cls = cls.__base__
        assert issubclass(base_cls, LazyObject)
        new_descriptors = {
            name: descriptor
            for name, descriptor in cls.__dict__.items()
            if isinstance(descriptor, LazyDescriptor)
        }
        cls._lazy_descriptors = base_cls._lazy_descriptors | new_descriptors

        # Use dict.fromkeys to preserve order (by first occurrance).
        cls._slot_copiers = {
            slot_name: slot_copier
            for base_cls in reversed(cls.__mro__)
            if issubclass(base_cls, LazyObject)
            for slot_name in base_cls.__slots__
            if (slot_copier := base_cls._special_slot_copiers.get(slot_name, copy.copy)) is not None
        }

        for name, descriptor in new_descriptors.items():
            signature = inspect.signature(descriptor._method, locals={cls.__name__: cls}, eval_str=True)

            return_annotation = signature.return_annotation
            if descriptor._is_multiple:
                assert return_annotation.__origin__ is tuple
                element_type_annotation, ellipsis = return_annotation.__args__
                assert ellipsis is ...
            else:
                element_type_annotation = return_annotation
            descriptor._element_type_annotation = element_type_annotation

            try:
                element_type = element_type_annotation.__origin__
            except AttributeError:
                element_type = element_type_annotation
            descriptor._element_type = element_type

            #assert all(
            #    parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            #    for parameter in signature.parameters.values()
            #)
            descriptor._descriptor_chains = tuple(
                get_descriptor_chain(
                    name_chain=tuple(f"_{name_segment}_" for name_segment in parameter_name.split("__")),
                    root_class=cls,
                    parameter=parameter
                )
                for parameter_name, parameter in signature.parameters.items()
            )
            if descriptor._is_variable:
                assert not descriptor._descriptor_chains

            overridden_descriptor = base_cls._lazy_descriptors.get(name)
            if overridden_descriptor is not None:
                descriptor._element_registration = overridden_descriptor._element_registration
                assert descriptor._can_override(overridden_descriptor)

    def __init__(self) -> None:
        super().__init__()
        self._lazy_slots: dict[str, LazySlot] = {}
        self._is_frozen: bool = False
        for descriptor in type(self)._lazy_descriptors.values():
            descriptor._init(self)

    def _copy(self):
        cls = type(self)
        result = cls.__new__(cls)
        result._lazy_slots = {}
        result._is_frozen = False
        for descriptor in cls._lazy_descriptors.values():
            descriptor._init(result)
            if descriptor._is_variable:
                descriptor._set_elements(result, descriptor._get_elements(self))
        for slot_name, slot_copier in cls._slot_copiers.items():
            result.__setattr__(slot_name, slot_copier(self.__getattribute__(slot_name)))
        return result


class Lazy:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @classmethod
    def _descriptor_singular(
        cls,
        is_variable: bool,
        frozen: bool,
        cache_capacity: int,
        hasher: Callable[..., Hashable]
    ) -> Callable[[Callable[..., _T]], LazyDescriptor[_T, _T]]:

        def singular_decomposer(
            data: _T
        ) -> tuple[_T, ...]:
            return (data,)

        def singular_composer(
            elements: tuple[_T, ...]
        ) -> _T:
            (element,) = elements
            return element

        def result(
            method: Callable[[], _T]
        ) -> LazyDescriptor[_T, _T]:
            return LazyDescriptor(
                method=method,
                is_multiple=False,
                decomposer=singular_decomposer,
                composer=singular_composer,
                is_variable=is_variable,
                hasher=hasher,
                frozen=frozen,
                cache_capacity=cache_capacity
            )

        return result

    @classmethod
    def _descriptor_multiple(
        cls,
        is_variable: bool,
        hasher: Callable[..., Hashable],
        frozen: bool,
        cache_capacity: int
    ) -> Callable[[Callable[..., tuple[_T, ...]]], LazyDescriptor[tuple[_T, ...], _T]]:

        def multiple_decomposer(
            data: tuple[_T, ...]
        ) -> tuple[_T, ...]:
            return data

        def multiple_composer(
            elements: tuple[_T, ...]
        ) -> tuple[_T, ...]:
            return elements

        def result(
            method: Callable[[], tuple[_T, ...]]
        ) -> LazyDescriptor[tuple[_T, ...], _T]:
            return LazyDescriptor(
                method=method,
                is_multiple=True,
                decomposer=multiple_decomposer,
                composer=multiple_composer,
                is_variable=is_variable,
                hasher=hasher,
                frozen=frozen,
                cache_capacity=cache_capacity
            )

        return result

    @classmethod
    def variable(
        cls,
        hasher: Callable[..., Hashable] = id,
        frozen: bool = True
    ) -> Callable[[Callable[[], _T]], LazyDescriptor[_T, _T]]:
        return cls._descriptor_singular(
            is_variable=True,
            hasher=hasher,
            frozen=frozen,
            cache_capacity=1
        )

    @classmethod
    def variable_collection(
        cls,
        hasher: Callable[..., Hashable] = id,
        frozen: bool = True
    ) -> Callable[[Callable[[], tuple[_T, ...]]], LazyDescriptor[tuple[_T, ...], _T]]:
        return cls._descriptor_multiple(
            is_variable=True,
            hasher=hasher,
            frozen=frozen,
            cache_capacity=1
        )

    @classmethod
    def property(
        cls,
        hasher: Callable[..., Hashable] = id,
        cache_capacity: int = 128
    ) -> Callable[[Callable[..., _T]], LazyDescriptor[_T, _T]]:
        return cls._descriptor_singular(
            is_variable=False,
            hasher=hasher,
            frozen=True,
            cache_capacity=cache_capacity
        )

    @classmethod
    def property_collection(
        cls,
        hasher: Callable[..., Hashable] = id,
        cache_capacity: int = 128
    ) -> Callable[[Callable[..., tuple[_T, ...]]], LazyDescriptor[tuple[_T, ...], _T]]:
        return cls._descriptor_multiple(
            is_variable=False,
            hasher=hasher,
            frozen=True,
            cache_capacity=cache_capacity
        )

    @staticmethod
    def naive_hasher(
        element: Hashable
    ) -> Hashable:
        return element

    @staticmethod
    def array_hasher(
        element: np.ndarray
    ) -> bytes:
        return element.tobytes()

    @staticmethod
    def branch_hasher(
        element: LazyObject
    ) -> Hashable:
        return (type(element), tuple(
            tuple(
                id(variable_element)
                for variable_element in descriptor._get_elements(element)
            )
            for descriptor in type(element)._lazy_descriptors.values()
            if descriptor._is_variable
        ))
