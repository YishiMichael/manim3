import copy
import inspect
#import itertools as it
import re
import weakref
from abc import ABC
from typing import (
    #Any,
    Callable,
    ClassVar,
    #Concatenate,
    Generic,
    Hashable,
    #Iterable,
    Iterator,
    #ParamSpec,
    TypeVar,
    overload
)

import numpy as np


_T = TypeVar("_T")
_KT = TypeVar("_KT", bound=Hashable)
_VT = TypeVar("_VT")
_DataT = TypeVar("_DataT")
#_T = TypeVar("_T")
#_HashInT = TypeVar("_HashInT")
#_HashOutT = TypeVar("_HashOutT", bound=Hashable)
#_Parameters = ParamSpec("_Parameters")


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
        #self._cache_dict: weakref.WeakKeyDictionary[_KT, _VT] = weakref.WeakKeyDictionary()

    #def get(
    #    self,
    #    key: _KT
    #) -> _VT | None:
    #    return self._cache_dict.get(key)

    def set(
        self,
        key: _KT,
        value: _VT
    ) -> None:
        #cache_dict = self._cache_dict
        assert key not in self
        if len(self) == self._capacity:
            self.pop(next(iter(self)))
        self[key] = value

    #@staticmethod
    #def _restrict_size(
    #    method: "Callable[Concatenate[Cache, _Parameters], _T]",
    #) -> "Callable[Concatenate[Cache, _Parameters], _T]":

    #    @wraps(method)
    #    def new_method(
    #        self: "Cache",
    #        *args: _Parameters.args,
    #        **kwargs: _Parameters.kwargs
    #    ) -> _T:
    #        result = method(self, *args, **kwargs)
    #        capacity = self._capacity
    #        while len(self) > capacity:
    #            self.pop(next(iter(self)))
    #        return result

    #    return new_method

    #@_restrict_size
    #def __setitem__(
    #    self,
    #    key: _KT,
    #    value: _VT
    #) -> None:
    #    return super().__setitem__(key, value)


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


#class LazyList(list[_T]):
#    __slots__ = ("_slot_backref",)
#
#    def __init__(
#        self,
#        elements: Iterable[_T]
#    ) -> None:
#        super().__init__(elements)
#        self._slot_backref: weakref.ref[LazySlot[LazyList[_T]]] | None = None

    #def _iter_elements(self) -> Iterator[_T]:
    #    yield from self._elements

    #def _write(
    #    self,
    #    src: "LazyList[_T]"
    #) -> None:
    #    self._elements = list(src._elements)

    #def _copy_container(self) -> "LazyList[_T]":
    #    return LazyList(
    #        elements=self._elements
    #    )

    # list methods
    # Only those considered necessary are ported.

    #@staticmethod
    #def _force_expire(
    #    method: "Callable[Concatenate[LazyList[_T], _Parameters], _T]",
    #) -> "Callable[Concatenate[LazyList[_T], _Parameters], _T]":

    #    @wraps(method)
    #    def new_method(
    #        self: "LazyList[_T]",
    #        *args: _Parameters.args,
    #        **kwargs: _Parameters.kwargs
    #    ) -> _T:
    #        assert (slot_backref := self._slot_backref) is not None
    #        assert (slot := slot_backref()) is not None
    #        slot.
    #        #assert slot._is_writable
    #        #slot._expire_property_slots()
    #        return method(self, *args, **kwargs)

    #    return new_method

    #@overload
    #def __setitem__(
    #    self,
    #    index: int,
    #    value: _T
    #) -> None: ...

    #@overload
    #def __setitem__(
    #    self,
    #    index: slice,
    #    value: Iterable[_T]
    #) -> None: ...

    #@_force_expire
    #def __setitem__(
    #    self,
    #    index: Any,
    #    value: Any
    #) -> None:
    #    super().__setitem__(index, value)

    #@_force_expire
    #def insert(
    #    self,
    #    index: int,
    #    value: _T
    #) -> None:
    #    return super().insert(index, value)

    #@_force_expire
    #def append(
    #    self,
    #    value: _T
    #) -> None:
    #    return super().append(value)

    #@_force_expire
    #def extend(
    #    self,
    #    values: Iterable[_T]
    #) -> None:
    #    return super().extend(values)

    #@_force_expire
    #def reverse(self) -> None:
    #    return super().reverse()

    #@_force_expire
    #def pop(
    #    self,
    #    index: int = -1
    #) -> _T:
    #    return super().pop(index)

    #@_force_expire
    #def remove(
    #    self,
    #    value: _T
    #) -> None:
    #    return super().remove(value)

    #@_force_expire
    #def clear(self) -> None:
    #    return super().clear()

    ## Additional methods for convenience.

    #@_force_expire
    #def eliminate(
    #    self,
    #    values: Iterable[_T]
    #) -> None:
    #    for value in values:
    #        self.remove(value)

    #@_force_expire
    #def reset(
    #    self,
    #    values: Iterable[_T]
    #) -> None:
    #    self.clear()
    #    self.extend(values)


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

    #def _link(
    #    self,
    #    slots: "set[LazySlot]"
    #) -> None:
    #    assert not self._linked_slots
    #    self._linked_slots.update(slots)
    #    for slot in slots:
    #        slot._linked_slots.add(self)

    #def _unlink(self) -> None:
    #    for slot in self._linked_slots:
    #        slot._linked_slots.remove(self)
    #    self._linked_slots.clear()
    #    #self._linked_slots.remove(slot)
    #    #slot._linked_slots.remove(self)

    # variable slot

    #def _expire_property_slots(self) -> None:
    #    for expired_property_slot in self._iter_linked_slots():
    #        expired_property_slot._expire()

    # property slot

    #def _expire(self) -> None:
    #    for variable_slot in self._iter_linked_slots():
    #        self._unlink(variable_slot)
    #    self._set_data(None)

    #def _refresh(
    #    self,
    #    data: _DataT,
    #    variable_slots: "Iterable[LazySlot]"
    #) -> None:
    #    for variable_slot in variable_slots:
    #        self._link(variable_slot)
    #    self._set_data(data)


#class HashProcessor(ABC, Generic[_T, _HashInT, _HashOutT]):
#    __slots__ = ()
#
#    def __new__(cls) -> None:
#        pass
#
#    @classmethod
#    @abstractmethod
#    def _hash(
#        cls,
#        element: _T
#    ) -> _HashInT, _HashOutT:
#        pass
#
#    #@classmethod
#    #@abstractmethod
#    #def _frozen(
#    #    cls,
#    #    element: _T
#    #) -> _T:
#    #    pass


#class StorageProcessor(ABC, Generic[_DataT, _T]):
#    __slots__ = ()

#    def __new__(cls) -> None:
#        pass

#    @classmethod
#    @abstractmethod
#    def _iter_elements(
#        cls,
#        data: _DataT
#    ) -> Iterator[_T]:
#        pass

#    @classmethod
#    @abstractmethod
#    def _assemble_elements(
#        cls,
#        elements: Iterator[_T]
#    ) -> _DataT:
#        pass


#class UnitaryStorageProcessor(Generic[_T], StorageProcessor[_T, _T]):
#    __slots__ = ()

#    @classmethod
#    def _iter_elements(
#        cls,
#        data: _T
#    ) -> Iterator[_T]:
#        yield data

#    @classmethod
#    def _assemble_elements(
#        cls,
#        elements: Iterator[_T]
#    ) -> _T:
#        return next(elements)
#        #slot._set_data(data)


#class DynamicStorageProcessor(Generic[_T], StorageProcessor[list[_T], _T]):
#    __slots__ = ()

#    @classmethod
#    def _iter_elements(
#        cls,
#        data: list[_T]
#    ) -> Iterator[_T]:
#        yield from data

#    @classmethod
#    def _assemble_elements(
#        cls,
#        elements: Iterator[_T]
#    ) -> list[_T]:
#        return list(elements)
#        ##assert data._slot_backref is None
#        #data._slot_backref = slot_backref
#        ##slot._set_data(data)
#        #return data


#class LogicProcessor(ABC, Generic[_DataT, _T]):
#    __slots__ = ()

#    def __new__(cls) -> None:
#        pass

#    @classmethod
#    @abstractmethod
#    def _init_slot(
#        cls,
#        slot: LazySlot[_DataT]
#        #method: Callable[..., _DataT],
#        #storage_processor_cls: type[StorageProcessor[_DataT, _T]]
#    ) -> None:
#        pass


#class VariableLogicProcessor(LogicProcessor[_DataT, _T]):
#    __slots__ = ()

#    @classmethod
#    def _init_slot(
#        cls,
#        slot: LazySlot[_DataT]
#        #method: Callable[..., _DataT],
#        #storage_processor_cls: type[StorageProcessor[_DataT, _T]]
#    ) -> None:
#        pass
#        #data = method()
#        #storage_processor_cls._prepare_data(data, weakref.ref(slot))


#class PropertyLogicProcessor(LogicProcessor[_DataT, _T]):
#    __slots__ = ()

#    @classmethod
#    def _init_slot(
#        cls,
#        slot: LazySlot[_DataT]
#        #method: Callable[..., _DataT],
#        #storage_processor_cls: type[StorageProcessor[_DataT, _T]]
#    ) -> None:
#        slot._is_writable = False


#class FrozenProcessor(ABC, Generic[_T]):
#    __slots__ = ()

#    def __new__(cls) -> None:
#        pass

#    @classmethod
#    @abstractmethod
#    def _frozen(
#        cls,
#        element: _T
#    ) -> None:
#        pass


#class BlankFrozenProcessor(FrozenProcessor[_T]):
#    __slots__ = ()

#    @classmethod
#    def _frozen(
#        cls,
#        element: _T
#    ) -> None:
#        pass


#class DescendantFrozenProcessor(FrozenProcessor[_LazyObjectT]):
#    __slots__ = ()

#    @classmethod
#    def _frozen(
#        cls,
#        element: _LazyObjectT
#    ) -> None:
#        pass


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
        #"_logic_processor_cls",
        #"_storage_processor_cls",
        #"_frozen_processor_cls",
        "_cache",
        "_parameter_key_registration",
        "_element_registration",
        #"_finalize_method",
        "_element_type",
        "_element_type_annotation",
        #"_parameter_name_chains",
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
        #element_type: type[_T],
        #logic_processor_cls: type[LogicProcessor[_DataT, _T]],
        #storage_processor_cls: type[StorageProcessor[_DataT, _T]],
        #frozen_processor_cls: type[FrozenProcessor[_T]],
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
        #self._logic_processor_cls: type[LogicProcessor[_DataT, _T]] = logic_processor_cls
        #self._storage_processor_cls: type[StorageProcessor[_DataT, _T]] = storage_processor_cls
        #self._frozen_processor_cls: type[FrozenProcessor[_T]] = frozen_processor_cls
        self._cache: Cache[Registered[Hashable], tuple[Registered[_T], ...]] = Cache(capacity=cache_capacity)
        #self._finalize_method: Callable[[type[_InstanceT], _T], None] | None = None
        self._parameter_key_registration: Registration[Hashable, Hashable] = Registration()
        # Shared when overridden.
        self._element_registration: Registration[Hashable, _T] = Registration()

        self._element_type: type[_T] = NotImplemented
        self._element_type_annotation: type = NotImplemented
        #self._parameter_name_chains: tuple[tuple[str, ...], ...] = NotImplemented
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
        #print(self._is_multiple, descriptor._is_multiple)
        #print(self._decomposer, descriptor._decomposer)
        #print(self._is_multiple, descriptor._is_multiple)
        #print(self._is_variable, descriptor._is_variable)
        #print(self._hasher, descriptor._hasher)
        #print(self._frozen, descriptor._frozen)
        #print(self._element_type, descriptor._element_type)
        #print()
        return (
            self._is_multiple is descriptor._is_multiple
            #and self._decomposer is descriptor._decomposer
            #and self._composer is descriptor._composer
            #and self._is_variable is descriptor._is_variable
            and self._hasher is descriptor._hasher
            and self._frozen is descriptor._frozen
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
        #self._logic_processor_cls._init_slot(slot)
        #slot._set_data(data)
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
            #branch_structure: list[tuple[int, ...]] = []
            #linked_variable_slots: set[LazySlot] = set()
            for descriptor in descriptor_chain:
                leaf_items = tuple(
                    get_leaf_items(leaf, descriptor._name)
                    for leaf in leaf_tuple
                )
                #overridden_descriptors = tuple(
                #    type(leaf)._lazy_descriptors[descriptor._name]
                #    for leaf in leaf_tuple
                #)
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
                ##leaf_items = tuple(
                ##    get_elements_and_linked_variable_slots_from_leaf(leaf, descriptor._name)
                ##    for leaf in leaf_tuple
                ##)
                #elements_list: list = []
                #for leaf in leaf_tuple:
                #    overridden_descriptor = type(leaf)._lazy_descriptors[descriptor._name]
                #    elements_list.append(overridden_descriptor._get_elements(leaf))
                #    #if overridden_descriptor._is_variable:
                #    #    linked_variable_slots.update(
                #    #        overridden_descriptor._get_slot(leaf)
                #    #        for leaf in leaf_tuple
                #    #    )
                #    #else:
                #    #    linked_variable_slots.update(set().union(*(
                #    #        overridden_descriptor._get_slot(leaf)._iter_linked_slots()
                #    #        for leaf in leaf_tuple
                #    #    )))
                ##elements_list = [
                ##    list(descriptor._decomposer(
                ##        type(leaf)._lazy_descriptors[descriptor._name]._get(leaf)
                ##    ))
                ##    for leaf in leaf_tuple
                ##]
                ##if descriptor._is_variable:
                ##    linked_variable_slots.extend(
                ##        descriptor._get_slot(leaf)
                ##        for leaf in leaf_tuple
                ##    )
                ##else:
                ##    linked_variable_slots.extend(it.chain.from_iterable(
                ##        descriptor._get_slot(leaf)._iter_linked_slots()
                ##        for leaf in leaf_tuple
                ##    ))
                #leaf_tuple = tuple(it.chain.from_iterable(elements_list))
            #    if descriptor._is_multiple:
            #        branch_structure.append(tuple(len(elements) for elements, _ in leaf_items))
            #return PseudoTree(leaf_tuple, tuple(branch_structure)), linked_variable_slots

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

        #def pseudo_tree_key(
        #    pseudo_tree: tuple[list, list[list[int]]]
        #) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        #    leaf_tuple, branch_structure = pseudo_tree
        #    return (
        #        tuple(id(leaf) for leaf in leaf_tuple),
        #        tuple(tuple(branch_sizes) for branch_sizes in branch_structure)
        #    )

        #def construct_tree(
        #    pseudo_tree: tuple[list, list[list[int]]]
        #) -> list:
        #    leaf_tuple, branch_structure = pseudo_tree
        #    result = leaf_tuple
        #    for branch_sizes in reversed(branch_structure):
        #        result = [
        #            result[begin_index:end_index]
        #            for begin_index, end_index in it.pairwise([0, *it.accumulate(branch_sizes)])
        #        ]
        #    return result

        slot = self._get_slot(instance)
        if (registered_elements := slot._get()) is None:
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
            #else:
            #    data = self._composer(iter(elements))
            # As variables do not have parameters,
            # it's guaranteed that if `linked_variable_slots` is not empty,
            # `slot` is a property slot and is indeed linked to variable slots.
            #slot._link(linked_variable_slots)
            #for variable_slot in linked_variable_slots:
            #    slot._link(variable_slot)
            slot._set(
                elements=registered_elements,
                parameter_key=registered_parameter_key,
                linked_slots=linked_variable_slots
            )
            #data = self._composer(iter(elements))
        return tuple(registered_element._value for registered_element in registered_elements)

    def _set_elements(
        self,
        instance: "LazyObject",
        elements: tuple[_T, ...]
    ) -> None:
        slot = self._get_slot(instance)
        assert slot._is_writable, "Attempting to write to a readonly slot"
        # Guaranteed to be a variable slot. Expire linked property slots.
        for expired_property_slot in slot._iter_linked_slots():
            #expired_property_slot._expire()
            #expired_property_slot._unlink()
            #for variable_slot in expired_property_slot._iter_linked_slots():
            #    expired_property_slot._unlink(variable_slot)
            expired_property_slot._expire()
        #slot._expire_property_slots()  # ???
        #self._set_slot_data(slot, data)
        #elements = self._register(data)
        registered_elements = self._register_elements(elements)
        slot._set(
            elements=registered_elements,
            parameter_key=None,
            linked_slots=set()
        )

    #def _get_slot_data(
    #    self,
    #    slot: LazySlot[_DataT]
    #) -> _DataT:
    #    pass

    #def _frozen_element(
    #    self,
    #    element: _T
    #) -> None:
    #    if not self._frozen:
    #        return
    #    if not isinstance(element, LazyObject):
    #        return
    #    for descriptor in type(element)._iter_variable_descriptors():
    #        descriptor._get_slot(element)._is_writable = False

    #def _register_element(
    #    self,
    #    element: _T
    #) -> _T:
    #    #hash_processor_cls = self._hash_processor_cls
    #    #hash_table = self._registration
    #    #hash_key = hash_processor_cls._hash(element)
    #    #if (hasher := self._hasher) is None:
    #    #    self._frozen_element(element)
    #    #    return element
    #    key = self._hasher(element)
    #    registration = self._registration
    #    if (registered_element := registration.get(key)) is None:
    #        self._frozen_element(element)
    #        registered_element = element
    #        registration[key] = element
    #    return registered_element
    #    #return self._registration.setdefault(
    #    #    self._hash_processor_cls._hash(element),
    #    #    self._hash_processor_cls._frozen(element)
    #    #)
    #    #if (registered_element := hash_table.get(hash_key)) is None:
    #    #    hash_processor_cls._frozen(element)
    #    #    hash_table[hash_key] = element
    #    #    registered_element = element
    #    #return registered_element

    #def _get_linked_variable_slots(
    #    self,
    #    instance: "LazyObject"
    #) -> set[LazySlot]:
    #    slot = self._get_slot(instance)
    #    return {slot} if self._is_variable else set(slot._iter_linked_slots())

    def _register_parameter_key(
        self,
        parameter_key: Hashable
    ) -> Registered[Hashable]:
        return self._parameter_key_registration.register(parameter_key, parameter_key)
        #parameter_key_registration = self._parameter_key_registration
        #if (registered_parameter_key := parameter_key_registration.get(parameter_key)) is None:
        #    registered_parameter_key = Registered(parameter_key)
        #    parameter_key_registration[parameter_key] = registered_parameter_key
        #return registered_parameter_key

    def _register_elements(
        self,
        elements: tuple[_T, ...]
    ) -> tuple[Registered[_T], ...]:

        #def register_element(
        #    element: _T,
        #    hash_table: weakref.WeakValueDictionary[_HashInT, _HashOutT, _T],
        #    hash_processor_cls: type[HashProcessor[_T, _HashInT, _HashOutT]]
        #) -> _T:
        #    hash_key = hash_processor_cls._hash(element)
        #    if (registered_element := hash_table.get(hash_key)) is None:
        #        hash_processor_cls._frozen(element)
        #        hash_table[hash_key] = element
        #        registered_element = element
        #    return registered_element

        #storage_processor_cls = self._storage_processor_cls
        ##frozen_processor_cls = self._frozen_processor_cls
        #hash_processor_cls = self._hash_processor_cls
        #hash_table = self._registration
        ##old_elements = list(storage_processor_cls._iter_elements(slot._get_data()))

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
                #if slot._is_writable:
                #    slot._is_writable = False
                #    if (variable_elements := slot._get_elements()) is not None:
                #        for variable_element in variable_elements:
                #            freeze(variable_element)

        #def register_element(
        #    element: _T,
        #    element_registration: Registration[Hashable, _T],
        #    hasher: Callable[[_T], Hashable],
        #    frozen: bool
        #) -> Registered[_T]:
        #    #if element in element_registration.values():
        #    #    return element
        #    if frozen:
        #        freeze(element)
        #    key = hasher(element)
        #    return element_registration.register(key, element)
        #    #if (registered_element := element_registration.get(key)) is None:
        #    #    registered_element = Registered(element)
        #    #    element_registration[key] = registered_element
        #    #return registered_element

        if self._frozen:
            for element in elements:
                freeze(element)
        element_registration = self._element_registration
        hasher = self._hasher
        #frozen = self._frozen
        return tuple(
            element_registration.register(hasher(element), element)
            for element in elements
        )

        #elements = list(storage_processor_cls._iter_elements(data))
        #if len(old_elements) == len(elements) and all(
        #    hash_processor_cls._hash(old_element) == hash_processor_cls._hash(new_element)
        #    for old_element, new_element in zip(old_elements, elements, strict=True)
        #):
        #    return
        #for expired_property_slot in slot._iter_linked_slots():
        #    for variable_slot in expired_property_slot._iter_linked_slots():
        #        expired_property_slot._unlink(variable_slot)
        #    expired_property_slot._set_data(None)
        #slot._set_data(data)
        #for element in elements:
        #    hash_processor_cls._frozen(element)
        #storage_processor_cls._prepare_data(registered_data, weakref.ref(slot))
        #return registered_data
        #slot._set_data(registered_data)


#class LazyVariableDescriptor(LazyDescriptor[_InstanceT, _DataT, _T, _HashInT, _HashOutT]):
#    __slots__ = ()

    #def _init_instance(
    #    self,
    #    instance: _InstanceT
    #) -> None:
    #    slot = LazySlot()
    #    data = self._method()
    #    self._storage_processor_cls._prepare_data(slot, data)
    #    #slot._set_data(data)
    #    self._set_slot(instance, slot)

    #def _initial_data(
    #    self,
    #    instance_cls: type[_InstanceT]
    #) -> _DataT | None:
    #    return self._method(instance_cls)

    #def _get_slot_data(
    #    self,
    #    slot: LazySlot[_DataT]
    #) -> _DataT:
    #    return slot._get_data()

    #def _set_slot_data(
    #    self,
    #    slot: LazySlot[_DataT],
    #    data: _DataT
    #) -> None:
    #    assert slot._is_writable
    #    storage_processor_cls = self._storage_processor_cls
    #    #frozen_processor_cls = self._frozen_processor_cls
    #    hash_processor_cls = self._hash_processor_cls
    #    #old_elements = list(storage_processor_cls._iter_elements(slot._get_data()))
    #    elements = list(storage_processor_cls._iter_elements(data))
    #    #if len(old_elements) == len(elements) and all(
    #    #    hash_processor_cls._hash(old_element) == hash_processor_cls._hash(new_element)
    #    #    for old_element, new_element in zip(old_elements, elements, strict=True)
    #    #):
    #    #    return
    #    slot._expire_property_slots()
    #    #for expired_property_slot in slot._iter_linked_slots():
    #    #    for variable_slot in expired_property_slot._iter_linked_slots():
    #    #        expired_property_slot._unlink(variable_slot)
    #    #    expired_property_slot._set_data(None)
    #    #slot._set_data(data)
    #    for element in elements:
    #        hash_processor_cls._frozen(element)
    #    storage_processor_cls._prepare_data(slot, data)


#class LazyPropertyDescriptor(LazyDescriptor[_InstanceT, _DataT, _T, _HashInT, _HashOutT]):
#    __slots__ = ()

#    def __init__(
#        self,
#        method: Callable[..., _DataT],
#        #element_type: type[_T],
#        logic_processor_cls: type[LogicProcessor],
#        storage_processor_cls: type[StorageProcessor[_DataT, _T]],
#        frozen_processor_cls: type[FrozenProcessor],
#        hash_processor_cls: type[HashProcessor[_T, _HashInT, _HashOutT]],
#        cache_capacity: int = 128
#    ) -> None:
#        super().__init__(
#            method=method,
#            #element_type=element_type,
#            logic_processor_cls=logic_processor_cls,
#            storage_processor_cls=storage_processor_cls,
#            frozen_processor_cls=frozen_processor_cls,
#            hash_processor_cls=hash_processor_cls
#        )
#        self._cache: Cache[Hashable, _DataT] = Cache(capacity=cache_capacity)
#        self._finalize_method: Callable[[type[_InstanceT], _T], None] | None = None
#        self._descriptor_name_chains: tuple[tuple[str, ...], ...] = NotImplemented

#    def _initial_data(
#        self,
#        instance_cls: type[_InstanceT]
#    ) -> _DataT | None:
#        return None

#    def _get_slot_data(
#        self,
#        instance_cls: type[_InstanceT],
#        slot: LazySlot[_DataT]
#    ) -> _DataT:
#        return slot._get_data()

#    def _set_slot_data(
#        self,
#        instance_cls: type[_InstanceT],
#        slot: LazySlot[_DataT],
#        data: _DataT
#    ) -> None:
#        raise ValueError("Attempting to set a property")

#    def finalizer(
#        self,
#        finalize_method: Callable[[type[_InstanceT], _DataT], None]
#    ) -> Callable[[type[_InstanceT], _DataT], None]:
#        self._finalize_method = finalize_method.__func__
#        return finalize_method


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
            parameter_annotation: type,
            root_class: type[LazyObject]
        ) -> tuple[LazyDescriptor, ...]:
            descriptor_chain: list[LazyDescriptor] = []
            element_type = root_class
            #expected_annotation = element_type
            collection_level = 0
            for descriptor_name in name_chain:
                #print(name_chain, element_type)
                assert issubclass(element_type, LazyObject)
                descriptor = element_type._lazy_descriptors[descriptor_name]
                descriptor_chain.append(descriptor)
                element_type = descriptor._element_type
                #expected_annotation = descriptor._element_type_annotation
                if descriptor._is_multiple:
                    collection_level += 1
            expected_annotation = descriptor_chain[-1]._element_type_annotation
            for _ in range(collection_level):
                expected_annotation = tuple[expected_annotation, ...]
            #print(cls, name_chain, expected_annotation, parameter_annotation)
            assert expected_annotation == parameter_annotation or isinstance(expected_annotation, TypeVar)
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
            #print(isinstance(element_type_annotation, GenericAlias), element_type_annotation)
            descriptor._element_type = element_type

            descriptor._descriptor_chains = tuple(
                get_descriptor_chain(
                    name_chain=tuple(f"_{name_segment}_" for name_segment in parameter_name.split("__")),
                    parameter_annotation=parameter.annotation,
                    root_class=cls
                )
                for parameter_name, parameter in signature.parameters.items()
            )
            if descriptor._is_variable:
                assert not descriptor._descriptor_chains
            #descriptor_chains: list[tuple[LazyDescriptor, ...]] = []
            #for parameter_name, parameter in signature.parameters.items():
            #    name_chain = tuple(f"_{name_segment}_" for name_segment in parameter_name.split("__"))
            #    descriptor_chain = get_descriptor_chain(
            #        name_chain=name_chain,
            #        parameter_annotation=parameter.annotation,
            #        root_class=cls
            #    )
            #    descriptor_chains.append(descriptor_chain)

            #    #element_type = cls
            #    #collection_level: int = 0
            #    #for descriptor_name in name_chain:
            #    #    assert issubclass(element_type, LazyObject)
            #    #    descriptor = element_type._lazy_descriptors[descriptor_name]
            #    #    assert not isinstance(descriptor.converter, LazyExternalConverter)
            #    #    if isinstance(descriptor.converter, LazyCollectionConverter):
            #    #        collection_level += 1
            #    #    element_type = descriptor.element_type
            #    #descriptor = element_type._lazy_descriptors[descriptor_name_chain[-1]]
            #    #requires_unwrapping = isinstance(descriptor.converter, LazyExternalConverter)
            #    #requires_unwrapping_list.append(requires_unwrapping)

            #    #if requires_unwrapping:
            #    #    assert descriptor.element_type is LazyWrapper
            #    #expected_annotation = descriptor.return_annotation
            #    #for _ in range(collection_level):
            #    #    expected_annotation = list[expected_annotation]
            #    #assert expected_annotation == parameter.annotation
            #descriptor._descriptor_chains = tuple(descriptor_chains)

            overridden_descriptor = base_cls._lazy_descriptors.get(name)
            if overridden_descriptor is not None:
                #if "__weakref__" in dir(element_type):
                #    descriptor._registration = weakref.WeakValueDictionary()  # TODO: weakref to dataclass
                #else:
                #descriptor._registration = weakref.WeakValueDictionary()
                #else:
                descriptor._element_registration = overridden_descriptor._element_registration
                assert descriptor._can_override(overridden_descriptor)

        #base_classes = tuple(
        #    base_class
        #    for base_class in reversed(cls.__mro__)
        #    if issubclass(base_class, LazyObject)
        #)
        #base_descriptors = {
        #    descriptor._name: descriptor
        #    for base_class in base_classes
        #    for descriptor in base_class._lazy_descriptors
        #}
        #new_descriptors
        #new_descriptor_items = [
        #    (name, descriptor, inspect.signature(descriptor._method, locals={cls.__name__: cls}, eval_str=True))
        #    for name, descriptor in cls.__dict__.items()
        #    if isinstance(descriptor, LazyDescriptor)
        #]

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
            #src_value = copy.copy(self.__getattribute__(slot_name))
            ## TODO: This looks like a temporary patch... Is there any better practice?
            #if isinstance(src_value, weakref.WeakSet):
            #    # Use `WeakSet.copy` instead of `copy.copy` for better behavior.
            #    dst_value = src_value.copy()
            #else:
            #    dst_value = copy.copy(src_value)
            #result.__setattr__(slot_name, dst_value)
        return result

    #@classmethod
    #def _iter_variable_descriptors(cls) -> Iterator[LazyDescriptor]:
    #    for descriptor in cls._lazy_descriptors.values():
    #        if descriptor._is_variable:
    #            yield descriptor


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
