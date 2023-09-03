import inspect
import itertools as it
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


#_T = TypeVar("_T")
_KT = TypeVar("_KT", bound=Hashable)
_VT = TypeVar("_VT")
_DataT = TypeVar("_DataT")
_ElementT = TypeVar("_ElementT")
_HashableT = TypeVar("_HashableT", bound=Hashable)
_LazyDescriptorT = TypeVar("_LazyDescriptorT", bound="LazyDescriptor")
#_LazyObjectT = TypeVar("_LazyObjectT", bound="LazyObject")
#_Parameters = ParamSpec("_Parameters")


class Cache(Generic[_KT, _VT]):
    # Use `Cache` instead of `weakref.WeakValueDictionary` to
    # keep strong references to objects for a while even after
    # all external references to them disappear.
    __slots__ = (
        "_capacity",
        "_cache_dict"
    )

    def __init__(
        self,
        capacity: int
    ) -> None:
        super().__init__()
        self._capacity: int = capacity
        self._cache_dict: dict[_KT, _VT] = {}

    def get(
        self,
        key: _KT
    ) -> _VT | None:
        return self._cache_dict.get(key)

    def set(
        self,
        key: _KT,
        value: _VT
    ) -> None:
        cache_dict = self._cache_dict
        assert key not in cache_dict
        if len(cache_dict) == self._capacity:
            cache_dict.pop(next(iter(cache_dict)))
        cache_dict[key] = value

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
        "_leaf_list",
        "_branch_structure"
    )

    def __init__(
        self,
        leaf_list: list,
        branch_structure: list[list[int]]
    ) -> None:
        super().__init__()
        self._leaf_list: list = leaf_list
        self._branch_structure: list[list[int]] = branch_structure

    def key(self) -> Hashable:
        return (
            tuple(id(leaf) for leaf in self._leaf_list),
            tuple(tuple(branch_sizes) for branch_sizes in self._branch_structure)
        )

    def tree(self) -> list:
        result = self._leaf_list
        for branch_sizes in reversed(self._branch_structure):
            result = [
                result[begin_index:end_index]
                for begin_index, end_index in it.pairwise([0, *it.accumulate(branch_sizes)])
            ]
        return result


#class LazyList(list[_ElementT]):
#    __slots__ = ("_slot_backref",)
#
#    def __init__(
#        self,
#        elements: Iterable[_ElementT]
#    ) -> None:
#        super().__init__(elements)
#        self._slot_backref: weakref.ref[LazySlot[LazyList[_ElementT]]] | None = None

    #def _iter_elements(self) -> Iterator[_ElementT]:
    #    yield from self._elements

    #def _write(
    #    self,
    #    src: "LazyList[_ElementT]"
    #) -> None:
    #    self._elements = list(src._elements)

    #def _copy_container(self) -> "LazyList[_ElementT]":
    #    return LazyList(
    #        elements=self._elements
    #    )

    # list methods
    # Only those considered necessary are ported.

    #@staticmethod
    #def _force_expire(
    #    method: "Callable[Concatenate[LazyList[_ElementT], _Parameters], _T]",
    #) -> "Callable[Concatenate[LazyList[_ElementT], _Parameters], _T]":

    #    @wraps(method)
    #    def new_method(
    #        self: "LazyList[_ElementT]",
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
    #    value: _ElementT
    #) -> None: ...

    #@overload
    #def __setitem__(
    #    self,
    #    index: slice,
    #    value: Iterable[_ElementT]
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
    #    value: _ElementT
    #) -> None:
    #    return super().insert(index, value)

    #@_force_expire
    #def append(
    #    self,
    #    value: _ElementT
    #) -> None:
    #    return super().append(value)

    #@_force_expire
    #def extend(
    #    self,
    #    values: Iterable[_ElementT]
    #) -> None:
    #    return super().extend(values)

    #@_force_expire
    #def reverse(self) -> None:
    #    return super().reverse()

    #@_force_expire
    #def pop(
    #    self,
    #    index: int = -1
    #) -> _ElementT:
    #    return super().pop(index)

    #@_force_expire
    #def remove(
    #    self,
    #    value: _ElementT
    #) -> None:
    #    return super().remove(value)

    #@_force_expire
    #def clear(self) -> None:
    #    return super().clear()

    ## Additional methods for convenience.

    #@_force_expire
    #def eliminate(
    #    self,
    #    values: Iterable[_ElementT]
    #) -> None:
    #    for value in values:
    #        self.remove(value)

    #@_force_expire
    #def reset(
    #    self,
    #    values: Iterable[_ElementT]
    #) -> None:
    #    self.clear()
    #    self.extend(values)


class LazySlot(Generic[_DataT]):
    __slots__ = (
        "_data",
        "_linked_slots",
        "_is_writable"
    )

    def __init__(self) -> None:
        super().__init__()
        self._data: _DataT | None = None
        self._linked_slots: weakref.WeakSet[LazySlot] = weakref.WeakSet()
        self._is_writable: bool = True

    def _get_data(self) -> _DataT | None:
        #assert (data := self._data) is not None
        return self._data

    def _set_data(
        self,
        data: _DataT | None
    ) -> None:
        self._data = data

    def _iter_linked_slots(self) -> "Iterator[LazySlot]":
        return iter(self._linked_slots)

    def _link(
        self,
        slot: "LazySlot"
    ) -> None:
        self._linked_slots.add(slot)
        slot._linked_slots.add(self)

    def _unlink(
        self,
        slot: "LazySlot"
    ) -> None:
        self._linked_slots.remove(slot)
        slot._linked_slots.remove(self)

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


#class HashProcessor(ABC, Generic[_ElementT, _HashableT]):
#    __slots__ = ()
#
#    def __new__(cls) -> None:
#        pass
#
#    @classmethod
#    @abstractmethod
#    def _hash(
#        cls,
#        element: _ElementT
#    ) -> _HashableT:
#        pass
#
#    #@classmethod
#    #@abstractmethod
#    #def _frozen(
#    #    cls,
#    #    element: _ElementT
#    #) -> _ElementT:
#    #    pass


#class StorageProcessor(ABC, Generic[_DataT, _ElementT]):
#    __slots__ = ()

#    def __new__(cls) -> None:
#        pass

#    @classmethod
#    @abstractmethod
#    def _iter_elements(
#        cls,
#        data: _DataT
#    ) -> Iterator[_ElementT]:
#        pass

#    @classmethod
#    @abstractmethod
#    def _assemble_elements(
#        cls,
#        elements: Iterator[_ElementT]
#    ) -> _DataT:
#        pass


#class UnitaryStorageProcessor(Generic[_ElementT], StorageProcessor[_ElementT, _ElementT]):
#    __slots__ = ()

#    @classmethod
#    def _iter_elements(
#        cls,
#        data: _ElementT
#    ) -> Iterator[_ElementT]:
#        yield data

#    @classmethod
#    def _assemble_elements(
#        cls,
#        elements: Iterator[_ElementT]
#    ) -> _ElementT:
#        return next(elements)
#        #slot._set_data(data)


#class DynamicStorageProcessor(Generic[_ElementT], StorageProcessor[list[_ElementT], _ElementT]):
#    __slots__ = ()

#    @classmethod
#    def _iter_elements(
#        cls,
#        data: list[_ElementT]
#    ) -> Iterator[_ElementT]:
#        yield from data

#    @classmethod
#    def _assemble_elements(
#        cls,
#        elements: Iterator[_ElementT]
#    ) -> list[_ElementT]:
#        return list(elements)
#        ##assert data._slot_backref is None
#        #data._slot_backref = slot_backref
#        ##slot._set_data(data)
#        #return data


#class LogicProcessor(ABC, Generic[_DataT, _ElementT]):
#    __slots__ = ()

#    def __new__(cls) -> None:
#        pass

#    @classmethod
#    @abstractmethod
#    def _init_slot(
#        cls,
#        slot: LazySlot[_DataT]
#        #method: Callable[..., _DataT],
#        #storage_processor_cls: type[StorageProcessor[_DataT, _ElementT]]
#    ) -> None:
#        pass


#class VariableLogicProcessor(LogicProcessor[_DataT, _ElementT]):
#    __slots__ = ()

#    @classmethod
#    def _init_slot(
#        cls,
#        slot: LazySlot[_DataT]
#        #method: Callable[..., _DataT],
#        #storage_processor_cls: type[StorageProcessor[_DataT, _ElementT]]
#    ) -> None:
#        pass
#        #data = method()
#        #storage_processor_cls._prepare_data(data, weakref.ref(slot))


#class PropertyLogicProcessor(LogicProcessor[_DataT, _ElementT]):
#    __slots__ = ()

#    @classmethod
#    def _init_slot(
#        cls,
#        slot: LazySlot[_DataT]
#        #method: Callable[..., _DataT],
#        #storage_processor_cls: type[StorageProcessor[_DataT, _ElementT]]
#    ) -> None:
#        slot._is_writable = False


#class FrozenProcessor(ABC, Generic[_ElementT]):
#    __slots__ = ()

#    def __new__(cls) -> None:
#        pass

#    @classmethod
#    @abstractmethod
#    def _frozen(
#        cls,
#        element: _ElementT
#    ) -> None:
#        pass


#class BlankFrozenProcessor(FrozenProcessor[_ElementT]):
#    __slots__ = ()

#    @classmethod
#    def _frozen(
#        cls,
#        element: _ElementT
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


class LazyDescriptor(ABC, Generic[_DataT, _ElementT, _HashableT]):
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
        #"_finalize_method",
        "_element_type",
        #"_parameter_name_chains",
        "_descriptor_chains",
        "_registration"
    )

    def __init__(
        self,
        method: Callable[..., _DataT],
        is_multiple: bool,
        decomposer: Callable[[_DataT], Iterator[_ElementT]],
        composer: Callable[[Iterator[_ElementT]], _DataT],
        is_variable: bool,
        hasher: Callable[[_ElementT], _HashableT] | None,
        frozen: bool,
        cache_capacity: int
        #element_type: type[_ElementT],
        #logic_processor_cls: type[LogicProcessor[_DataT, _ElementT]],
        #storage_processor_cls: type[StorageProcessor[_DataT, _ElementT]],
        #frozen_processor_cls: type[FrozenProcessor[_ElementT]],
    ) -> None:
        assert hasher is None or frozen
        super().__init__()
        self._method: Callable[..., _DataT] = method
        self._name: str = method.__name__
        self._is_multiple: bool = is_multiple
        self._decomposer: Callable[[_DataT], Iterator[_ElementT]] = decomposer
        self._composer: Callable[[Iterator[_ElementT]], _DataT] = composer
        self._is_variable: bool = is_variable
        self._hasher: Callable[[_ElementT], _HashableT] | None = hasher
        self._frozen: bool = frozen
        #self._logic_processor_cls: type[LogicProcessor[_DataT, _ElementT]] = logic_processor_cls
        #self._storage_processor_cls: type[StorageProcessor[_DataT, _ElementT]] = storage_processor_cls
        #self._frozen_processor_cls: type[FrozenProcessor[_ElementT]] = frozen_processor_cls
        self._cache: Cache[Hashable, _DataT] = Cache(capacity=cache_capacity)
        #self._finalize_method: Callable[[type[_InstanceT], _ElementT], None] | None = None
        self._element_type: type[_ElementT] = NotImplemented
        #self._parameter_name_chains: tuple[tuple[str, ...], ...] = NotImplemented
        self._descriptor_chains: tuple[tuple[LazyDescriptor, ...], ...] = NotImplemented
        self._registration: weakref.WeakValueDictionary[_HashableT, _ElementT] = NotImplemented  # Shared when overridden.

    @overload
    def __get__(
        self: _LazyDescriptorT,
        instance: None,
        owner: "type[LazyObject] | None" = None
    ) -> _LazyDescriptorT: ...  # TODO: typing

    @overload
    def __get__(
        self,
        instance: "LazyObject",
        owner: "type[LazyObject] | None" = None
    ) -> _DataT: ...

    def __get__(
        self: _LazyDescriptorT,
        instance: "LazyObject" | None,
        owner: "type[LazyObject] | None" = None
    ) -> _LazyDescriptorT | _DataT:
        if instance is None:
            return self
        return self._get(instance)

    def __set__(
        self,
        instance: "LazyObject",
        data: _DataT
    ) -> None:
        return self._set(instance, data)

    def __delete__(
        self,
        instance: "LazyObject"
    ) -> None:
        raise TypeError("Cannot delete attributes of a lazy object")

    def _get_slot(
        self,
        instance: "LazyObject"
    ) -> LazySlot[_DataT]:
        return instance._lazy_slots[self._name]

    def _set_slot(
        self,
        instance: "LazyObject",
        slot: LazySlot[_DataT]
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

    def _get(
        self,
        instance: "LazyObject"
    ) -> _DataT:

        def get_pseudo_tree_and_linked_variable_slots(
            descriptor_chain: tuple[LazyDescriptor, ...],
            instance: LazyObject
        ) -> tuple[PseudoTree, list[LazySlot]]:
            leaf_list: list = [instance]
            branch_structure: list[list[int]] = []
            linked_variable_slots: list[LazySlot] = []
            for descriptor in descriptor_chain:
                elements_list = [
                    list(descriptor._decomposer(type(leaf)._lazy_descriptors[descriptor._name]._get(leaf)))
                    for leaf in leaf_list
                ]
                if descriptor._is_variable:
                    linked_variable_slots.extend(
                        descriptor._get_slot(leaf)
                        for leaf in leaf_list
                    )
                else:
                    linked_variable_slots.extend(it.chain.from_iterable(
                        descriptor._get_slot(leaf)._iter_linked_slots()
                        for leaf in leaf_list
                    ))
                leaf_list = list(it.chain.from_iterable(elements_list))
                if descriptor._is_multiple:
                    branch_structure.append([len(elements) for elements in elements_list])
            return PseudoTree(leaf_list, branch_structure), linked_variable_slots

        def get_pseudo_trees_and_linked_variable_slots(
            descriptor_chains: tuple[tuple[LazyDescriptor, ...], ...],
            instance: LazyObject
        ) -> tuple[list[PseudoTree], list[LazySlot]]:
            parameter_items = [
                get_pseudo_tree_and_linked_variable_slots(descriptor_chain, instance)
                for descriptor_chain in descriptor_chains
            ]
            return [pseudo_tree for pseudo_tree, _ in parameter_items], list(it.chain.from_iterable(
                parameter_linked_variable_slots for _, parameter_linked_variable_slots in parameter_items
            ))

        #def pseudo_tree_key(
        #    pseudo_tree: tuple[list, list[list[int]]]
        #) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        #    leaf_list, branch_structure = pseudo_tree
        #    return (
        #        tuple(id(leaf) for leaf in leaf_list),
        #        tuple(tuple(branch_sizes) for branch_sizes in branch_structure)
        #    )

        #def construct_tree(
        #    pseudo_tree: tuple[list, list[list[int]]]
        #) -> list:
        #    leaf_list, branch_structure = pseudo_tree
        #    result = leaf_list
        #    for branch_sizes in reversed(branch_structure):
        #        result = [
        #            result[begin_index:end_index]
        #            for begin_index, end_index in it.pairwise([0, *it.accumulate(branch_sizes)])
        #        ]
        #    return result

        slot = self._get_slot(instance)
        if (data := slot._get_data()) is None:
            pseudo_trees, linked_variable_slots = get_pseudo_trees_and_linked_variable_slots(
                self._descriptor_chains, instance
            )
            key = tuple(
                pseudo_tree.key()
                for pseudo_tree in pseudo_trees
            )
            if (data := self._cache.get(key)) is None:
                data = self._register_data(self._method(*(
                    pseudo_tree.tree()
                    for pseudo_tree in pseudo_trees
                )))
                self._cache.set(key, data)
            # As variables do not have parameters,
            # it's guaranteed that if `linked_variable_slots` is not empty,
            # `slot` is a property slot and is indeed linked to variable slots.
            for variable_slot in linked_variable_slots:
                slot._link(variable_slot)
            slot._set_data(data)
        return data

    def _set(
        self,
        instance: "LazyObject",
        data: _DataT
    ) -> None:
        slot = self._get_slot(instance)
        assert slot._is_writable
        # Guaranteed to be a variable slot. Expire linked property slots.
        for expired_property_slot in slot._iter_linked_slots():
            #expired_property_slot._expire()
            for variable_slot in expired_property_slot._iter_linked_slots():
                expired_property_slot._unlink(variable_slot)
            expired_property_slot._set_data(None)
        #slot._expire_property_slots()  # ???
        #self._set_slot_data(slot, data)
        registered_data = self._register_data(data)
        slot._set_data(registered_data)

    #def _get_slot_data(
    #    self,
    #    slot: LazySlot[_DataT]
    #) -> _DataT:
    #    pass

    def _register_element(
        self,
        element: _ElementT
    ) -> _ElementT:
        #hash_processor_cls = self._hash_processor_cls
        #hash_table = self._registration
        #hash_key = hash_processor_cls._hash(element)
        if (hasher := self._hasher) is None:
            return element
        key = hasher(element)
        registration = self._registration
        if (registered_element := registration.get(key)) is None:
            if self._frozen and isinstance(element, LazyObject):
                for slot in element._iter_variable_slots():
                    slot._is_writable = False
            registered_element = element
            registration[key] = element
        return registered_element
        #return self._registration.setdefault(
        #    self._hash_processor_cls._hash(element),
        #    self._hash_processor_cls._frozen(element)
        #)
        #if (registered_element := hash_table.get(hash_key)) is None:
        #    hash_processor_cls._frozen(element)
        #    hash_table[hash_key] = element
        #    registered_element = element
        #return registered_element

    def _register_data(
        self,
        data: _DataT
    ) -> _DataT:

        #def register_element(
        #    element: _ElementT,
        #    hash_table: weakref.WeakValueDictionary[_HashableT, _ElementT],
        #    hash_processor_cls: type[HashProcessor[_ElementT, _HashableT]]
        #) -> _ElementT:
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

        return self._composer(
            self._register_element(element)
            for element in self._decomposer(data)
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


#class LazyVariableDescriptor(LazyDescriptor[_InstanceT, _DataT, _ElementT, _HashableT]):
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


#class LazyPropertyDescriptor(LazyDescriptor[_InstanceT, _DataT, _ElementT, _HashableT]):
#    __slots__ = ()

#    def __init__(
#        self,
#        method: Callable[..., _DataT],
#        #element_type: type[_ElementT],
#        logic_processor_cls: type[LogicProcessor],
#        storage_processor_cls: type[StorageProcessor[_DataT, _ElementT]],
#        frozen_processor_cls: type[FrozenProcessor],
#        hash_processor_cls: type[HashProcessor[_ElementT, _HashableT]],
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
#        self._finalize_method: Callable[[type[_InstanceT], _ElementT], None] | None = None
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
        "__weakref__",
        "_lazy_slots"
    )

    _lazy_descriptors: ClassVar[dict[str, LazyDescriptor]] = {}
    #_py_slots: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls) -> None:

        def get_descriptor_chain(
            name_chain: tuple[str, ...],
            root_class: type[LazyObject]
        ) -> tuple[list[LazyDescriptor], type]:
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
            expected_annotation = element_type
            for _ in range(collection_level):
                expected_annotation = list[expected_annotation]
            return descriptor_chain, expected_annotation

        super().__init_subclass__()
        base_cls = cls.__base__
        assert issubclass(base_cls, LazyObject)
        new_descriptors = {
            descriptor._name: descriptor
            for descriptor in cls.__dict__.items()
            if isinstance(descriptor, LazyDescriptor)
        }
        cls._lazy_descriptors = base_cls._lazy_descriptors | new_descriptors

        for name, descriptor in new_descriptors.items():
            signature = inspect.signature(descriptor._method, locals={cls.__name__: cls}, eval_str=True)

            return_annotation = signature.return_annotation
            if descriptor._is_multiple:
                assert return_annotation.__origin__ is list
                element_type = return_annotation.__args__[0]
            else:
                element_type = return_annotation
            descriptor._element_type = element_type

            descriptor_chains: list[tuple[LazyDescriptor, ...]] = []
            for parameter_name, parameter in signature.parameters.items():
                name_chain = tuple(f"_{name_segment}_" for name_segment in parameter_name.split("__"))
                descriptor_chain, expected_annotation = get_descriptor_chain(name_chain, cls)
                assert expected_annotation == parameter.annotation
                descriptor_chains.append(tuple(descriptor_chain))

                #element_type = cls
                #collection_level: int = 0
                #for descriptor_name in name_chain:
                #    assert issubclass(element_type, LazyObject)
                #    descriptor = element_type._lazy_descriptors[descriptor_name]
                #    assert not isinstance(descriptor.converter, LazyExternalConverter)
                #    if isinstance(descriptor.converter, LazyCollectionConverter):
                #        collection_level += 1
                #    element_type = descriptor.element_type
                #descriptor = element_type._lazy_descriptors[descriptor_name_chain[-1]]
                #requires_unwrapping = isinstance(descriptor.converter, LazyExternalConverter)
                #requires_unwrapping_list.append(requires_unwrapping)

                #if requires_unwrapping:
                #    assert descriptor.element_type is LazyWrapper
                #expected_annotation = descriptor.return_annotation
                #for _ in range(collection_level):
                #    expected_annotation = list[expected_annotation]
                #assert expected_annotation == parameter.annotation
            descriptor._descriptor_chains = tuple(descriptor_chains)

            overridden_descriptor = base_cls._lazy_descriptors.get(name)
            if overridden_descriptor is None:
                descriptor._registration = weakref.WeakValueDictionary()
            else:
                descriptor._registration = overridden_descriptor._registration

                assert descriptor._is_multiple is overridden_descriptor._is_multiple
                assert descriptor._decomposer is overridden_descriptor._decomposer
                assert descriptor._composer is overridden_descriptor._composer
                assert descriptor._is_variable is overridden_descriptor._is_variable
                assert descriptor._hasher is overridden_descriptor._hasher
                assert descriptor._frozen is overridden_descriptor._frozen
                assert (
                    descriptor._element_type == overridden_descriptor._element_type
                    or issubclass(descriptor._element_type, overridden_descriptor._element_type)
                )

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
        for descriptor in type(self)._lazy_descriptors.values():
            descriptor._init(self)

    def _iter_variable_slots(self) -> Iterator[LazySlot]:
        for descriptor in type(self)._lazy_descriptors.values():
            if descriptor._is_variable:
                yield descriptor._get_slot(self)


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
        hasher: Callable[[_ElementT], _HashableT] | None
    ) -> Callable[[Callable[..., _ElementT]], LazyDescriptor[_ElementT, _ElementT, _HashableT]]:

        def singular_decomposer(
            data: _ElementT
        ) -> Iterator[_ElementT]:
            yield data

        def singular_composer(
            elements: Iterator[_ElementT]
        ) -> _ElementT:
            return next(elements)

        def result(
            method: Callable[[], _ElementT]
        ) -> LazyDescriptor[_ElementT, _ElementT, _HashableT]:
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
        hasher: Callable[[_ElementT], _HashableT] | None,
        frozen: bool,
        cache_capacity: int
    ) -> Callable[[Callable[..., list[_ElementT]]], LazyDescriptor[list[_ElementT], _ElementT, _HashableT]]:

        def multiple_decomposer(
            data: list[_ElementT]
        ) -> Iterator[_ElementT]:
            yield from data

        def multiple_composer(
            elements: Iterator[_ElementT]
        ) -> list[_ElementT]:
            return list(elements)

        def result(
            method: Callable[[], list[_ElementT]]
        ) -> LazyDescriptor[list[_ElementT], _ElementT, _HashableT]:
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
        hasher: Callable[[_ElementT], _HashableT] | None = None,
        frozen: bool = True
    ) -> Callable[[Callable[[], _ElementT]], LazyDescriptor[_ElementT, _ElementT, _HashableT]]:
        return cls._descriptor_singular(
            is_variable=True,
            hasher=hasher,
            frozen=frozen,
            cache_capacity=1
        )

    @classmethod
    def variable_list(
        cls,
        hasher: Callable[[_ElementT], _HashableT] | None = None,
        frozen: bool = True
    ) -> Callable[[Callable[[], list[_ElementT]]], LazyDescriptor[list[_ElementT], _ElementT, _HashableT]]:
        return cls._descriptor_multiple(
            is_variable=True,
            hasher=hasher,
            frozen=frozen,
            cache_capacity=1
        )

    @classmethod
    def property(
        cls,
        hasher: Callable[[_ElementT], _HashableT] | None = None,
        cache_capacity: int = 128
    ) -> Callable[[Callable[..., _ElementT]], LazyDescriptor[_ElementT, _ElementT, _HashableT]]:
        return cls._descriptor_singular(
            is_variable=False,
            hasher=hasher,
            frozen=True,
            cache_capacity=cache_capacity
        )

    @classmethod
    def property_list(
        cls,
        hasher: Callable[[_ElementT], _HashableT] | None = None,
        cache_capacity: int = 128
    ) -> Callable[[Callable[..., list[_ElementT]]], LazyDescriptor[list[_ElementT], _ElementT, _HashableT]]:
        return cls._descriptor_multiple(
            is_variable=False,
            hasher=hasher,
            frozen=True,
            cache_capacity=cache_capacity
        )
