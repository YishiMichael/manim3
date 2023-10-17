from __future__ import annotations


import weakref
from abc import (
    ABC,
    abstractmethod
)
from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    Iterator,
    Never,
    Self,
    overload
)

from .lazy_slot import LazySlot

if TYPE_CHECKING:
    from .lazy_object import LazyObject


class Registered[T]:
    __slots__ = (
        "__weakref__",
        "_value"
    )

    def __init__(
        self: Self,
        value: T
    ) -> None:
        super().__init__()
        self._value: T = value


class Registration[KT: Hashable, VT](weakref.WeakValueDictionary[KT, Registered[VT]]):
    __slots__ = ()

    def register(
        self: Self,
        key: KT,
        value: VT
    ) -> Registered[VT]:
        if (registered_value := self.get(key)) is None:
            registered_value = Registered(value)
            self[key] = registered_value
        return registered_value


class Cache[KT: Hashable, VT](weakref.WeakKeyDictionary[KT, VT]):
    __slots__ = ("_capacity",)

    def __init__(
        self: Self,
        capacity: int
    ) -> None:
        super().__init__()
        self._capacity: int = capacity

    def set(
        self: Self,
        key: KT,
        value: VT
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
        self: Self,
        leaf_tuple: tuple,
        branch_structure: tuple[tuple[int, ...], ...]
    ) -> None:
        super().__init__()
        self._leaf_tuple: tuple = leaf_tuple
        self._branch_structure: tuple[tuple[int, ...], ...] = branch_structure

    def key(
        self: Self
    ) -> Hashable:
        return (
            tuple(id(leaf) for leaf in self._leaf_tuple),
            self._branch_structure
        )

    def tree(
        self: Self
    ) -> tuple:

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


class LazyDescriptor[T, DataT](ABC):
    __slots__ = (
        "_method",
        "_name",
        "_descriptor_info_chains",
        #"_is_plural",
        #"_decomposer",
        #"_composer",
        "_is_variable",
        "_hasher",
        "_freeze",
        "_freezer",
        "_cache",
        "_parameter_key_registration",
        "_element_registration"
        #"_element_type",
        #"_element_type_annotation",
        #"_descriptor_chains"
    )

    def __init__(
        self: Self,
        method: Callable[..., DataT],
        #is_plural: bool,
        #decomposer: Callable[[DataT], tuple[T, ...]],
        #composer: Callable[[tuple[T, ...]], DataT],
        is_variable: bool,
        hasher: Callable[[T], Hashable],
        freeze: bool,
        cache_capacity: int
    ) -> None:
        assert isinstance(method, staticmethod)
        assert hasher is id or freeze
        super().__init__()
        self._method: Callable[..., DataT] = method.__func__
        self._name: str = method.__name__
        self._descriptor_info_chains: tuple[tuple[tuple[str, bool], ...], ...] = NotImplemented
        #self._is_plural: bool = is_plural
        #self._decomposer: Callable[[DataT], tuple[T, ...]] = decomposer
        #self._composer: Callable[[tuple[T, ...]], DataT] = composer
        self._is_variable: bool = is_variable
        self._hasher: Callable[[T], Hashable] = hasher
        self._freeze: bool = freeze
        self._freezer: Callable[[T], None] = type(self)._empty_freezer
        self._cache: Cache[Registered[Hashable], tuple[Registered[T], ...]] = Cache(capacity=cache_capacity)
        self._parameter_key_registration: Registration[Hashable, Hashable] = Registration()
        self._element_registration: Registration[Hashable, T] = Registration()

        #self._element_type: type[T] | None = None
        #self._element_type_annotation: type = NotImplemented
        #self._descriptor_chains: tuple[tuple[LazyDescriptor, ...], ...] = NotImplemented

    @overload
    def __get__(
        self: Self,
        instance: None,
        owner: type[LazyObject] | None = None
    ) -> Self: ...

    @overload
    def __get__(
        self: Self,
        instance: LazyObject,
        owner: type[LazyObject] | None = None
    ) -> DataT: ...

    def __get__(
        self: Self,
        instance: LazyObject | None,
        owner: type[LazyObject] | None = None
    ) -> Self | DataT:
        if instance is None:
            return self
        return self._composer(self._get_elements(instance))

    def __set__(
        self: Self,
        instance: LazyObject,
        data: DataT
    ) -> None:
        return self._set_elements(instance, self._decomposer(data))

    def __delete__(
        self: Self,
        instance: LazyObject
    ) -> Never:
        raise TypeError("Cannot delete attributes of a lazy object")

    @classmethod
    @property
    @abstractmethod
    def _is_plural(
        cls: type[Self]
    ) -> bool:
        pass

    @classmethod
    @abstractmethod
    def _decomposer(
        cls: type[Self],
        data: DataT
    ) -> tuple[T, ...]:
        pass

    @classmethod
    @abstractmethod
    def _composer(
        cls: type[Self],
        elements: tuple[T, ...]
    ) -> DataT:
        pass

    #def _can_override(
    #    self: Self,
    #    descriptor: LazyDescriptor
    #) -> bool:
    #    return (
    #        self._is_plural is descriptor._is_plural
    #        #and self._hasher is descriptor._hasher
    #        #and (self._freeze or not descriptor._freeze)
    #        and (
    #            self._element_type_annotation == descriptor._element_type_annotation
    #            or self._element_type is None
    #            or descriptor._element_type is None
    #            or issubclass(self._element_type, descriptor._element_type)
    #        )
    #    )

    def _register_parameter_key(
        self: Self,
        parameter_key: Hashable
    ) -> Registered[Hashable]:
        return self._parameter_key_registration.register(parameter_key, parameter_key)

    def _register_elements(
        self: Self,
        elements: tuple[T, ...]
    ) -> tuple[Registered[T], ...]:
        if self._freeze:
            freezer = self._freezer
            for element in elements:
                freezer(element)
        element_registration = self._element_registration
        hasher = self._hasher
        return tuple(
            element_registration.register(hasher(element), element)
            for element in elements
        )

    @classmethod
    def _empty_freezer(
        cls: type[Self],
        element: T
    ) -> None:
        return

    def _init(
        self: Self,
        instance: LazyObject
    ) -> None:
        slot = LazySlot()
        slot._is_writable = self._is_variable
        self._set_slot(instance, slot)

    def _get_slot(
        self: Self,
        instance: LazyObject
    ) -> LazySlot[T]:
        return instance._lazy_slots[self._name]

    def _set_slot(
        self: Self,
        instance: LazyObject,
        slot: LazySlot[T]
    ) -> None:
        instance._lazy_slots[self._name] = slot

    def _get_elements(
        self: Self,
        instance: LazyObject
    ) -> tuple[T, ...]:

        def get_leaf_items(
            descriptor_name: str,
            leaf: LazyObject
        ) -> tuple[tuple, set[LazySlot]]:
            descriptor = type(leaf)._lazy_descriptors[descriptor_name]
            slot = descriptor._get_slot(leaf)
            elements = descriptor._get_elements(leaf)
            leaf_associated_variable_slots = {slot} if descriptor._is_variable else set(slot.iter_associated_slots())
            return elements, leaf_associated_variable_slots

        def iter_descriptor_items(
            descriptor_info_chain: tuple[tuple[str, bool], ...],
            instance: LazyObject
        ) -> Iterator[tuple[tuple, tuple[int, ...] | None, set[LazySlot]]]:
            leaf_tuple: tuple = (instance,)
            for descriptor_name, is_plural in descriptor_info_chain:
                leaf_items = tuple(
                    get_leaf_items(descriptor_name, leaf)
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
                ) if is_plural else None
                descriptor_associated_variable_slots = set().union(*(
                    leaf_associated_variable_slots
                    for _, leaf_associated_variable_slots in leaf_items
                ))
                yield leaf_tuple, branch_sizes, descriptor_associated_variable_slots

        def get_parameter_items(
            descriptor_info_chain: tuple[tuple[str, bool], ...],
            instance: LazyObject
        ) -> tuple[PseudoTree, set[LazySlot]]:
            descriptor_items = tuple(iter_descriptor_items(descriptor_info_chain, instance))
            leaf_tuple, _, _ = descriptor_items[-1]
            pseudo_tree = PseudoTree(
                leaf_tuple=leaf_tuple,
                branch_structure=tuple(
                    branch_sizes
                    for _, branch_sizes, _ in descriptor_items
                    if branch_sizes is not None
                )
            )
            parameter_associated_variable_slots = set().union(*(
                descriptor_associated_variable_slots
                for _, _, descriptor_associated_variable_slots in descriptor_items
            ))
            return pseudo_tree, parameter_associated_variable_slots

        def get_pseudo_trees_and_associated_variable_slots(
            descriptor_info_chains: tuple[tuple[tuple[str, bool], ...], ...],
            instance: LazyObject
        ) -> tuple[tuple[PseudoTree, ...], set[LazySlot]]:
            parameter_items = tuple(
                get_parameter_items(descriptor_info_chain, instance)
                for descriptor_info_chain in descriptor_info_chains
            )
            pseudo_trees = tuple(pseudo_tree for pseudo_tree, _ in parameter_items)
            associated_variable_slots = set().union(*(
                parameter_associated_variable_slots for _, parameter_associated_variable_slots in parameter_items
            ))
            return pseudo_trees, associated_variable_slots

        slot = self._get_slot(instance)
        if (registered_elements := slot.get()) is None:
            # If there's at least a parameter, `slot` is guaranteed to be a property slot.
            # Associate it with variable slots.
            pseudo_trees, associated_variable_slots = get_pseudo_trees_and_associated_variable_slots(
                self._descriptor_info_chains, instance
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
                if self._freeze:
                    self._cache.set(registered_parameter_key, registered_elements)
            slot.set(
                elements=registered_elements,
                parameter_key=registered_parameter_key,
                associated_slots=associated_variable_slots
            )
        return tuple(registered_element._value for registered_element in registered_elements)

    def _set_elements(
        self: Self,
        instance: LazyObject,
        elements: tuple[T, ...]
    ) -> None:
        slot = self._get_slot(instance)
        assert slot._is_writable, "Attempting to write to a readonly slot"
        registered_elements = self._register_elements(elements)
        if registered_elements == slot.get():
            return
        # `slot` is guaranteed to be a variable slot. Expire associated property slots.
        for expired_property_slot in slot.iter_associated_slots():
            expired_property_slot.expire()
        slot.set(
            elements=registered_elements,
            parameter_key=None,
            associated_slots=set()
        )


class LazySingularDescriptor[T](LazyDescriptor[T, T]):
    __slots__ = ()

    @classmethod
    @property
    def _is_plural(
        cls: type[Self]
    ) -> bool:
        return False

    @classmethod
    def _decomposer(
        cls: type[Self],
        data: T
    ) -> tuple[T, ...]:
        return (data,)

    @classmethod
    def _composer(
        cls: type[Self],
        elements: tuple[T, ...]
    ) -> T:
        (element,) = elements
        return element


class LazyPluralDescriptor[T](LazyDescriptor[T, tuple[T, ...]]):
    __slots__ = ()

    @classmethod
    @property
    def _is_plural(
        cls: type[Self]
    ) -> bool:
        return True

    @classmethod
    def _decomposer(
        cls: type[Self],
        data: tuple[T, ...]
    ) -> tuple[T, ...]:
        return data

    @classmethod
    def _composer(
        cls: type[Self],
        elements: tuple[T, ...]
    ) -> tuple[T, ...]:
        return elements
