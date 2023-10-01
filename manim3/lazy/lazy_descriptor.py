import re
import weakref
from abc import ABC
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Hashable,
    Iterator,
    TypeVar,
    overload
)

from .lazy_slot import LazySlot

if TYPE_CHECKING:
    from .lazy_object import LazyObject


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


class LazyDescriptor(ABC, Generic[_DataT, _T]):
    __slots__ = (
        "_method",
        "_name",
        "_is_multiple",
        "_decomposer",
        "_composer",
        "_is_variable",
        "_hasher",
        "_freeze",
        "_freezer",
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
        freeze: bool,
        cache_capacity: int
    ) -> None:
        assert re.fullmatch(r"_([^_]+_)+", method.__name__)
        assert hasher is id or freeze
        super().__init__()
        self._method: Callable[..., _DataT] = method
        self._name: str = method.__name__
        self._is_multiple: bool = is_multiple
        self._decomposer: Callable[[_DataT], tuple[_T, ...]] = decomposer
        self._composer: Callable[[tuple[_T, ...]], _DataT] = composer
        self._is_variable: bool = is_variable
        self._hasher: Callable[[_T], Hashable] = hasher
        self._freeze: bool = freeze
        self._freezer: Callable[[_T], None] = type(self)._empty_freezer
        self._cache: Cache[Registered[Hashable], tuple[Registered[_T], ...]] = Cache(capacity=cache_capacity)
        self._parameter_key_registration: Registration[Hashable, Hashable] = Registration()
        # Shared when overridden.
        self._element_registration: Registration[Hashable, _T] = Registration()

        self._element_type: type[_T] | None = None
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
            and (self._freeze or not descriptor._freeze)
            and (
                self._element_type_annotation == descriptor._element_type_annotation
                or self._element_type is None
                or descriptor._element_type is None
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
            leaf: "LazyObject",
            descriptor_name: str
        ) -> tuple[tuple, set[LazySlot]]:
            descriptor = type(leaf)._lazy_descriptors[descriptor_name]
            slot = descriptor._get_slot(leaf)
            elements = descriptor._get_elements(leaf)
            leaf_linked_variable_slots = {slot} if descriptor._is_variable else set(slot._iter_linked_slots())
            return elements, leaf_linked_variable_slots

        def iter_descriptor_items(
            descriptor_chain: tuple[LazyDescriptor, ...],
            instance: "LazyObject"
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
            instance: "LazyObject"
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
            instance: "LazyObject"
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
                if self._freeze:
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
        registered_elements = self._register_elements(elements)
        if registered_elements == slot._get():
            return
        # `slot` is guaranteed to be a variable slot. Expire linked property slots.
        for expired_property_slot in slot._iter_linked_slots():
            expired_property_slot._expire()
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

    @staticmethod
    def _empty_freezer(
        element: _T
    ) -> None:
        return
