from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    Iterator,
    Self
)

from .cache import Cache
from .registration import (
    Registered,
    Registration
)

if TYPE_CHECKING:
    from .lazy_object import LazyObject
    from .lazy_overriding import LazyOverriding
    from .lazy_slot import LazySlot


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


class LazyRoutine[T, DataT](ABC):
    __slots__ = (
        "__weakref__",
        "_overriding",
        "_method",
        "_parameter_overriding_chains",
        "_is_variable",
        "_freeze",
        "_cache",
        "_parameter_key_registration"
    )

    def __init__(
        self: Self,
        overriding: LazyOverriding[T, DataT],
        method: Callable[..., DataT],
        parameter_overriding_chains: tuple[tuple[LazyOverriding, ...], ...],
        is_variable: bool,
        freeze: bool,
        cache_capacity: int
    ) -> None:
        super().__init__()
        self._overriding: LazyOverriding[T, DataT] = overriding
        self._method: Callable[..., DataT] = method
        self._parameter_overriding_chains: tuple[tuple[LazyOverriding, ...], ...] = parameter_overriding_chains
        self._is_variable: bool = is_variable
        self._freeze: bool = freeze
        self._cache: Cache[Registered[Hashable], tuple[Registered[T], ...]] = Cache(capacity=cache_capacity)
        self._parameter_key_registration: Registration[Hashable, Hashable] = Registration()

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

    def _register_parameter_key(
        self: Self,
        parameter_key: Hashable
    ) -> Registered[Hashable]:
        return self._parameter_key_registration.register(parameter_key, parameter_key)

    def _register_elements(
        self: Self,
        elements: tuple[T, ...]
    ) -> tuple[Registered[T], ...]:
        return self._overriding._register_elements(elements, self._freeze)

    def get_elements(
        self: Self,
        instance: LazyObject
    ) -> tuple[T, ...]:

        def get_leaf_items(
            parameter_overriding: LazyOverriding,
            leaf: LazyObject
        ) -> tuple[tuple, set[LazySlot]]:
            #descriptor = type(leaf)._lazy_descriptors[descriptor_name]
            slot = parameter_overriding.get_slot(leaf)
            routine = parameter_overriding._routines[type(leaf)]
            elements = routine.get_elements(leaf)
            leaf_associated_variable_slots = {slot} if routine._is_variable else set(slot.iter_associated_slots())
            return elements, leaf_associated_variable_slots

        def iter_routine_items(
            parameter_overriding_chain: tuple[LazyOverriding, ...],
            instance: LazyObject
        ) -> Iterator[tuple[tuple, tuple[int, ...] | None, set[LazySlot]]]:
            leaf_tuple: tuple = (instance,)
            for parameter_overriding in parameter_overriding_chain:
                leaf_items = tuple(
                    get_leaf_items(parameter_overriding, leaf)
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
                ) if parameter_overriding._is_plural else None
                routine_associated_variable_slots = set().union(*(
                    leaf_associated_variable_slots
                    for _, leaf_associated_variable_slots in leaf_items
                ))
                yield leaf_tuple, branch_sizes, routine_associated_variable_slots

        def get_parameter_items(
            parameter_overriding_chain: tuple[LazyOverriding, ...],
            instance: LazyObject
        ) -> tuple[PseudoTree, set[LazySlot]]:
            routine_items = tuple(iter_routine_items(parameter_overriding_chain, instance))
            leaf_tuple, _, _ = routine_items[-1]
            pseudo_tree = PseudoTree(
                leaf_tuple=leaf_tuple,
                branch_structure=tuple(
                    branch_sizes
                    for _, branch_sizes, _ in routine_items
                    if branch_sizes is not None
                )
            )
            parameter_associated_variable_slots = set().union(*(
                routine_associated_variable_slots
                for _, _, routine_associated_variable_slots in routine_items
            ))
            return pseudo_tree, parameter_associated_variable_slots

        def get_pseudo_trees_and_associated_variable_slots(
            parameter_overriding_chains: tuple[tuple[LazyOverriding, ...], ...],
            instance: LazyObject
        ) -> tuple[tuple[PseudoTree, ...], set[LazySlot]]:
            parameter_items = tuple(
                get_parameter_items(parameter_overriding_chain, instance)
                for parameter_overriding_chain in parameter_overriding_chains
            )
            pseudo_trees = tuple(pseudo_tree for pseudo_tree, _ in parameter_items)
            associated_variable_slots = set().union(*(
                parameter_associated_variable_slots for _, parameter_associated_variable_slots in parameter_items
            ))
            return pseudo_trees, associated_variable_slots

        slot = self._overriding.get_slot(instance)
        if (registered_elements := slot.get()) is None:
            # If there's at least a parameter, `slot` is guaranteed to be a property slot.
            # Associate it with variable slots.
            pseudo_trees, associated_variable_slots = get_pseudo_trees_and_associated_variable_slots(
                self._parameter_overriding_chains, instance
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

    def set_elements(
        self: Self,
        instance: LazyObject,
        elements: tuple[T, ...]
    ) -> None:
        slot = self._overriding.get_slot(instance)
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

    #def init(
    #    self: Self,
    #    instance: LazyObject
    #) -> None:
    #    slot = LazySlot()
    #    slot._is_writable = self._is_variable
    #    self._overriding.set_slot(instance, slot)

    def descriptor_get(
        self: Self,
        instance: LazyObject
    ) -> DataT:
        return self._composer(self.get_elements(instance))

    def descriptor_set(
        self: Self,
        instance: LazyObject,
        data: DataT
    ) -> None:
        self.set_elements(instance, self._decomposer(data))


class LazySingularRoutine[T](LazyRoutine[T, T]):
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


class LazyPluralRoutine[T](LazyRoutine[T, tuple[T, ...]]):
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
