from __future__ import annotations


import itertools
import weakref
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Hashable,
    Iterator,
    Never,
    Self,
    overload
)

from lru import LRU

if TYPE_CHECKING:
    from .lazy_object import LazyObject
    from .lazy_slot import LazySlot


class Memoized[T]:
    __slots__ = (
        "__weakref__",
        "_id",
        "_value"
    )

    _id_counter: ClassVar[itertools.count[int]] = itertools.count()

    def __init__(
        self: Self,
        value: T
    ) -> None:
        super().__init__()
        self._id: int = next(type(self)._id_counter)
        self._value: T = value

    def get_id(
        self: Self
    ) -> int:
        return self._id

    def get_value(
        self: Self
    ) -> T:
        return self._value


class Memoization[KT: Hashable, VT](weakref.WeakValueDictionary[KT, Memoized[VT]]):
    __slots__ = ()

    def memoize(
        self: Self,
        key: KT,
        value: VT
    ) -> Memoized[VT]:
        if (memoized_value := self.get(key)) is None:
            memoized_value = Memoized(value)
            self[key] = memoized_value
        return memoized_value


type TupleTree[T] = T | tuple[TupleTree[T], ...]


class Tree[T]:
    __slots__ = (
        "_content",
        "_children"
    )

    def __init__(
        self: Self,
        content: T,
        children: tuple[Tree[T], ...] | None = None
    ) -> None:
        super().__init__()
        self._content: T = content
        self._children: tuple[Tree[T], ...] | None = children

    def iter_leaves(
        self: Self
    ) -> Iterator[Tree[T]]:
        if (children := self._children) is not None:
            for child in children:
                yield from child.iter_leaves()
        else:
            yield self

    def as_tuple_tree[ConvertedT](
        self: Self,
        func: Callable[[T], ConvertedT]
    ) -> TupleTree[ConvertedT]:
        if (children := self._children) is not None:
            return tuple(
                child.as_tuple_tree(func)
                for child in children
            )
        return func(self._content)


class LazyDescriptor[T, DataT]:
    __slots__ = (
        "__weakref__",
        "_method",
        "_is_property",
        "_plural",
        "_freeze",
        "_deepcopy",
        "_lru_cache",
        "_parameter_key_memoization",
        "_element_memoization",
        "_name",
        "_parameter_name_chains",
        "_decomposer",
        "_composer",
        "_hasher"
    )

    def __init__(
        self: Self,
        method: Callable[..., DataT],
        is_property: bool,
        plural: bool,
        freeze: bool,
        deepcopy: bool,
        cache_capacity: int
    ) -> None:
        super().__init__()
        self._method: Callable[..., DataT] = method
        self._is_property: bool = is_property
        self._plural: bool = plural
        self._freeze: bool = freeze
        self._deepcopy: bool = deepcopy
        self._lru_cache: LRU[Memoized[Hashable], tuple[Memoized[T], ...]] | None = LRU(cache_capacity) if cache_capacity else None
        self._parameter_key_memoization: Memoization[Hashable, Hashable] = Memoization()
        self._element_memoization: Memoization[Hashable, T] = Memoization()
        self._name: str = NotImplemented
        self._parameter_name_chains: tuple[tuple[str, ...], ...] = NotImplemented
        self._decomposer: Callable[[DataT], tuple[T, ...]] = NotImplemented
        self._composer: Callable[[tuple[T, ...]], DataT] = NotImplemented
        self._hasher: Callable[[T], Hashable] = NotImplemented

    @overload
    def __get__(
        self: Self,
        instance: LazyObject,
        owner: type[LazyObject] | None = None
    ) -> DataT: ...

    @overload
    def __get__(
        self: Self,
        instance: None,
        owner: type[LazyObject] | None = None
    ) -> Self: ...

    def __get__(
        self: Self,
        instance: LazyObject | None,
        owner: type[LazyObject] | None = None
    ) -> Self | DataT:
        if instance is None:
            return self
        return self._composer(self.get_elements(instance))

    def __set__(
        self: Self,
        instance: LazyObject,
        data: DataT
    ) -> None:
        self.set_elements(instance, self._decomposer(data))

    def __delete__(
        self: Self,
        instance: LazyObject
    ) -> Never:
        raise TypeError("Cannot delete attributes of a lazy object")

    def _memoize_parameter_key(
        self: Self,
        parameter_key: Hashable
    ) -> Memoized[Hashable]:
        return self._parameter_key_memoization.memoize(parameter_key, parameter_key)

    def _memoize_elements(
        self: Self,
        elements: tuple[T, ...]
    ) -> tuple[Memoized[T], ...]:
        element_memoization = self._element_memoization
        hasher = self._hasher
        return tuple(
            element_memoization.memoize(hasher(element), element)
            for element in elements
        )

    def _get_memoized_elements(
        self: Self,
        instance: LazyObject
    ) -> tuple[Memoized[T], ...]:
        slot = self.get_slot(instance)
        if (memoized_elements := slot.get()) is None:
            # If there's at least a parameter, `slot` is guaranteed to be a property slot.
            # Associate it with variable slots.
            tree_root = Memoized(instance)
            trees = tuple(Tree(tree_root) for _ in self._parameter_name_chains)
            associated_slots: set[LazySlot] = set()
            for parameter_name_chain, tree in zip(self._parameter_name_chains, trees, strict=True):
                for name in parameter_name_chain:
                    for leaf in tree.iter_leaves():
                        leaf_object = leaf._content.get_value()
                        leaf_slot = leaf_object._get_lazy_slot(name)
                        descriptor = leaf_slot.get_descriptor()
                        children_memoized_elements = descriptor._get_memoized_elements(leaf_object)
                        if descriptor._plural:
                            leaf._children = tuple(Tree(element) for element in children_memoized_elements)
                        else:
                            (leaf._content,) = children_memoized_elements
                        if descriptor._is_property:
                            associated_slots.update(leaf_slot.iter_associated_slots())
                        else:
                            associated_slots.add(leaf_slot)

            memoized_parameter_key = self._memoize_parameter_key(tuple(
                tree.as_tuple_tree(Memoized.get_id) for tree in trees
            ))
            if (lru_cache := self._lru_cache) is None or (memoized_elements := lru_cache.get(memoized_parameter_key)) is None:
                memoized_elements = self._memoize_elements(self._decomposer(self._method(*(
                    tree.as_tuple_tree(Memoized.get_value) for tree in trees
                ))))
                if lru_cache is not None:
                    lru_cache[memoized_parameter_key] = memoized_elements
            slot.set(
                elements=memoized_elements,
                parameter_key=memoized_parameter_key,
                associated_slots=associated_slots
            )
        return memoized_elements

    def _set_memoized_elements(
        self: Self,
        instance: LazyObject,
        memoized_elements: tuple[Memoized[T], ...]
    ) -> None:
        assert not self._is_property
        slot = self.get_slot(instance)
        # Guaranteed to be a variable slot. Expire associated property slots.
        for expired_property_slot in slot.iter_associated_slots():
            expired_property_slot.expire()
        slot.set(
            elements=memoized_elements,
            parameter_key=None,
            associated_slots=set()
        )

    def get_slot(
        self: Self,
        instance: LazyObject
    ) -> LazySlot:
        return instance._get_lazy_slot(self._name)

    def get_elements(
        self: Self,
        instance: LazyObject
    ) -> tuple[T, ...]:
        return tuple(memoized_element.get_value() for memoized_element in self._get_memoized_elements(instance))

    def set_elements(
        self: Self,
        instance: LazyObject,
        elements: tuple[T, ...]
    ) -> None:
        self._set_memoized_elements(instance, self._memoize_elements(elements))
