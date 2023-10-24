from __future__ import annotations


import inspect
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

if TYPE_CHECKING:
    from .lazy_object import LazyObject
    from .lazy_slot import LazySlot


class Memoized[T]:
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


class Cache[KT: Hashable, VT](weakref.WeakKeyDictionary[KT, VT]):
    __slots__ = ("_capacity",)

    def __init__(
        self: Self,
        capacity: int
    ) -> None:
        super().__init__()
        assert capacity > 0
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


type TupleTree[T] = T | tuple[TupleTree[T], ...]


class Tree[T](ABC):
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

    def as_tuple_tree(
        self: Self
    ) -> TupleTree[T]:
        if (children := self._children) is not None:
            return tuple(
                child.as_tuple_tree()
                for child in children
            )
        return self._content

    def convert[ConvertedT](
        self: Self,
        func: Callable[[T], ConvertedT]
    ) -> Tree[ConvertedT]:
        return Tree(
            content=func(self._content),
            children=tuple(
                child.convert(func)
                for child in children
            ) if (children := self._children) is not None else None
        )


class LazyDescriptor[T, DataT](ABC):
    __slots__ = (
        "__weakref__",
        "_method",
        "_name",
        "_parameter_name_chains",
        "_is_variable",
        "_hasher",
        "_freezer",
        "_freeze",
        "_cache",
        "_parameter_key_memoization",
        "_element_memoization"
    )

    def __init__(
        self: Self,
        method: Callable[..., DataT],
        is_variable: bool,
        hasher: Callable[[T], Hashable],
        freezer: Callable[[T], None],
        freeze: bool,
        cache_capacity: int
    ) -> None:
        assert isinstance(method, staticmethod)
        assert hasher is id or freeze
        method = method.__func__
        name = method.__name__
        assert name.startswith("_") and name.endswith("_") and "__" not in name
        parameter_name_chains = tuple(
            tuple(f"_{name_body}_" for name_body in parameter_name.split("__"))
            for parameter_name in inspect.getargs(method.__code__).args
        )
        assert not is_variable or not parameter_name_chains

        super().__init__()
        self._method: Callable[..., DataT] = method
        self._name: str = name
        self._parameter_name_chains: tuple[tuple[str, ...], ...] = parameter_name_chains
        self._is_variable: bool = is_variable
        self._hasher: Callable[[T], Hashable] = hasher
        self._freezer: Callable[[T], None] = freezer
        self._freeze: bool = freeze
        self._cache: Cache[Memoized[Hashable], tuple[Memoized[T], ...]] = Cache(capacity=cache_capacity)
        self._parameter_key_memoization: Memoization[Hashable, Hashable] = Memoization()
        self._element_memoization: Memoization[Hashable, T] = Memoization()

    @overload
    def __get__(
        self: Self,
        instance: None,
        owner: type[LazyObject] | None
    ) -> Self: ...

    @overload
    def __get__(
        self: Self,
        instance: LazyObject,
        owner: type[LazyObject] | None
    ) -> DataT: ...

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

    def _memoize_parameter_key(
        self: Self,
        parameter_key: Hashable
    ) -> Memoized[Hashable]:
        return self._parameter_key_memoization.memoize(parameter_key, parameter_key)

    def _memoize_elements(
        self: Self,
        elements: tuple[T, ...]
    ) -> tuple[Memoized[T], ...]:
        if self._freeze:
            freezer = self._freezer
            for element in elements:
                freezer(element)
        element_memoization = self._element_memoization
        hasher = self._hasher
        return tuple(
            element_memoization.memoize(hasher(element), element)
            for element in elements
        )

    def get_slot(
        self: Self,
        instance: LazyObject
    ) -> LazySlot:
        return instance._get_slot(self._name)

    def get_elements(
        self: Self,
        instance: LazyObject
    ) -> tuple[T, ...]:
        slot = self.get_slot(instance)
        if (memoized_elements := slot.get()) is None:
            # If there's at least a parameter, `slot` is guaranteed to be a property slot.
            # Associate it with variable slots.
            trees = tuple(Tree(instance) for _ in self._parameter_name_chains)
            associated_slots: set[LazySlot] = set()
            for parameter_name_chain, tree in zip(self._parameter_name_chains, trees, strict=True):
                for name in parameter_name_chain:
                    for leaf in tree.iter_leaves():
                        leaf_object = leaf._content
                        leaf_slot = leaf_object._get_slot(name)
                        descriptor = leaf_slot.get_descriptor()
                        elements = descriptor.get_elements(leaf_object)
                        if descriptor._is_plural:
                            leaf._children = tuple(Tree(element) for element in elements)
                        else:
                            (element,) = elements
                            leaf._content = element
                        if descriptor._is_variable:
                            associated_slots.add(leaf_slot)
                        else:
                            associated_slots.update(leaf_slot.iter_associated_slots())

            memoized_parameter_key = self._memoize_parameter_key(tuple(
                tree.convert(id).as_tuple_tree() for tree in trees
            ))
            if (memoized_elements := self._cache.get(memoized_parameter_key)) is None:
                memoized_elements = self._memoize_elements(self._decomposer(self._method(*(
                    tree.as_tuple_tree() for tree in trees
                ))))
                if self._freeze:
                    self._cache.set(memoized_parameter_key, memoized_elements)
            slot.set(
                elements=memoized_elements,
                parameter_key=memoized_parameter_key,
                associated_slots=associated_slots
            )
        return tuple(memoized_element._value for memoized_element in memoized_elements)

    def set_elements(
        self: Self,
        instance: LazyObject,
        elements: tuple[T, ...]
    ) -> None:
        slot = self.get_slot(instance)
        slot.check_writability()
        memoized_elements = self._memoize_elements(elements)
        if memoized_elements == slot.get():
            return
        # `slot` passes the writability check, hence is guaranteed to be a variable slot.
        # Expire associated property slots.
        for expired_property_slot in slot.iter_associated_slots():
            expired_property_slot.expire()
        slot.set(
            elements=memoized_elements,
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
