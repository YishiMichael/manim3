__all__ = ["Node"]


from typing import Generator, Iterable, Iterator, TypeVar, overload


_T = TypeVar("_T")
Self = TypeVar("Self", bound="Node")


class Node:
    def __init__(self, *args, **kwargs) -> None:
        self.__parents__: list = []
        self.__children__: list = []
        return super().__init__(*args, **kwargs)

    def __iter__(self: Self) -> Iterator[Self]:
        return iter(self._children)

    @overload
    def __getitem__(self: Self, i: int) -> Self: ...

    @overload
    def __getitem__(self: Self, i: slice) -> list[Self]: ...

    def __getitem__(self: Self, i: int | slice) -> Self | list[Self]:
        return self._children.__getitem__(i)

    @classmethod
    def _remove_redundancies(cls, l: Iterable[_T]) -> list[_T]:
        """
        Used instead of list(set(l)) to maintain order
        Keeps the first occurrence of each element
        """
        return list(dict.fromkeys(l))

    # getters

    @property
    def _parents(self: Self) -> list[Self]:
        return self.__parents__

    @property
    def _children(self: Self) -> list[Self]:
        return self.__children__

    #@_parents_.updater
    #def _append_parent(self: Self, node: Self) -> None:
    #    self._parents_.append(node)

    #@_parents_.updater
    #def _remove_parent(self: Self, node: Self) -> None:
    #    self._parents_.remove(node)

    #@_children_.updater
    #def _append_child(self: Self, node: Self) -> None:
    #    self._children_.append(node)

    #@_children_.updater
    #def _insert_child(self: Self, index: int, node: Self) -> None:
    #    self._children_.insert(index, node)

    #@_children_.updater
    #def _remove_child(self: Self, node: Self) -> None:
    #    self._children_.remove(node)

    def get_parents(self: Self) -> list[Self]:
        return self._parents[:]

    def get_children(self: Self) -> list[Self]:
        return self._children[:]

    def _iter_ancestors(self: Self) -> Generator[Self, None, None]:
        yield self
        for parent_node in self._parents:
            yield from parent_node._iter_ancestors()

    def _iter_descendants(self: Self) -> Generator[Self, None, None]:
        yield self
        for child_node in self._children:
            yield from child_node._iter_descendants()

    def get_ancestors(self: Self, *, broadcast: bool = True) -> list[Self]:  # TODO: order
        if not broadcast:
            return [self]
        return self._remove_redundancies(self._iter_ancestors())

    def get_descendants(self: Self, *, broadcast: bool = True) -> list[Self]:
        if not broadcast:
            return [self]
        return self._remove_redundancies(self._iter_descendants())

    # setters

    def includes(self: Self, node: Self) -> bool:
        return node in self._iter_descendants()

    def _bind_child(self: Self, node: Self, *, index: int | None = None) -> None:
        if node.includes(self):
            raise ValueError(f"'{node}' has already included '{self}'")
        if index is not None:
            self._children.insert(index, node)
        else:
            self._children.append(node)
        node._parents.append(self)

    def _unbind_child(self: Self, node: Self) -> None:
        self._children.remove(node)
        node._parents.remove(self)

    #def clear_bindings(self) -> None:
    #    for parent in self.parent:
    #        parent.children.remove(self)
    #    for child in self.children:
    #        child.parent.remove(self)
    #    #for parent in self.parent:
    #    #    for child in self.children:
    #    #        parent._bind_child(child, loop_check=False)
    #    self.parent.clear()
    #    self.children.clear()

    def index(self: Self, node: Self) -> int:
        return self._children.index(node)

    def insert(self: Self, index: int, node: Self) -> Self:
        self._bind_child(node, index=index)
        return self

    def add(self: Self, *nodes: Self) -> Self:
        for node in nodes:
            self._bind_child(node)
        return self

    def remove(self: Self, *nodes: Self) -> Self:
        for node in nodes:
            self._unbind_child(node)
        return self

    def pop(self: Self, index: int = -1) -> Self:
        node = self[index]
        self._unbind_child(node)
        return node

    def clear(self: Self) -> Self:
        for child in self.get_children():
            self._unbind_child(child)
        return self

    def clear_parents(self: Self) -> Self:
        for parent in self.get_parents():
            parent._unbind_child(self)
        return self

    def set_children(self: Self, nodes: Iterable[Self]) -> Self:
        self.clear()
        self.add(*nodes)
        return self
