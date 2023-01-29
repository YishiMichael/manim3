__all__ = ["Node"]


from typing import (
    Generator,
    Iterable,
    Iterator,
    TypeVar,
    overload
)


Self = TypeVar("Self", bound="Node")


class Node:
    def __init__(self) -> None:
        self.__parents__: list = []
        self.__children__: list = []
        super().__init__()

    def __iter__(self: Self) -> Iterator[Self]:
        return iter(self._children)

    @overload
    def __getitem__(self: Self, i: int) -> Self: ...

    @overload
    def __getitem__(self: Self, i: slice) -> list[Self]: ...

    def __getitem__(self: Self, i: int | slice) -> Self | list[Self]:
        return self._children.__getitem__(i)

    @property
    def _parents(self: Self) -> list[Self]:
        return self.__parents__

    @property
    def _children(self: Self) -> list[Self]:
        return self.__children__

    def iter_parents(self: Self) -> Iterator[Self]:
        return iter(self._parents)

    def iter_children(self: Self) -> Iterator[Self]:
        return iter(self._children)

    def _iter_ancestors(self: Self) -> Generator[Self, None, None]:
        yield self
        for parent_node in self._parents:
            yield from parent_node._iter_ancestors()

    def _iter_descendants(self: Self) -> Generator[Self, None, None]:
        yield self
        for child_node in self._children:
            yield from child_node._iter_descendants()

    def iter_ancestors(self: Self, *, broadcast: bool = True) -> Generator[Self, None, None]:
        yield self
        if not broadcast:
            return
        occurred: set[Self] = {self}
        for node in self._iter_ancestors():
            if node in occurred:
                continue
            yield node
            occurred.add(node)

    def iter_descendants(self: Self, *, broadcast: bool = True) -> Generator[Self, None, None]:
        yield self
        if not broadcast:
            return
        occurred: set[Self] = {self}
        for node in self._iter_descendants():
            if node in occurred:
                continue
            yield node
            occurred.add(node)

    def includes(self: Self, node: Self) -> bool:
        return node in self.iter_descendants()

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
        for child in self.iter_children():
            self._unbind_child(child)
        return self

    def clear_parents(self: Self) -> Self:
        for parent in self.iter_parents():
            parent._unbind_child(self)
        return self

    def set_children(self: Self, nodes: Iterable[Self]) -> Self:
        self.clear()
        self.add(*nodes)
        return self
