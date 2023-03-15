__all__ = ["DAGNode"]


from abc import ABC
from typing import (
    Generator,
    TypeVar
)


Self = TypeVar("Self", bound="DAGNode")


class DAGNode(ABC):
    """
    Node of a doubly linked DAG (directed acyclic graph)
    """
    __slots__ = (
        "_children",
        "_parents"
    )

    def __init__(self) -> None:
        super().__init__()
        self._children: list = []
        self._parents: list = []

    def _iter_descendants(self: Self) -> Generator[Self, None, None]:
        occurred: set[Self] = set()

        def iter_descendants_atom(
            node: Self
        ) -> Generator[Self, None, None]:
            if node in occurred:
                return
            occurred.add(node)
            yield node
            for child in node._children:
                yield from iter_descendants_atom(child)

        yield from iter_descendants_atom(self)

    def _iter_ancestors(self: Self) -> Generator[Self, None, None]:
        occurred: set[Self] = set()

        def iter_ancestors_atom(
            node: Self
        ) -> Generator[Self, None, None]:
            if node in occurred:
                return
            occurred.add(node)
            yield node
            for child in node._parents:
                yield from iter_ancestors_atom(child)

        yield from iter_ancestors_atom(self)

    def _bind(
        self: Self,
        node: Self
    ) -> None:
        if node in self._iter_ancestors():
            raise ValueError(f"Node `{node}` has already included `{self}`")
        if node in self._children:
            raise ValueError(f"Node `{node}` is already a child of `{self}`")
        self._children.append(node)
        node._parents.append(self)

    def _unbind(
        self: Self,
        node: Self
    ) -> None:
        if node not in self._children:
            raise ValueError(f"Node `{node}` is not a child of `{self}`")
        self._children.remove(node)
        node._parents.remove(self)

    def _clear(self) -> None:
        for child in self._children:
            child._parents.remove(self)
        self._children.clear()
