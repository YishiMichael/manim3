from dataclasses import dataclass
from typing import Generator, Iterable, Iterator, Self, TypeVar, Union

from cameras.camera import Camera
from utils.arrays import Mat3, Mat4, Vec2, Vec3, Vec4


__all__ = [
    "Mobject"
]


T = TypeVar("T")
Uniform = Union[
    float,
    int,
    Mat3,
    Mat4,
    Vec2,
    Vec3,
    Vec4
]
Attribute = Union[
    list[float],
    list[int],
    list[Mat3],
    list[Mat4],
    list[Vec2],
    list[Vec3],
    list[Vec4]
]


@dataclass
class ShaderData:
    vertex_shader: str
    fragment_shader: str
    uniforms: dict[str, Uniform]
    vertex_indices: list[int]
    vertex_attributes: dict[str, Attribute]
    render_primitive: int


class Mobject:
    def __init__(self: Self) -> None:
        self.__parents: list[Self] = []
        self.__children: list[Self] = []

    def __iter__(self: Self) -> Iterator[Self]:
        return iter(self.get_children())

    # family

    def get_parents(self: Self) -> list[Self]:
        return self.__parents

    def get_children(self: Self) -> list[Self]:
        return self.__children

    def _bind_child(self: Self, node: Self, index: int | None = None) -> Self:
        if node.includes(self):
            raise ValueError(f"'{node}' has already included '{self}'")
        if index is not None:
            self.__children.insert(index, node)
        else:
            self.__children.append(node)
        node.__parents.append(self)
        return self

    def _unbind_child(self: Self, node: Self) -> Self:
        self.__children.remove(node)
        node.__parents.remove(self)
        return self

    @staticmethod
    def remove_redundancies(l: Iterable[T]) -> list[T]:
        """
        Used instead of list(set(l)) to maintain order
        Keeps the first occurrence of each element
        """
        return list(dict.fromkeys(l))

    def _iter_ancestors(self: Self) -> Generator[Self, None, None]:
        yield self
        for parent in self.get_parents():
            yield from parent._iter_ancestors()

    def _iter_descendents(self: Self) -> Generator[Self, None, None]:
        yield self
        for child in self.get_children():
            yield from child._iter_descendents()

    def get_ancestors(self: Self) -> list[Self]:  # TODO: order
        return self.remove_redundancies(self._iter_ancestors())

    def get_descendents(self: Self) -> list[Self]:
        return self.remove_redundancies(self._iter_descendents())

    def includes(self: Self, node: Self) -> bool:
        return node in self._iter_descendents()

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
        return self.get_children().index(node)

    def insert(self: Self, index: int, *nodes: Self) -> Self:
        for i, node in enumerate(nodes, start=index):
            self._bind_child(node, index=i)
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
        node = self.children[index]
        self._unbind_child(node)
        return node

    def clear(self: Self) -> Self:
        for child in self.children:
            self._unbind_child(child)
        return self

    def clear_parents(self: Self) -> Self:
        for parent in self.parent:
            parent._unbind_child(self)
        return self

    # shader

    def setup_shader_data(self: Self, camera: Camera) -> ShaderData:
        raise NotImplementedError
