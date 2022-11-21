from abc import ABC
import copy
from dataclasses import dataclass
from functools import reduce
import pyrr
from typing import Generator, Iterable, Iterator, TypeVar
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

#from ..animations.animation import Animation
from ..cameras.camera import Camera
from ..shader_utils import ShaderData
from ..constants import ORIGIN, RIGHT
from ..custom_typing import *


__all__ = [
    "BoundingBox3D",
    "Mobject",
    "Group"
]


T = TypeVar("T")


@dataclass
class BoundingBox3D:
    origin: Vector3Type
    radius: Vector3Type


class Mobject(ABC):
    def __init__(self: Self) -> None:
        self.parents: list[Self] = []
        self.children: list[Self] = []

        self.matrix: pyrr.Matrix44 = pyrr.Matrix44.identity()

        self.animations: list["Animation"] = []  # TODO: circular typing

        # shader context settings
        self.enable_depth_test: bool = True
        self.enable_blend: bool = True
        self.cull_face: str = "back"
        self.wireframe: bool = False

    def __iter__(self: Self) -> Iterator[Self]:
        return iter(self.get_children())

    def __getitem__(self: Self, i: int | slice) -> Self:
        if isinstance(i, int):
            return self.children.__getitem__(i)
        return Group(*self.children.__getitem__(i))

    # family

    def get_parents(self: Self) -> list[Self]:
        return self.parents

    def get_children(self: Self) -> list[Self]:
        return self.children

    def _bind_child(self: Self, node: Self, index: int | None = None) -> Self:
        if node.includes(self):
            raise ValueError(f"'{node}' has already included '{self}'")
        if index is not None:
            self.children.insert(index, node)
        else:
            self.children.append(node)
        node.parents.append(self)
        return self

    def _unbind_child(self: Self, node: Self) -> Self:
        self.children.remove(node)
        node.parents.remove(self)
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

    def get_ancestors(self: Self, *, broadcast: bool = True) -> list[Self]:  # TODO: order
        if not broadcast:
            return [self]
        return self.remove_redundancies(self._iter_ancestors())

    def get_descendents(self: Self, *, broadcast: bool = True) -> list[Self]:
        if not broadcast:
            return [self]
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
        for child in self.children[:]:
            self._unbind_child(child)
        return self

    def clear_parents(self: Self) -> Self:
        for parent in self.parent:
            parent._unbind_child(self)
        return self

    def set_children(self: Self, children: Iterable[Self]) -> Self:
        self.clear()
        self.add(*children)
        return self

    def copy(self: Self) -> Self:
        return copy.copy(self)  # TODO

    # matrix & transform

    @staticmethod
    def matrix_from_translation(vector: Vector3Type) -> pyrr.Matrix44:
        return pyrr.Matrix44.from_translation(vector)

    @staticmethod
    def matrix_from_scale(factor_vector: Vector3Type) -> pyrr.Matrix44:
        return pyrr.Matrix44.from_scale(factor_vector)

    @staticmethod
    def matrix_from_rotation(rotation: Rotation) -> pyrr.Matrix44:
        return pyrr.Matrix44.from_matrix33(rotation.as_matrix())

    def get_local_sample_points(self: Self) -> Vector3ArrayType:
        # Implemented in subclasses
        return np.zeros((0, 3))

    def get_bounding_box(
        self: Self,
        *,
        broadcast: bool = True
    ) -> BoundingBox3D:
        points = [
            pyrr.matrix44.apply_to_vector(
                mobject.matrix, point
            )
            for mobject in self.get_descendents(broadcast=broadcast)
            for point in mobject.get_local_sample_points()
        ]
        if not points:
            warnings.warn("Trying to calculate the bounding box of some mobject with no points")
            origin = ORIGIN
            radius = ORIGIN
        else:
            minimum, maximum = pyrr.aabb.create_from_points(points)
            origin = (maximum + minimum) / 2.0
            radius = (maximum - minimum) / 2.0
        # For zero-width dimensions of radius, thicken a little bit to avoid zero division
        radius[np.isclose(radius, 0.0)] = 1e-8
        return BoundingBox3D(
            origin=origin,
            radius=radius
        )

    def get_bounding_box_size(
        self: Self,
        *,
        broadcast: bool = True
    ) -> Vector3Type:
        aabb = self.get_bounding_box(broadcast=broadcast)
        return aabb.radius * 2.0

    def get_bounding_box_point(
        self: Self,
        direction: Vector3Type,
        *,
        broadcast: bool = True
    ) -> Vector3Type:
        aabb = self.get_bounding_box(broadcast=broadcast)
        return aabb.origin + direction * aabb.radius

    def get_center(
        self: Self,
        *,
        broadcast: bool = True
    ) -> Vector3Type:
        return self.get_bounding_box_point(ORIGIN, broadcast=broadcast)

    def apply_raw_matrix(
        self: Self,
        matrix: pyrr.Matrix44,
        *,
        broadcast: bool = True
    ) -> Self:
        #if np.isclose(np.linalg.det(matrix), 0.0):
        #    warnings.warn("Applying a singular matrix transform")
        for mobject in self.get_descendents(broadcast=broadcast):
            mobject.matrix = mobject.matrix @ matrix
        return self

    def preapply_raw_matrix(
        self: Self,
        matrix: pyrr.Matrix44,
        *,
        broadcast: bool = True
    ) -> Self:
        #if np.isclose(np.linalg.det(matrix), 0.0):
        #    warnings.warn("Applying a singular matrix transform")
        for mobject in self.get_descendents(broadcast=broadcast):
            mobject.matrix = matrix @ mobject.matrix
        return self

    def apply_matrix(
        self: Self,
        matrix: pyrr.Matrix44,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ) -> Self:
        if about_point is None:
            if about_edge is None:
                about_edge = ORIGIN
            about_point = self.get_bounding_box_point(about_edge, broadcast=broadcast)
        elif about_edge is not None:
            raise AttributeError("Cannot specify both parameters `about_point` and `about_edge`")

        matrix = reduce(pyrr.Matrix44.__matmul__, (
            self.matrix_from_translation(-about_point),
            matrix,
            self.matrix_from_translation(about_point)
        ))
        self.apply_raw_matrix(
            matrix,
            broadcast=broadcast
        )
        return self

    def shift(
        self: Self,
        vector: Vector3Type,
        *,
        coor_mask: Vector3Type | None = None,
        broadcast: bool = True
    ) -> Self:
        if coor_mask is not None:
            vector *= coor_mask
        matrix = self.matrix_from_translation(vector)
        # `about_point` and `about_edge` are meaningless when shifting
        self.apply_matrix(
            matrix,
            broadcast=broadcast
        )
        return self

    def scale(
        self: Self,
        factor: Real | Vector3Type,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ) -> Self:
        if isinstance(factor, Real):
            factor_vector = pyrr.Vector3()
            factor_vector.fill(factor)
        else:
            factor_vector = pyrr.Vector3(factor)
        matrix = self.matrix_from_scale(factor_vector)
        self.apply_matrix(
            matrix,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def rotate(
        self: Self,
        rotation: Rotation,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ) -> Self:
        matrix = self.matrix_from_rotation(rotation)
        self.apply_matrix(
            matrix,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def move_to(
        self: Self,
        mobject_or_point: Self | Vector3Type,
        aligned_edge: Vector3Type = ORIGIN,
        *,
        coor_mask: Vector3Type | None = None,
        broadcast: bool = True
    ) -> Self:
        if isinstance(mobject_or_point, Mobject):
            target_point = mobject_or_point.get_bounding_box_point(aligned_edge, broadcast=broadcast)
        else:
            target_point = mobject_or_point
        point_to_align = self.get_bounding_box_point(aligned_edge, broadcast=broadcast)
        vector = target_point - point_to_align
        self.shift(
            vector,
            coor_mask=coor_mask,
            broadcast=broadcast
        )
        return self

    def next_to(
        self: Self,
        mobject_or_point: Self | Vector3Type,
        direction: Vector3Type = RIGHT,
        buff: float = 0.25,
        *,
        coor_mask: Vector3Type | None = None,
        broadcast: bool = True
    ) -> Self:
        if isinstance(mobject_or_point, Mobject):
            target_point = mobject_or_point.get_bounding_box_point(direction, broadcast=broadcast)
        else:
            target_point = mobject_or_point
        point_to_align = self.get_bounding_box_point(-direction, broadcast=broadcast)
        vector = target_point - point_to_align + buff * direction
        self.shift(
            vector,
            coor_mask=coor_mask,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_size(
        self: Self,
        target_size: Vector3Type,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ) -> Self:
        factor_vector = target_size / self.get_bounding_box_size(broadcast=broadcast)
        self.scale(
            factor_vector,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_dim(
        self: Self,
        target_length: Real,
        dim: int,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ) -> Self:
        factor_vector = np.ones(3)
        factor_vector[dim] = target_length / self.get_bounding_box_size(broadcast=broadcast)[dim]
        self.scale(
            factor_vector,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_width(
        self: Self,
        target_length: Real,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ) -> Self:
        self.stretch_to_fit_dim(
            target_length,
            0,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_height(
        self: Self,
        target_length: Real,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ) -> Self:
        self.stretch_to_fit_dim(
            target_length,
            1,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_depth(
        self: Self,
        target_length: Real,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ) -> Self:
        self.stretch_to_fit_dim(
            target_length,
            2,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    # animations

    def animate(self: Self, animation: "Animation") -> Self:
        self.animations.append(animation)
        animation.start(self)
        return self

    def update(self: Self, dt: Real) -> Self:
        for animation in self.animations[:]:
            animation.update_dt(self, dt)
            if animation.expired():
                self.animations.remove(animation)
        return self

    # shader

    def setup_shader_data(self: Self, camera: Camera) -> ShaderData | None:
        # To be implemented in subclasses
        return None


class Group(Mobject):
    def __init__(self: Self, *mobjects: Mobject):
        super().__init__()
        self.add(*mobjects)

    # TODO
    def _bind_child(self: Self, node: Self, index: int | None = None) -> Self:
        assert isinstance(node, Mobject)
        super()._bind_child(node, index=index)
