import copy
from dataclasses import dataclass
from functools import reduce
from typing import Generator, Iterable, Iterator, TypeVar
import warnings

import numpy as np
import pyrr
from scipy.spatial.transform import Rotation

#from ..animations.animation import Animation
from ..cameras.camera import Camera
from ..cameras.perspective_camera import PerspectiveCamera
from ..utils.lazy import lazy_property_initializer
from ..utils.renderable import Renderable
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


class Mobject(Renderable):
    def __init__(self) -> None:
        self.parents: list["Mobject"] = []
        self.children: list["Mobject"] = []

        #self.matrix: pyrr.Matrix44 = pyrr.Matrix44.identity()
        self.animations: list["Animation"] = []  # TODO: circular typing
        super().__init__()

    def __iter__(self) -> Iterator["Mobject"]:
        return iter(self.get_children())

    def __getitem__(self, i: int | slice):
        if isinstance(i, int):
            return self.children.__getitem__(i)
        return Group(*self.children.__getitem__(i))

    # family

    def get_parents(self) -> list["Mobject"]:
        return self.parents

    def get_children(self) -> list["Mobject"]:
        return self.children

    def _bind_child(self, node, index: int | None = None):
        if node.includes(self):
            raise ValueError(f"'{node}' has already included '{self}'")
        if index is not None:
            self.children.insert(index, node)
        else:
            self.children.append(node)
        node.parents.append(self)
        return self

    def _unbind_child(self, node):
        self.children.remove(node)
        node.parents.remove(self)
        return self

    @classmethod
    def remove_redundancies(cls, l: Iterable[T]) -> list[T]:
        """
        Used instead of list(set(l)) to maintain order
        Keeps the first occurrence of each element
        """
        return list(dict.fromkeys(l))

    def _iter_ancestors(self) -> Generator["Mobject", None, None]:
        yield self
        for parent in self.get_parents():
            yield from parent._iter_ancestors()

    def _iter_descendents(self) -> Generator["Mobject", None, None]:
        yield self
        for child in self.get_children():
            yield from child._iter_descendents()

    def get_ancestors(self, *, broadcast: bool = True) -> list["Mobject"]:  # TODO: order
        if not broadcast:
            return [self]
        return self.remove_redundancies(self._iter_ancestors())

    def get_descendents(self, *, broadcast: bool = True) -> list["Mobject"]:
        if not broadcast:
            return [self]
        return self.remove_redundancies(self._iter_descendents())

    def includes(self, node) -> bool:
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

    def index(self, node) -> int:
        return self.get_children().index(node)

    def insert(self, index: int, *nodes):
        for i, node in enumerate(nodes, start=index):
            self._bind_child(node, index=i)
        return self

    def add(self, *nodes):
        for node in nodes:
            self._bind_child(node)
        return self

    def remove(self, *nodes):
        for node in nodes:
            self._unbind_child(node)
        return self

    def pop(self, index: int = -1):
        node = self.children[index]
        self._unbind_child(node)
        return node

    def clear(self):
        for child in self.children[:]:
            self._unbind_child(child)
        return self

    def clear_parents(self):
        for parent in self.parent:
            parent._unbind_child(self)
        return self

    def set_children(self, children: Iterable["Mobject"]):
        self.clear()
        self.add(*children)
        return self

    def copy(self):
        return copy.copy(self)  # TODO

    # matrix & transform

    @lazy_property_initializer
    def _matrix_() -> pyrr.Matrix44:
        return pyrr.Matrix44.identity()

    #@_matrix.setter
    #def _matrix(self, arg: pyrr.Matrix44) -> None:
    #    pass

    @_matrix_.updater
    def set_local_matrix(self, matrix: pyrr.Matrix44):
        self._matrix_ = matrix
        return self

    @classmethod
    def matrix_from_translation(cls, vector: Vector3Type) -> pyrr.Matrix44:
        return pyrr.Matrix44.from_translation(vector)

    @classmethod
    def matrix_from_scale(cls, factor_vector: Vector3Type) -> pyrr.Matrix44:
        return pyrr.Matrix44.from_scale(factor_vector)

    @classmethod
    def matrix_from_rotation(cls, rotation: Rotation) -> pyrr.Matrix44:
        return pyrr.Matrix44.from_matrix33(rotation.as_matrix())

    @lazy_property_initializer
    def _local_sample_points_() -> Vector3ArrayType:
        # Implemented in subclasses
        return np.zeros((0, 3))
        #raise NotImplementedError

    # TODO: lazify bounding boxes

    def get_bounding_box(
        self,
        *,
        broadcast: bool = True
    ) -> BoundingBox3D:
        points = [
            pyrr.matrix44.apply_to_vector(
                mobject._matrix_, point
            )
            for mobject in self.get_descendents(broadcast=broadcast)
            for point in mobject._local_sample_points_
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
        self,
        *,
        broadcast: bool = True
    ) -> Vector3Type:
        aabb = self.get_bounding_box(broadcast=broadcast)
        return aabb.radius * 2.0

    def get_bounding_box_point(
        self,
        direction: Vector3Type,
        *,
        broadcast: bool = True
    ) -> Vector3Type:
        aabb = self.get_bounding_box(broadcast=broadcast)
        return aabb.origin + direction * aabb.radius

    def get_center(
        self,
        *,
        broadcast: bool = True
    ) -> Vector3Type:
        return self.get_bounding_box_point(ORIGIN, broadcast=broadcast)

    def apply_raw_matrix(
        self,
        matrix: pyrr.Matrix44,
        *,
        broadcast: bool = True
    ):
        #if np.isclose(np.linalg.det(matrix), 0.0):
        #    warnings.warn("Applying a singular matrix transform")
        for mobject in self.get_descendents(broadcast=broadcast):
            mobject.set_local_matrix(mobject._matrix_ @ matrix)
        return self

    def preapply_raw_matrix(
        self,
        matrix: pyrr.Matrix44,
        *,
        broadcast: bool = True
    ):
        #if np.isclose(np.linalg.det(matrix), 0.0):
        #    warnings.warn("Applying a singular matrix transform")
        for mobject in self.get_descendents(broadcast=broadcast):
            mobject.set_local_matrix(matrix @ mobject._matrix_)
        return self

    def apply_matrix(
        self,
        matrix: pyrr.Matrix44,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ):
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
        self,
        vector: Vector3Type,
        *,
        coor_mask: Vector3Type | None = None,
        broadcast: bool = True
    ):
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
        self,
        factor: Real | Vector3Type,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ):
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
        self,
        rotation: Rotation,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ):
        matrix = self.matrix_from_rotation(rotation)
        self.apply_matrix(
            matrix,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def move_to(
        self,
        mobject_or_point: "Mobject | Vector3Type",
        aligned_edge: Vector3Type = ORIGIN,
        *,
        coor_mask: Vector3Type | None = None,
        broadcast: bool = True
    ):
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
        self,
        mobject_or_point: "Mobject | Vector3Type",
        direction: Vector3Type = RIGHT,
        buff: float = 0.25,
        *,
        coor_mask: Vector3Type | None = None,
        broadcast: bool = True
    ):
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
        self,
        target_size: Vector3Type,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ):
        factor_vector = target_size / self.get_bounding_box_size(broadcast=broadcast)
        self.scale(
            factor_vector,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_dim(
        self,
        target_length: Real,
        dim: int,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ):
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
        self,
        target_length: Real,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ):
        self.stretch_to_fit_dim(
            target_length,
            0,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_height(
        self,
        target_length: Real,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ):
        self.stretch_to_fit_dim(
            target_length,
            1,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_depth(
        self,
        target_length: Real,
        *,
        about_point: Vector3Type | None = None,
        about_edge: Vector3Type | None = None,
        broadcast: bool = True
    ):
        self.stretch_to_fit_dim(
            target_length,
            2,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    # animations

    def animate(self, animation: "Animation"):
        self.animations.append(animation)
        animation.start(self)
        return self

    def update_dt(self, dt: Real):
        for animation in self.animations[:]:
            animation.update_dt(self, dt)
            if animation.expired():
                self.animations.remove(animation)
        return self

    # shader

    @lazy_property_initializer
    def _camera_() -> Camera:
        return PerspectiveCamera()

    #@_camera.setter
    #def _camera(self, camera: Camera) -> None:
    #    pass

    #@lazy_property
    #def _shader_data(self) -> ShaderData | None:
    #    # To be implemented in subclasses
    #    return None


class Group(Mobject):
    def __init__(self, *mobjects: Mobject):
        super().__init__()
        self.add(*mobjects)

    # TODO
    def _bind_child(self, node, index: int | None = None):
        assert isinstance(node, Mobject)
        super()._bind_child(node, index=index)
        return self
