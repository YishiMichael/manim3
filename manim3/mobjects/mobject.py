__all__ = ["Mobject"]


#import copy
from dataclasses import dataclass
from functools import reduce
from typing import (
    Generator,
    Iterable,
    Iterator,
    overload
)
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

#from ..animations.animation import Animation
from ..constants import (
    ORIGIN,
    RIGHT
)
from ..custom_typing import (
    Mat4T,
    Real,
    Vec3T,
    Vec3sT
)
from ..utils.lazy import (
    lazy_property,
    lazy_property_updatable,
    lazy_property_writable
)
from ..utils.node import Node
from ..utils.render_procedure import UniformBlockBuffer
from ..utils.renderable import Renderable


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class BoundingBox3D:
    origin: Vec3T
    radius: Vec3T


class MobjectNode(Node):
    def __init__(self, mobject: "Mobject"):
        self._mobject: Mobject = mobject
        super().__init__()


class Mobject(Renderable):
    def __init__(self) -> None:
        self._node: MobjectNode = MobjectNode(self)
        super().__init__()

    #    #self.matrix: pyrr.Matrix44 = pyrr.Matrix44.identity()
    #    self.render_passes: list["RenderPass"] = []
    #    self.animations: list["Animation"] = []  # TODO: circular typing
    #    super().__init__()

    def __iter__(self) -> Iterator["Mobject"]:
        return self.iter_children()

    def __getitem__(self, i: int | slice):
        if isinstance(i, int):
            return self._node.__getitem__(i)._mobject
        return self.__class__().add(*(node._mobject for node in self._node.__getitem__(i)))

    #def copy(self):
    #    return copy.copy(self)  # TODO

    # family matters

    def iter_parents(self) -> "Generator[Mobject, None, None]":
        for node in self._node.iter_parents():
            yield node._mobject

    def iter_children(self) -> "Generator[Mobject, None, None]":
        for node in self._node.iter_children():
            yield node._mobject

    def iter_ancestors(self, *, broadcast: bool = True) -> "Generator[Mobject, None, None]":
        for node in self._node.iter_ancestors(broadcast=broadcast):
            yield node._mobject

    def iter_descendants(self, *, broadcast: bool = True) -> "Generator[Mobject, None, None]":
        for node in self._node.iter_descendants(broadcast=broadcast):
            yield node._mobject

    def includes(self, mobject: "Mobject") -> bool:
        return self._node.includes(mobject._node)

    def index(self, mobject: "Mobject") -> int:
        return self._node.index(mobject._node)

    def insert(self, index: int, mobject: "Mobject"):
        self._node.insert(index, mobject._node)
        return self

    def add(self, *mobjects: "Mobject"):
        self._node.add(*(mobject._node for mobject in mobjects))
        return self

    def remove(self, *mobjects: "Mobject"):
        self._node.remove(*(mobject._node for mobject in mobjects))
        return self

    def pop(self, index: int = -1):
        self._node.pop(index=index)
        return self

    def clear(self):
        self._node.clear()
        return self

    def clear_parents(self):
        self._node.clear_parents()
        return self

    def set_children(self, mobjects: Iterable["Mobject"]):
        self._node.set_children(mobject._node for mobject in mobjects)
        return self

    # matrix & transform

    @staticmethod
    def matrix_from_translation(vector: Vec3T) -> Mat4T:
        m = np.identity(4)
        m[3, :3] = vector
        return m

    @staticmethod
    def matrix_from_scale(factor: Real | Vec3T) -> Mat4T:
        m = np.identity(4)
        m[:3, :3] *= factor
        return m

    @staticmethod
    def matrix_from_rotation(rotation: Rotation) -> Mat4T:
        m = np.identity(4)
        m[:3, :3] = rotation.as_matrix()
        return m

    @overload
    @staticmethod
    def apply_affine(matrix: Mat4T, vector: Vec3T) -> Vec3T: ...

    @overload
    @staticmethod
    def apply_affine(matrix: Mat4T, vector: Vec3sT) -> Vec3sT: ...

    @staticmethod
    def apply_affine(matrix: Mat4T, vector: Vec3T | Vec3sT) -> Vec3T | Vec3sT:
        if len(vector.shape) == 1:
            v = vector[:, None]
        else:
            v = vector[:, :].T
        v = np.concatenate((v, np.ones((1, v.shape[1]))))
        v = matrix.T @ v
        if not np.allclose(v[3], 1.0):
            v /= v[3]
        v = v[:3]
        if len(vector.shape) == 1:
            result = v.squeeze(axis=1)
        else:
            result = v.T
        return result

    @lazy_property_writable
    @staticmethod
    def _model_matrix_() -> Mat4T:
        return np.identity(4)

    def _apply_transform_locally(self, matrix: Mat4T):
        self._model_matrix_ = self._model_matrix_ @ matrix
        return self

    def _get_local_sample_points(self) -> Vec3sT:
        # Implemented in subclasses
        return np.zeros((0, 3))

    def get_bounding_box(
        self,
        *,
        broadcast: bool = True
    ) -> BoundingBox3D:
        points_array = np.concatenate([
            self.apply_affine(self._model_matrix_, mobject._get_local_sample_points())
            for mobject in self.iter_descendants(broadcast=broadcast)
        ])
        if not points_array.shape[0]:
            warnings.warn("Trying to calculate the bounding box of some mobject with no points")
            origin = ORIGIN
            radius = ORIGIN
        else:
            minimum = points_array.min(axis=0)
            maximum = points_array.max(axis=0)
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
    ) -> Vec3T:
        aabb = self.get_bounding_box(broadcast=broadcast)
        return aabb.radius * 2.0

    def get_bounding_box_point(
        self,
        direction: Vec3T,
        *,
        broadcast: bool = True
    ) -> Vec3T:
        aabb = self.get_bounding_box(broadcast=broadcast)
        return aabb.origin + direction * aabb.radius

    def get_center(
        self,
        *,
        broadcast: bool = True
    ) -> Vec3T:
        return self.get_bounding_box_point(ORIGIN, broadcast=broadcast)

    #def apply_matrix_directly(
    #    self,
    #    matrix: Mat4T,
    #    *,
    #    broadcast: bool = True
    #):
    #    #if np.isclose(np.linalg.det(matrix), 0.0):
    #    #    warnings.warn("Applying a singular matrix transform")
    #    for mobject in self.get_descendants(broadcast=broadcast):
    #        mobject.apply_relative_transform(matrix)
    #    return self

    def apply_transform(
        self,
        matrix: Mat4T,
        *,
        broadcast: bool = True
    ):
        for mobject in self.iter_descendants(broadcast=broadcast):
            mobject._apply_transform_locally(matrix)
        return self

    def apply_relative_transform(
        self,
        matrix: Mat4T,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        if about_point is None:
            if about_edge is None:
                about_edge = ORIGIN
            about_point = self.get_bounding_box_point(about_edge, broadcast=broadcast)
        elif about_edge is not None:
            raise AttributeError("Cannot specify both parameters `about_point` and `about_edge`")

        matrix = reduce(np.ndarray.__matmul__, (
            self.matrix_from_translation(-about_point),
            matrix,
            self.matrix_from_translation(about_point)
        ))
        #if np.isclose(np.linalg.det(matrix), 0.0):
        #    warnings.warn("Applying a singular matrix transform")
        self.apply_transform(matrix, broadcast=broadcast)
        return self

    def shift(
        self,
        vector: Vec3T,
        *,
        coor_mask: Vec3T | None = None,
        broadcast: bool = True
    ):
        if coor_mask is not None:
            vector *= coor_mask
        matrix = self.matrix_from_translation(vector)
        # `about_point` and `about_edge` are meaningless when shifting
        self.apply_relative_transform(
            matrix,
            broadcast=broadcast
        )
        return self

    def scale(
        self,
        factor: Real | Vec3T,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        matrix = self.matrix_from_scale(factor)
        self.apply_relative_transform(
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
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        matrix = self.matrix_from_rotation(rotation)
        self.apply_relative_transform(
            matrix,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def move_to(
        self,
        mobject_or_point: "Mobject | Vec3T",
        aligned_edge: Vec3T = ORIGIN,
        *,
        coor_mask: Vec3T | None = None,
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

    def center(
        self,
        *,
        coor_mask: Vec3T | None = None,
        broadcast: bool = True
    ):
        self.move_to(
            ORIGIN,
            coor_mask=coor_mask,
            broadcast=broadcast
        )
        return self

    def next_to(
        self,
        mobject_or_point: "Mobject | Vec3T",
        direction: Vec3T = RIGHT,
        buff: float = 0.25,
        *,
        coor_mask: Vec3T | None = None,
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
        target_size: Vec3T,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
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
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
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
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
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
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
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
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
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

    def _adjust_frame(
        self,
        original_width: Real,
        original_height: Real,
        specified_width: Real | None,
        specified_height: Real | None,
        specified_frame_scale: Real | None
    ):
        # Called when initializing a planar mobject
        if specified_width is None and specified_height is None:
            width = original_width
            height = original_height
            if specified_frame_scale is not None:
                width *= specified_frame_scale
                height *= specified_frame_scale
        elif specified_width is not None and specified_height is None:
            width = specified_width
            height = specified_width / original_width * original_height
        elif specified_width is None and specified_height is not None:
            width = specified_height / original_height * original_width
            height = specified_height
        elif specified_width is not None and specified_height is not None:
            width = specified_width
            height = specified_height
        else:
            raise  # never
        self.center()
        self.stretch_to_fit_size(np.array((width, height, 0.0)))
        return self

    # animations

    @lazy_property_updatable
    @staticmethod
    def _animations_() -> list["Animation"]:
        return []

    @_animations_.updater
    def animate(self, animation: "Animation"):
        self._animations_.append(animation)
        animation.start(self)
        return self

    @_animations_.updater
    def _update_dt(self, dt: Real):
        for animation in self._animations_[:]:
            animation.update_dt(self, dt)
            if animation.expired():
                self._animations_.remove(animation)
        return self

    # render

    @lazy_property_writable
    @staticmethod
    def _apply_oit_() -> bool:
        return False

    @lazy_property
    @staticmethod
    def _ub_model_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_model", [
            "mat4 u_model_matrix"
        ])

    @lazy_property
    @staticmethod
    def _ub_model_(
        ub_model_o: UniformBlockBuffer,
        model_matrix: Mat4T
    ) -> UniformBlockBuffer:
        ub_model_o.write({
            "u_model_matrix": model_matrix
        })
        return ub_model_o


#class Group(Mobject):
#    def __init__(self, *mobjects: Mobject):
#        super().__init__()
#        self.add(*mobjects)

#    # TODO
#    def _bind_child(self, node, index: int | None = None):
#        assert isinstance(node, Mobject)
#        super()._bind_child(node, index=index)
#        return self
