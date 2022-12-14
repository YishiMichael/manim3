__all__ = ["Mobject"]


import copy
from dataclasses import dataclass
from functools import reduce
from typing import (
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
from ..geometries.geometry import Geometry
from ..utils.lazy import (
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.node import Node
from ..utils.renderable import (
    Framebuffer,
    IntermediateDepthTextures,
    IntermediateFramebuffer,
    IntermediateTextures,
    Renderable
)
from ..utils.scene_config import SceneConfig


@dataclass
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
        return iter(self.get_children())

    #def __getitem__(self, i: int | slice) -> "Mobject":
    #    if isinstance(i, int):
    #        return self.get_children().__getitem__(i)
    #    return Mobject().add(*self.get_children().__getitem__(i))

    def copy(self):
        return copy.copy(self)  # TODO

    # family matters

    def get_parents(self) -> "list[Mobject]":
        return [node._mobject for node in self._node.get_parents()]

    def get_children(self) -> "list[Mobject]":
        return [node._mobject for node in self._node.get_children()]

    def get_ancestors(self, *, broadcast: bool = True) -> "list[Mobject]":
        return [node._mobject for node in self._node.get_ancestors(broadcast=broadcast)]

    def get_descendants(self, *, broadcast: bool = True) -> "list[Mobject]":
        return [node._mobject for node in self._node.get_descendants(broadcast=broadcast)]

    def get_descendants_excluding_self(self) -> "list[Mobject]":
        return [node._mobject for node in self._node.get_descendants_excluding_self()]

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

    #@lazy_property_initializer
    #@classmethod
    #def _geometry_matrix_(cls) -> Mat4T:
    #    return np.identity(4)

    #@lazy_property
    #@classmethod
    #def _model_matrix_(cls, matrix: Mat4T, geometry_matrix: Mat4T) -> Mat4T:
    #    return matrix @ geometry_matrix

    #@lazy_property
    #@classmethod
    #def _model_matrix_buffer_(cls, model_matrix: Mat4T) -> moderngl.Buffer:
    #    return cls._make_buffer(model_matrix)

    #@_model_matrix_buffer_.releaser
    #@staticmethod
    #def _model_matrix_buffer_releaser(model_matrix_buffer: moderngl.Buffer) -> None:
    #    model_matrix_buffer.release()

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

    @lazy_property_initializer
    @staticmethod
    def _geometry_() -> Geometry:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _model_matrix_() -> Mat4T:
        return np.identity(4)

    @_model_matrix_.updater
    def apply_transform_locally(self, matrix: Mat4T):
        self._model_matrix_ = self._model_matrix_ @ matrix
        return self

    def get_bounding_box(
        self,
        *,
        broadcast: bool = True
    ) -> BoundingBox3D:
        points_array = np.concatenate([
            self.apply_affine(self._model_matrix_, mobject._geometry_._position_)
            for mobject in self.get_descendants(broadcast=broadcast)
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
    #        mobject.apply_transform(matrix)
    #    return self

    def apply_transform(
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
        for mobject in self.get_descendants(broadcast=broadcast):
            mobject.apply_transform_locally(matrix)
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
        self.apply_transform(
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
        self.apply_transform(
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
        self.apply_transform(
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

    # render

    def _render(self, scene_config: SceneConfig, target_framebuffer: Framebuffer) -> None:
        # Implemented in subclasses
        pass

    def _render_full(self, scene_config: SceneConfig, target_framebuffer: Framebuffer) -> None:
        render_passes = self._render_passes_
        #if not render_passes:
        #    #target_framebuffer.clear()
        #    target_framebuffer.use()
        #    self._render(scene_config, target_framebuffer)
        #    return

        with IntermediateTextures.register_n(len(render_passes)) as textures:
            with IntermediateDepthTextures.register_n(len(render_passes)) as depth_textures:
                input_framebuffers = [
                    IntermediateFramebuffer(
                        color_attachments=[texture],
                        depth_attachment=depth_texture
                    )
                    for texture, depth_texture in zip(textures, depth_textures)
                ]
                output_framebuffers: list[Framebuffer] = input_framebuffers[:]
                output_framebuffers.append(target_framebuffer)
                #framebuffers[0].use()
                self._render(scene_config, output_framebuffers[0])
                for render_pass, input_framebuffer, output_framebuffer in zip(render_passes, input_framebuffers, output_framebuffers[1:]):
                    #output_framebuffer.use()
                    render_pass._render(
                        input_framebuffer=input_framebuffer,
                        output_framebuffer=output_framebuffer,
                        mobject=self,
                        scene_config=scene_config
                    )

    @lazy_property_initializer
    @staticmethod
    def _render_passes_() -> list["RenderPass"]:
        return []

    @_render_passes_.updater
    def add_pass(self, *render_passes: "RenderPass"):
        for render_pass in render_passes:
            self._render_passes_.append(render_pass)
        return self

    @_render_passes_.updater
    def remove_pass(self, *render_passes: "RenderPass"):
        for render_pass in render_passes:
            self._render_passes_.remove(render_pass)
        return self

    # animations

    @lazy_property_initializer
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

    # shader

    #@lazy_property_initializer_writable
    #@classmethod
    #def _camera_(cls) -> Camera:
    #    return PerspectiveCamera()

    #@_camera.setter
    #def _camera(self, camera: Camera) -> None:
    #    pass


#class Group(Mobject):
#    def __init__(self, *mobjects: Mobject):
#        super().__init__()
#        self.add(*mobjects)

#    # TODO
#    def _bind_child(self, node, index: int | None = None):
#        assert isinstance(node, Mobject)
#        super()._bind_child(node, index=index)
#        return self
