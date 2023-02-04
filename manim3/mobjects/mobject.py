__all__ = ["Mobject"]


#import copy
from dataclasses import dataclass
from functools import reduce
from typing import (
    Callable,
    Generator,
    Generic,
    Iterable,
    Iterator,
    TypeVar,
    overload
)
import warnings

import moderngl
import numpy as np
from scipy.spatial.transform import Rotation

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
from ..passes.render_pass import RenderPass
from ..rendering.render_procedure import (
    RenderProcedure,
    UniformBlockBuffer
)
#from ..rendering.renderable import Renderable
from ..scenes.scene_config import SceneConfig
from ..utils.lazy import (
    LazyBase,
    lazy_property,
    lazy_property_updatable,
    lazy_property_writable
)
from ..utils.node import Node
from ..utils.space import SpaceUtils


_T = TypeVar("_T")
_MobjectT = TypeVar("_MobjectT", bound="Mobject")


class MobjectData(Generic[_MobjectT, _T]):
    def __init__(self, interpolate_method: Callable[[_T, _T, Real], _T], init_value: _T):
        self.interpolate_method: Callable[[_T, _T, Real], _T] = interpolate_method
        self.pointer_dict: dict[_MobjectT, int] = {}
        self.references_dict: dict[int, list[_MobjectT]] = {}
        self.value_dict: dict[int, _T] = {0: init_value}
        self.pointer: int = 0

    @overload
    def __get__(self, instance: None, owner: type[_MobjectT] | None = None) -> "MobjectData[_MobjectT, _T]": ...

    @overload
    def __get__(self, instance: _MobjectT, owner: type[_MobjectT] | None = None) -> _T: ...

    def __get__(self, instance: _MobjectT | None, owner: type[_MobjectT] | None = None) -> "MobjectData[_MobjectT, _T] | _T":
        if instance is None:
            return self
        return self.value_dict[self.pointer_dict[instance]]


class mobject_data(Generic[_MobjectT, _T]):
    def __init__(self, interpolate_method: Callable[[_T, _T, Real], _T]):
        self.interpolate_method: Callable[[_T, _T, Real], _T] = interpolate_method

    def __call__(self, init_method: Callable[[], _T]) -> MobjectData[_MobjectT, _T]:
        init_value = init_method.__func__()
        return MobjectData(self.interpolate_method, init_value)


class mobject_property(Generic[_MobjectT, _T]):
    def __init__(self, interpolate_method: Callable[[_T, _T, Real], _T], init_value: _T):
        self.interpolate_method: Callable[[_T, _T, Real], _T] = interpolate_method


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class BoundingBox3D:
    maximum: Vec3T
    minimum: Vec3T

    @property
    def origin(self) -> Vec3T:
        return (self.maximum + self.minimum) / 2.0

    @property
    def radius(self) -> Vec3T:
        radius = (self.maximum - self.minimum) / 2.0
        # For zero-width dimensions of radius, thicken a little bit to avoid zero division
        radius[np.isclose(radius, 0.0)] = 1e-8
        return radius


class MobjectNode(Node):
    def __init__(self, mobject: "Mobject"):
        self._mobject: Mobject = mobject
        super().__init__()


class Mobject(LazyBase):
    def __init__(self) -> None:
        self._node: MobjectNode = MobjectNode(self)
        super().__init__()

    #    #self.matrix: pyrr.Matrix44 = pyrr.Matrix44.identity()
    #    self.render_passes: list["RenderPass"] = []
    #    self.animations: list["Animation"] = []  # TODO: circular typing
    #    super().__init__()

    def __iter__(self) -> Iterator["Mobject"]:
        return self.iter_children()

    @overload
    def __getitem__(self, index: int) -> "Mobject": ...

    @overload
    def __getitem__(self, index: slice) -> "list[Mobject]": ...

    def __getitem__(self, index: int | slice) -> "Mobject | list[Mobject]":
        if isinstance(index, int):
            return self._node.__getitem__(index)._mobject
        return [node._mobject for node in self._node.__getitem__(index)]

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

    @lazy_property_writable
    @staticmethod
    def _model_matrix_() -> Mat4T:
        return np.identity(4)

    def _apply_transform_locally(self, matrix: Mat4T):
        self._model_matrix_ = matrix @ self._model_matrix_
        return self

    @lazy_property
    @staticmethod
    def _local_sample_points_() -> Vec3sT:
        # Implemented in subclasses
        return np.zeros((0, 3))

    @lazy_property
    @staticmethod
    def _has_local_sample_points_(local_sample_points: Vec3sT) -> bool:
        return bool(len(local_sample_points))

    @lazy_property
    @staticmethod
    def _local_world_bounding_box_(
        model_matrix: Mat4T,
        local_sample_points: Vec3sT,
        has_local_sample_points: bool
    ) -> BoundingBox3D | None:
        if not has_local_sample_points:
            return None
        world_sample_points = SpaceUtils.apply_affine(model_matrix, local_sample_points)
        return BoundingBox3D(
            maximum=world_sample_points.max(axis=0),
            minimum=world_sample_points.min(axis=0)
        )

    def get_bounding_box(
        self,
        *,
        broadcast: bool = True
    ) -> BoundingBox3D:
        points_array = np.concatenate([
            (aabb.maximum, aabb.minimum)
            for mobject in self.iter_descendants(broadcast=broadcast)
            if (aabb := mobject._local_world_bounding_box_) is not None
        ])
        if not len(points_array):
            warnings.warn("Trying to calculate the bounding box of some mobject with no points")
            maximum = ORIGIN
            minimum = ORIGIN
        else:
            maximum = points_array.max(axis=0)
            minimum = points_array.min(axis=0)
        return BoundingBox3D(
            maximum=maximum,
            minimum=minimum
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
            SpaceUtils.matrix_from_translation(about_point),
            matrix,
            SpaceUtils.matrix_from_translation(-about_point)
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
        matrix = SpaceUtils.matrix_from_translation(vector)
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
        matrix = SpaceUtils.matrix_from_scale(factor)
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
        matrix = SpaceUtils.matrix_from_rotation(rotation)
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
        scale_factor = np.ones(2)
        if specified_width is None and specified_height is None:
            if specified_frame_scale is not None:
                scale_factor *= specified_frame_scale
        elif specified_width is not None and specified_height is None:
            scale_factor *= specified_width / original_width
        elif specified_width is None and specified_height is not None:
            scale_factor *= specified_height / original_height
        elif specified_width is not None and specified_height is not None:
            scale_factor *= np.array((
                specified_width / original_width,
                specified_height / original_height
            ))
        else:
            raise ValueError  # never
        self.center().scale(np.append(scale_factor, 1.0))
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
        return ub_model_o.write({
            "u_model_matrix": model_matrix.T
        })

    @lazy_property_writable
    @staticmethod
    def _render_samples_() -> int:
        return 0

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        # Implemented in subclasses
        # This function is not responsible for clearing the `target_framebuffer`.
        # On the other hand, one shall clear the framebuffer before calling this function.
        pass

    def _render_with_samples(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        samples = self._render_samples_
        if not samples:
            self._render(scene_config, target_framebuffer)
            return

        with RenderProcedure.texture(samples=4) as msaa_color_texture, \
                RenderProcedure.depth_texture(samples=4) as msaa_depth_texture, \
                RenderProcedure.framebuffer(
                    color_attachments=[msaa_color_texture],
                    depth_attachment=msaa_depth_texture
                ) as msaa_framebuffer:
            self._render(scene_config, msaa_framebuffer)
            RenderProcedure.downsample_framebuffer(msaa_framebuffer, target_framebuffer)

    def _render_with_passes(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        render_passes = self._render_passes_
        if not render_passes:
            self._render_with_samples(scene_config, target_framebuffer)
            return

        with RenderProcedure.texture() as intermediate_texture_0, \
                RenderProcedure.texture() as intermediate_texture_1:
            textures = (intermediate_texture_0, intermediate_texture_1)
            target_texture_id = 0
            with RenderProcedure.framebuffer(
                        color_attachments=[intermediate_texture_0],
                        depth_attachment=target_framebuffer.depth_attachment
                    ) as initial_framebuffer:
                self._render_with_samples(scene_config, initial_framebuffer)
            for render_pass in render_passes[:-1]:
                target_texture_id = 1 - target_texture_id
                with RenderProcedure.framebuffer(
                            color_attachments=[textures[target_texture_id]],
                            depth_attachment=None
                        ) as intermediate_framebuffer:
                    render_pass._render(
                        texture=textures[1 - target_texture_id],
                        target_framebuffer=intermediate_framebuffer
                    )
            target_framebuffer.depth_mask = False  # TODO: shall we disable writing to depth?
            render_passes[-1]._render(
                texture=textures[target_texture_id],
                target_framebuffer=target_framebuffer
            )
            target_framebuffer.depth_mask = True

    @lazy_property_updatable
    @staticmethod
    def _render_passes_() -> list[RenderPass]:
        return []

    @_render_passes_.updater
    def add_pass(self, *render_passes: RenderPass):
        for render_pass in render_passes:
            self._render_passes_.append(render_pass)
        return self

    @_render_passes_.updater
    def remove_pass(self, *render_passes: RenderPass):
        for render_pass in render_passes:
            self._render_passes_.remove(render_pass)
        return self


#class Group(Mobject):
#    def __init__(self, *mobjects: Mobject):
#        super().__init__()
#        self.add(*mobjects)

#    # TODO
#    def _bind_child(self, node, index: int | None = None):
#        assert isinstance(node, Mobject)
#        super()._bind_child(node, index=index)
#        return self
