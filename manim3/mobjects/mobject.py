__all__ = ["Mobject"]


from dataclasses import dataclass
from functools import reduce
import itertools as it
from typing import (
    Generator,
    Iterable,
    Iterator,
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
    Vec2T,
    Vec3T,
    Vec3sT
)
from ..passes.render_pass import RenderPass
from ..rendering.framebuffer_batch import FramebufferBatch
from ..rendering.framebuffer_batches import (
    ColorFramebufferBatch,
    SimpleFramebufferBatch
)
from ..rendering.glsl_variables import UniformBlockBuffer
from ..scenes.scene_config import SceneConfig
from ..utils.lazy import (
    LazyBase,
    NewData,
    lazy_basedata,
    lazy_property,
    lazy_slot
)
from ..utils.space import SpaceUtils


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


class Mobject(LazyBase):
    __slots__ = ()

    def __iter__(self) -> "Iterator[Mobject]":
        return iter(self._children)

    @overload
    def __getitem__(self, index: int) -> "Mobject": ...

    @overload
    def __getitem__(self, index: slice) -> "list[Mobject]": ...

    def __getitem__(self, index: int | slice) -> "Mobject | list[Mobject]":
        return self._children.__getitem__(index)

    # family matters
    # These methods implement a double-directed, loop-free graph

    @lazy_slot
    @staticmethod
    def _parents() -> "list[Mobject]":
        return []

    @lazy_slot
    @staticmethod
    def _children() -> "list[Mobject]":
        return []

    def _bind_child(self, node: "Mobject", *, index: int | None = None) -> None:
        if self in node.iter_descendants():
            raise ValueError(f"'{node}' has already included '{self}'")
        if index is not None:
            self._children.insert(index, node)
        else:
            self._children.append(node)
        node._parents.append(self)

    def _unbind_child(self, node: "Mobject") -> None:
        self._children.remove(node)
        node._parents.remove(self)

    def iter_parents(self) -> "Iterator[Mobject]":
        return iter(self._parents)

    def iter_children(self) -> "Iterator[Mobject]":
        return iter(self._children)

    def iter_ancestors_with_duplicates(self) -> "Generator[Mobject, None, None]":
        yield self
        for parent_node in self._parents:
            yield from parent_node.iter_ancestors_with_duplicates()

    def iter_descendants_with_duplicates(self) -> "Generator[Mobject, None, None]":
        yield self
        for child_node in self._children:
            yield from child_node.iter_descendants_with_duplicates()

    def iter_ancestors(self, *, broadcast: bool = True) -> "Generator[Mobject, None, None]":
        yield self
        if not broadcast:
            return
        occurred: set[Mobject] = {self}
        for node in self.iter_ancestors_with_duplicates():
            if node in occurred:
                continue
            yield node
            occurred.add(node)

    def iter_descendants(self, *, broadcast: bool = True) -> "Generator[Mobject, None, None]":
        yield self
        if not broadcast:
            return
        occurred: set[Mobject] = {self}
        for node in self.iter_descendants_with_duplicates():
            if node in occurred:
                continue
            yield node
            occurred.add(node)

    def index(self, node: "Mobject") -> int:
        return self._children.index(node)

    def insert(self, index: int, node: "Mobject"):
        self._bind_child(node, index=index)
        return self

    def add(self, *nodes: "Mobject"):
        for node in nodes:
            self._bind_child(node)
        return self

    def remove(self, *nodes: "Mobject"):
        for node in nodes:
            self._unbind_child(node)
        return self

    def pop(self, index: int = -1):
        node = self[index]
        self._unbind_child(node)
        return node

    def clear(self):
        for child in self.iter_children():
            self._unbind_child(child)
        return self

    def clear_parents(self):
        for parent in self.iter_parents():
            parent._unbind_child(self)
        return self

    def set_children(self, nodes: "Iterable[Mobject]"):
        self.clear()
        self.add(*nodes)
        return self

    def copy_standalone(self):
        result = self._copy()
        result._parents = []
        result._children = []
        for child in self._children:
            child_copy = child.copy_standalone()
            result._bind_child(child_copy)
        return result

    def copy(self):
        result = self.copy_standalone()
        for parent in self._parents:
            parent._bind_child(result)
        return result

    # matrix & transform

    @lazy_basedata
    @staticmethod
    def _model_matrix_() -> Mat4T:
        return np.identity(4)

    @lazy_basedata
    @staticmethod
    def _local_sample_points_() -> Vec3sT:
        # Implemented in subclasses
        return np.zeros((0, 3))

    @lazy_property
    @staticmethod
    def _ub_model_(
        model_matrix: Mat4T
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_model",
            fields=[
                "mat4 u_model_matrix"
            ],
            data={
                "u_model_matrix": model_matrix.T
            }
        )

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
        points_array = np.array(list(it.chain(*(
            (aabb.maximum, aabb.minimum)
            for mobject in self.iter_descendants(broadcast=broadcast)
            if (aabb := mobject._local_world_bounding_box_) is not None
        ))))
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

    def apply_transform(
        self,
        matrix: Mat4T,
        *,
        broadcast: bool = True
    ):
        # Avoid redundant caculations
        transform_dict: dict[NewData[Mat4T], NewData[Mat4T]] = {}
        for mobject in self.iter_descendants(broadcast=broadcast):
            original_matrix = Mobject._model_matrix_._get_data(mobject)
            if (transformed_matrix := transform_dict.get(original_matrix)) is None:
                transformed_matrix = NewData(matrix @ mobject._model_matrix_)
                transform_dict[original_matrix] = transformed_matrix
            mobject._model_matrix_ = transformed_matrix
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
        self.apply_transform(
            matrix=matrix,
            broadcast=broadcast
        )
        return self

    # shift relatives

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
            matrix=matrix,
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
            vector=vector,
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
            mobject_or_point=ORIGIN,
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
            vector=vector,
            coor_mask=coor_mask,
            broadcast=broadcast
        )
        return self

    # scale relatives

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
            matrix=matrix,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def scale_about_origin(
        self,
        factor: Real | Vec3T,
        *,
        broadcast: bool = True
    ):
        self.scale(
            factor=factor,
            about_point=ORIGIN,
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
            factor=factor_vector,
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
            factor=factor_vector,
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
            target_length=target_length,
            dim=0,
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
            target_length=target_length,
            dim=1,
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
            target_length=target_length,
            dim=2,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    @classmethod
    def _get_frame_scale_vector(
        cls,
        original_width: Real,
        original_height: Real,
        specified_width: Real | None,
        specified_height: Real | None,
        specified_frame_scale: Real | None
    ) -> Vec2T:
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
        return scale_factor
        #self.center().scale(np.append(scale_factor, 1.0))
        #return self

    # rotate relatives

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
            matrix=matrix,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def rotate_about_origin(
        self,
        rotation: Rotation,
        *,
        broadcast: bool = True
    ):
        self.rotate(
            rotation=rotation,
            about_point=ORIGIN,
            broadcast=broadcast
        )
        return self

    # render

    @lazy_slot
    @staticmethod
    def _apply_oit() -> bool:
        return False

    @lazy_slot
    @staticmethod
    def _render_samples() -> int:
        return 0

    @lazy_slot
    @staticmethod
    def _render_passes() -> list[RenderPass]:
        return []

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        # Implemented in subclasses
        # This function is not responsible for clearing the `target_framebuffer`.
        # On the other hand, one shall clear the framebuffer before calling this function.
        pass

    def _render_with_samples(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        samples = self._render_samples
        if not samples:
            self._render(scene_config, target_framebuffer)
            return

        with SimpleFramebufferBatch(samples=samples) as msaa_batch:
            self._render(scene_config, msaa_batch.framebuffer)
            FramebufferBatch.downsample_framebuffer(msaa_batch.framebuffer, target_framebuffer)

    def _render_with_passes(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        render_passes = self._render_passes
        if not render_passes:
            self._render_with_samples(scene_config, target_framebuffer)
            return

        with ColorFramebufferBatch() as batch_0, ColorFramebufferBatch() as batch_1:
            batches = (batch_0, batch_1)
            target_id = 0
            self._render_with_samples(scene_config, batch_0.framebuffer)
            for render_pass in render_passes[:-1]:
                target_id = 1 - target_id
                render_pass._render(
                    texture=batches[1 - target_id].color_texture,
                    target_framebuffer=batches[target_id].framebuffer
                )
            target_framebuffer.depth_mask = False  # TODO: shall we disable writing to depth?
            render_passes[-1]._render(
                texture=batches[target_id].color_texture,
                target_framebuffer=target_framebuffer
            )
            target_framebuffer.depth_mask = True

    def add_pass(self, *render_passes: RenderPass):
        self._render_passes.extend(render_passes)
        return self

    def remove_pass(self, *render_passes: RenderPass):
        for render_pass in render_passes:
            self._render_passes.remove(render_pass)
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
