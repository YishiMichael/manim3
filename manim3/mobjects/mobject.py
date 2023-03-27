__all__ = ["Mobject"]


from dataclasses import dataclass
from functools import reduce
import itertools as it
from typing import (
    Generator,
    Iterator,
    overload
)
import warnings

import moderngl
import numpy as np
from scipy.spatial.transform import Rotation

from ..constants import (
    ORIGIN,
    PI,
    RIGHT
)
from ..custom_typing import (
    Mat4T,
    Vec2T,
    Vec3T,
    Vec3sT
)
from ..lazy.core import (
    LazyDynamicVariableDescriptor,
    LazyObject,
    LazyUnitaryVariableDescriptor,
    LazyWrapper
)
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..passes.render_pass import RenderPass
from ..rendering.framebuffer_batch import ColorFramebufferBatch
from ..rendering.gl_buffer import UniformBlockBuffer
from ..utils.scene_state import SceneState
from ..utils.space import SpaceUtils


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class BoundingBox:
    maximum: Vec3T
    minimum: Vec3T

    @property
    def origin(self) -> Vec3T:
        return (self.maximum + self.minimum) / 2.0

    @property
    def radius(self) -> Vec3T:
        radius = (self.maximum - self.minimum) / 2.0
        # For zero-width dimensions of radius, thicken a little bit to avoid zero division.
        radius[np.isclose(radius, 0.0)] = 1e-8
        return radius


class Mobject(LazyObject):
    __slots__ = (
        "_parents",
        "_real_ancestors"
    )

    def __init__(self) -> None:
        super().__init__()
        # If `_parents` and `_real_ancestors` are implemented with `LazyDynamicContainer` also,
        # loops will pop up in the DAG of the lazy system.
        self._parents: set[Mobject] = set()
        self._real_ancestors: set[Mobject] = set()

    def __iter__(self) -> "Iterator[Mobject]":
        return iter(self._children_)

    @overload
    def __getitem__(
        self,
        index: int
    ) -> "Mobject": ...

    @overload
    def __getitem__(
        self,
        index: slice
    ) -> "list[Mobject]": ...

    def __getitem__(
        self,
        index: int | slice
    ) -> "Mobject | list[Mobject]":
        return self._children_.__getitem__(index)

    # family matters
    # These methods implement a DAG (directed acyclic graph).

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _children_(cls) -> "list[Mobject]":
        return []

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _real_descendants_(cls) -> "list[Mobject]":
        return []

    def iter_children(self) -> "Generator[Mobject, None, None]":
        yield from self._children_

    def iter_parents(self) -> "Generator[Mobject, None, None]":
        yield from self._parents

    def iter_descendants(
        self,
        *,
        broadcast: bool = True
    ) -> "Generator[Mobject, None, None]":
        yield self
        if broadcast:
            yield from self._real_descendants_

    def iter_ancestors(
        self,
        *,
        broadcast: bool = True
    ) -> "Generator[Mobject, None, None]":
        yield self
        if broadcast:
            yield from self._real_ancestors

    def add(
        self,
        *mobjects: "Mobject"
    ):
        all_descendants = list(it.chain(*(
            mobject.iter_descendants()
            for mobject in mobjects
        )))
        self._children_.add(*mobjects)
        for ancestor_mobject in self.iter_ancestors():
            ancestor_mobject._real_descendants_.add(*all_descendants)
        for mobject in mobjects:
            mobject._parents.add(self)
        for descendant_mobject in all_descendants:
            descendant_mobject._real_ancestors.update(self.iter_ancestors())
        return self

    def discard(
        self,
        *mobjects: "Mobject"
    ):
        all_descendants = list(it.chain(*(
            mobject.iter_descendants()
            for mobject in mobjects
        )))
        self._children_.discard(*mobjects)
        for ancestor_mobject in self.iter_ancestors():
            ancestor_mobject._real_descendants_.discard(*all_descendants)
        for mobject in mobjects:
            mobject._parents.discard(self)
        for descendant_mobject in all_descendants:
            descendant_mobject._real_ancestors.difference_update(self.iter_ancestors())
        return self

    #def index(self, node: "Mobject") -> int:
    #    return self._children_.index(node)

    #def insert(self, index: int, node: "Mobject"):
    #    self._bind_child(node, index=index)
    #    return self

    #def add(self, *nodes: "Mobject"):
    #    for node in nodes:
    #        self._bind_child(node)
    #    return self

    #def remove(self, *nodes: "Mobject"):
    #    for node in nodes:
    #        self._unbind_child(node)
    #    return self

    #def pop(self, index: int = -1):
    #    node = self[index]
    #    self.remove(node)
    #    return node

    def clear(self):
        self.discard(*self.iter_children())
        return self

    def becomes(
        self,
        src: "Mobject"
    ):
        if self is src:
            return self

        self.clear()
        self._becomes(src)
        self.add(*src.iter_children())
        return self

    def copy(
        self,
        broadcast: bool = True
    ):
        result = self._copy()
        descendants: list[Mobject] = [self]
        descendants_copy: list[Mobject] = [result]
        if broadcast:
            real_descendants_copy = [
                descendant._copy()
                for descendant in self._real_descendants_
            ]
            descendants.extend(self._real_descendants_)
            descendants_copy.extend(real_descendants_copy)

        def get_matched_descendant_mobject(
            mobject: Mobject
        ) -> Mobject:
            if mobject not in descendants:
                return mobject
            return descendants_copy[descendants.index(mobject)]

        for descendant, descendant_copy in zip(descendants, descendants_copy, strict=True):
            descendant_copy._parents.clear()
            descendant_copy._parents.update(
                get_matched_descendant_mobject(mobject)
                for mobject in descendant._parents
            )
            descendant_copy._real_ancestors.clear()
            descendant_copy._real_ancestors.update(
                get_matched_descendant_mobject(mobject)
                for mobject in descendant._real_ancestors
            )

        for descriptor in self.__class__._LAZY_VARIABLE_DESCRIPTORS:
            if not issubclass(descriptor.element_type, Mobject):
                continue
            if isinstance(descriptor, LazyUnitaryVariableDescriptor):
                mobject = descriptor.__get__(self)
                descriptor.__set__(result, get_matched_descendant_mobject(mobject))
            elif isinstance(descriptor, LazyDynamicVariableDescriptor):
                descriptor.__set__(result, (
                    get_matched_descendant_mobject(mobject)
                    for mobject in descriptor.__get__(self)
                ))

        if not broadcast:
            result.clear()
        return result

    # matrix & transform

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _model_matrix_(cls) -> Mat4T:
        return np.identity(4)

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _local_sample_points_(cls) -> Vec3sT:
        # Implemented in subclasses.
        return np.zeros((0, 3))

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _ub_model_(
        cls,
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

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _has_local_sample_points_(
        cls,
        local_sample_points: Vec3sT
    ) -> bool:
        return bool(len(local_sample_points))

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _bounding_box_without_descendants_(
        cls,
        model_matrix: Mat4T,
        local_sample_points: Vec3sT,
        has_local_sample_points: bool
    ) -> BoundingBox | None:
        if not has_local_sample_points:
            return None
        world_sample_points = SpaceUtils.apply_affine(model_matrix, local_sample_points)
        return BoundingBox(
            maximum=world_sample_points.max(axis=0),
            minimum=world_sample_points.min(axis=0)
        )

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _bounding_box_with_descendants_(
        cls,
        bounding_box_without_descendants: BoundingBox | None,
        real_descendants__bounding_box_without_descendants: list[BoundingBox | None]
    ) -> BoundingBox | None:
        points_array = np.array(list(it.chain(*(
            (aabb.maximum, aabb.minimum)
            for aabb in (
                bounding_box_without_descendants,
                *real_descendants__bounding_box_without_descendants
            )
            if aabb is not None
        ))))
        if not len(points_array):
            return None
        return BoundingBox(
            maximum=points_array.max(axis=0),
            minimum=points_array.min(axis=0)
        )

    def get_bounding_box(
        self,
        *,
        broadcast: bool = True
    ) -> BoundingBox:
        if broadcast:
            result = self._bounding_box_with_descendants_.value
        else:
            result = self._bounding_box_without_descendants_.value
        if result is None:
            warnings.warn("Trying to calculate the bounding box of some mobject with no points")
            return BoundingBox(
                maximum=ORIGIN,
                minimum=ORIGIN
            )
        return result

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
        # Avoid redundant caculations.
        transform_dict: dict[LazyWrapper[Mat4T], LazyWrapper[Mat4T]] = {}
        for mobject in self.iter_descendants(broadcast=broadcast):
            original_matrix = mobject._model_matrix_
            if (transformed_matrix := transform_dict.get(original_matrix)) is None:
                transformed_matrix = LazyWrapper(matrix @ original_matrix.value)
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
        # `about_point` and `about_edge` are meaningless when shifting.
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
        factor: float | Vec3T,
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
        factor: float | Vec3T,
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
        target_length: float,
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
        target_length: float,
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
        target_length: float,
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
        target_length: float,
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
        *,
        original_width: float,
        original_height: float,
        specified_frame_scale: float | None,
        specified_width: float | None,
        specified_height: float | None
    ) -> Vec2T:
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

    def flip(
        self,
        axis: Vec3T,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        self.rotate(
            rotation=Rotation.from_rotvec(SpaceUtils.normalize(axis) * PI),
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    # render

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _apply_oit_(cls) -> bool:
        return False

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _render_passes_(cls) -> list[RenderPass]:
        return []

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _scene_state_(cls) -> SceneState:
        return SceneState()

    def _render(
        self,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        # Implemented in subclasses.
        pass

    def _render_with_passes(
        self,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        render_passes = self._render_passes_
        if not render_passes:
            self._render(target_framebuffer)
            return

        with ColorFramebufferBatch() as batch_0, ColorFramebufferBatch() as batch_1:
            batches = (batch_0, batch_1)
            target_id = 0
            self._render(batch_0.framebuffer)
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

    def add_pass(
        self,
        *render_passes: RenderPass
    ):
        self._render_passes_.add(*render_passes)
        return self

    def remove_pass(
        self,
        *render_passes: RenderPass
    ):
        self._render_passes_.remove(*render_passes)
        return self
