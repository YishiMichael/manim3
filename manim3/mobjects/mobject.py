__all__ = ["Mobject"]


from abc import ABC
from dataclasses import dataclass
from functools import reduce
import itertools as it
from typing import (
    Generator,
    Generic,
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
from ..utils.scene_config import SceneConfig
from ..utils.space import SpaceUtils


_T = TypeVar("_T")


class PseudoDynamicContainer(ABC, Generic[_T]):
    # Provides a interface similar to `LazyDynamicContainer`.
    # If `_parents` and `_real_ancestors` are implemented with `LazyDynamicContainer` also,
    # loops will pop up in the DAG in the lazy system.

    __slots__ = ("_elements",)

    def __init__(
        self,
        *elements: _T
    ) -> None:
        super().__init__()
        self._elements: list[_T] = []
        self.add(*elements)

    def __iter__(self) -> Iterator[_T]:
        return self._elements.__iter__()

    def __len__(self) -> int:
        return self._elements.__len__()

    @overload
    def __getitem__(
        self,
        index: int
    ) -> _T:
        ...

    @overload
    def __getitem__(
        self,
        index: slice
    ) -> list[_T]:
        ...

    def __getitem__(
        self,
        index: int | slice
    ) -> _T | list[_T]:
        return self._elements.__getitem__(index)

    def __copy__(self):
        return PseudoDynamicContainer(*self._elements)

    def add(
        self,
        *elements: _T
    ):
        if not elements:
            return self
        for element in elements:
            if element in self._elements:
                continue
            self._elements.append(element)
        return self

    def remove(
        self,
        *elements: _T
    ):
        if not elements:
            return self
        for element in elements:
            if element not in self._elements:
                continue
            self._elements.remove(element)
        return self


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
        self._parents: PseudoDynamicContainer[Mobject] = PseudoDynamicContainer()
        self._real_ancestors: PseudoDynamicContainer[Mobject] = PseudoDynamicContainer()

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
        all_descendants = [
            descendant_mobject
            for mobject in mobjects
            for descendant_mobject in mobject.iter_descendants()
        ]
        self._children_.add(*mobjects)
        for ancestor_mobject in self.iter_ancestors():
            ancestor_mobject._real_descendants_.add(*all_descendants)
        for mobject in mobjects:
            mobject._parents.add(self)
        for descendant_mobject in all_descendants:
            descendant_mobject._real_ancestors.add(*self.iter_ancestors())
        return self

    def remove(
        self,
        *mobjects: "Mobject"
    ):
        all_descendants = [
            descendant_mobject
            for mobject in mobjects
            for descendant_mobject in mobject.iter_descendants()
        ]
        self._children_.remove(*mobjects)
        for ancestor_mobject in self.iter_ancestors():
            ancestor_mobject._real_descendants_.remove(*all_descendants)
        for mobject in mobjects:
            mobject._parents.remove(self)
        for descendant_mobject in all_descendants:
            descendant_mobject._real_ancestors.remove(*self.iter_ancestors())
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
        self.remove(*self.iter_children())
        return self

    def copy(self):
        result = self._copy()
        real_descendants_copy = [
            descendant._copy()
            for descendant in self._real_descendants_
        ]
        descendants = [self, *self._real_descendants_]
        descendants_copy = [result, *real_descendants_copy]

        def get_matched_descendant_mobject(
            mobject: Mobject
        ) -> Mobject:
            if mobject not in descendants:
                return mobject
            return descendants_copy[descendants.index(mobject)]

        for descendant, descendant_copy in zip(descendants, descendants_copy, strict=True):
            descendant_copy._parents = PseudoDynamicContainer(*(
                get_matched_descendant_mobject(mobject)
                for mobject in descendant._parents
            ))
            descendant_copy._real_ancestors = PseudoDynamicContainer(*(
                get_matched_descendant_mobject(mobject)
                for mobject in descendant._real_ancestors
            ))

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
    def _scene_config_(cls) -> SceneConfig:
        return SceneConfig()

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
