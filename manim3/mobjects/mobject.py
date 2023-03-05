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
    RIGHT
)
from ..custom_typing import (
    Mat4T,
    Real,
    Vec2T,
    Vec3T,
    Vec3sT
)
from ..lazy.core import (
    LazyCollection,
    LazyCollectionVariableDescriptor,
    LazyObject,
    LazyObjectVariableDescriptor
)
from ..lazy.interface import (
    Lazy,
    LazyMode,
    LazyWrapper
)
from ..passes.render_pass import RenderPass
#from ..rendering.framebuffer_batch import FramebufferBatch
from ..rendering.framebuffer_batches import (
    ColorFramebufferBatch,
    #SimpleFramebufferBatch
)
from ..rendering.glsl_buffers import UniformBlockBuffer
from ..utils.scene_config import SceneConfig
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


class Mobject(LazyObject):
    __slots__ = (
        "_parents",
        "_real_ancestors"
        #"_apply_oit",
        #"_render_samples"
    )

    def __init__(self) -> None:
        super().__init__()
        self._parents: list[Mobject] = []
        self._real_ancestors: list[Mobject] = []
        #self._apply_oit: bool = False
        #self._render_samples: int = 0

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
    # These methods implement a Directed Acyclic Graph

    #@lazy_value
    #@staticmethod
    #def _parents() -> "LazyCollection[Mobject]":
    #    return LazyCollection()

    @Lazy.variable(LazyMode.COLLECTION)
    def _children_(cls) -> "LazyCollection[Mobject]":
        return LazyCollection()

    #@Lazy.variable(LazyMode.COLLECTION)
    #@staticmethod
    #def _parents_() -> "LazyCollection[Mobject]":
    #    return LazyCollection()

    @Lazy.variable(LazyMode.COLLECTION)
    def _real_descendants_(cls) -> "LazyCollection[Mobject]":
        return LazyCollection()

    #@Lazy.variable(LazyMode.COLLECTION)
    #@staticmethod
    #def _real_ancestors_() -> "LazyCollection[Mobject]":
    #    return LazyCollection()

    def iter_children(self) -> "Generator[Mobject, None, None]":
        yield from self._children_

    def iter_parents(self) -> "Generator[Mobject, None, None]":
        yield from self._parents

    #def iter_ancestors_with_duplicates(self) -> "Generator[Mobject, None, None]":
    #    yield self
    #    for parent_node in self._parents:
    #        yield from parent_node.iter_ancestors_with_duplicates()

    #def iter_descendants_with_duplicates(self) -> "Generator[Mobject, None, None]":
    #    yield self
    #    for child_node in self._children_:
    #        yield from child_node.iter_descendants_with_duplicates()

    def iter_descendants(
        self,
        *,
        broadcast: bool = True
    ) -> "Generator[Mobject, None, None]":
        yield self
        if broadcast:
            yield from self._real_descendants_
        #occurred: set[Mobject] = {self}
        #for node in self.iter_descendants_with_duplicates():
        #    if node in occurred:
        #        continue
        #    yield node
        #    occurred.add(node)

    def iter_ancestors(
        self,
        *,
        broadcast: bool = True
    ) -> "Generator[Mobject, None, None]":
        yield self
        if broadcast:
            yield from self._real_ancestors
        #occurred: set[Mobject] = {self}
        #for node in self.iter_ancestors_with_duplicates():
        #    if node in occurred:
        #        continue
        #    yield node
        #    occurred.add(node)

    def add(
        self,
        *mobjects: "Mobject"
    ):
        #filtered_mobjects = [
        #    mob for mob in mobjects
        #    if mob not in self._children_
        #]
        #for mobject in mobjects:
        #    if self in mobject.iter_descendants():
        #        raise ValueError(f"'{mobject}' has already included '{self}'")
        self._children_.add(*mobjects)
        for ancestor_mobject in self.iter_ancestors():
            #print()
            #print(ancestor_mobject)
            #print(list(ancestor_mobject._real_descendants_))
            #print(mobjects)
            ancestor_mobject._real_descendants_.add(*mobjects)
        for mobject in mobjects:
            mobject._parents.append(self)
            for descendant_mobject in self.iter_descendants():
                descendant_mobject._real_ancestors.append(self)
        return self

    def remove(
        self,
        *mobjects: "Mobject"
    ):
        #filtered_mobjects = [
        #    mob for mob in mobjects
        #    if mob in self._children_
        #]
        self._children_.remove(*mobjects)
        for ancestor_mobject in self.iter_ancestors():
            ancestor_mobject._real_descendants_.remove(*mobjects)
        for mobject in mobjects:
            mobject._parents.remove(self)
            for descendant_mobject in self.iter_descendants():
                descendant_mobject._real_ancestors.remove(self)
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

    #def clear_parents(self):
    #    for parent in self.iter_parents():
    #        parent._unbind_child(self)
    #    return self

    #def set_children(self, nodes: "Iterable[Mobject]"):
    #    self.clear()
    #    self.add(*nodes)
    #    return self

    #def copy_standalone(self):
    #    result = self._copy()
    #    result._parents = []
    #    result._children_ = []
    #    for child in self._children_:
    #        child_copy = child.copy_standalone()
    #        result._bind_child(child_copy)
    #    return result

    #def copy(self):
    #    return self._copy()

    def copy(self):
        #real_descendants = list(self._real_descendants_)
        result = self._copy()
        #real_descendants_copy: list[Mobject] = []
        #for descendant in real_descendants:
        #    descendant_copy = descendant._copy()
        #    descendant_copy._parents = 
        #    real_descendants_copy.append(descendant_copy)
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
            descendant_copy._parents = [
                get_matched_descendant_mobject(mobject)
                for mobject in descendant._parents
            ]
            descendant_copy._real_ancestors = [
                get_matched_descendant_mobject(mobject)
                for mobject in descendant._real_ancestors
            ]

        #result = descendants_copy[0]
        for descriptor in self.__class__._LAZY_DESCRIPTORS:
            if not issubclass(descriptor.element_type, Mobject):
                continue
            if isinstance(descriptor, LazyObjectVariableDescriptor):
                mobject = descriptor.__get__(self)
                descriptor.__set__(result, get_matched_descendant_mobject(mobject))
            elif isinstance(descriptor, LazyCollectionVariableDescriptor):
                descriptor.__set__(result, LazyCollection(*(
                    get_matched_descendant_mobject(mobject)
                    for mobject in descriptor.__get__(self)
                )))

        #for mobject in self._parents:
        #    mobject.add(result)
        #for mobject in self._real_ancestors:
        #    mobject._real_descendants_.add(result)
        #for descendant, descendant_copy in zip(descendants, descendants_copy, strict=True):
        #    descendant_copy._children_ = LazyCollection(*(
        #        descendants_copy[descendants.index(descendant_child)]
        #        for descendant_child in descendant._children_
        #    ))
        #    descendant_copy._parents = [
        #        (
        #            descendants_copy[descendants.index(descendant_parent)]
        #            if descendant_parent in descendants
        #            else descendant_parent
        #        )
        #        for descendant_parent in descendant._parents
        #    ]
        #result = self.copy_standalone()
        #for parent in self._parents:
        #    parent._bind_child(result)
        #result = descendants_copy[0]
        #assert isinstance(result, self.__class__)
        return result

    # matrix & transform

    @Lazy.variable(LazyMode.UNWRAPPED)
    def _model_matrix_(cls) -> Mat4T:
        return np.identity(4)

    @Lazy.variable(LazyMode.UNWRAPPED)
    def _local_sample_points_(cls) -> Vec3sT:
        # Implemented in subclasses
        return np.zeros((0, 3))

    @Lazy.property(LazyMode.OBJECT)
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
    def _has_local_sample_points_(
        cls,
        local_sample_points: Vec3sT
    ) -> bool:
        return bool(len(local_sample_points))

    @Lazy.property(LazyMode.UNWRAPPED)
    def _local_world_bounding_box_(
        cls,
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

    # TODO: Can we lazyfy bounding_box, as long as _children_ is lazified...?
    def get_bounding_box(
        self,
        *,
        broadcast: bool = True
    ) -> BoundingBox3D:
        points_array = np.array(list(it.chain(*(
            (aabb.maximum, aabb.minimum)
            for mobject in self.iter_descendants(broadcast=broadcast)
            if (aabb := mobject._local_world_bounding_box_.value) is not None
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

    @Lazy.variable(LazyMode.UNWRAPPED)
    def _apply_oit_(cls) -> bool:
        return False

    #@lazy_slot
    #@staticmethod
    #def _apply_oit() -> bool:
    #    return False

    #@lazy_slot
    #@staticmethod
    #def _render_samples() -> int:
    #    return 0

    @Lazy.variable(LazyMode.COLLECTION)
    def _render_passes_(cls) -> LazyCollection[RenderPass]:
        return LazyCollection()

    def _render(
        self,
        scene_config: SceneConfig,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        # Implemented in subclasses
        # This function is not responsible for clearing the `target_framebuffer`.
        # On the other hand, one shall clear the framebuffer before calling this function.
        pass

    #def _render_with_samples(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
    #    samples = self._render_samples
    #    if not samples:
    #        self._render(scene_config, target_framebuffer)
    #        return

    #    with SimpleFramebufferBatch(samples=samples) as msaa_batch:
    #        self._render(scene_config, msaa_batch.framebuffer)
    #        FramebufferBatch.downsample_framebuffer(msaa_batch.framebuffer, target_framebuffer)

    def _render_with_passes(
        self,
        scene_config: SceneConfig,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        render_passes = self._render_passes_
        if not render_passes:
            self._render(scene_config, target_framebuffer)
            return

        with ColorFramebufferBatch() as batch_0, ColorFramebufferBatch() as batch_1:
            batches = (batch_0, batch_1)
            target_id = 0
            self._render(scene_config, batch_0.framebuffer)
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


#class Group(Mobject):
#    def __init__(self, *mobjects: Mobject):
#        super().__init__()
#        self.add(*mobjects)

#    # TODO
#    def _bind_child(self, node, index: int | None = None):
#        assert isinstance(node, Mobject)
#        super()._bind_child(node, index=index)
#        return self
