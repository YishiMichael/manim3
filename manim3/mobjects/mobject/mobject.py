import itertools as it
import weakref
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    TypedDict,
    overload
)

import numpy as np

from ...animatables.models.model import Model
#from ...constants.constants import (
#    ORIGIN,
#    PI
#)
from ...constants.custom_typing import (
    ColorT,
    NP_3f8,
    NP_44f8,
    NP_x3f8,
    NP_xf8
)
from ...lazy.lazy import Lazy
from ...lazy.lazy_descriptor import LazyDescriptor
#from ...lazy.lazy_object import LazyObject
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ...rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ...toplevel.toplevel import Toplevel
from ...utils.space_utils import SpaceUtils
from ..cameras.camera import Camera
from ..graph_mobjects.graphs.graph import Graph
from ..lights.lighting import Lighting
from ..mesh_mobjects.meshes.mesh import Mesh
from ..shape_mobjects.shapes.shape import Shape
#from .mobject_attributes.array_attribute import ArrayAttribute
#from .mobject_attributes.mobject_attribute import (
#    InterpolateHandler,
#    MobjectAttribute
#)
#from .remodel_handlers.remodel_handler import RemodelHandler
#from .remodel_handlers.rotate_remodel_handler import RotateRemodelHandler
#from .remodel_handlers.scale_remodel_handler import ScaleRemodelHandler
#from .remodel_handlers.shift_remodel_handler import ShiftRemodelHandler

if TYPE_CHECKING:
    from typing_extensions import Unpack

    
    #from .abouts.about import About
    #from .aligns.align import Align


class StyleKwargs(TypedDict, total=False):
    # polymorphism variables
    color: ColorT
    opacity: float
    weight: float

    # Mobject
    camera: Camera

    # MeshMobject
    mesh: Mesh
    ambient_strength: float
    specular_strength: float
    shininess: float
    lighting: Lighting

    # ShapeMobject
    shape: Shape

    # GraphMobject
    graph: Graph
    width: float


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class BoundingBox:
    maximum: NP_3f8
    minimum: NP_3f8

    #@property
    #def center(self) -> NP_3f8:
    #    return (self.maximum + self.minimum) / 2.0

    #@property
    #def radii(self) -> NP_3f8:
    #    radii = (self.maximum - self.minimum) / 2.0
    #    # For zero-width dimensions of radii, thicken a little bit to avoid zero division.
    #    radii[np.isclose(radii, 0.0)] = 1e-8
    #    return radii


class Mobject(Model):
    __slots__ = (
        "__weakref__",
        "_parents",
        "_real_ancestors"
    )
    _special_slot_copiers: ClassVar[dict[str, Callable | None]] = {
        "__weakref__": None,
        "_parents": weakref.WeakSet.copy,
        "_real_ancestors": weakref.WeakSet.copy
    }

    _attribute_descriptors: ClassVar[dict[str, LazyDescriptor[MobjectAttribute, MobjectAttribute]]] = {}
    _equivalent_cls_mro_index: ClassVar[int] = 0

    def __init_subclass__(cls):
        super().__init_subclass__()

        cls._attribute_descriptors = {
            name: descriptor
            for name, descriptor in cls._lazy_descriptors.items()
            if not descriptor._is_multiple
            and descriptor._is_variable
            and issubclass(descriptor._element_type, MobjectAttribute)
        }

        base_cls = cls.__base__
        assert issubclass(base_cls, Mobject)
        if cls._attribute_descriptors == base_cls._attribute_descriptors:
            cls._equivalent_cls_mro_index = base_cls._equivalent_cls_mro_index + 1

    def __init__(self) -> None:
        super().__init__()
        # If `_parents` and `_real_ancestors` are implemented with `LazyDynamicContainer` also,
        # loops will pop up in the DAG of the lazy system.
        self._parents: weakref.WeakSet[Mobject] = weakref.WeakSet()
        self._real_ancestors: weakref.WeakSet[Mobject] = weakref.WeakSet()

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
    ) -> "tuple[Mobject, ...]": ...

    def __getitem__(
        self,
        index: int | slice
    ) -> "Mobject | tuple[Mobject, ...]":
        return self._children_.__getitem__(index)

    # family matters
    # These methods implement a DAG (directed acyclic graph).

    @Lazy.variable_collection(freeze=False)
    @staticmethod
    def _children_() -> "tuple[Mobject, ...]":
        return ()

    @Lazy.variable_collection(freeze=False)
    @staticmethod
    def _real_descendants_() -> "tuple[Mobject, ...]":
        return ()

    @classmethod
    def _refresh_families(
        cls,
        *mobjects: "Mobject"
    ) -> None:

        def iter_descendants_by_children(
            mobject: Mobject
        ) -> Iterator[Mobject]:
            yield mobject
            for child in mobject._children_:
                yield from iter_descendants_by_children(child)

        def iter_ancestors_by_parents(
            mobject: Mobject
        ) -> Iterator[Mobject]:
            yield mobject
            for parent in mobject._parents:
                yield from iter_ancestors_by_parents(parent)

        for ancestor in dict.fromkeys(it.chain.from_iterable(
            iter_ancestors_by_parents(mobject)
            for mobject in mobjects
        )):
            ancestor._real_descendants_ = tuple(dict.fromkeys(it.chain.from_iterable(
                iter_descendants_by_children(child)
                for child in ancestor._children_
            )))
        for descendant in dict.fromkeys(it.chain.from_iterable(
            iter_descendants_by_children(mobject)
            for mobject in mobjects
        )):
            descendant._real_ancestors.clear()
            descendant._real_ancestors.update(dict.fromkeys(it.chain.from_iterable(
                iter_ancestors_by_parents(parent)
                for parent in descendant._parents
            )))

    def iter_children(self) -> "Iterator[Mobject]":
        yield from self._children_

    def iter_parents(self) -> "Iterator[Mobject]":
        yield from self._parents

    def iter_descendants(
        self,
        *,
        broadcast: bool = True
    ) -> "Iterator[Mobject]":
        yield self
        if broadcast:
            yield from self._real_descendants_

    def iter_ancestors(
        self,
        *,
        broadcast: bool = True
    ) -> "Iterator[Mobject]":
        yield self
        if broadcast:
            yield from self._real_ancestors

    def add(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject not in self._children_
        ]
        if (invalid_mobjects := [
            mobject for mobject in filtered_mobjects
            if mobject in self.iter_ancestors()
        ]):
            raise ValueError(f"Circular relationship occurred when adding {invalid_mobjects} to {self}")
        children = list(self._children_)
        for mobject in filtered_mobjects:
            children.append(mobject)
            mobject._parents.add(self)
        self._children_ = tuple(children)
        self._refresh_families(self)
        return self

    def discard(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject in self._children_
        ]
        children = list(self._children_)
        for mobject in filtered_mobjects:
            children.remove(mobject)
            mobject._parents.remove(self)
        self._children_ = tuple(children)
        self._refresh_families(self, *filtered_mobjects)
        return self

    def clear(self):
        self.discard(*self.iter_children())
        return self

    def copy(self):
        # Copy all descendants. The result is not bound to any mobject.
        result = self._copy()
        real_descendants = list(self._real_descendants_)
        real_descendants_copy = [
            descendant._copy()
            for descendant in real_descendants
        ]
        descendants = [self, *real_descendants]
        descendants_copy = [result, *real_descendants_copy]

        def match_copies(
            mobjects: Iterable[Mobject]
        ) -> Iterator[Mobject]:
            return (
                descendants_copy[descendants.index(mobject)] if mobject in descendants else mobject
                for mobject in mobjects
            )

        result._children_ = tuple(match_copies(self._children_))
        result._parents.clear()
        for real_descendant, real_descendant_copy in zip(real_descendants, real_descendants_copy, strict=True):
            real_descendant_copy._children_ = tuple(match_copies(real_descendant._children_))
            real_descendant_copy._parents.clear()
            real_descendant_copy._parents.update(match_copies(real_descendant._parents))

        self._refresh_families(*descendants_copy)
        return result

    # model matrix

    #@Lazy.variable(hasher=Lazy.branch_hasher)
    #@staticmethod
    #def _model_matrix_() -> ArrayAttribute[NP_44f8]:
    #    return ArrayAttribute(np.identity(4))

    @Lazy.property()
    @staticmethod
    def _model_uniform_block_buffer_(
        model_matrix: NP_44f8
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

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _local_sample_positions_() -> NP_x3f8:
        # Implemented in subclasses.
        return np.zeros((0, 3))

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _world_sample_positions_(
        model_matrix: NP_44f8,
        local_sample_positions: NP_x3f8,
    ) -> NP_x3f8:
        return SpaceUtils.apply_affine(model_matrix, local_sample_positions)

    #@Lazy.property()
    #@staticmethod
    #def _bounding_box_without_descendants_(
    #    model_matrix__array: NP_44f8,
    #    local_sample_positions: NP_x3f8
    #) -> BoundingBox | None:
    #    if not len(local_sample_positions):
    #        return None
    #    world_sample_positions = SpaceUtils.apply_affine(model_matrix__array, local_sample_positions)
    #    return BoundingBox(
    #        maximum=world_sample_positions.max(axis=0),
    #        minimum=world_sample_positions.min(axis=0)
    #    )

    #@Lazy.property()
    #@staticmethod
    #def _bounding_box_with_descendants_(
    #    bounding_box_without_descendants: BoundingBox | None,
    #    real_descendants__bounding_box_without_descendants: tuple[BoundingBox | None, ...]
    #) -> BoundingBox | None:
    #    positions_array = np.fromiter((it.chain.from_iterable(
    #        (bounding_box.maximum, bounding_box.minimum)
    #        for bounding_box in (
    #            bounding_box_without_descendants,
    #            *real_descendants__bounding_box_without_descendants
    #        )
    #        if bounding_box is not None
    #    )), dtype=np.dtype((np.float64, (3,))))
    #    if not len(positions_array):
    #        return None
    #    return BoundingBox(
    #        maximum=positions_array.max(axis=0),
    #        minimum=positions_array.min(axis=0)
    #    )

    @Lazy.property()
    @staticmethod
    def _bounding_box_(
        world_sample_positions: NP_x3f8,
        real_descendants__world_sample_positions: tuple[NP_x3f8, ...],
    ) -> BoundingBox:
        positions_array = np.concatenate((
            world_sample_positions,
            *real_descendants__world_sample_positions
        ))
        if not len(positions_array):
            return BoundingBox(
                maximum=np.zeros((1, 3)),
                minimum=np.zeros((1, 3))
            )
        return BoundingBox(
            maximum=positions_array.max(axis=0),
            minimum=positions_array.min(axis=0)
        )

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _centroid_(
        bounding_box: BoundingBox
    ) -> NP_3f8:
        return (bounding_box.maximum + bounding_box.minimum) / 2.0

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _radii_(
        bounding_box: BoundingBox
    ) -> NP_3f8:
        return (bounding_box.maximum - bounding_box.minimum) / 2.0
        # For zero-width dimensions of radii, thicken a little bit to avoid zero division.
        #radii[np.isclose(radii, 0.0)] = 1e-8
        #return radii

    #def get_bounding_box(self) -> BoundingBox:
    #    if (result := self._bounding_box_) is None:
    #        raise ValueError("Trying to calculate the bounding box of an empty mobject")
    #    return result

    #def get_bounding_box_size(self) -> NP_3f8:
    #    bounding_box = self.get_bounding_box()
    #    return bounding_box.radii * 2.0

    #def get_bounding_box_position(
    #    self,
    #    direction: NP_3f8
    #) -> NP_3f8:
    #    bounding_box = self._bounding_box_
    #    return bounding_box.center + direction * bounding_box.radii

    #def get_center(self) -> NP_3f8:
    #    return self.get_bounding_box_position(ORIGIN)

    # remodel

    #def _get_remodel_bound_handlers(
    #    self,
    #    remodel_handler: RemodelHandler,
    #    about: "About | None" = None
    #) -> "list[RemodelBoundHandler]":
    #    if about is not None:
    #        about_position = about._get_about_position(mobject=self)
    #        pre_remodel = ShiftRemodelHandler(-about_position)._remodel()
    #        post_remodel = ShiftRemodelHandler(about_position)._remodel()
    #    else:
    #        pre_remodel = np.identity(4)
    #        post_remodel = np.identity(4)
    #    return [
    #        RemodelBoundHandler(
    #            mobject=mobject,
    #            remodel_handler=remodel_handler,
    #            pre_remodel=pre_remodel,
    #            post_remodel=post_remodel
    #        )
    #        for mobject in self.iter_descendants()
    #    ]

    #def shift(
    #    self,
    #    vector: NP_3f8,
    #    # `about` is meaningless for shifting.
    #    *,
    #    alpha: float | NP_3f8 = 1.0
    #):
    #    for remodel_bound_handler in self._get_remodel_bound_handlers(
    #        remodel_handler=ShiftRemodelHandler(vector)
    #    ):
    #        remodel_bound_handler._remodel(alpha)
    #    return self

    #def scale(
    #    self,
    #    factor: float | NP_3f8,
    #    about: "About | None" = None,
    #    *,
    #    alpha: float | NP_3f8 = 1.0
    #):
    #    for remodel_bound_handler in self._get_remodel_bound_handlers(
    #        remodel_handler=ScaleRemodelHandler(factor),
    #        about=about
    #    ):
    #        remodel_bound_handler._remodel(alpha)
    #    return self

    #def rotate(
    #    self,
    #    rotvec: NP_3f8,
    #    about: "About | None" = None,
    #    *,
    #    alpha: float | NP_3f8 = 1.0
    #):
    #    for remodel_bound_handler in self._get_remodel_bound_handlers(
    #        remodel_handler=RotateRemodelHandler(rotvec),
    #        about=about
    #    ):
    #        remodel_bound_handler._remodel(alpha)
    #    return self

    #def move_to(
    #    self,
    #    align: "Align",
    #    *,
    #    alpha: float | NP_3f8 = 1.0
    #):
    #    self.shift(
    #        vector=align._get_shift_vector(mobject=self, direction_sign=1.0),
    #        alpha=alpha
    #    )
    #    return self

    #def next_to(
    #    self,
    #    align: "Align",
    #    *,
    #    alpha: float | NP_3f8 = 1.0
    #):
    #    self.shift(
    #        vector=align._get_shift_vector(mobject=self, direction_sign=-1.0),
    #        alpha=alpha
    #    )
    #    return self

    #def scale_to(
    #    self,
    #    target: float | NP_3f8,
    #    about: "About | None" = None,
    #    *,
    #    alpha: float | NP_3f8 = 1.0
    #):
    #    factor = target / self.get_bounding_box_size()
    #    self.scale(
    #        factor=factor,
    #        about=about,
    #        alpha=alpha
    #    )
    #    return self

    #def match_bounding_box(
    #    self,
    #    mobject: "Mobject"
    #):
    #    self.shift(-self.get_center()).scale_to(mobject.get_bounding_box_size()).shift(mobject.get_center())
    #    return self

    #def flip(
    #    self,
    #    axis: NP_3f8,
    #    about: "About | None" = None
    #):
    #    self.rotate(
    #        rotvec=SpaceUtils.normalize(axis) * PI,
    #        about=about
    #    )
    #    return self

    # attributes

    @classmethod
    @property
    def _equivalent_cls(cls) -> "type[Mobject]":
        return cls.__mro__[cls._equivalent_cls_mro_index]

    @classmethod
    def _get_interpolate_bound_handler(
        cls,
        dst_mobject: "Mobject",
        src_mobject_0: "Mobject",
        src_mobject_1: "Mobject"
    ) -> "InterpolateBoundHandler":
        assert all(cls is type(mobject)._equivalent_cls for mobject in (dst_mobject, src_mobject_0, src_mobject_1))
        interpolate_handler_dict: dict[LazyDescriptor, InterpolateHandler] = {}
        for descriptor in cls._attribute_descriptors.values():
            data_0 = descriptor.__get__(src_mobject_0)
            data_1 = descriptor.__get__(src_mobject_1)
            if data_0 is data_1:
                continue
            if not descriptor._element_type._interpolate_implemented:
                raise ValueError(f"Uninterpolable variables of `{descriptor._name}` don't match")
            interpolate_handler = descriptor._element_type._interpolate(data_0, data_1)
            interpolate_handler_dict[descriptor] = interpolate_handler
        return InterpolateBoundHandler(dst_mobject, interpolate_handler_dict)

    #@classmethod
    #def _split_into(
    #    cls,
    #    dst_mobject_list: "list[Mobject]",
    #    src_mobject: "Mobject",
    #    alphas: NP_xf8
    #) -> None:
    #    assert all(cls is type(mobject)._equivalent_cls for mobject in (*dst_mobject_list, src_mobject))
    #    for descriptor in cls._attribute_descriptors.values():
    #        src_data = descriptor.__get__(src_mobject)
    #        if not descriptor._element_type._split_implemented:
    #            for dst_mobject in dst_mobject_list:
    #                descriptor.__set__(dst_mobject, src_data)
    #            continue
    #        dst_data_list = descriptor._element_type._split(src_data, alphas)
    #        for dst_mobject, dst_data in zip(dst_mobject_list, dst_data_list, strict=True):
    #            descriptor.__set__(dst_mobject, dst_data)

    #@classmethod
    #def _concatenate_into(
    #    cls,
    #    dst_mobject: "Mobject",
    #    src_mobject_list: "list[Mobject]"
    #) -> None:
    #    assert all(cls is type(mobject)._equivalent_cls for mobject in (dst_mobject, *src_mobject_list))
    #    for descriptor in cls._attribute_descriptors.values():
    #        src_data_list = [
    #            descriptor.__get__(src_mobject)
    #            for src_mobject in src_mobject_list
    #        ]
    #        if not descriptor._element_type._concatenate_implemented:
    #            unique_src_data_list = list(dict.fromkeys(src_data_list))
    #            if not unique_src_data_list:
    #                continue
    #            src_data = unique_src_data_list.pop()
    #            descriptor.__set__(dst_mobject, src_data)
    #            assert not unique_src_data_list, f"Uncatenatable variables of `{descriptor._name}` don't match"
    #            continue
    #        dst_data = descriptor._element_type._concatenate(src_data_list)
    #        descriptor.__set__(dst_mobject, dst_data)

    #def split(
    #    self,
    #    alphas: NP_xf8
    #):
    #    equivalent_cls = type(self)._equivalent_cls
    #    dst_mobject_list = [equivalent_cls() for _ in range(len(alphas) + 1)]
    #    equivalent_cls._split_into(
    #        dst_mobject_list=dst_mobject_list,
    #        src_mobject=self,
    #        alphas=alphas
    #    )
    #    for descriptor in equivalent_cls._attribute_descriptors.values():
    #        descriptor._init(self)
    #    self.add(*dst_mobject_list)
    #    return self

    #def concatenate(self):
    #    self._equivalent_cls._concatenate_into(
    #        dst_mobject=self,
    #        src_mobject_list=list(self.iter_children())
    #    )
    #    self.clear()
    #    return self

    def set(
        self,
        *,
        broadcast: bool = True,
        type_filter: "type[Mobject] | None" = None,
        **kwargs: "Unpack[StyleKwargs]"
    ):
        for mobject in self.iter_descendants(broadcast=broadcast):
            if type_filter is not None and not isinstance(mobject, type_filter):
                continue
            for key, value in kwargs.items():
                if (descriptor := type(mobject)._attribute_descriptors.get(f"_{key}_")) is None:
                    continue
                descriptor.__set__(mobject, descriptor._element_type._convert_input(value))
        return self

    # render

    @Lazy.variable(freeze=False)
    @staticmethod
    def _camera_() -> Camera:
        return Toplevel.scene._camera

    @abstractmethod
    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        pass


#class RemodelBoundHandler:
#    __slots__ = (
#        "_mobject",
#        "_remodel_handler",
#        "_pre_remodel",
#        "_post_remodel",
#        "_original_model"
#    )

#    def __init__(
#        self,
#        mobject: Mobject,
#        remodel_handler: RemodelHandler,
#        pre_remodel: NP_44f8,
#        post_remodel: NP_44f8
#    ) -> None:
#        super().__init__()
#        self._mobject: Mobject = mobject
#        self._remodel_handler: RemodelHandler = remodel_handler
#        self._pre_remodel: NP_44f8 = pre_remodel
#        self._post_remodel: NP_44f8 = post_remodel
#        self._original_model: NP_44f8 = mobject._model_matrix_._array_

#    def _remodel(
#        self,
#        alpha: float | NP_3f8 = 1.0
#    ) -> None:
#        self._mobject._model_matrix_ = ArrayAttribute(
#            self._post_remodel @ self._remodel_handler._remodel(alpha) @ self._pre_remodel @ self._original_model
#        )


#class InterpolateBoundHandler:
#    __slots__ = (
#        "_dst_mobject",
#        "_interpolate_handler_dict"
#    )

#    def __init__(
#        self,
#        dst_mobject: Mobject,
#        interpolate_handler_dict: dict[LazyDescriptor, InterpolateHandler]
#    ) -> None:
#        super().__init__()
#        self._dst_mobject: Mobject = dst_mobject
#        self._interpolate_handler_dict: dict[LazyDescriptor, InterpolateHandler] = interpolate_handler_dict

#    def _interpolate(
#        self,
#        alpha: float
#    ) -> None:
#        dst_mobject = self._dst_mobject
#        for descriptor, interpolate_handler in self._interpolate_handler_dict.items():
#            descriptor.__set__(dst_mobject, interpolate_handler._interpolate(alpha))
