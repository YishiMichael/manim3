from abc import (
    ABC,
    abstractmethod
)
from dataclasses import dataclass
import itertools as it
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    overload
)
import weakref

import numpy as np

from ..constants import (
    ORIGIN,
    PI
)
from ..custom_typing import (
    ColorT,
    NP_3f8,
    NP_44f8,
    NP_x3f8
)
from ..geometries.geometry import Geometry
from ..lazy.lazy import (
    Lazy,
    LazyObject,
    LazyVariableDescriptor
)
from ..rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ..shape.multi_line_string import MultiLineString
from ..shape.shape import Shape
from ..utils.color import ColorUtils
from ..utils.model_interpolant import ModelInterpolant
from ..utils.space import SpaceUtils
from .mobject_style_meta import MobjectStyleMeta

if TYPE_CHECKING:
    from .cameras.camera import Camera
    from .lights.lighting import Lighting


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class BoundingBox:
    maximum: NP_3f8
    minimum: NP_3f8

    @property
    def center(self) -> NP_3f8:
        return (self.maximum + self.minimum) / 2.0

    @property
    def radii(self) -> NP_3f8:
        radii = (self.maximum - self.minimum) / 2.0
        # For zero-width dimensions of radii, thicken a little bit to avoid zero division.
        radii[np.isclose(radii, 0.0)] = 1e-8
        return radii


class AboutABC(ABC):
    __slots__ = ()

    @abstractmethod
    def _get_about_point(
        self,
        mobject: "Mobject"
    ) -> NP_3f8:
        pass


class AlignABC(ABC):
    __slots__ = (
        "_about",
        "_direction",
        "_buff"
    )

    def __init__(
        self,
        about: AboutABC,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__()
        self._about: AboutABC = about
        self._direction: NP_3f8 = direction
        self._buff: float | NP_3f8 = buff

    def _get_shift_vector(
        self,
        mobject: "Mobject",
        direction_sign: float
    ) -> NP_3f8:
        target_point = self._about._get_about_point(mobject)
        direction = direction_sign * self._direction
        point_to_align = mobject.get_bounding_box_point(direction) + self._buff * direction
        return target_point - point_to_align


class Mobject(LazyObject):
    __slots__ = (
        "_parents",
        "_real_ancestors"
    )

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
    ) -> "list[Mobject]": ...

    def __getitem__(
        self,
        index: int | slice
    ) -> "Mobject | list[Mobject]":
        return self._children_.__getitem__(index)

    # family matters
    # These methods implement a DAG (directed acyclic graph).

    @Lazy.variable_collection
    @classmethod
    def _children_(cls) -> "list[Mobject]":
        return []

    @Lazy.variable_collection
    @classmethod
    def _real_descendants_(cls) -> "list[Mobject]":
        return []

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
            ancestor._real_descendants_.reset(dict.fromkeys(it.chain.from_iterable(
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
        self._children_.extend(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._parents.add(self)
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
        self._children_.eliminate(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._parents.remove(self)
        self._refresh_families(self, *filtered_mobjects)
        return self

    def added_by(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject not in self._parents
        ]
        if (invalid_mobjects := [
            mobject for mobject in filtered_mobjects
            if mobject in self.iter_descendants()
        ]):
            raise ValueError(f"Circular relationship occurred when adding {self} to {invalid_mobjects}")
        self._parents.update(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._children_.append(self)
        self._refresh_families(self)
        return self

    def discarded_by(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject in self._parents
        ]
        self._parents.difference_update(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._children_.remove(self)
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

        result._children_.reset(match_copies(self._children_))
        result._parents.clear()
        for real_descendant, real_descendant_copy in zip(real_descendants, real_descendants_copy, strict=True):
            real_descendant_copy._children_.reset(match_copies(real_descendant._children_))
            real_descendant_copy._parents.clear()
            real_descendant_copy._parents.update(match_copies(real_descendant._parents))

        self._refresh_families(*descendants_copy)
        return result

    # model matrix

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_44f8
    )
    @Lazy.variable_array
    @classmethod
    def _model_matrix_(cls) -> NP_44f8:
        return np.identity(4)

    @Lazy.property
    @classmethod
    def _model_uniform_block_buffer_(
        cls,
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

    @Lazy.property_array
    @classmethod
    def _local_sample_points_(cls) -> NP_x3f8:
        # Implemented in subclasses.
        return np.zeros((0, 3))

    @Lazy.property_external
    @classmethod
    def _bounding_box_without_descendants_(
        cls,
        model_matrix: NP_44f8,
        local_sample_points: NP_x3f8
    ) -> BoundingBox | None:
        if not len(local_sample_points):
            return None
        world_sample_points = SpaceUtils.apply_affine(model_matrix, local_sample_points)
        return BoundingBox(
            maximum=world_sample_points.max(axis=0),
            minimum=world_sample_points.min(axis=0)
        )

    @Lazy.property_external
    @classmethod
    def _bounding_box_with_descendants_(
        cls,
        bounding_box_without_descendants: BoundingBox | None,
        real_descendants__bounding_box_without_descendants: list[BoundingBox | None]
    ) -> BoundingBox | None:
        points_array = np.array(list(it.chain.from_iterable(
            (bounding_box.maximum, bounding_box.minimum)
            for bounding_box in (
                bounding_box_without_descendants,
                *real_descendants__bounding_box_without_descendants
            )
            if bounding_box is not None
        )))
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
            result = self._bounding_box_with_descendants_
        else:
            result = self._bounding_box_without_descendants_
        assert result is not None, "Trying to calculate the bounding box of some mobject with no points"
        return result

    def get_bounding_box_size(
        self,
        *,
        broadcast: bool = True
    ) -> NP_3f8:
        bounding_box = self.get_bounding_box(broadcast=broadcast)
        return bounding_box.radii * 2.0

    def get_bounding_box_point(
        self,
        direction: NP_3f8,
        *,
        broadcast: bool = True
    ) -> NP_3f8:
        bounding_box = self.get_bounding_box(broadcast=broadcast)
        return bounding_box.center + direction * bounding_box.radii

    def get_center(
        self,
        *,
        broadcast: bool = True
    ) -> NP_3f8:
        return self.get_bounding_box_point(ORIGIN, broadcast=broadcast)

    # transform

    def _apply_transform_callback(
        self,
        model_interpolant: ModelInterpolant,
        about: AboutABC | None = None
    ) -> Callable[[float | NP_3f8], None]:
        if about is None:
            pre_transform = np.identity(4)
            post_transform = np.identity(4)
        else:
            about_point = about._get_about_point(mobject=self)
            pre_transform = ModelInterpolant.from_shift(-about_point)()
            post_transform = ModelInterpolant.from_shift(about_point)()

        mobject_to_model_matrix = {
            mobject: mobject._model_matrix_
            for mobject in self.iter_descendants()
        }

        def callback(
            alpha: float | NP_3f8
        ) -> None:
            matrix = post_transform @ model_interpolant(alpha) @ pre_transform
            for mobject, model_matrix in mobject_to_model_matrix.items():
                mobject._model_matrix_ = matrix @ model_matrix

        return callback

    def apply_transform(
        self,
        matrix: NP_44f8,
        about: AboutABC | None = None
    ):
        self._apply_transform_callback(ModelInterpolant(lambda alpha: matrix), about=about)(1.0)
        return self

    def shift(
        self,
        vector: NP_3f8,
        # `about` is meaningless for shifting.
        *,
        alpha: float | NP_3f8 = 1.0
    ):
        self._apply_transform_callback(ModelInterpolant.from_shift(vector))(alpha)
        return self

    def move_to(
        self,
        align: AlignABC,
        *,
        alpha: float | NP_3f8 = 1.0
    ):
        self.shift(
            vector=align._get_shift_vector(mobject=self, direction_sign=1.0),
            alpha=alpha
        )
        return self

    def next_to(
        self,
        align: AlignABC,
        *,
        alpha: float | NP_3f8 = 1.0
    ):
        self.shift(
            vector=align._get_shift_vector(mobject=self, direction_sign=-1.0),
            alpha=alpha
        )
        return self

    def scale(
        self,
        factor: float | NP_3f8,
        about: AboutABC | None = None,
        *,
        alpha: float | NP_3f8 = 1.0
    ):
        self._apply_transform_callback(ModelInterpolant.from_scale(factor), about=about)(alpha)
        return self

    def scale_to(
        self,
        target: float | NP_3f8,
        about: AboutABC | None = None,
        alpha: float | NP_3f8 = 1.0
    ):
        factor = target / self.get_bounding_box_size()
        self.scale(
            factor=factor,
            about=about,
            alpha=alpha
        )
        return self

    def match_bounding_box(
        self,
        mobject: "Mobject"
    ):
        self.shift(-self.get_center()).scale_to(mobject.get_bounding_box_size()).shift(mobject.get_center())
        return self

    def rotate(
        self,
        rotvec: NP_3f8,
        about: AboutABC | None = None,
        *,
        alpha: float | NP_3f8 = 1.0
    ):
        self._apply_transform_callback(ModelInterpolant.from_rotate(rotvec), about=about)(alpha)
        return self

    def flip(
        self,
        axis: NP_3f8,
        about: AboutABC | None = None
    ):
        self.rotate(
            rotvec=SpaceUtils.normalize(axis) * PI,
            about=about
        )
        return self

    # meta methods

    def concatenate(self):
        MobjectStyleMeta._concatenate(*self.iter_children())(self)()
        self.clear()
        return self

    def set_style(
        self,
        *,
        # polymorphism variables
        color: ColorT | None = None,
        opacity: float | None = None,
        weight: float | None = None,

        # Mobject
        model_matrix: NP_44f8 | None = None,

        # RenderableMobject
        camera: "Camera | None" = None,

        # MeshMobject
        geometry: Geometry | None = None,
        lighting: "Lighting | None" = None,
        ambient_strength: float | None = None,
        specular_strength: float | None = None,
        shininess: float | None = None,

        # ShapeMobject
        shape: Shape | None = None,

        # StrokeMobject
        multi_line_string: MultiLineString | None = None,
        width: float | None = None,

        # setting configs
        broadcast: bool = True,
        type_filter: "type[Mobject] | None" = None
    ):

        def standardize_input(
            value: Any
        ) -> np.ndarray:
            if not isinstance(value, float | int | np.ndarray):
                return value
            return value * np.ones((), dtype=np.float64)

        if color is not None:
            color = ColorUtils.standardize_color(color)
        style = {
            f"_{key}_": standardize_input(value)
            for key, value in {
                "color": color,
                "opacity": opacity,
                "weight": weight,
                "model_matrix": model_matrix,
                "camera": camera,
                "geometry": geometry,
                "lighting": lighting,
                "ambient_strength": ambient_strength,
                "specular_strength": specular_strength,
                "shininess": shininess,
                "shape": shape,
                "multi_line_string": multi_line_string,
                "width": width
            }.items() if value is not None
        }

        for mobject in self.iter_descendants(broadcast=broadcast):
            if type_filter is not None and not isinstance(mobject, type_filter):
                continue
            for key, value in style.items():
                if (descriptor := type(mobject)._lazy_descriptor_dict.get(key)) is None:
                    continue
                if not isinstance(descriptor, LazyVariableDescriptor):
                    continue
                descriptor.__set__(mobject, value)
        return self
