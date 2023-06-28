import itertools as it
from typing import (
    Iterable,
    Iterator,
    overload
)
import weakref

from ..lazy.lazy import Lazy
from ..models.model import (
    Model,
    StyleMeta
)
from ..models.cameras.camera import Camera
from ..models.cameras.perspective_camera import PerspectiveCamera
from ..rendering.framebuffer import OITFramebuffer


class Mobject(Model):
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

    #def iter_children_by_type(
    #    self,
    #    mobject_type: type[_MobjectT]
    #) -> Iterator[_MobjectT]:
    #    for mobject in self.iter_children():
    #        if isinstance(mobject, mobject_type):
    #            yield mobject

    #def iter_descendants_by_type(
    #    self,
    #    mobject_type: type[_MobjectT],
    #    broadcast: bool = True
    #) -> Iterator[_MobjectT]:
    #    for mobject in self.iter_descendants(broadcast=broadcast):
    #        if isinstance(mobject, mobject_type):
    #            yield mobject

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

    # bounding box

    #@StyleMeta.register(
    #    interpolate_method=SpaceUtils.lerp_44f8
    #)
    #@Lazy.variable_array
    #@classmethod
    #def _model_matrix_(cls) -> NP_44f8:
    #    return np.identity(4)

    #@Lazy.property
    #@classmethod
    #def _model_uniform_block_buffer_(
    #    cls,
    #    model_matrix: NP_44f8
    #) -> UniformBlockBuffer:
    #    return UniformBlockBuffer(
    #        name="ub_model",
    #        fields=[
    #            "mat4 u_model_matrix"
    #        ],
    #        data={
    #            "u_model_matrix": model_matrix.T
    #        }
    #    )

    #@Lazy.property_array
    #@classmethod
    #def _local_sample_points_(cls) -> NP_x3f8:
    #    # Implemented in subclasses.
    #    return np.zeros((0, 3))

    ##@Lazy.property_hashable
    ##@classmethod
    ##def _has_local_sample_points_(
    ##    cls,
    ##    local_sample_points: NP_x3f8
    ##) -> bool:
    ##    return bool(len(local_sample_points))

    #@Lazy.property_external
    #@classmethod
    #def _bounding_box_without_descendants_(
    #    cls,
    #    model_matrix: NP_44f8,
    #    local_sample_points: NP_x3f8
    #) -> BoundingBox | None:
    #    if not len(local_sample_points):
    #        return None
    #    world_sample_points = SpaceUtils.apply_affine(model_matrix, local_sample_points)
    #    return BoundingBox(
    #        maximum=world_sample_points.max(axis=0),
    #        minimum=world_sample_points.min(axis=0)
    #    )

    #@Lazy.property_external
    #@classmethod
    #def _bounding_box_with_descendants_(
    #    cls,
    #    bounding_box_without_descendants: BoundingBox | None,
    #    real_descendants__bounding_box_without_descendants: list[BoundingBox | None]
    #) -> BoundingBox | None:
    #    points_array = np.array(list(it.chain.from_iterable(
    #        (bounding_box.maximum, bounding_box.minimum)
    #        for bounding_box in (
    #            bounding_box_without_descendants,
    #            *real_descendants__bounding_box_without_descendants
    #        )
    #        if bounding_box is not None
    #    )))
    #    if not len(points_array):
    #        return None
    #    return BoundingBox(
    #        maximum=points_array.max(axis=0),
    #        minimum=points_array.min(axis=0)
    #    )

    #def get_bounding_box(
    #    self,
    #    *,
    #    broadcast: bool = True
    #) -> BoundingBox:
    #    if broadcast:
    #        result = self._bounding_box_with_descendants_
    #    else:
    #        result = self._bounding_box_without_descendants_
    #    assert result is not None, "Trying to calculate the bounding box of some mobject with no points"
    #    return result

    #def get_bounding_box_size(
    #    self,
    #    *,
    #    broadcast: bool = True
    #) -> NP_3f8:
    #    bounding_box = self.get_bounding_box(broadcast=broadcast)
    #    return bounding_box.radii * 2.0

    #def get_bounding_box_point(
    #    self,
    #    direction: NP_3f8,
    #    *,
    #    broadcast: bool = True
    #) -> NP_3f8:
    #    bounding_box = self.get_bounding_box(broadcast=broadcast)
    #    return bounding_box.center + direction * bounding_box.radii

    #def get_center(
    #    self,
    #    *,
    #    broadcast: bool = True
    #) -> NP_3f8:
    #    return self.get_bounding_box_point(ORIGIN, broadcast=broadcast)

    ## transform

    #def _make_callback_relative(
    #    self,
    #    matrix_callback: Callable[[float | NP_3f8], NP_44f8],
    #    about: AboutABC | None
    #) -> Callable[[float | NP_3f8], NP_44f8]:
    #    if about is None:
    #        return matrix_callback
    #    about_point = about._get_about_point(mobject=self)
    #    pre_transform = SpaceUtils.matrix_from_translation(-about_point)
    #    post_transform = SpaceUtils.matrix_from_translation(about_point)

    #    def callback(
    #        alpha: float | NP_3f8
    #    ) -> NP_44f8:
    #        return post_transform @ matrix_callback(alpha) @ pre_transform

    #    return callback

    #def _apply_transform_callback(
    #    self,
    #    matrix_callback: Callable[[float], NP_44f8]
    #) -> Callable[[float], None]:
    #    mobject_to_model_matrix = {
    #        mobject: mobject._model_matrix_
    #        for mobject in self.iter_descendants()
    #    }

    #    def callback(
    #        alpha: float
    #    ) -> None:
    #        for mobject, model_matrix in mobject_to_model_matrix.items():
    #            mobject._model_matrix_ = matrix_callback(alpha) @ model_matrix

    #    return callback

    #def apply_transform(
    #    self,
    #    matrix: NP_44f8,
    #):
    #    if np.isclose(np.linalg.det(matrix), 0.0):
    #        warnings.warn("Applying a singular matrix transform")
    #    self._apply_transform_callback(lambda alpha: matrix)(1.0)
    #    return self

    ## shift relatives

    #def _shift_callback(
    #    self,
    #    vector: NP_3f8
    #    # `about` is meaningless for shifting.
    #) -> Callable[[float | NP_3f8], NP_44f8]:
    #    return SpaceUtils.matrix_callback_from_translation(vector)

    #def shift(
    #    self,
    #    vector: NP_3f8,
    #    *,
    #    alpha: float | NP_3f8 = 1.0
    #):
    #    matrix = self._shift_callback(vector)(alpha)
    #    self.apply_transform(matrix)
    #    return self

    #def move_to(
    #    self,
    #    align: AlignABC,
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
    #    align: AlignABC,
    #    *,
    #    alpha: float | NP_3f8 = 1.0
    #):
    #    self.shift(
    #        vector=align._get_shift_vector(mobject=self, direction_sign=-1.0),
    #        alpha=alpha
    #    )
    #    return self

    ## scale relatives

    #def _scale_callback(
    #    self,
    #    factor: float | NP_3f8,
    #    about: AboutABC | None = None
    #) -> Callable[[float | NP_3f8], NP_44f8]:
    #    return self._make_callback_relative(
    #        matrix_callback=SpaceUtils.matrix_callback_from_scale(factor),
    #        about=about
    #    )

    #def scale(
    #    self,
    #    factor: float | NP_3f8,
    #    about: AboutABC | None = None,
    #    *,
    #    alpha: float | NP_3f8 = 1.0
    #):
    #    matrix = self._scale_callback(factor, about)(alpha)
    #    self.apply_transform(matrix)
    #    return self

    #def scale_to(
    #    self,
    #    target: float | NP_3f8,
    #    about: AboutABC | None = None,
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
    #    self.move_to(AlignMobject(mobject)).scale_to(mobject.get_bounding_box_size())
    #    return self

    ## rotate relatives

    #def _rotate_callback(
    #    self,
    #    rotvec: NP_3f8,
    #    about: AboutABC | None = None
    #) -> Callable[[float | NP_3f8], NP_44f8]:
    #    return self._make_callback_relative(
    #        matrix_callback=SpaceUtils.matrix_callback_from_rotation(rotvec),
    #        about=about
    #    )

    #def rotate(
    #    self,
    #    rotvec: NP_3f8,
    #    about: AboutABC | None = None,
    #    *,
    #    alpha: float | NP_3f8 = 1.0
    #):
    #    matrix = self._rotate_callback(rotvec, about)(alpha)
    #    self.apply_transform(matrix)
    #    return self

    #def flip(
    #    self,
    #    axis: NP_3f8,
    #    about: AboutABC | None = None
    #):
    #    self.rotate(
    #        rotvec=SpaceUtils.normalize(axis) * PI,
    #        about=about
    #    )
    #    return self

    # render

    @StyleMeta.register()
    @Lazy.variable
    @classmethod
    def _camera_(cls) -> Camera:
        return PerspectiveCamera()

    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        pass

    # meta methods

    def concatenate(self):
        StyleMeta._concatenate(*self.iter_children())(self)()
        self.clear()
        return self
