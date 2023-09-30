from abc import abstractmethod
from typing import TypeVar
#from typing import TYPE_CHECKING

import numpy as np

from ...constants.constants import (
    ORIGIN,
    PI
)
from ...constants.custom_typing import (
    BoundaryT,
    NP_3f8,
    NP_44f8,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from ...lazy.lazy_object import LazyObject
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ...utils.space_utils import SpaceUtils
from ..arrays.model_matrix import (
    AffineApplier,
    ModelMatrix
)
from ..animatable.animatable import (
    Animatable,
    Updater
)
from ..animatable.piecewiser import Piecewiser

#if TYPE_CHECKING:
#    from ...models.model.model import model


_ModelT = TypeVar("_ModelT", bound="Model")


class Box(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        maximum: NP_3f8,
        minimum: NP_3f8
    ) -> None:
        super().__init__()
        self._maximum_ = maximum
        self._minimum_ = minimum

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _maximum_() -> NP_3f8:
        return NotImplemented

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _minimum_() -> NP_3f8:
        return NotImplemented

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _centroid_(
        maximum: NP_3f8,
        minimum: NP_3f8
    ) -> NP_3f8:
        return (maximum + minimum) / 2.0

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _radii_(
        maximum: NP_3f8,
        minimum: NP_3f8
    ) -> NP_3f8:
        return (maximum - minimum) / 2.0

    def get_position(
        self,
        direction: NP_3f8,
        buff: NP_3f8 = ORIGIN
    ) -> NP_3f8:
        return self._centroid_ + self._radii_ * direction + buff * direction

    def get_centroid(self) -> NP_3f8:
        return self._centroid_

    #@property
    #def center(self) -> NP_3f8:
    #    return (self.maximum + self.minimum) / 2.0

    #@property
    #def radii(self) -> NP_3f8:
    #    radii = (self.maximum - self.minimum) / 2.0
    #    # For zero-width dimensions of radii, thicken a little bit to avoid zero division.
    #    radii[np.isclose(radii, 0.0)] = 1e-8
    #    return radii


class Model(Animatable):
    __slots__ = ()

    @Lazy.variable(freeze=False)
    @staticmethod
    def _model_matrix_() -> ModelMatrix:
        return ModelMatrix()

    @Lazy.variable_collection()
    @staticmethod
    def _associated_models_() -> "tuple[Model, ...]":
        return ()

    @Lazy.property()
    @staticmethod
    def _model_uniform_block_buffer_(
        model_matrix__array: NP_44f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_model",
            fields=[
                "mat4 u_model_matrix"
            ],
            data={
                "u_model_matrix": model_matrix__array.T
            }
        )

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _local_sample_positions_() -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _world_sample_positions_(
        model_matrix__applier: AffineApplier,
        local_sample_positions: NP_x3f8,
    ) -> NP_x3f8:
        return model_matrix__applier.apply_multiple(local_sample_positions)

    #@Lazy.property(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _box_reference_points_(
    #    world_sample_positions: NP_x3f8,
    #) -> NP_x3f8:
    #    return world_sample_positions

    #@Lazy.variable(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _centroid_() -> NP_3f8:
    #    return np.zeros((3,))

    #@Lazy.variable(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _radii_() -> NP_3f8:
    #    return np.zeros((3,))

    @Lazy.property()
    @staticmethod
    def _box_(
        world_sample_positions: NP_x3f8,
        associated_models__world_sample_positions: tuple[NP_x3f8, ...]
    ) -> Box:
        sample_positions = np.concatenate((
            world_sample_positions,
            *associated_models__world_sample_positions
        ))
        if not len(sample_positions):
            return Box(
                maximum=np.zeros((3,)),
                minimum=np.zeros((3,))
            )
        return Box(
            maximum=sample_positions.max(axis=0),
            minimum=sample_positions.min(axis=0)
        )

    #@Lazy.property(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _centroid_(
    #    box: Box
    #) -> NP_3f8:
    #    return (box.maximum + box.minimum) / 2.0

    #@Lazy.property(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _radii_(
    #    box: Box
    #) -> NP_3f8:
    #    return (box.maximum - box.minimum) / 2.0
    #    # For zero-width dimensions of radii, thicken a little bit to avoid zero division.
    #    #radii[np.isclose(radii, 0.0)] = 1e-8
    #    #return radii

    #def get_box_position(
    #    self,
    #    direction: NP_3f8,
    #    buff: NP_3f8 = ORIGIN
    #) -> NP_3f8:
    #    return self._box_.get_position(direction, buff)

    #def get_centroid(self) -> NP_3f8:
    #    return self._box_._centroid_

    def _get_interpolate_updater(
        self: _ModelT,
        src_0: _ModelT,
        src_1: _ModelT
    ) -> Updater:
        return super()._get_interpolate_updater(src_0, src_1).add(*(
            super(Model, dst_associated)._get_interpolate_updater(src_0_assiciated, src_1_assiciated)
            for dst_associated, src_0_assiciated, src_1_assiciated in zip(
                self._associated_models_, src_0._associated_models_, src_1._associated_models_, strict=True
            )
        ))

    def _get_piecewise_updater(
        self: _ModelT,
        src: _ModelT,
        piecewiser: Piecewiser
    ) -> Updater:
        return super()._get_piecewise_updater(src, piecewiser).add(*(
            super(Model, dst_associated)._get_piecewise_updater(src_assiciated, piecewiser)
            for dst_associated, src_assiciated in zip(
                self._associated_models_, src._associated_models_, strict=True
            )
        ))

    def copy(self):
        result = super().copy()
        associated_models = self._associated_models_
        associated_models_copy = tuple(
            super(Model, associated_model).copy()
            for associated_model in associated_models
        )
        result._associated_models_ = associated_models_copy
        for associated_model_copy in associated_models_copy:
            associated_model_copy._associated_models_ = tuple(
                associated_models_copy[associated_models.index(associated_model)]
                for associated_model in associated_model_copy._associated_models_
            )
        return result

    @property
    def box(self) -> Box:
        return self._box_

    #@classmethod
    #def _split(
    #    cls: type[_ModelT],
    #    dst_tuple: tuple[_ModelT, ...],
    #    src: _ModelT,
    #    alphas: NP_xf8
    #) -> None:
    #    super()._split(dst_tuple, src, alphas)
    #    for dst_tuple_associated, src_assiciated in zip(
    #        tuple(dst._associated_models_ for dst in dst_tuple), src._associated_models_, strict=True
    #    ):
    #        super()._split(dst_tuple_associated, src_assiciated, alphas)

    #@classmethod
    #def _concatenate(
    #    cls: type[_ModelT],
    #    dst: _ModelT,
    #    src_tuple: tuple[_ModelT, ...]
    #) -> None:
    #    super()._concatenate(dst, src_tuple)
    #    for dst_associated, src_tuple_assiciated in zip(
    #        dst._associated_models_, tuple(src._associated_models_ for src in src_tuple), strict=True
    #    ):
    #        super()._concatenate(dst_associated, src_tuple_assiciated)

    #def _apply_directly(
    #    self,
    #    matrix: NP_44f8
    #):
    #    self._model_matrix_._apply(matrix)
    #    for associated_model in self._associated_models_:
    #        associated_model._model_matrix_._apply(matrix)
    #    return self

    # animations

    def shift(
        self,
        vector: NP_3f8,
        mask: float | NP_3f8 = 1.0
    ):
        self._stack_updater(ModelShiftUpdater(
            model=self,
            vector=vector,
            mask=mask * np.ones((3,))
        ))
        return self

    def shift_to(
        self,
        aligned_model: "Model",
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0,
        mask: float | NP_3f8 = 1.0,
        align_direction_sign: float = 0.0
    ):
        #signed_direction = align_direction_sign * direction
        self._stack_updater(ModelShiftToUpdater(
            model=self,
            aligned_model=aligned_model,
            direction=direction,
            buff=buff * np.ones((3,)),
            mask=mask * np.ones((3,)),
            align_direction_sign=align_direction_sign
            #buff_vector=self.get_box_position(signed_direction) + buff * signed_direction,
            #initial_model=self
        ))
        return self

    def move_to(
        self,
        aligned_model: "Model",
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0,
        mask: float | NP_3f8 = 1.0
    ):
        self.shift_to(
            aligned_model=aligned_model,
            direction=direction,
            buff=buff,
            mask=mask,
            align_direction_sign=1.0
        )
        return self

    def next_to(
        self,
        aligned_model: "Model",
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0,
        mask: float | NP_3f8 = 1.0
    ):
        self.shift_to(
            aligned_model=aligned_model,
            direction=direction,
            buff=buff,
            mask=mask,
            align_direction_sign=-1.0
        )
        return self

    def scale(
        self,
        factor: float | NP_3f8,
        about_model: "Model | None" = None,
        about_direction: NP_3f8 = ORIGIN,
        mask: float | NP_3f8 = 1.0
    ):
        if about_model is None:
            about_model = self
        self._stack_updater(ModelScaleUpdater(
            model=self,
            factor=factor * np.ones((3,)),
            about_model=about_model,
            about_direction=about_direction,
            mask=mask * np.ones((3,))
        ))
        return self

    def scale_to(
        self,
        aligned_model: "Model",
        about_model: "Model | None" = None,
        about_direction: NP_3f8 = ORIGIN,
        mask: float | NP_3f8 = 1.0
    ):
        if about_model is None:
            about_model = self
        self._stack_updater(ModelScaleToUpdater(
            model=self,
            aligned_model=aligned_model,
            #radii=self._radii_,
            about_model=about_model,
            about_direction=about_direction,
            mask=mask * np.ones((3,))
            #initial_model=self
        ))
        #factor = target / self.get_box_size()
        #self.scale(
        #    vector=target_size / (2.0 * self_copy._radii_),
        #    about_model=about_model,
        #    about_direction=about_direction,
        #    mask=mask
        #)
        return self

    def match_box(
        self,
        model: "Model",
        mask: float | NP_3f8 = 1.0
    ):
        self.scale_to(model, mask=mask).shift_to(model, mask=mask)
        #self.shift(-self.get_center()).scale_to(model.get_box_size()).shift(model.get_center())
        return self

    def rotate(
        self,
        rotvec: NP_3f8,
        about_model: "Model | None" = None,
        about_direction: NP_3f8 = ORIGIN,
        mask: float | NP_3f8 = 1.0
    ):
        if about_model is None:
            about_model = self
        self._stack_updater(ModelRotateUpdater(
            model=self,
            rotvec=rotvec,
            about_model=about_model,
            about_direction=about_direction,
            mask=mask * np.ones((3,))
        ))
        return self

    def flip(
        self,
        axis: NP_3f8,
        about_model: "Model | None" = None,
        about_direction: NP_3f8 = ORIGIN,
        mask: float | NP_3f8 = 1.0
    ):
        self.rotate(
            rotvec=SpaceUtils.normalize(axis) * PI,
            about_model=about_model,
            about_direction=about_direction,
            mask=mask
        )
        return self

    def apply(
        self,
        matrix: NP_44f8,
        about_model: "Model | None" = None,
        about_direction: NP_3f8 = ORIGIN
    ):
        if about_model is None:
            about_model = self
        self._stack_updater(ModelApplyUpdater(
            model=self,
            matrix=matrix,
            about_model=about_model,
            about_direction=about_direction
        ))
        return self

    def pose(
        self,
        aligned_model: "Model",
        about_model: "Model | None" = None,
        about_direction: NP_3f8 = ORIGIN
    ):
        if about_model is None:
            about_model = self
        self._stack_updater(ModelPoseUpdater(
            model=self,
            aligned_model=aligned_model,
            #matrix=self._matrix_,
            about_model=about_model,
            about_direction=about_direction
            #initial_model=self
        ))
        return self


class ModelUpdater(Updater):
    __slots__ = (
        "_model_matrices",
        "_about_model"
    )

    def __init__(
        self,
        model: Model,
        #factor: float | NP_3f8,
        about_model: Model,
        about_direction: NP_3f8
    ) -> None:
        #if not isinstance(factor, np.ndarray):
        #    factor *= np.ones((3,))
        #assert (factor >= 0.0).all(), "Scale factor must be positive"
        #factor = np.maximum(factor, 1e-8)
        #super().__init__()
        #self._factor: NP_3f8 = factor
        super().__init__()
        self._model_matrices: dict[ModelMatrix, NP_44f8] = {
            model_assiciated._model_matrix_: model_assiciated._model_matrix_._array_
            for model_assiciated in (model, *model._associated_models_)
        }
        #self._model: Model = model
        self._about_model: Model = about_model
        self._about_direction_ = about_direction

    @Lazy.variable()
    @staticmethod
    def _about_box_() -> Box:
        return NotImplemented

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _about_direction_() -> NP_3f8:
        return np.zeros((3,))

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _about_point_(
        about_box: Box,
        about_direction: NP_3f8
    ) -> NP_3f8:
        return about_box.get_position(about_direction)

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _pre_shift_matrix_(
        about_point: NP_3f8
    ) -> NP_44f8:
        return SpaceUtils.matrix_from_shift(-about_point)

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _post_shift_matrix_(
        about_point: NP_3f8
    ) -> NP_44f8:
        return SpaceUtils.matrix_from_shift(about_point)

    @abstractmethod
    def _get_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        pass

    def update(
        self,
        alpha: float
    ) -> None:
        super().update(alpha)
        #model = self._model
        self._about_box_ = self._about_model.box
        matrix = self._post_shift_matrix_ @ self._get_matrix(alpha) @ self._pre_shift_matrix_
        for model_matrix in self._model_matrices:
            model_matrix._array_ = matrix @ model_matrix._array_

    def update_boundary(
        self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self.update(float(boundary))

    def restore(self) -> None:
        for model_matrix, initial_matrix in self._model_matrices.items():
            model_matrix._array_ = initial_matrix
        super().restore()  # TODO: order

    #def initial_update(self) -> None:
    #    self.update(0.0)

    #def final_update(self) -> None:
    #    self.update(1.0)


#class ModelAbstractShiftUpdater(ModelUpdater):
#    __slots__ = ("_mask",)

#    def __init__(
#        self,
#        mask: NP_3f8
#    ) -> None:
#        super().__init__(
#            about_model=Model(),
#            about_direction=ORIGIN
#        )
#        #self._vector: NP_3f8 = vector
#        self._mask: NP_3f8 = mask

#    @Lazy.property(hasher=Lazy.array_hasher)
#    @staticmethod
#    def _vector_() -> NP_3f8:
#        return NotImplemented

#    def _get_matrix(
#        self,
#        alpha: float
#    ) -> NP_44f8:
#        return SpaceUtils.matrix_from_shift(self._vector_ * (self._mask * alpha))


#class ModelAbstractScaleUpdater(ModelUpdater):
#    __slots__ = ("_mask",)

#    def __init__(
#        self,
#        #factor: float | NP_3f8,
#        about_model: Model,
#        about_direction: NP_3f8,
#        mask: NP_3f8
#    ) -> None:
#        #if not isinstance(factor, np.ndarray):
#        #    factor *= np.ones((3,))
#        #assert (factor >= 0.0).all(), "Scale factor must be positive"
#        #factor = np.maximum(factor, 1e-8)
#        #super().__init__()
#        #self._factor: NP_3f8 = factor
#        #self._about_model_ = about_model
#        #self._about_direction_ = about_direction
#        super().__init__(
#            about_model=about_model,
#            about_direction=about_direction
#        )
#        self._mask: NP_3f8 = mask

#    #@Lazy.variable(hasher=Lazy.array_hasher)
#    #@staticmethod
#    #def _mask_() -> NP_3f8:
#    #    return NotImplemented

#    @Lazy.property(hasher=Lazy.array_hasher)
#    @staticmethod
#    def _factor_() -> NP_3f8:
#        return NotImplemented

#    def _get_matrix(
#        self,
#        alpha: float
#    ) -> NP_44f8:
#        return SpaceUtils.matrix_from_scale(self._factor_ ** (self._mask * alpha))


#class ModelAbstractRotateUpdater(ModelUpdater):
#    __slots__ = ("_mask",)

#    def __init__(
#        self,
#        #vector: NP_3f8,
#        about_model: Model,
#        about_direction: NP_3f8,
#        mask: NP_3f8
#    ) -> None:
#        #super().__init__()
#        #self._vector: NP_3f8 = vector
#        super().__init__(
#            about_model=about_model,
#            about_direction=about_direction
#        )
#        self._mask: NP_3f8 = mask
#        #self._mask_ = mask

#    #@Lazy.variable(hasher=Lazy.array_hasher)
#    #@staticmethod
#    #def _mask_() -> NP_3f8:
#    #    return NotImplemented

#    @Lazy.property(hasher=Lazy.array_hasher)
#    @staticmethod
#    def _rotvec_() -> NP_3f8:
#        return NotImplemented

#    def _get_matrix(
#        self,
#        alpha: float
#    ) -> NP_44f8:
#        return SpaceUtils.matrix_from_rotate(self._rotvec_ * (self._mask * alpha))


#class ModelAbstractTransformationUpdater(ModelUpdater):
#    __slots__ = ()

#    def __init__(
#        self,
#        #vector: NP_3f8,
#        about_model: Model,
#        about_direction: NP_3f8
#    ) -> None:
#        #super().__init__()
#        #self._vector: NP_3f8 = vector
#        super().__init__(
#            about_model=about_model,
#            about_direction=about_direction
#        )
#        #self._mask: NP_3f8 = mask
#        #self._mask_ = mask

#    #@Lazy.variable(hasher=Lazy.array_hasher)
#    #@staticmethod
#    #def _mask_() -> NP_3f8:
#    #    return NotImplemented

#    @Lazy.property(hasher=Lazy.array_hasher)
#    @staticmethod
#    def _matrix_() -> NP_44f8:
#        return NotImplemented

#    def _get_matrix(
#        self,
#        alpha: float
#    ) -> NP_44f8:
#        return self._matrix_ * alpha


class ModelShiftUpdater(ModelUpdater):
    __slots__ = ("_mask",)

    def __init__(
        self,
        model: Model,
        vector: NP_3f8,
        mask: NP_3f8
    ) -> None:
        super().__init__(
            model=model,
            about_model=Model(),
            about_direction=ORIGIN
        )
        self._vector_ = vector
        self._mask: NP_3f8 = mask

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _vector_() -> NP_3f8:
        return NotImplemented

    def _get_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        return SpaceUtils.matrix_from_shift(self._vector_ * (self._mask * alpha))

    #@Lazy.variable(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _mask_() -> NP_3f8:
    #    return NotImplemented

    #@Lazy.property(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _shift_vector_(
    #    vector: NP_3f8,
    #    mask: NP_3f8
    #) -> NP_3f8:
    #    return vector * mask


class ModelShiftToUpdater(ModelUpdater):
    __slots__ = (
        "_aligned_model",
        "_mask"
    )

    def __init__(
        self,
        model: Model,
        aligned_model: Model,
        direction: NP_3f8,
        buff: NP_3f8,
        #buff_vector: NP_3f8,
        mask: NP_3f8,
        align_direction_sign: float
        #initial_model: Model
    ) -> None:
        super().__init__(
            model=model,
            about_model=Model(),
            about_direction=ORIGIN
        )
        self._direction_ = direction
        self._buff_vector_ = (
            model.box.get_position(align_direction_sign * direction, buff)
        )
        self._aligned_model: Model = aligned_model
        self._mask: NP_3f8 = mask

    @Lazy.variable()
    @staticmethod
    def _aligned_box_() -> Box:
        return NotImplemented

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _direction_() -> NP_3f8:
        return NotImplemented

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _buff_vector_() -> NP_3f8:
        return NotImplemented

    #@Lazy.variable(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _mask_() -> NP_3f8:
    #    return NotImplemented

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _vector_(
        aligned_box: Box,
        direction: NP_3f8,
        buff_vector: NP_3f8
    ) -> NP_3f8:
        return aligned_box.get_position(direction) - buff_vector

    def _get_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        self._aligned_box_ = self._aligned_model.box
        return SpaceUtils.matrix_from_shift(self._vector_ * (self._mask * alpha))


class ModelScaleUpdater(ModelUpdater):
    __slots__ = ("_mask",)

    def __init__(
        self,
        model: Model,
        factor: NP_3f8,
        about_model: Model,
        about_direction: NP_3f8,
        mask: NP_3f8
    ) -> None:
        assert (factor >= 0.0).all(), "Scale vector must be positive"
        factor = np.maximum(factor, 1e-8)
        super().__init__(
            model=model,
            about_model=about_model,
            about_direction=about_direction
        )
        self._factor_ = factor
        self._mask: NP_3f8 = mask

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _factor_() -> NP_3f8:
        return NotImplemented

    def _get_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        return SpaceUtils.matrix_from_scale(self._factor_ ** (self._mask * alpha))

    #@Lazy.variable(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _mask_() -> NP_3f8:
    #    return NotImplemented

    #@Lazy.property(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _scale_vector_(
    #    vector: NP_3f8,
    #    mask: NP_3f8
    #) -> NP_3f8:
    #    return np.maximum(vector * mask, 1e-8)


class ModelScaleToUpdater(ModelUpdater):
    __slots__ = (
        "_aligned_model",
        "_mask"
    )

    def __init__(
        self,
        model: Model,
        aligned_model: Model,
        #radii: NP_3f8,
        about_model: Model,
        about_direction: NP_3f8,
        mask: NP_3f8
        #initial_model: Model
    ) -> None:
        #assert (target_size >= 0.0).all(), "Scale vector must be positive"
        #target_size = np.maximum(target_size, 1e-8)
        super().__init__(
            model=model,
            about_model=about_model,
            about_direction=about_direction
        )
        self._initial_radii_ = model.box._radii_
        self._aligned_model: Model = aligned_model
        self._mask: NP_3f8 = mask
        #self._target_size_ = target_size
        #self._mask_ = mask

    @Lazy.variable()
    @staticmethod
    def _aligned_box_() -> Box:
        return NotImplemented

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _initial_radii_() -> NP_3f8:
        return NotImplemented

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _factor_(
        aligned_box__radii: NP_3f8,
        initial_radii: NP_3f8
    ) -> NP_3f8:
        return aligned_box__radii / np.maximum(initial_radii, 1e-8)

    def _get_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        self._aligned_box_ = self._aligned_model.box
        return SpaceUtils.matrix_from_scale(self._factor_ ** (self._mask * alpha))


class ModelRotateUpdater(ModelUpdater):
    __slots__ = ("_mask",)

    def __init__(
        self,
        model: Model,
        rotvec: NP_3f8,
        about_model: Model,
        about_direction: NP_3f8,
        mask: NP_3f8
    ) -> None:
        super().__init__(
            model=model,
            about_model=about_model,
            about_direction=about_direction
        )
        self._rotvec_ = rotvec
        self._mask: NP_3f8 = mask
        #self._mask_ = mask

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _rotvec_() -> NP_3f8:
        return NotImplemented

    def _get_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        return SpaceUtils.matrix_from_rotate(self._rotvec_ * (self._mask * alpha))

    #@Lazy.variable(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _mask_() -> NP_3f8:
    #    return NotImplemented

    #@Lazy.property(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _rotate_vector_(
    #    vector: NP_3f8,
    #    mask: NP_3f8
    #) -> NP_3f8:
    #    return vector * mask


class ModelApplyUpdater(ModelUpdater):
    __slots__ = ()

    def __init__(
        self,
        model: Model,
        matrix: NP_44f8,
        about_model: Model,
        about_direction: NP_3f8
    ) -> None:
        super().__init__(
            model=model,
            about_model=about_model,
            about_direction=about_direction
        )
        self._matrix_ = matrix

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _matrix_() -> NP_44f8:
        return NotImplemented

    def _get_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        return self._matrix_ * alpha


class ModelPoseUpdater(ModelUpdater):
    __slots__ = ("_aligned_model",)

    def __init__(
        self,
        model: Model,
        aligned_model: Model,
        #matrix: NP_44f8,
        about_model: Model,
        about_direction: NP_3f8
        #initial_model: Model
    ) -> None:
        super().__init__(
            model=model,
            about_model=about_model,
            about_direction=about_direction
        )
        self._initial_model_matrix_inverse_ = np.linalg.inv(model._model_matrix_._array_)
        self._aligned_model: Model = aligned_model

    @Lazy.variable()
    @staticmethod
    def _aligned_model_matrix_() -> ModelMatrix:
        return NotImplemented

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _initial_model_matrix_inverse_() -> NP_44f8:
        return NotImplemented

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _matrix_(
        aligned_model_matrix__array: NP_44f8,
        initial_model_matrix_inverse: NP_44f8
    ) -> NP_44f8:
        return aligned_model_matrix__array @ initial_model_matrix_inverse

    def _get_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        self._aligned_model_matrix_ = self._aligned_model._model_matrix_
        return self._matrix_ * alpha
