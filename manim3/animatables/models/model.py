from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Callable,
    TypeVar
)
#from typing import TYPE_CHECKING

import numpy as np

from ...constants.constants import (
    ORIGIN,
    PI
)
from ...constants.custom_typing import (
    NP_3f8,
    NP_44f8,
    NP_x3f8,
    NP_xf8,
    NP_xi4
)
from ...lazy.lazy import Lazy
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ...utils.space_utils import SpaceUtils
from ..arrays.model_matrix import ModelMatrix
from ..animatable import (
    Animatable,
    Updater
)

#if TYPE_CHECKING:
#    from ...models.model.model import model


_ModelT = TypeVar("_ModelT", bound="Model")


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


class Model(Animatable):
    __slots__ = ()

    @Lazy.variable(freeze=False)
    @staticmethod
    def _model_matrix_() -> ModelMatrix:
        return ModelMatrix()

    @Lazy.property_collection()
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
        model_matrix: ModelMatrix,
        local_sample_positions: NP_x3f8,
    ) -> NP_x3f8:
        return model_matrix._apply_affine_multiple(local_sample_positions)

    #@Lazy.property(hasher=Lazy.array_hasher)
    #@staticmethod
    #def _bounding_box_reference_points_(
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
    def _bounding_box_(
        world_sample_positions: NP_x3f8,
        associated_models__world_sample_positions: tuple[NP_x3f8, ...]
    ) -> BoundingBox:
        bounding_box_reference_points = np.concatenate((
            world_sample_positions,
            *associated_models__world_sample_positions
        ))
        if not len(bounding_box_reference_points):
            return BoundingBox(
                maximum=np.zeros((1, 3)),
                minimum=np.zeros((1, 3))
            )
        return BoundingBox(
            maximum=bounding_box_reference_points.max(axis=0),
            minimum=bounding_box_reference_points.min(axis=0)
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

    def get_bounding_box_position(
        self,
        direction: NP_3f8,
        buff: NP_3f8 = ORIGIN
    ) -> NP_3f8:
        return self._centroid_ + self._radii_ * direction + buff * direction

    def get_centroid(self) -> NP_3f8:
        return self._centroid_

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
        piecewise_func: Callable[[float], tuple[NP_xf8, NP_xi4]]
    ) -> Updater:
        return super()._get_piecewise_updater(src, piecewise_func).add(*(
            super(Model, dst_associated)._get_piecewise_updater(src_assiciated, piecewise_func)
            for dst_associated, src_assiciated in zip(
                self._associated_models_, src._associated_models_, strict=True
            )
        ))

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
            #buff_vector=self.get_bounding_box_position(signed_direction) + buff * signed_direction,
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
        #factor = target / self.get_bounding_box_size()
        #self.scale(
        #    vector=target_size / (2.0 * self_copy._radii_),
        #    about_model=about_model,
        #    about_direction=about_direction,
        #    mask=mask
        #)
        return self

    def match_bounding_box(
        self,
        model: "Model",
        mask: float | NP_3f8 = 1.0
    ):
        self.scale_to(model, mask=mask).shift_to(model, mask=mask)
        #self.shift(-self.get_center()).scale_to(model.get_bounding_box_size()).shift(model.get_center())
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
    __slots__ = ("_initial_matrices",)

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
        self._initial_matrices: dict[ModelMatrix, NP_44f8] = {
            model_assiciated._model_matrix_: model_assiciated._model_matrix_._array_
            for model_assiciated in (model, *model._associated_models_)
        }
        #self._model: Model = model
        self._about_model_ = about_model
        self._about_direction_ = about_direction

    @Lazy.variable(freeze=False)
    @staticmethod
    def _about_model_() -> Model:
        return NotImplemented

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _about_direction_() -> NP_3f8:
        return NotImplemented

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _about_point_(
        about_model: Model,
        about_direction: NP_3f8
    ) -> NP_3f8:
        return about_model.get_bounding_box_position(about_direction)

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
        #model = self._model
        matrix = self._get_matrix(alpha)
        for model_matrix, initial_matrix in self._initial_matrices.items():
            model_matrix._array_ = (
                self._post_shift_matrix_ @ matrix @ self._pre_shift_matrix_ @ initial_matrix
            )

    def initial_update(self) -> None:
        self.update(0.0)

    def final_update(self) -> None:
        self.update(1.0)


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
    __slots__ = ("_mask",)

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
        self._aligned_model_ = aligned_model
        self._direction_ = direction
        self._buff_vector_ = (
            model.get_bounding_box_position(align_direction_sign * direction, buff)
        )
        self._mask: NP_3f8 = mask

    @Lazy.variable(freeze=False)
    @staticmethod
    def _aligned_model_() -> Model:
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
        aligned_model: Model,
        direction: NP_3f8,
        buff_vector: NP_3f8
    ) -> NP_3f8:
        return aligned_model.get_bounding_box_position(direction) - buff_vector

    def _get_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
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
    __slots__ = ("_mask",)

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
        self._aligned_model_ = aligned_model
        self._initial_radii_ = model._radii_
        self._mask: NP_3f8 = mask
        #self._target_size_ = target_size
        #self._mask_ = mask

    @Lazy.variable(freeze=False)
    @staticmethod
    def _aligned_model_() -> Model:
        return NotImplemented

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _initial_radii_() -> NP_3f8:
        return NotImplemented

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _factor_(
        aligned_model__radii: NP_3f8,
        initial_radii: NP_3f8
    ) -> NP_3f8:
        return aligned_model__radii / np.maximum(initial_radii, 1e-8)

    def _get_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
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
    __slots__ = ()

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
        self._aligned_model_ = aligned_model
        self._initial_model_matrix_inverse_ = np.linalg.inv(model._model_matrix_._array_)

    @Lazy.variable(freeze=False)
    @staticmethod
    def _aligned_model_() -> Model:
        return NotImplemented

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _initial_model_matrix_inverse_() -> NP_44f8:
        return NotImplemented

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _matrix_(
        aligned_model__model_matrix: NP_44f8,
        initial_matrix_inverse: NP_44f8
    ) -> NP_44f8:
        return initial_matrix_inverse @ aligned_model__model_matrix

    def _get_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        return self._matrix_ * alpha
