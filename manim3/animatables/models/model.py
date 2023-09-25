from abc import abstractmethod
from dataclasses import dataclass
#from typing import TYPE_CHECKING

import numpy as np

from ...constants.constants import (
    ORIGIN,
    PI
)
from ...constants.custom_typing import (
    NP_3f8,
    NP_44f8,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..animatable import (
    Animatable,
    Updater
)

#if TYPE_CHECKING:
#    from ...models.model.model import model


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

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _model_matrix_() -> NP_44f8:
        return np.identity(4)

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _local_sample_positions_() -> NP_x3f8:
        #return np.array((ORIGIN,))
        return np.zeros((0, 3))

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _world_sample_positions_(
        model_matrix: NP_44f8,
        local_sample_positions: NP_x3f8,
    ) -> NP_x3f8:
        return SpaceUtils.apply_affine(model_matrix, local_sample_positions)

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _bounding_box_reference_points_(
        world_sample_positions: NP_x3f8,
    ) -> NP_x3f8:
        return world_sample_positions

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
        bounding_box_reference_points: NP_x3f8
    ) -> BoundingBox:
        #positions_array = np.concatenate((
        #    world_sample_positions,
        #    *real_descendants__world_sample_positions
        #))
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
        buff: float | NP_3f8 = 0.0
    ) -> NP_3f8:
        return self._centroid_ + self._radii_ * direction + buff * direction

    def get_centroid(self) -> NP_3f8:
        return self._centroid_

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
        buff_direction_sign: float = 0.0,
        mask: float | NP_3f8 = 1.0
    ):
        #signed_direction = buff_direction_sign * direction
        self._stack_updater(ModelShiftToUpdater(
            model=self,
            aligned_model=aligned_model,
            direction=direction,
            buff=buff * np.ones((3,)),
            buff_direction_sign=buff_direction_sign,
            #buff_vector=self.get_bounding_box_position(signed_direction) + buff * signed_direction,
            mask=mask * np.ones((3,))
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
            buff_direction_sign=1.0,
            mask=mask
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
            buff_direction_sign=-1.0,
            mask=mask
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

    def transformation(
        self,
        transmat: NP_44f8,
        about_model: "Model | None" = None,
        about_direction: NP_3f8 = ORIGIN
    ):
        if about_model is None:
            about_model = self
        self._stack_updater(ModelTransformationUpdater(
            model=self,
            transmat=transmat,
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


class ModelUpdater(Updater[Model]):
    __slots__ = ()

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
        super().__init__(model)
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

    @abstractmethod
    def _get_transformation_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        pass

    def update(
        self,
        model: Model,
        alpha: float
    ) -> None:
        transformation = self._get_transformation_matrix(alpha)
        about_point = self._about_point_
        model._model_matrix_ = (
            SpaceUtils.matrix_from_shift(about_point) @ transformation @ SpaceUtils.matrix_from_shift(-about_point) @ model._model_matrix_
        )


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

#    def _get_transformation_matrix(
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

#    def _get_transformation_matrix(
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

#    def _get_transformation_matrix(
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
#    def _transmat_() -> NP_44f8:
#        return NotImplemented

#    def _get_transformation_matrix(
#        self,
#        alpha: float
#    ) -> NP_44f8:
#        return self._transmat_ * alpha


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

    def _get_transformation_matrix(
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
        buff_direction_sign: float,
        #buff_vector: NP_3f8,
        mask: NP_3f8
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
            model.get_bounding_box_position(buff_direction_sign * direction, buff)
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

    def _get_transformation_matrix(
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

    def _get_transformation_matrix(
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

    def _get_transformation_matrix(
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

    def _get_transformation_matrix(
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


class ModelTransformationUpdater(ModelUpdater):
    __slots__ = ()

    def __init__(
        self,
        model: Model,
        transmat: NP_44f8,
        about_model: Model,
        about_direction: NP_3f8
    ) -> None:
        super().__init__(
            model=model,
            about_model=about_model,
            about_direction=about_direction
        )
        self._transmat_ = transmat

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _transmat_() -> NP_44f8:
        return NotImplemented

    def _get_transformation_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        return self._transmat_ * alpha


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
        self._initial_model_matrix_inverse_ = np.linalg.inv(model._model_matrix_)

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
    def _transmat_(
        aligned_model__model_matrix: NP_44f8,
        initial_matrix_inverse: NP_44f8
    ) -> NP_44f8:
        return initial_matrix_inverse @ aligned_model__model_matrix

    def _get_transformation_matrix(
        self,
        alpha: float
    ) -> NP_44f8:
        return self._transmat_ * alpha
