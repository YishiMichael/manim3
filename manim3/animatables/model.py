from __future__ import annotations


from abc import abstractmethod
from typing import (
    Iterator,
    Self,
    TypedDict,
    Unpack
)

import numpy as np

from ..constants.constants import (
    ORIGIN,
    PI
)
from ..constants.custom_typing import (
    ColorType,
    NP_3f8,
    NP_44f8,
    NP_x3f8
)
from ..lazy.lazy import Lazy
from ..lazy.lazy_object import LazyObject
from ..rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from .animatable.actions import (
    Action,
    ConverterDescriptorParameters
)
from .animatable.animatable import (
    Animatable,
    AnimatableActions,
    DynamicAnimatable
)
from .animatable.animation import (
    AnimateKwargs,
    Animation
)
from .arrays.model_matrix import ModelMatrix


class SetKwargs(TypedDict, total=False):
    # polymorphism variables
    color: ColorType
    opacity: float
    weight: float

    # MeshMobject
    ambient_strength: float
    specular_strength: float
    shininess: float

    # GraphMobject
    width: float


class Box(LazyObject):
    __slots__ = ()

    def __init__(
        self: Self,
        maximum: NP_3f8,
        minimum: NP_3f8
    ) -> None:
        super().__init__()
        self._maximum_ = maximum
        self._minimum_ = minimum

    @Lazy.variable()
    @staticmethod
    def _maximum_() -> NP_3f8:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _minimum_() -> NP_3f8:
        return NotImplemented

    @Lazy.property()
    @staticmethod
    def _centroid_(
        maximum: NP_3f8,
        minimum: NP_3f8
    ) -> NP_3f8:
        return (maximum + minimum) / 2.0

    @Lazy.property()
    @staticmethod
    def _radii_(
        maximum: NP_3f8,
        minimum: NP_3f8
    ) -> NP_3f8:
        return np.maximum((maximum - minimum) / 2.0, 1e-8)  # Avoid zero-divisions.

    def get(
        self: Self,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0
    ) -> NP_3f8:
        return self._centroid_ + self._radii_ * direction + buff * direction

    def get_radii(
        self: Self
    ) -> NP_3f8:
        return self._radii_


class ModelActions(AnimatableActions):
    __slots__ = ()

    @Action.register(ConverterDescriptorParameters)
    @classmethod
    def set(
        cls: type[Self],
        dst: Model,
        broadcast: bool = True,
        **kwargs: Unpack[SetKwargs]
    ) -> Iterator[Animation]:
        for descriptor, descriptor_parameters in cls.set._descriptor_dict.items():
            if (input_value := kwargs.get(descriptor._name.strip("_"))) is None:
                continue
            converter = descriptor_parameters.converter
            target = converter(input_value) if converter is not None else input_value
            for sibling in dst._iter_siblings(broadcast=broadcast):
                if descriptor not in sibling._lazy_descriptors:
                    continue
                initial = descriptor.__get__(sibling)
                yield from type(initial).interpolate._action(
                    dst=initial,
                    src_0=initial.copy(),
                    src_1=target
                )

    @Action.register()
    @classmethod
    def shift(
        cls: type[Self],
        dst: Model,
        vector: NP_3f8,
        mask: float | NP_3f8 = 1.0
    ) -> Iterator[Animation]:
        yield ModelShiftAnimation(
            model=dst,
            vector=vector,
            mask=mask * np.ones((3,))
        )

    @Action.register()
    @classmethod
    def move_to(
        cls: type[Self],
        dst: Model,
        target: Model,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0,
        mask: float | NP_3f8 = 1.0
    ) -> Iterator[Animation]:
        yield from cls.shift(
            dst=dst,
            vector=target.box.get(direction) - dst.box.get(direction, buff),
            mask=mask
        )

    @Action.register()
    @classmethod
    def next_to(
        cls: type[Self],
        dst: Model,
        target: Model,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0,
        mask: float | NP_3f8 = 1.0
    ) -> Iterator[Animation]:
        yield from cls.shift(
            dst=dst,
            vector=target.box.get(direction) - dst.box.get(-direction, buff),
            mask=mask
        )

    @Action.register()
    @classmethod
    def scale(
        cls: type[Self],
        dst: Model,
        factor: float | NP_3f8,
        about: Model | None = None,
        direction: NP_3f8 = ORIGIN,
        mask: float | NP_3f8 = 1.0
    ) -> Iterator[Animation]:
        if about is None:
            about = dst
        yield ModelScaleAnimation(
            model=dst,
            factor=factor * np.ones((3,)),
            about=about,
            direction=direction,
            mask=mask * np.ones((3,))
        )

    @Action.register()
    @classmethod
    def scale_about_origin(
        cls: type[Self],
        dst: Model,
        factor: float | NP_3f8,
        mask: float | NP_3f8 = 1.0
    ) -> Iterator[Animation]:
        yield from cls.scale(
            dst=dst,
            factor=factor,
            about=Model(),
            mask=mask
        )

    @Action.register()
    @classmethod
    def scale_to(
        cls: type[Self],
        dst: Model,
        target: Model,
        about: Model | None = None,
        direction: NP_3f8 = ORIGIN,
        mask: float | NP_3f8 = 1.0
    ) -> Iterator[Animation]:
        yield from cls.scale(
            dst=dst,
            factor=target.box.get_radii() / dst.box.get_radii(),
            about=about,
            direction=direction,
            mask=mask
        )

    @Action.register()
    @classmethod
    def rotate(
        cls: type[Self],
        dst: Model,
        rotvec: NP_3f8,
        about: Model | None = None,
        direction: NP_3f8 = ORIGIN,
        mask: float | NP_3f8 = 1.0
    ) -> Iterator[Animation]:
        if about is None:
            about = dst
        yield ModelRotateAnimation(
            model=dst,
            rotvec=rotvec,
            about=about,
            direction=direction,
            mask=mask * np.ones((3,))
        )

    @Action.register()
    @classmethod
    def rotate_about_origin(
        cls: type[Self],
        dst: Model,
        rotvec: NP_3f8,
        mask: float | NP_3f8 = 1.0
    ) -> Iterator[Animation]:
        yield from cls.rotate(
            dst=dst,
            rotvec=rotvec,
            about=Model(),
            mask=mask
        )

    @Action.register()
    @classmethod
    def flip(
        cls: type[Self],
        dst: Model,
        axis: NP_3f8,
        about: Model | None = None,
        direction: NP_3f8 = ORIGIN,
        mask: float | NP_3f8 = 1.0
    ) -> Iterator[Animation]:
        yield from cls.rotate(
            dst=dst,
            rotvec=axis / np.linalg.norm(axis) * PI,
            about=about,
            direction=direction,
            mask=mask
        )

    @Action.register()
    @classmethod
    def apply(
        cls: type[Self],
        dst: Model,
        matrix: NP_44f8,
        about: Model | None = None,
        direction: NP_3f8 = ORIGIN
    ) -> Iterator[Animation]:
        if about is None:
            about = dst
        yield ModelApplyAnimation(
            model=dst,
            matrix=matrix,
            about=about,
            direction=direction
        )


class Model(Animatable):
    __slots__ = ()

    @AnimatableActions.interpolate.register_descriptor()
    @Lazy.volatile()
    @staticmethod
    def _model_matrix_() -> ModelMatrix:
        return ModelMatrix()

    @AnimatableActions.interpolate.register_descriptor()
    @AnimatableActions.piecewise.register_descriptor()
    @Lazy.volatile(plural=True)
    @staticmethod
    def _proper_siblings_() -> tuple[Model, ...]:
        return ()

    @Lazy.property()
    @staticmethod
    def _model_uniform_block_buffer_(
        model_matrix__array: NP_44f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_model",
            field_declarations=(
                "mat4 u_model_matrix",
            ),
            data_dict={
                "u_model_matrix": model_matrix__array.T
            }
        )

    @Lazy.property()
    @staticmethod
    def _local_sample_positions_() -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.property()
    @staticmethod
    def _world_sample_positions_(
        model_matrix__array: NP_44f8,
        local_sample_positions: NP_x3f8,
    ) -> NP_x3f8:
        return ModelMatrix._apply_multiple(model_matrix__array, local_sample_positions)

    @Lazy.property()
    @staticmethod
    def _box_(
        world_sample_positions: NP_x3f8,
        proper_siblings__world_sample_positions: tuple[NP_x3f8, ...]
    ) -> Box:
        sample_positions = np.concatenate((
            world_sample_positions,
            *proper_siblings__world_sample_positions
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

    def _iter_siblings(
        self: Self,
        *,
        broadcast: bool = True
    ) -> Iterator[Model]:
        yield self
        if broadcast:
            yield from self._proper_siblings_

    @property
    def box(
        self: Self
    ) -> Box:
        return self._box_

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
    ) -> DynamicModel[Self]:
        return DynamicModel(self, **kwargs)

    set = ModelActions.set.build_action_descriptor()
    shift = ModelActions.shift.build_action_descriptor()
    move_to = ModelActions.move_to.build_action_descriptor()
    next_to = ModelActions.next_to.build_action_descriptor()
    scale = ModelActions.scale.build_action_descriptor()
    scale_about_origin = ModelActions.scale_about_origin.build_action_descriptor()
    scale_to = ModelActions.scale_to.build_action_descriptor()
    rotate = ModelActions.rotate.build_action_descriptor()
    rotate_about_origin = ModelActions.rotate_about_origin.build_action_descriptor()
    flip = ModelActions.flip.build_action_descriptor()
    apply = ModelActions.apply.build_action_descriptor()


class DynamicModel[ModelT: Model](DynamicAnimatable[ModelT]):
    __slots__ = ()

    set = ModelActions.set.build_dynamic_action_descriptor()
    shift = ModelActions.shift.build_dynamic_action_descriptor()
    move_to = ModelActions.move_to.build_dynamic_action_descriptor()
    next_to = ModelActions.next_to.build_dynamic_action_descriptor()
    scale = ModelActions.scale.build_dynamic_action_descriptor()
    scale_about_origin = ModelActions.scale_about_origin.build_dynamic_action_descriptor()
    scale_to = ModelActions.scale_to.build_dynamic_action_descriptor()
    rotate = ModelActions.rotate.build_dynamic_action_descriptor()
    rotate_about_origin = ModelActions.rotate_about_origin.build_dynamic_action_descriptor()
    flip = ModelActions.flip.build_dynamic_action_descriptor()
    apply = ModelActions.apply.build_dynamic_action_descriptor()


class ModelAnimation(Animation):
    __slots__ = (
        "_pre_shift_matrix",
        "_post_shift_matrix",
        "_model_matrices"
    )

    def __init__(
        self: Self,
        model: Model,
        about: Model,
        direction: NP_3f8
    ) -> None:
        super().__init__()
        about_point = about.box.get(direction)
        self._pre_shift_matrix: NP_44f8 = ModelMatrix._matrix_from_shift(-about_point)
        self._post_shift_matrix: NP_44f8 = ModelMatrix._matrix_from_shift(about_point)
        self._model_matrices: dict[ModelMatrix, NP_44f8] = {
            sibling._model_matrix_: sibling._model_matrix_._array_
            for sibling in (model, *model._proper_siblings_)
        }

    @abstractmethod
    def _get_matrix(
        self: Self,
        alpha: float
    ) -> NP_44f8:
        pass

    def update(
        self: Self,
        alpha: float
    ) -> None:
        super().update(alpha)
        matrix = self._post_shift_matrix @ self._get_matrix(alpha) @ self._pre_shift_matrix
        for model_matrix, initial_model_matrix_array in self._model_matrices.items():
            model_matrix._array_ = matrix @ initial_model_matrix_array


class ModelShiftAnimation(ModelAnimation):
    __slots__ = (
        "_vector",
        "_mask"
    )

    def __init__(
        self: Self,
        model: Model,
        vector: NP_3f8,
        mask: NP_3f8
    ) -> None:
        super().__init__(
            model=model,
            about=Model(),
            direction=ORIGIN
        )
        self._vector: NP_3f8 = vector
        self._mask: NP_3f8 = mask

    def _get_matrix(
        self: Self,
        alpha: float
    ) -> NP_44f8:
        return ModelMatrix._matrix_from_shift(self._vector * (self._mask * alpha))


class ModelScaleAnimation(ModelAnimation):
    __slots__ = (
        "_factor",
        "_mask"
    )

    def __init__(
        self: Self,
        model: Model,
        factor: NP_3f8,
        about: Model,
        direction: NP_3f8,
        mask: NP_3f8
    ) -> None:
        assert (factor >= 0.0).all(), "Scale vector must be positive"
        factor = np.maximum(factor, 1e-8)
        super().__init__(
            model=model,
            about=about,
            direction=direction
        )
        self._factor: NP_3f8 = factor
        self._mask: NP_3f8 = mask

    def _get_matrix(
        self: Self,
        alpha: float
    ) -> NP_44f8:
        return ModelMatrix._matrix_from_scale(self._factor ** (self._mask * alpha))


class ModelRotateAnimation(ModelAnimation):
    __slots__ = (
        "_rotvec",
        "_mask"
    )

    def __init__(
        self: Self,
        model: Model,
        rotvec: NP_3f8,
        about: Model,
        direction: NP_3f8,
        mask: NP_3f8
    ) -> None:
        super().__init__(
            model=model,
            about=about,
            direction=direction
        )
        self._rotvec: NP_3f8 = rotvec
        self._mask: NP_3f8 = mask

    def _get_matrix(
        self: Self,
        alpha: float
    ) -> NP_44f8:
        return ModelMatrix._matrix_from_rotate(self._rotvec * (self._mask * alpha))


class ModelApplyAnimation(ModelAnimation):
    __slots__ = ("_matrix",)

    def __init__(
        self: Self,
        model: Model,
        matrix: NP_44f8,
        about: Model,
        direction: NP_3f8
    ) -> None:
        super().__init__(
            model=model,
            about=about,
            direction=direction
        )
        self._matrix: NP_44f8 = matrix

    def _get_matrix(
        self: Self,
        alpha: float
    ) -> NP_44f8:
        return (1.0 - alpha) * np.identity(4) + alpha * self._matrix
