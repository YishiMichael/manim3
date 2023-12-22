from __future__ import annotations


from typing import Self

import numpy as np

from ..constants.constants import Z_AXIS
from ..constants.custom_typing import (
    NP_44f8,
    NP_x3f8
)
from ..lazy.lazy import Lazy
from ..rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ..toplevel.toplevel import Toplevel
from .model import Model
from .point import Point


class CameraFrame(Model):
    __slots__ = ()

    def __init__(
        self: Self,
        width: float,
        height: float
    ) -> None:
        super().__init__()
        self._width_ = width
        self._height_ = height

    @Lazy.variable()
    @staticmethod
    def _width_() -> float:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _height_() -> float:
        return NotImplemented

    @Lazy.property()
    @staticmethod
    def _local_sample_positions_(
        width: float,
        height: float
    ) -> NP_x3f8:
        return np.array((
            (width / 2.0, height / 2.0, 0.0),
            (-width / 2.0, -height / 2.0, 0.0)
        ))


class Camera(Point):
    __slots__ = ()

    def __init__(
        self: Self,
        width: float | None = None,
        height: float | None = None,
        distance: float | None = None,
        near: float | None = None,
        far: float | None = None
    ) -> None:
        match width, height:
            case float(), float():
                pass
            case float(), None:
                height = width / Toplevel._get_config().aspect_ratio
            case None, float():
                width = height * Toplevel._get_config().aspect_ratio
            case _:
                width, height = Toplevel._get_config().frame_size
        if distance is None:
            distance = Toplevel._get_config().camera_distance
        if near is None:
            near = Toplevel._get_config().camera_near
        if far is None:
            far = Toplevel._get_config().camera_far

        super().__init__(position=distance * Z_AXIS)
        self._width_ = width
        self._height_ = height
        self._distance_ = distance
        self._near_ = near
        self._far_ = far

    @Lazy.variable()
    @staticmethod
    def _width_() -> float:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _height_() -> float:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _distance_() -> float:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _near_() -> float:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _far_() -> float:
        return NotImplemented

    @Lazy.property()
    @staticmethod
    def _projection_matrix_(
        distance: float,
        width: float,
        height: float,
        far: float,
        near: float
    ) -> NP_44f8:
        sx = 2.0 * distance / width
        sy = 2.0 * distance / height
        sz = -(far + near) / (far - near)
        tz = -2.0 * far * near / (far - near)
        return np.array((
            ( sx, 0.0,  0.0, 0.0),
            (0.0,  sy,  0.0, 0.0),
            (0.0, 0.0,   sz,  tz),
            (0.0, 0.0, -1.0, 0.0)
        ))

    @Lazy.property()
    @staticmethod
    def _view_matrix_(
        model_matrix__array: NP_44f8
    ) -> NP_44f8:
        return np.linalg.inv(model_matrix__array)

    @Lazy.property()
    @staticmethod
    def _camera_uniform_block_buffer_(
        projection_matrix: NP_44f8,
        view_matrix: NP_44f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_camera",
            field_declarations=(
                "mat4 u_projection_matrix",
                "mat4 u_view_matrix"
            ),
            data_dict={
                "u_projection_matrix": projection_matrix.T,
                "u_view_matrix": view_matrix.T
            }
        )

    @Lazy.property()
    @staticmethod
    def _frame_(
        width: float,
        height: float
    ) -> CameraFrame:
        return CameraFrame(width, height)

    @property
    def frame(
        self: Self
    ) -> CameraFrame:
        return self._frame_
