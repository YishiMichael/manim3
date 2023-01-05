__all__ = ["OrthographicCamera"]


#import numpy as np
import pyrr

from ..cameras.camera import Camera
from ..constants import CAMERA_FAR, CAMERA_NEAR
from ..constants import FRAME_X_RADIUS, FRAME_Y_RADIUS
from ..utils.lazy import lazy_property, lazy_property_initializer_writable
from ..custom_typing import *


class OrthographicCamera(Camera):
    def __init__(
        self,
        left: Real = -FRAME_X_RADIUS,
        right: Real = FRAME_X_RADIUS,
        top: Real = FRAME_Y_RADIUS,
        bottom: Real = -FRAME_Y_RADIUS,
        near: Real = CAMERA_NEAR,
        far: Real = CAMERA_FAR
    ):
        super().__init__()
        self._left_ = left
        self._right_ = right
        self._top_ = top
        self._bottom_ = bottom
        self._near_ = near
        self._far_ = far

    @lazy_property_initializer_writable
    @staticmethod
    def _left_() -> Real:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _right_() -> Real:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _top_() -> Real:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _bottom_() -> Real:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _near_() -> Real:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _far_() -> Real:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _projection_matrix_(
        left: Real,
        right: Real,
        top: Real,
        bottom: Real,
        near: Real,
        far: Real
    ) -> Matrix44Type:
        return pyrr.matrix44.create_orthogonal_projection(
            left,
            right,
            top,
            bottom,
            near,
            far
        )
