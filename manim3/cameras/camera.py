__all__ = ["Camera"]


import moderngl
import numpy as np
import pyrr

from ..constants import CAMERA_ALTITUDE
from ..constants import ORIGIN, OUT, UP
from ..utils.renderable import Renderable
from ..utils.lazy import lazy_property, lazy_property_initializer, lazy_property_initializer_writable
from ..custom_typing import *


class Camera(Renderable):
    #def __init__(self):
    #    self.view_matrix: Matrix44Type = pyrr.matrix44.create_look_at(CAMERA_ALTITUDE * OUT, ORIGIN, UP)

    @lazy_property_initializer
    @classmethod
    def _projection_matrix_(cls) -> Matrix44Type:
        return NotImplemented

    @lazy_property_initializer_writable
    @classmethod
    def _view_matrix_(cls) -> Matrix44Type:
        return pyrr.matrix44.create_look_at(CAMERA_ALTITUDE * OUT, ORIGIN, UP)

    @lazy_property
    @classmethod
    def _in_projection_matrix_buffer_(cls, projection_matrix: Matrix44Type) -> moderngl.Buffer:
        return cls._make_buffer(projection_matrix.astype(np.float64))

    @_in_projection_matrix_buffer_.releaser
    @staticmethod
    def _in_projection_matrix_buffer_releaser(in_projection_matrix_buffer: moderngl.Buffer) -> None:
        in_projection_matrix_buffer.release()

    @lazy_property
    @classmethod
    def _in_view_matrix_buffer_(cls, view_matrix: Matrix44Type) -> moderngl.Buffer:
        return cls._make_buffer(view_matrix.astype(np.float64))

    @_in_view_matrix_buffer_.releaser
    @staticmethod
    def _in_view_matrix_buffer_releaser(in_view_matrix_buffer: moderngl.Buffer) -> None:
        in_view_matrix_buffer.release()
