__all__ = ["Camera"]


import moderngl
import numpy as np
import pyrr

from ..constants import CAMERA_ALTITUDE
from ..constants import ORIGIN, OUT, UP
from ..utils.context_singleton import ContextSingleton
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
        return pyrr.matrix44.create_look_at(CAMERA_ALTITUDE * OUT, ORIGIN, UP, dtype=np.float32)

    @lazy_property
    @classmethod
    def _camera_matrices_buffer_(cls, projection_matrix: Matrix44Type, view_matrix: Matrix44Type) -> moderngl.Buffer:
        buffer = ContextSingleton().buffer(reserve=128)
        buffer.write(projection_matrix.tobytes(), offset=0)
        buffer.write(view_matrix.tobytes(), offset=64)
        return buffer

    @_camera_matrices_buffer_.releaser
    @staticmethod
    def _camera_matrices_buffer_releaser(uniform_camera_matrices_buffer: moderngl.Buffer) -> None:
        uniform_camera_matrices_buffer.release()

    #@lazy_property
    #@classmethod
    #def _view_matrix_buffer_(cls, view_matrix: Matrix44Type) -> moderngl.Buffer:
    #    return cls._make_buffer(view_matrix.astype(np.float64))

    #@_view_matrix_buffer_.releaser
    #@staticmethod
    #def _view_matrix_buffer_releaser(view_matrix_buffer: moderngl.Buffer) -> None:
    #    view_matrix_buffer.release()
