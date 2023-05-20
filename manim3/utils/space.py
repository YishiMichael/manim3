from typing import (
    Any,
    Callable,
    Literal,
    overload
)

import numpy as np
from scipy.spatial.transform import Rotation

from ..custom_typing import (
    FloatsT,
    Mat3T,
    Mat4T,
    Vec2T,
    Vec2sT,
    Vec3T,
    Vec3sT,
    Vec4T,
    Vec4sT
)


class SpaceUtils:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @overload
    @classmethod
    def norm(
        cls,
        vector: Vec2T | Vec3T | Vec4T
    ) -> float: ...

    @overload
    @classmethod
    def norm(
        cls,
        vector: Vec2sT | Vec3sT | Vec4sT
    ) -> FloatsT: ...

    @classmethod
    def norm(
        cls,
        vector: Vec2T | Vec3T | Vec4T | Vec2sT | Vec3sT | Vec4sT
    ) -> float | FloatsT:
        if vector.ndim == 1:
            return float(np.linalg.norm(vector))
        return np.linalg.norm(vector, axis=1)

    @overload
    @classmethod
    def normalize(
        cls,
        vector: Vec2T
    ) -> Vec2T: ...

    @overload
    @classmethod
    def normalize(
        cls,
        vector: Vec3T
    ) -> Vec3T: ...

    @overload
    @classmethod
    def normalize(
        cls,
        vector: Vec4T
    ) -> Vec4T: ...

    @overload
    @classmethod
    def normalize(
        cls,
        vector: Vec2sT
    ) -> Vec2sT: ...

    @overload
    @classmethod
    def normalize(
        cls,
        vector: Vec3sT
    ) -> Vec3sT: ...

    @overload
    @classmethod
    def normalize(
        cls,
        vector: Vec4sT
    ) -> Vec4sT: ...

    @classmethod
    def normalize(
        cls,
        vector: Vec2T | Vec3T | Vec4T | Vec2sT | Vec3sT | Vec4sT
    ) -> Vec2T | Vec3T | Vec4T | Vec2sT | Vec3sT | Vec4sT:
        if vector.ndim == 1:
            return vector / np.linalg.norm(vector)
        return vector / np.linalg.norm(vector, axis=1)[:, None]

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: float,
        tensor_1: float
    ) -> Callable[[float], float]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: float,
        tensor_1: Vec2T
    ) -> Callable[[float | Vec2T], Vec2T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec2T,
        tensor_1: float
    ) -> Callable[[float | Vec2T], Vec2T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec2T,
        tensor_1: Vec2T
    ) -> Callable[[float | Vec2T], Vec2T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: float,
        tensor_1: Vec3T
    ) -> Callable[[float | Vec3T], Vec3T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec3T,
        tensor_1: float
    ) -> Callable[[float | Vec3T], Vec3T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec3T,
        tensor_1: Vec3T
    ) -> Callable[[float | Vec3T], Vec3T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: float,
        tensor_1: Vec4T
    ) -> Callable[[float | Vec4T], Vec4T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec4T,
        tensor_1: float
    ) -> Callable[[float | Vec4T], Vec4T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec4T,
        tensor_1: Vec4T
    ) -> Callable[[float | Vec4T], Vec4T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: float,
        tensor_1: Mat3T
    ) -> Callable[[float | Mat3T], Mat3T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Mat3T,
        tensor_1: float
    ) -> Callable[[float | Mat3T], Mat3T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Mat3T,
        tensor_1: Mat3T
    ) -> Callable[[float | Mat3T], Mat3T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: float,
        tensor_1: Mat4T
    ) -> Callable[[float | Mat4T], Mat4T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Mat4T,
        tensor_1: float
    ) -> Callable[[float | Mat4T], Mat4T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Mat4T,
        tensor_1: Mat4T
    ) -> Callable[[float | Mat4T], Mat4T]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: float,
        tensor_1: FloatsT
    ) -> Callable[[float], FloatsT]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: FloatsT,
        tensor_1: float
    ) -> Callable[[float], FloatsT]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: float | Vec2T,
        tensor_1: Vec2sT
    ) -> Callable[[float | Vec2T], Vec2sT]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec2sT,
        tensor_1: float | Vec2T
    ) -> Callable[[float | Vec2T], Vec2sT]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: float | Vec3T,
        tensor_1: Vec3sT
    ) -> Callable[[float | Vec3T], Vec3sT]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec3sT,
        tensor_1: float | Vec3T
    ) -> Callable[[float | Vec3T], Vec3sT]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: float | Vec4T,
        tensor_1: Vec4sT
    ) -> Callable[[float | Vec4T], Vec4sT]: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec4sT,
        tensor_1: float | Vec4T
    ) -> Callable[[float | Vec4T], Vec4sT]: ...

    @classmethod
    def lerp(
        cls,
        tensor_0: Any,
        tensor_1: Any
    ) -> Callable[[Any], Any]:

        def callback(
            alpha: Any
        ) -> Any:
            return (1.0 - alpha) * tensor_0 + alpha * tensor_1

        return callback

    # Type specifications of `lerp`.

    @classmethod
    def lerp_float(
        cls,
        tensor_0: float,
        tensor_1: float
    ) -> Callable[[float], float]:
        return cls.lerp(tensor_0, tensor_1)

    @classmethod
    def lerp_vec3(
        cls,
        tensor_0: Vec3T,
        tensor_1: Vec3T
    ) -> Callable[[float], Vec3T]:
        return cls.lerp(tensor_0, tensor_1)

    @classmethod
    def lerp_mat4(
        cls,
        tensor_0: Mat4T,
        tensor_1: Mat4T
    ) -> Callable[[float], Mat4T]:
        return cls.lerp(tensor_0, tensor_1)

    @overload
    @classmethod
    def apply_affine(
        cls,
        matrix: Mat3T,
        vector: Vec2T
    ) -> Vec2T: ...

    @overload
    @classmethod
    def apply_affine(
        cls,
        matrix: Mat4T,
        vector: Vec3T
    ) -> Vec3T: ...

    @overload
    @classmethod
    def apply_affine(
        cls,
        matrix: Mat3T,
        vector: Vec2sT
    ) -> Vec2sT: ...

    @overload
    @classmethod
    def apply_affine(
        cls,
        matrix: Mat4T,
        vector: Vec3sT
    ) -> Vec3sT: ...

    @classmethod
    def apply_affine(
        cls,
        matrix: Mat3T | Mat4T,
        vector: Vec2T | Vec3T | Vec2sT | Vec3sT
    ) -> Vec2T | Vec3T | Vec2sT | Vec3sT:
        if vector.ndim == 1:
            v = vector[:, None]
        else:
            v = vector[:, :].T
        v = np.concatenate((v, np.ones((1, v.shape[1]))))
        v = matrix @ v
        if not np.allclose(v[-1], 1.0):
            v /= v[-1]
        v = v[:-1]
        if vector.ndim == 1:
            result = v.squeeze(axis=1)
        else:
            result = v.T
        return result

    @classmethod
    def matrix_callback_from_translation(
        cls,
        vector: Vec3T
    ) -> Callable[[float | Vec3T], Mat4T]:
        lerp_callback = cls.lerp(0.0, vector)

        def callback(
            alpha: float | Vec3T
        ) -> Mat4T:
            m = np.identity(4)
            m[:3, 3] = lerp_callback(alpha)
            return m

        return callback

    @classmethod
    def matrix_callback_from_scale(
        cls,
        factor: float | Vec3T
    ) -> Callable[[float | Vec3T], Mat4T]:
        if not isinstance(factor, np.ndarray):
            factor *= np.ones(3)
        lerp_callback = cls.lerp(1.0, factor)

        def callback(
            alpha: float | Vec3T
        ) -> Mat4T:
            m = np.identity(4)
            m[:3, :3] = np.diag(lerp_callback(alpha))
            return m

        return callback

    @classmethod
    def matrix_callback_from_rotation(
        cls,
        rotvec: Vec3T
    ) -> Callable[[float | Vec3T], Mat4T]:
        lerp_callback = cls.lerp(0.0, rotvec)

        def callback(
            alpha: float | Vec3T
        ) -> Mat4T:
            m = np.identity(4)
            m[:3, :3] = Rotation.from_rotvec(lerp_callback(alpha)).as_matrix()
            return m

        return callback

    @classmethod
    def matrix_from_translation(
        cls,
        vector: Vec3T
    ) -> Mat4T:
        return cls.matrix_callback_from_translation(vector)(1.0)

    @classmethod
    def increase_dimension(
        cls,
        vectors: Vec2sT,
        *,
        z_value: float = 0.0
    ) -> Vec3sT:
        result = np.zeros((len(vectors), 3))
        result[:, :2] = vectors
        result[:, 2] = z_value
        return result

    @overload
    @classmethod
    def decrease_dimension(
        cls,
        vectors: Vec3sT,
        *,
        extract_z: Literal[True]
    ) -> tuple[Vec2sT, FloatsT]: ...

    @overload
    @classmethod
    def decrease_dimension(
        cls,
        vectors: Vec3sT,
        *,
        extract_z: Literal[False] = False
    ) -> Vec2sT: ...

    @classmethod
    def decrease_dimension(
        cls,
        vectors: Vec3sT,
        *,
        extract_z: bool = False
    ) -> tuple[Vec2sT, FloatsT] | Vec2sT:
        result = vectors[:, :2]
        if not extract_z:
            return result
        z_value = vectors[:, 2]
        return result, z_value

    @classmethod
    def _get_frame_scale_vector(
        cls,
        *,
        original_width: float,
        original_height: float,
        specified_width: float | None,
        specified_height: float | None,
        specified_frame_scale: float | None
    ) -> tuple[float, float]:
        match specified_width, specified_height:
            case float(), float():
                return specified_width / original_width, specified_height / original_height
            case float(), None:
                scale_factor = specified_width / original_width
            case None, float():
                scale_factor = specified_height / original_height
            case None, None:
                scale_factor = specified_frame_scale if specified_frame_scale is not None else 1.0
            case _:
                raise ValueError  # never
        return scale_factor, scale_factor
