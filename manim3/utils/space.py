__all__ = ["SpaceUtils"]


from abc import (
    ABC,
    abstractmethod
)
from functools import partial
from typing import (
    Callable,
    Literal,
    Union,
    overload
)

import numpy as np
from scipy.spatial.transform import (
    Rotation,
    Slerp
)

from ..custom_typing import (
    FloatsT,
    Mat3T,
    Mat4T,
    Vec2T,
    Vec3T,
    Vec4T,
    Vec2sT,
    Vec3sT,
    Vec4sT
)


class SpaceUtils(ABC):
    __slots__ = ()

    @abstractmethod
    def __new__(cls) -> None:
        pass

    @classmethod
    def matrix_from_translation(
        cls,
        vector: Vec3T
    ) -> Mat4T:
        m = np.identity(4)
        m[:3, 3] = vector
        return m

    @classmethod
    def matrix_from_scale(
        cls,
        factor: float | Vec3T
    ) -> Mat4T:
        m = np.identity(4)
        m[:3, :3] *= factor
        return m

    @classmethod
    def matrix_from_rotation(
        cls,
        rotation: Rotation
    ) -> Mat4T:
        m = np.identity(4)
        m[:3, :3] = rotation.as_matrix()
        return m

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
        tensor_1: float,
        alpha: float
    ) -> float: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: FloatsT,
        tensor_1: float,
        alpha: float
    ) -> FloatsT: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: float,
        tensor_1: FloatsT,
        alpha: float
    ) -> FloatsT: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec2T,
        tensor_1: Vec2T,
        alpha: float
    ) -> Vec2T: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec2sT,
        tensor_1: Vec2T,
        alpha: float
    ) -> Vec2sT: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec2T,
        tensor_1: Vec2sT,
        alpha: float
    ) -> Vec2sT: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec3T,
        tensor_1: Vec3T,
        alpha: float
    ) -> Vec3T: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec3sT,
        tensor_1: Vec3T,
        alpha: float
    ) -> Vec3sT: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec3T,
        tensor_1: Vec3sT,
        alpha: float
    ) -> Vec3sT: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec4T,
        tensor_1: Vec4T,
        alpha: float
    ) -> Vec4T: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec4sT,
        tensor_1: Vec4T,
        alpha: float
    ) -> Vec4sT: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Vec4T,
        tensor_1: Vec4sT,
        alpha: float
    ) -> Vec4sT: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Mat3T,
        tensor_1: Mat3T,
        alpha: float
    ) -> Mat3T: ...

    @overload
    @classmethod
    def lerp(
        cls,
        tensor_0: Mat4T,
        tensor_1: Mat4T,
        alpha: float
    ) -> Mat4T: ...

    @classmethod
    def lerp(
        cls,
        tensor_0: float | FloatsT | Vec2T | Vec2sT | Vec3T | Vec3sT | Vec4T | Vec4sT | Mat3T | Mat4T,
        tensor_1: float | FloatsT | Vec2T | Vec2sT | Vec3T | Vec3sT | Vec4T | Vec4sT | Mat3T | Mat4T,
        alpha: float
    ) -> float | FloatsT | Vec2T | Vec2sT | Vec3T | Vec3sT | Vec4T | Vec4sT | Mat3T | Mat4T:
        return (1.0 - alpha) * tensor_0 + alpha * tensor_1

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: float,
        tensor_1: float
    ) -> Callable[[float], float]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: FloatsT,
        tensor_1: float
    ) -> Callable[[float], FloatsT]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: float,
        tensor_1: FloatsT
    ) -> Callable[[float], FloatsT]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: Vec2T,
        tensor_1: Vec2T
    ) -> Callable[[float], Vec2T]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: Vec2sT,
        tensor_1: Vec2T
    ) -> Callable[[float], Vec2sT]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: Vec2T,
        tensor_1: Vec2sT
    ) -> Callable[[float], Vec2sT]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: Vec3T,
        tensor_1: Vec3T
    ) -> Callable[[float], Vec3T]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: Vec3sT,
        tensor_1: Vec3T
    ) -> Callable[[float], Vec3sT]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: Vec3T,
        tensor_1: Vec3sT
    ) -> Callable[[float], Vec3sT]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: Vec4T,
        tensor_1: Vec4T
    ) -> Callable[[float], Vec4T]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: Vec4sT,
        tensor_1: Vec4T
    ) -> Callable[[float], Vec4sT]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: Vec4T,
        tensor_1: Vec4sT
    ) -> Callable[[float], Vec4sT]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: Mat3T,
        tensor_1: Mat3T
    ) -> Callable[[float], Mat3T]: ...

    @overload
    @classmethod
    def lerp_callback(
        cls,
        tensor_0: Mat4T,
        tensor_1: Mat4T
    ) -> Callable[[float], Mat4T]: ...

    @classmethod
    def lerp_callback(
        cls,
        tensor_0: float | FloatsT | Vec2T | Vec2sT | Vec3T | Vec3sT | Vec4T | Vec4sT | Mat3T | Mat4T,
        tensor_1: float | FloatsT | Vec2T | Vec2sT | Vec3T | Vec3sT | Vec4T | Vec4sT | Mat3T | Mat4T
    ) -> Union[
        Callable[[float], float],
        Callable[[float], FloatsT],
        Callable[[float], Vec2T],
        Callable[[float], Vec2sT],
        Callable[[float], Vec3T],
        Callable[[float], Vec3sT],
        Callable[[float], Vec4T],
        Callable[[float], Vec4sT],
        Callable[[float], Mat3T],
        Callable[[float], Mat4T]
    ]:
        return partial(cls.lerp, tensor_0, tensor_1)

    @classmethod
    def rotational_interpolate_callback(
        cls,
        matrix_0: Mat4T,
        matrix_1: Mat4T
    ) -> Callable[[float], Mat4T]:
        rotation_part_0 = matrix_0[:3, :3]
        translation_0 = matrix_0[:3, 3]
        rotation_0 = Rotation.from_matrix(rotation_part_0)
        shear_0 = rotation_part_0 @ np.linalg.inv(rotation_0.as_matrix())
        rotation_part_1 = matrix_1[:3, :3]
        translation_1 = matrix_1[:3, 3]
        rotation_1 = Rotation.from_matrix(rotation_part_1)
        shear_1 = rotation_part_1 @ np.linalg.inv(rotation_1.as_matrix())
        slerp = Slerp((0.0, 1.0), Rotation.concatenate((rotation_0, rotation_1)))

        def callback(alpha: float) -> Mat4T:
            m = np.identity(4)
            m[:3, :3] = cls.lerp(shear_0, shear_1, alpha) @ Rotation.as_matrix(slerp(alpha))
            m[:3, 3] = cls.lerp(translation_0, translation_1, alpha)
            return m

        return callback

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
