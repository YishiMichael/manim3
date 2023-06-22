from typing import (
    Callable,
    Literal,
    overload
)

import numpy as np
from scipy.spatial.transform import Rotation

from ..custom_typing import (
    NP_f8,
    NP_xf8,
    NP_33f8,
    NP_44f8,
    NP_2f8,
    NP_x2f8,
    NP_3f8,
    NP_x3f8,
    NP_4f8,
    NP_x4f8
)


class SpaceUtils:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @overload
    @classmethod
    def norm(
        cls,
        vector: NP_2f8 | NP_3f8 | NP_4f8
    ) -> NP_f8: ...

    @overload
    @classmethod
    def norm(
        cls,
        vector: NP_x2f8 | NP_x3f8 | NP_x4f8
    ) -> NP_xf8: ...

    @classmethod
    def norm(
        cls,
        vector: NP_2f8 | NP_3f8 | NP_4f8 | NP_x2f8 | NP_x3f8 | NP_x4f8
    ) -> NP_f8 | NP_xf8:
        #if vector.ndim == 1:
        return np.array(np.linalg.norm(vector, axis=-1))
        #return np.linalg.norm(vector, axis=1)

    @overload
    @classmethod
    def normalize(
        cls,
        vector: NP_2f8
    ) -> NP_2f8: ...

    @overload
    @classmethod
    def normalize(
        cls,
        vector: NP_3f8
    ) -> NP_3f8: ...

    @overload
    @classmethod
    def normalize(
        cls,
        vector: NP_4f8
    ) -> NP_4f8: ...

    @overload
    @classmethod
    def normalize(
        cls,
        vector: NP_x2f8
    ) -> NP_x2f8: ...

    @overload
    @classmethod
    def normalize(
        cls,
        vector: NP_x3f8
    ) -> NP_x3f8: ...

    @overload
    @classmethod
    def normalize(
        cls,
        vector: NP_x4f8
    ) -> NP_x4f8: ...

    @classmethod
    def normalize(
        cls,
        vector: NP_2f8 | NP_3f8 | NP_4f8 | NP_x2f8 | NP_x3f8 | NP_x4f8
    ) -> NP_2f8 | NP_3f8 | NP_4f8 | NP_x2f8 | NP_x3f8 | NP_x4f8:
        if vector.ndim == 1:
            return vector / np.linalg.norm(vector)
        return vector / np.linalg.norm(vector, axis=1)[:, None]

    @classmethod
    def lerp(
        cls,
        tensor_0: np.ndarray,
        tensor_1: np.ndarray
    ) -> Callable[[float | np.ndarray], np.ndarray]:

        def callback(
            alpha: float | np.ndarray
        ) -> np.ndarray:
            return (1.0 - alpha) * tensor_0 + alpha * tensor_1

        return callback

    # Type specifications of `lerp`.

    @classmethod
    def lerp_f8(
        cls,
        tensor_0: NP_f8,
        tensor_1: NP_f8
    ) -> Callable[[float], NP_f8]:
        return cls.lerp(tensor_0, tensor_1)

    @classmethod
    def lerp_3f8(
        cls,
        tensor_0: NP_3f8,
        tensor_1: NP_3f8
    ) -> Callable[[float], NP_3f8]:
        return cls.lerp(tensor_0, tensor_1)

    #@classmethod
    #def lerp_x3f8(
    #    cls,
    #    tensor_0: NP_x3f8,
    #    tensor_1: NP_x3f8
    #) -> Callable[[NP_xf8], NP_x3f8]:
    #    return cls.lerp(tensor_0, tensor_1)

    @classmethod
    def lerp_44f8(
        cls,
        tensor_0: NP_44f8,
        tensor_1: NP_44f8
    ) -> Callable[[float], NP_44f8]:
        return cls.lerp(tensor_0, tensor_1)

    #@classmethod
    #def lerp_float_3f8(
    #    cls,
    #    tensor_0: float,
    #    tensor_1: NP_3f8
    #) -> Callable[[float | NP_3f8], NP_3f8]:
    #    return cls.lerp(tensor_0 * np.ones(()), tensor_1)

    @overload
    @classmethod
    def apply_affine(
        cls,
        matrix: NP_33f8,
        vector: NP_2f8
    ) -> NP_2f8: ...

    @overload
    @classmethod
    def apply_affine(
        cls,
        matrix: NP_44f8,
        vector: NP_3f8
    ) -> NP_3f8: ...

    @overload
    @classmethod
    def apply_affine(
        cls,
        matrix: NP_33f8,
        vector: NP_x2f8
    ) -> NP_x2f8: ...

    @overload
    @classmethod
    def apply_affine(
        cls,
        matrix: NP_44f8,
        vector: NP_x3f8
    ) -> NP_x3f8: ...

    @classmethod
    def apply_affine(
        cls,
        matrix: NP_33f8 | NP_44f8,
        vector: NP_2f8 | NP_3f8 | NP_x2f8 | NP_x3f8
    ) -> NP_2f8 | NP_3f8 | NP_x2f8 | NP_x3f8:
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
        vector: NP_3f8
    ) -> Callable[[float | NP_3f8], NP_44f8]:
        lerp_callback = cls.lerp(np.zeros(()), vector)

        def callback(
            alpha: float | NP_3f8
        ) -> NP_44f8:
            m = np.identity(4)
            m[:3, 3] = lerp_callback(alpha)
            return m

        return callback

    @classmethod
    def matrix_callback_from_scale(
        cls,
        factor: float | NP_3f8
    ) -> Callable[[float | NP_3f8], NP_44f8]:
        if not isinstance(factor, np.ndarray):
            factor *= np.ones((3,))
        lerp_callback = cls.lerp(np.ones(()), factor)

        def callback(
            alpha: float | NP_3f8
        ) -> NP_44f8:
            m = np.identity(4)
            m[:3, :3] = np.diag(lerp_callback(alpha))
            return m

        return callback

    @classmethod
    def matrix_callback_from_rotation(
        cls,
        rotvec: NP_3f8
    ) -> Callable[[float | NP_3f8], NP_44f8]:
        lerp_callback = cls.lerp(np.zeros(()), rotvec)

        def callback(
            alpha: float | NP_3f8
        ) -> NP_44f8:
            m = np.identity(4)
            m[:3, :3] = Rotation.from_rotvec(lerp_callback(alpha)).as_matrix()
            return m

        return callback

    @classmethod
    def matrix_from_translation(
        cls,
        vector: NP_3f8
    ) -> NP_44f8:
        return cls.matrix_callback_from_translation(vector)(1.0)

    @classmethod
    def increase_dimension(
        cls,
        vectors: NP_x2f8,
        *,
        z_value: float = 0.0
    ) -> NP_x3f8:
        result = np.zeros((len(vectors), 3))
        result[:, :2] = vectors
        result[:, 2] = z_value
        return result

    @overload
    @classmethod
    def decrease_dimension(
        cls,
        vectors: NP_x3f8,
        *,
        extract_z: Literal[True]
    ) -> tuple[NP_x2f8, NP_xf8]: ...

    @overload
    @classmethod
    def decrease_dimension(
        cls,
        vectors: NP_x3f8,
        *,
        extract_z: Literal[False] = False
    ) -> NP_x2f8: ...

    @classmethod
    def decrease_dimension(
        cls,
        vectors: NP_x3f8,
        *,
        extract_z: bool = False
    ) -> tuple[NP_x2f8, NP_xf8] | NP_x2f8:
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
