from typing import (
    Literal,
    overload
)

import numpy as np
from scipy.spatial.transform import Rotation

from ..constants.custom_typing import (
    NP_2f8,
    NP_3f8,
    NP_44f8,
    NP_4f8,
    NP_xf8,
    NP_x2f8,
    NP_x3f8,
    NP_x4f8,
    NPE_f8
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
    ) -> NPE_f8: ...

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
    ) -> NPE_f8 | NP_xf8:
        return np.linalg.norm(vector, axis=-1)

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
        tensor_1: np.ndarray,
        alpha: float | np.ndarray
    ) -> np.ndarray:
        return (1.0 - alpha) * tensor_0 + alpha * tensor_1

    #@overload
    #@classmethod
    #def apply_affine(
    #    cls,
    #    matrix: NP_44f8,
    #    vector: NP_3f8
    #) -> NP_3f8: ...

    #@overload
    #@classmethod
    #def apply_affine(
    #    cls,
    #    matrix: NP_44f8,
    #    vector: NP_x3f8
    #) -> NP_x3f8: ...

    ## TODO: split
    #@classmethod
    #def apply_affine(
    #    cls,
    #    matrix: NP_44f8,
    #    vector: NP_3f8 | NP_x3f8
    #) -> NP_3f8 | NP_x3f8:
    #    if vector.ndim == 1:
    #        v = vector[:, None]
    #    else:
    #        v = vector[:, :].T
    #    v = np.concatenate((v, np.ones((1, v.shape[1]))))
    #    v = matrix @ v
    #    if not np.allclose(v[-1], 1.0):
    #        v /= v[-1]
    #    v = v[:-1]
    #    if vector.ndim == 1:
    #        result = v.squeeze(axis=1)
    #    else:
    #        result = v.T
    #    return result

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
    def get_frame_scale_vector(
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

    @classmethod
    def matrix_from_shift(
        cls,
        vector: NP_3f8
    ) -> NP_44f8:
        matrix = np.identity(4)
        matrix[:3, 3] = vector
        return matrix

    @classmethod
    def matrix_from_scale(
        cls,
        factor: NP_3f8
    ) -> NP_44f8:
        matrix = np.identity(4)
        matrix[:3, :3] = np.diag(factor)
        return matrix

    @classmethod
    def matrix_from_rotate(
        cls,
        rotvec: NP_3f8
    ) -> NP_44f8:
        matrix = np.identity(4)
        matrix[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
        return matrix
