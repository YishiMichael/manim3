"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


#__all__ = [
#    "Array",
#    "Vec2",
#    "Vec3",
#    "Vec4",
#    "Mat3",
#    "Mat4",
#    "Vec2s",
#    "Vec3s",
#    "Vec4s",
#    "Mat3s",
#    "Mat4s"
#]


# What could be a better way to annotate these array-like stuff?
Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
Vec = Array
Vec2 = Array
Vec3 = Array
Vec4 = Array
Mat = Array
Mat3 = Array
Mat4 = Array
Vec2s = Array
Vec3s = Array
Vec4s = Array
Mat3s = Array
Mat4s = Array

IArray = np.ndarray[tuple[int, ...], np.dtype[np.int32]]


def apply_affine_matrix(vec: Vec, mat: Mat) -> Vec:
    return (mat @ np.append(vec, 1.0))[:-1]


def matrix_inverse(mat: Mat) -> Mat:
    return np.linalg.inv(mat)


#def from_matrix_rotation(mat: Mat4) -> Mat4:
#    result = np.eye(4)
#    result[:3, :3] = mat[:3, :3] / np.linalg.norm(mat[:3, :3], axis=1)
#    return result


def from_shift(vec: Vec3) -> Mat4:
    result = np.eye(4)
    result[:-1, -1] = vec
    return result


def from_scale(vec: Vec3) -> Mat4:
    result = np.eye(4)
    result[:3, :3] = np.diag(vec)
    return result


def from_rotate(rotation: Rotation) -> Mat4:
    result = np.eye(4)
    result[:3, :3] = rotation.as_matrix()
    return result
"""

"""
class Array(np.ndarray):
    def __new__(cls, *args: Any) -> Self:
        if not args:
            arr = cls._get_default_arr()
        else:
            arr = np.array(args, dtype=float).reshape(cls._shape())
        return arr.view(cls)

    #def __iter__(self: Self) -> Iterator[float]:
    #    yield from self._arr.flatten()

    #def __len__(self: Self) -> int:
    #    return 0

    @classmethod
    def _shape(cls) -> Shape:
        return ()

    @classmethod
    def _get_default_arr(cls) -> np.ndarray:
        return np.zeros(cls._shape(), dtype=float)

    #def __getitem__(self: Self, key: Any):
    #    result = super().__getitem__(key)
    #    return self.wrap_array(result)

    #def __array_wrap__(self: Self, array: Any, context=None):
    #    result = super().__array_wrap__(array, context)
    #    return self.wrap_array(result)

    #@staticmethod
    #def wrap_array(array: np.ndarray) -> float | Vec2 | Vec3 | Vec4 | Mat3 | Mat4 | np.ndarray:
    #    shape = array.shape
    #    if shape == (1,):
    #        return float(array)
    #    if shape == (2,):
    #        return Vec2(array)
    #    if shape == (3,):
    #        return Vec3(array)
    #    if shape == (4,):
    #        return Vec4(array)
    #    if shape == (3, 3):
    #        return Mat3(array)
    #    if shape == (4, 4):
    #        return Mat4(array)
    #    return array

    # These direct inheritances just calm IDE down...

    def __pos__(self: Self) -> Self:
        return self.__class__(super().__pos__())

    def __neg__(self: Self) -> Self:
        return self.__class__(super().__neg__())

    def __add__(self: Self, other: Self | float) -> Self:
        return self.__class__(super().__add__(other))

    def __radd__(self: Self, other: Self | float) -> Self:
        return self.__class__(super().__radd__(other))

    def __sub__(self: Self, other: Self | float) -> Self:
        return self.__class__(super().__sub__(other))

    def __rsub__(self: Self, other: Self | float) -> Self:
        return self.__class__(super().__rsub__(other))

    def __mul__(self: Self, other: Self | float) -> Self:
        return self.__class__(super().__mul__(other))

    def __rmul__(self: Self, other: Self | float) -> Self:
        return self.__class__(super().__rmul__(other))

    def __truediv__(self: Self, other: Self | float) -> Self:
        return self.__class__(super().__truediv__(other))

    def __rtruediv__(self: Self, other: Self | float) -> Self:
        return self.__class__(super().__rtruediv__(other))

    def __pow__(self: Self, other: Self | float) -> Self:
        return self.__class__(super().__pow__(other))

    def __rpow__(self: Self, other: Self | float) -> Self:
        return self.__class__(super().__rpow__(other))

    def lerp(self: Self, other: Self, alpha: float) -> Self:
        return (1.0 - alpha) * self + alpha * other


class Vec(Array):
    #def __len__(self: Self) -> int:
    #    return self._size

    @classmethod
    def _size(cls) -> int:
        return 0

    @classmethod
    def _shape(cls) -> Shape:
        return (cls._size(),)

    def norm(self: Self, ord: float | None = None) -> float:
        return float(np.linalg.norm(self, ord))

    def normalize(self: Self, ord: float | None = None) -> Self:
        return self / self.norm(ord)


class Vec2(Vec):
    @classmethod
    def _size(cls) -> int:
        return 2

    def apply_affine(self: Self, mat: Mat3) -> Vec2:
        return Vec2((mat @ np.append(self, 1.0))[:-1])


class Vec3(Vec):
    @classmethod
    def _size(cls) -> int:
        return 3

    def apply(self: Self, mat: Mat3) -> Vec3:
        return Vec3(mat @ self)

    def apply_affine(self: Self, mat: Mat4) -> Vec3:
        return Vec3((mat @ np.append(self, 1.0))[:-1])

    @staticmethod
    def from_matrix_basis(mat: Mat4, n: int) -> Vec3:
        return Vec3(mat[n, :-1])

    @staticmethod
    def from_matrix_position(mat: Mat4) -> Vec3:
        return Vec3(mat[-1, :-1])

    #def scale(self: Self, factor: float) -> Vec3:
    #    return self * factor

    def transform_direction(self: Self, mat: Mat4) -> Vec3:
        return self.apply(Mat3.from_mat4(mat)).normalize()


class Vec4(Vec):
    @classmethod
    def _size(cls) -> int:
        return 4

    def apply(self: Self, mat: Mat4) -> Vec4:
        return Vec4(mat @ self)


class Mat(Array):
    #def __len__(self: Self) -> int:
    #    return self._size ** 2

    @classmethod
    def _size(cls) -> int:
        return 0

    @classmethod
    def _shape(cls) -> Shape:
        return (cls._size(), cls._size())

    @classmethod
    def _get_default_arr(cls) -> np.ndarray:
        return np.identity(cls._size(), dtype=float)

    def inverse(self: Self) -> Self:
        return self.__class__(np.linalg.inv(self))


class Mat3(Mat):
    @classmethod
    def _size(cls) -> int:
        return 3

    def apply(self: Self, mat: Mat3) -> Mat3:
        return Mat3(mat @ self)

    @staticmethod
    def from_mat4(mat: Mat4) -> Mat3:
        return Mat3(mat[:3, :3])


class Mat4(Mat):
    @classmethod
    def _size(cls) -> int:
        return 4

    def apply(self: Self, mat: Mat4) -> Mat4:
        return Mat4(mat @ self)

    @staticmethod
    def from_mat3(mat: Mat3) -> Mat4:
        result = np.identity(4)
        result[:3, :3] = mat
        return Mat4(result)

    @staticmethod
    def from_basis(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Mat4:
        return Mat4.from_mat3(Mat3(x_axis, y_axis, z_axis))

    @staticmethod
    def from_matrix_rotation(mat: Mat4) -> Mat4:
        return Mat4.from_basis(*(
            Vec3.from_matrix_basis(mat, i).normalize()
            for i in range(3)
        ))

    @staticmethod
    def from_shift(vec: Vec3) -> Mat4:
        x, y, z = vec
        return Mat4(
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1
        )

    def shift(self: Self, vec: Vec3) -> Mat4:
        return self.apply(Mat4.from_shift(vec))

    @staticmethod
    def from_scale(vec: Vec3) -> Mat4:
        x, y, z = vec
        return Mat4(
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1
        )

    def scale(self: Self, vec: Vec3) -> Mat4:
        return self.apply(Mat4.from_scale(vec))

    @staticmethod
    def from_rotate(angle: float, axis: Vec3) -> Mat4:
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1.0 - c
        x, y, z = axis
        return Mat4(
            t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0,
            t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0,
            0, 0, 0, 1
        )

    def rotate(self: Self, angle: float, axis: Vec3) -> Mat4:
        return self.apply(Mat4.from_rotate(angle, axis))
"""
