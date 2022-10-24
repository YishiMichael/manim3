from __future__ import annotations

import numpy as np
from typing import Any

__all__ = [
    "Vec2",
    "Vec3",
    "Vec4",
    "Mat3",
    "Mat4"
]


Self = Any
Shape = tuple[int, ...]


class Array(np.ndarray):
    def __new__(cls, *args: Any):
        if not args:
            array = cls.default_init()
        else:
            array = np.array(args, dtype=float).reshape(cls.__shape__())
        obj = array.view(cls)
        return obj

    @classmethod
    def __shape__(cls) -> Shape:
        return ()

    @classmethod
    def default_init(cls) -> Self:
        return np.zeros(cls.__shape__(), dtype=float)

    #def __getitem__(self, key):
    #    result = super().__getitem__(key)
    #    return wrap_array(result)

    #def __array_wrap__(self, array, context=None):
    #    result = super().__array_wrap__(array, context)
    #    return wrap_array(result)

    # These direct inheritances just calm IDE down...

    def __pos__(self: Self) -> Self:
        return super().__pos__()

    def __neg__(self: Self) -> Self:
        return super().__neg__()

    def __add__(self: Self, other: Any) -> Self:
        return super().__add__(other)

    def __radd__(self: Self, other: Any) -> Self:
        return super().__radd__(other)

    def __sub__(self: Self, other: Any) -> Self:
        return super().__sub__(other)

    def __rsub__(self: Self, other: Any) -> Self:
        return super().__rsub__(other)

    def __mul__(self: Self, other: Any) -> Self:
        return super().__mul__(other)

    def __rmul__(self: Self, other: Any) -> Self:
        return super().__rmul__(other)

    def __truediv__(self: Self, other: Any) -> Self:
        return super().__truediv__(other)

    def __rtruediv__(self: Self, other: Any) -> Self:
        return super().__rtruediv__(other)

    def __pow__(self: Self, other: Any) -> Self:
        return super().__pow__(other)

    def __rpow__(self: Self, other: Any) -> Self:
        return super().__rpow__(other)

    def __matmul__(self: Self, other: Any) -> Self:
        return super().__matmul__(other)

    def __rmatmul__(self: Self, other: Any) -> Self:
        return super().__rmatmul__(other)

    def norm(self: Self, ord: float | None = None) -> float:
        return float(np.linalg.norm(self, ord))

    def normalize(self: Self, ord: float | None = None) -> Self:
        return self / self.norm(ord)

    def apply(self: Self, mat: Mat) -> Self:
        return mat @ self

    def lerp(self: Self, other: Self, alpha: float) -> Self:
        return (1.0 - alpha) * self + alpha * other


class Vec(Array):
    @classmethod
    def __size__(cls) -> int:
        return 0

    @classmethod
    def __shape__(cls) -> Shape:
        return (cls.__size__(),)

    @classmethod
    def default_init(cls) -> Self:
        return np.zeros(cls.__shape__(), dtype=float)


class Mat(Array):
    @classmethod
    def __size__(cls) -> int:
        return 0

    @classmethod
    def __shape__(cls) -> Shape:
        return (cls.__size__(), cls.__size__())

    @classmethod
    def default_init(cls) -> Self:
        return np.identity(cls.__size__(), dtype=float)

    def inverse(self: Self) -> Self:
        return self.__class__(np.linalg.inv(self))


class Vec2(Vec):
    @classmethod
    def __size__(cls) -> int:
        return 2


class Vec3(Vec):
    @classmethod
    def __size__(cls) -> int:
        return 3

    def apply_affine(self: Self, mat: Mat4) -> Vec3:
        return Vec3((mat @ np.append(self, 1.0))[:-1])

    @classmethod
    def from_matrix_basis(cls, mat: Mat4, n: int) -> Vec3:
        return cls(mat[n, :-1])

    @classmethod
    def from_matrix_position(cls, mat: Mat4) -> Vec3:
        return cls.from_matrix_basis(mat, -1)

    def scale(self: Self, factor: float) -> Vec3:
        return Vec3(self * factor)

    def transform_direction(self: Self, mat: Mat4) -> Vec3:
        return self.apply(Mat3.from_mat4(mat)).normalize()


class Vec4(Vec):
    @classmethod
    def __size__(cls) -> int:
        return 4


class Mat3(Mat):
    @classmethod
    def __size__(cls) -> int:
        return 3

    @classmethod
    def from_mat4(cls, mat: Mat4) -> Mat3:
        return cls(mat[:3, :3])


class Mat4(Mat):
    @classmethod
    def __size__(cls) -> int:
        return 4

    @classmethod
    def from_mat3(cls, mat: Mat3) -> Mat4:
        result = cls()
        result[:3, :3] = mat
        return result

    @classmethod
    def from_basis(cls, x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Mat4:
        return cls.from_mat3(Mat3(x_axis, y_axis, z_axis))

    @classmethod
    def from_matrix_rotation(cls, mat: Mat4) -> Mat4:
        return Mat4.from_basis(*(
            Vec3.from_matrix_basis(mat, i).normalize()
            for i in range(3)
        ))

    @classmethod
    def from_shift(cls, vec: Vec3) -> Mat4:
        x, y, z = vec
        return cls(
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1
        )

    def shift(self: Self, vec: Vec3) -> Mat4:
        return self.apply(self.from_shift(vec))

    @classmethod
    def from_scale(cls, vec: Vec3) -> Mat4:
        x, y, z = vec
        return cls(
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1
        )

    def scale(self: Self, vec: Vec3) -> Mat4:
        return self.apply(self.from_scale(vec))

    @classmethod
    def from_rotate(cls, angle: float, axis: Vec3) -> Mat4:
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1.0 - c
        x, y, z = axis
        return cls(
            t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0,
            t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0,
            0, 0, 0, 1
        )

    def rotate(self: Self, angle: float, axis: Vec3) -> Mat4:
        return self.apply(self.from_rotate(angle, axis))


#SHAPE_TO_CLASS_DICT: dict[Shape, type] = {
#    cls.CLS_SHAPE: cls
#    for cls in (Vec2, Vec3, Vec4, Mat3, Mat4)
#}
#SHAPE_TO_CLASS_DICT[(1,)] = float


#def wrap_array(array: np.ndarray):
#    cls = SHAPE_TO_CLASS_DICT.get(array.shape)
#    if cls is None:
#        return array
#    return cls(*array.flatten())
