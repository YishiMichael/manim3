__all__ = ["ShapeStrokeGeometry"]


import itertools as it
from typing import Iterator

import numpy as np

from ..custom_typing import (
    FloatsT,
    Real,
    Vec2sT,
    VertexIndexType
)
from ..geometries.geometry import Geometry
from ..utils.lazy import lazy_property_initializer_writable
from ..utils.shape import (
    LineString,
    Shape
)


class ShapeStrokeGeometry(Geometry):
    def __init__(
        self,
        shape: Shape,
        width: Real,
        round_end: bool = True,
        single_sided: bool = False
    ):
        index, coords, distance = self._get_stroke_triangulation(shape, width, round_end, single_sided)
        position = np.insert(coords, 2, 0.0, axis=1)
        normal = np.repeat(np.array((0.0, 0.0, 1.0))[None], len(position), axis=0)
        super().__init__(
            index=index,
            position=position,
            normal=normal,
            uv=coords
        )
        self._distance_ = distance

    @lazy_property_initializer_writable
    @staticmethod
    def _distance_() -> FloatsT:
        return NotImplemented

    @classmethod
    def _get_stroke_triangulation(
        cls, shape: Shape, width: Real, round_end: bool, single_sided: bool
    ) -> tuple[VertexIndexType, Vec2sT, FloatsT]:
        item_list: list[tuple[VertexIndexType, Vec2sT, FloatsT]] = []
        coords_len = 0
        for line_string in shape._multi_line_string_._children_:
            index, coords, distance = cls._get_line_string_stroke_triangulation(line_string, width, round_end, single_sided)
            item_list.append((index + coords_len, coords, distance))
            coords_len += len(coords)

        if not item_list:
            return np.zeros((0,), dtype=np.uint32), np.zeros((0, 2)), np.zeros((0,))

        index_list, coords_list, distance_list = zip(*item_list)
        return np.concatenate(index_list), np.concatenate(coords_list), np.concatenate(distance_list)

    @classmethod
    def _get_line_string_stroke_triangulation(
        cls, line_string: LineString, width: Real, round_end: bool, single_sided: bool
    ) -> tuple[VertexIndexType, Vec2sT, FloatsT]:
        if np.isclose(width, 0.0):
            return np.zeros((0,), dtype=np.uint32), np.zeros((0, 2)), np.zeros((0,))

        index_list: list[tuple[int, int, int]] = []
        z_list: list[complex] = []
        distance_list: list[float] = []
        vertex_counter = it.count()

        def dump_vertex(zs: list[complex], on_path: bool) -> list[int]:
            distance = 0.0 if on_path else 1.0
            result: list[int] = []
            for z in zs:
                z_list.append(z)
                distance_list.append(distance)
                result.append(next(vertex_counter))
            return result

        path_zs = [complex(x, y) for x, y in line_string._coords_]
        if line_string._kind_ == "linear_ring":
            is_ring = True
            path_zs.pop()
        else:
            is_ring = False
        path_zs_index = dump_vertex(path_zs, True)

        def it_advance(iterator: Iterator) -> Iterator:
            next(iterator)
            return iterator

        def add_sector(
            p: complex, width: Real, angle_0: Real, angle_1: Real,
            ip: int, il1: int | None, il0: int | None
        ) -> None:
            # Make sure `d_angle` is between +-pi
            d_angle = (angle_1 - angle_0 + np.pi) % (2.0 * np.pi) - np.pi
            if d_angle * width >= 0:
                return
            # Approximate the arc
            # A full circle will be approximated as a 32-gon
            n_triangles = int(np.ceil((abs(d_angle) / np.pi) * 16))
            if not n_triangles:
                return
            angle_samples = np.linspace(angle_0, angle_0 + d_angle, n_triangles + 1)
            sector_samples = list(p + np.exp(angle_samples * 1.0j) * width)

            # Eliminate the first end last samples if they're generated already
            l1 = sector_samples.pop(0)
            if il1 is None:
                il1, = dump_vertex([l1], False)
            l0 = sector_samples.pop(-1)
            if il0 is None:
                il0, = dump_vertex([l0], False)
            sector_zs_index = dump_vertex(sector_samples, False)
            for is0, is1 in it.pairwise([il1, *sector_zs_index, il0]):
                index_list.append((ip, is0, is1))

        def add_single_side(width: Real) -> None:
            segments_len = len(path_zs_index) - 1
            if is_ring:
                segments_len += 1
            angles: list[float] = []
            line_zs_index_0: list[int] = []
            line_zs_index_1: list[int] = []
            for _, ((ip0, p0), (ip1, p1)) in zip(
                range(segments_len),
                it.pairwise(it.cycle(zip(path_zs_index, path_zs)))
            ):
                direction = p1 - p0
                angle = float(np.angle(direction * 1.0j))
                angles.append(angle)
                normal = np.exp(angle * 1.0j) * width
                l0 = p0 + normal
                l1 = p1 + normal
                il0, il1 = dump_vertex([l0, l1], False)
                line_zs_index_0.append(il0)
                line_zs_index_1.append(il1)
                index_list.append((ip0, ip1, il0))
                index_list.append((ip1, il1, il0))

            #segments = np.array([
            #    [z0, z1]
            #    for _, (z0, z1) in zip(range(segments_len), )
            #])
            #directions = segments[:, 1] - segments[:, 0]
            #directions = directions / np.linalg.norm(directions, axis=1)[:, None]
            #normals = 1.0j * directions
            #line_zs = (segments + normals[..., None] * width).flatten()

            joints_len = segments_len - 1
            if is_ring:
                joints_len += 1
            # Advance to the 2nd point, as the first point doesn't contain a joint
            for _, (ip, p), (angle_0, angle_1), il1, il0 in zip(
                range(joints_len),
                it_advance(it.cycle(zip(path_zs_index, path_zs))),
                it.pairwise(it.cycle(angles)),
                it.cycle(line_zs_index_1),
                it_advance(it.cycle(line_zs_index_0))
            ):
                add_sector(p, width, angle_0, angle_1, ip, il1, il0)

            if round_end and not is_ring:
                d_theta = np.sign(width) * np.pi / 2.0
                add_sector(path_zs[0], width, angles[0] + d_theta, angles[0], path_zs_index[0], None, line_zs_index_0[0])
                add_sector(path_zs[-1], width, angles[-1], angles[-1] - d_theta, path_zs_index[-1], line_zs_index_1[-1], None)

        add_single_side(width)
        if not single_sided:
            add_single_side(-width)

        index = np.array(index_list).flatten()
        coords = np.array([[z.real, z.imag] for z in z_list])
        distance = np.array(distance_list)
        return index, coords, distance
