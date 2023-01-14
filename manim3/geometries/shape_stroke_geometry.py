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

        def dump_result() -> tuple[VertexIndexType, Vec2sT, FloatsT]:
            if not index_list:
                return np.zeros((0,), dtype=np.uint32), np.zeros((0, 2)), np.zeros((0,))

            index = np.array(index_list).flatten()
            coords = np.array([[z.real, z.imag] for z in z_list])
            distance = np.array(distance_list)
            return index, coords, distance

        def it_advance(iterator: Iterator) -> Iterator:
            next(iterator)
            return iterator

        def add_sector(
            p: complex, width: Real, angle_0: Real, angle_1: Real,
            ip: int, il1: int, il0: int
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
            sector_zs_index = dump_vertex(sector_samples[1:-1], False)
            for is0, is1 in it.pairwise([il1, *sector_zs_index, il0]):
                index_list.append((ip, is0, is1))

        if np.isclose(width, 0.0):
            return dump_result()

        if line_string._kind_ == "point":
            if not round_end or single_sided:
                return dump_result()
            x, y = line_string._coords_[0]
            p = complex(x, y)
            ip, = dump_vertex([p], True)
            l0 = p + width
            l1 = p - width
            il0, il1 = dump_vertex([l0, l1], False)
            add_sector(p, width, 0.0, np.pi, ip, il0, il1)
            add_sector(p, width, -np.pi, 0.0, ip, il1, il0)
            return dump_result()

        path_zs = [complex(x, y) for x, y in line_string._coords_]
        if line_string._kind_ == "linear_ring":
            is_ring = True
            path_zs.pop()
        else:
            is_ring = False
        path_zs_index = dump_vertex(path_zs, True)

        segments_len = len(path_zs_index) - 1
        if is_ring:
            segments_len += 1
        joints_len = segments_len - 1
        if is_ring:
            joints_len += 1
        angles = [
            float(np.angle((p1 - p0) * 1.0j))
            for _, (p0, p1) in zip(
                range(segments_len),
                it.pairwise(it.cycle(path_zs))
            )
        ]

        def add_single_side(width: Real) -> tuple[list[int], list[int]]:
            line_zs_index_0: list[int] = []
            line_zs_index_1: list[int] = []
            for _, ((ip0, p0), (ip1, p1)), angle in zip(
                range(segments_len),
                it.pairwise(it.cycle(zip(path_zs_index, path_zs))),
                it.cycle(angles)
            ):
                normal = np.exp(angle * 1.0j) * width
                l0 = p0 + normal
                l1 = p1 + normal
                il0, il1 = dump_vertex([l0, l1], False)
                line_zs_index_0.append(il0)
                line_zs_index_1.append(il1)
                index_list.append((ip0, ip1, il0))
                index_list.append((ip1, il1, il0))

            # Advance to the 2nd point, as the first point doesn't contain a joint
            for _, (ip, p), (angle_0, angle_1), il1, il0 in zip(
                range(joints_len),
                it_advance(it.cycle(zip(path_zs_index, path_zs))),
                it.pairwise(it.cycle(angles)),
                it.cycle(line_zs_index_1),
                it_advance(it.cycle(line_zs_index_0))
            ):
                add_sector(p, width, angle_0, angle_1, ip, il1, il0)

            return line_zs_index_0, line_zs_index_1

        line_zs_index_00, line_zs_index_10 = add_single_side(width)
        if not single_sided:
            line_zs_index_01, line_zs_index_11 = add_single_side(-width)
            if round_end and not is_ring:
                d_theta = np.sign(width) * np.pi
                add_sector(
                    path_zs[-1], width, angles[-1], angles[-1] - d_theta,
                    path_zs_index[-1], line_zs_index_10[-1], line_zs_index_11[-1]
                )
                add_sector(
                    path_zs[0], width, angles[0] + d_theta, angles[0],
                    path_zs_index[0], line_zs_index_01[0], line_zs_index_00[0]
                )

        return dump_result()
