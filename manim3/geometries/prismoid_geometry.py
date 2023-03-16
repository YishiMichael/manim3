__all__ = ["PrismoidGeometry"]


import itertools as it

import numpy as np

from ..custom_typing import (
    Vec2T,
    Vec3T
)
from ..geometries.geometry import GeometryData
from ..geometries.shape_geometry import ShapeGeometry
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..utils.shape import Shape
from ..utils.space import SpaceUtils


class PrismoidGeometry(ShapeGeometry):
    __slots__ = ()

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _geometry_data_(
        cls,
        _shape_: Shape
    ) -> GeometryData:
        position_list: list[Vec3T] = []
        normal_list: list[Vec3T] = []
        uv_list: list[Vec2T] = []
        index_list: list[int] = []
        index_offset = 0
        for line_string in _shape_._multi_line_string_._line_strings_:
            coords = SpaceUtils.decrease_dimension(line_string._coords_.value)
            # Remove redundant adjacent points to ensure
            # all segments have non-zero lengths.
            # TODO: Shall we normalize winding?
            points: list[Vec2T] = [coords[0]]
            current_point = coords[0]
            for point in coords:
                if np.isclose(SpaceUtils.norm(point - current_point), 0.0):
                    continue
                current_point = point
                points.append(point)
            if np.isclose(SpaceUtils.norm(current_point - coords[0]), 0.0):
                points.pop()
            if len(points) <= 1:
                continue

            # Assemble side faces.
            ip_normal_pairs: list[tuple[int, Vec2T]] = []
            rotation_mat = np.array(((0.0, 1.0), (-1.0, 0.0)))
            for ip, (p_prev, p, p_next) in enumerate(zip(np.roll(points, 1, axis=0), points, np.roll(points, -1, axis=0))):
                n0 = rotation_mat @ SpaceUtils.normalize(p - p_prev)
                n1 = rotation_mat @ SpaceUtils.normalize(p_next - p)

                angle = abs(np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0)))
                if angle <= np.pi / 16.0:
                    n_avg = SpaceUtils.normalize(n0 + n1)
                    ip_normal_pairs.append((ip, n_avg))
                else:
                    # Vertices shall be duplicated if its connected faces have significantly different normal vectors.
                    ip_normal_pairs.append((ip, n0))
                    ip_normal_pairs.append((ip, n1))

            duplicated_points = np.array([points[ip] for ip, _ in ip_normal_pairs])
            normals = np.array([normal for _, normal in ip_normal_pairs])
            position_list.extend(SpaceUtils.increase_dimension(duplicated_points, z_value=1.0))
            position_list.extend(SpaceUtils.increase_dimension(duplicated_points, z_value=-1.0))
            normal_list.extend(SpaceUtils.increase_dimension(normals))
            normal_list.extend(SpaceUtils.increase_dimension(normals))
            uv_list.extend(duplicated_points)
            uv_list.extend(duplicated_points)

            l = len(ip_normal_pairs)
            for (i0, (ip0, _)), (i1, (ip1, _)) in it.islice(it.pairwise(it.cycle(enumerate(ip_normal_pairs))), l):
                if ip0 == ip1:
                    continue
                index_list.extend(
                    index_offset + i
                    for i in (i0, i0 + l, i1, i1 + l, i1, i0 + l)
                )
            index_offset += 2 * l

        # Assemble top and bottom faces.
        shape_index, shape_coords = _shape_._triangulation_.value
        for sign in (1.0, -1.0):
            position_list.extend(SpaceUtils.increase_dimension(shape_coords, z_value=sign))
            normal_list.extend(SpaceUtils.increase_dimension(np.zeros_like(shape_coords), z_value=sign))
            uv_list.extend(shape_coords)
            index_list.extend(index_offset + shape_index)
            index_offset += len(shape_coords)

        return GeometryData(
            index=np.array(index_list),
            position=np.array(position_list),
            normal=np.array(normal_list),
            uv=np.array(uv_list)
        )
