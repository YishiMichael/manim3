import itertools as it

import numpy as np

from ..custom_typing import (
    NP_2f8,
    NP_3f8
)
from ..shape.shape import Shape
from ..utils.iterables import IterUtils
from ..utils.space import SpaceUtils
from .geometry import Geometry


class PrismoidGeometry(Geometry):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape
    ) -> None:
        position_list: list[NP_3f8] = []
        normal_list: list[NP_3f8] = []
        uv_list: list[NP_2f8] = []
        index_list: list[int] = []
        index_offset = 0
        for line_string in shape._multi_line_string_._line_strings_:
            points = SpaceUtils.decrease_dimension(line_string._points_)
            # Remove redundant adjacent points to ensure
            # all segments have non-zero lengths.
            # TODO: Shall we normalize winding?
            points_list: list[NP_2f8] = [points[0]]
            current_point = points[0]
            for point in points:
                if np.isclose(SpaceUtils.norm(point - current_point), 0.0):
                    continue
                current_point = point
                points_list.append(point)
            if np.isclose(SpaceUtils.norm(current_point - points[0]), 0.0):
                points_list.pop()
            if len(points_list) <= 1:
                continue

            # Assemble side faces.
            triplets: list[tuple[int, NP_2f8, NP_2f8]] = []
            rotation_mat = np.array(((0.0, 1.0), (-1.0, 0.0)))
            for ip, (p_prev, p, p_next) in enumerate(zip(
                np.roll(points_list, 1, axis=0),
                points_list,
                np.roll(points_list, -1, axis=0),
                strict=True
            )):
                n0 = rotation_mat @ SpaceUtils.normalize(p - p_prev)
                n1 = rotation_mat @ SpaceUtils.normalize(p_next - p)

                angle = abs(np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0)))
                if angle <= np.pi / 16.0:
                    n_avg = SpaceUtils.normalize(n0 + n1)
                    triplets.append((ip, p, n_avg))
                else:
                    # Vertices shall be duplicated if its connected faces have significantly different normal vectors.
                    triplets.append((ip, p, n0))
                    triplets.append((ip, p, n1))

            ip_iterator, p_iterator, normal_iterator = IterUtils.unzip_triplets(triplets)
            duplicated_points = np.array(list(p_iterator))
            normals = np.array(list(normal_iterator))
            position_list.extend(SpaceUtils.increase_dimension(duplicated_points, z_value=1.0))
            position_list.extend(SpaceUtils.increase_dimension(duplicated_points, z_value=-1.0))
            normal_list.extend(SpaceUtils.increase_dimension(normals))
            normal_list.extend(SpaceUtils.increase_dimension(normals))
            uv_list.extend(duplicated_points)
            uv_list.extend(duplicated_points)

            ips = list(ip_iterator)
            l = len(ips)
            for (i0, ip0), (i1, ip1) in it.islice(it.pairwise(it.cycle(enumerate(ips))), l):
                if ip0 == ip1:
                    continue
                index_list.extend(
                    index_offset + i
                    for i in (i0, i0 + l, i1, i1 + l, i1, i0 + l)
                )
            index_offset += 2 * l

        # Assemble top and bottom faces.
        shape_index, shape_points = shape._triangulation_
        for sign in (1.0, -1.0):
            position_list.extend(SpaceUtils.increase_dimension(shape_points, z_value=sign))
            normal_list.extend(SpaceUtils.increase_dimension(np.zeros_like(shape_points), z_value=sign))
            uv_list.extend(shape_points)
            index_list.extend(index_offset + shape_index)
            index_offset += len(shape_points)

        super().__init__()
        self._index_ = np.array(index_list)
        self._position_ = np.array(position_list)
        self._normal_ = np.array(normal_list)
        self._uv_ = np.array(uv_list)
