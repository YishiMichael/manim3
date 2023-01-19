__all__ = ["PrismoidGeometry"]


import itertools as it

import numpy as np

from ..custom_typing import (
    Vec2T,
    Vec3T
)
from ..geometries.geometry import Geometry
from ..geometries.shape_geometry import ShapeGeometry
from ..utils.shape import Shape


class PrismoidGeometry(Geometry):
    def __init__(self, shape: Shape):
        position_list: list[Vec3T] = []
        normal_list: list[Vec3T] = []
        uv_list: list[Vec2T] = []
        index_list: list[int] = []
        index_offset = 0
        for line_string in shape._multi_line_string_._children_:
            coords = line_string._coords_
            if line_string._signed_area_ < 0:
                # Normalize winding
                coords = coords[::-1]
            # Remove redundant adjacent points to ensure
            # all segments have non-zero lengths
            points: list[Vec2T] = [coords[0]]
            current_point = coords[0]
            for point in coords:
                if np.isclose(np.linalg.norm(point - current_point), 0.0):
                    continue
                current_point = point
                points.append(point)
            if np.isclose(np.linalg.norm(current_point - coords[0]), 0.0):
                points.pop()
            if len(points) <= 1:
                continue

            ip_normal_pairs: list[tuple[int, Vec2T]] = []
            for ip, (p_prev, p, p_next) in enumerate(zip(np.roll(points, 1), points, np.roll(points, -1))):
                v0 = p - p_prev
                v1 = p_next - p
                n0 = np.array((v0[1], -v0[0])) / np.linalg.norm(v0)
                n1 = np.array((v1[1], -v1[0])) / np.linalg.norm(v1)

                angle = abs(np.arccos(np.dot(n0, n1)))
                if angle <= np.pi / 16.0:
                    n_avg = n0 + n1
                    ip_normal_pairs.append((ip, n_avg / np.linalg.norm(n_avg)))
                else:
                    # Notice, vertices shall be duplicated
                    # if its connected faces have significantly different normal vectors
                    ip_normal_pairs.append((ip, n0))
                    ip_normal_pairs.append((ip, n1))

            # Assemble side faces
            for ip, normal in ip_normal_pairs:
                p = points[ip]
                position_list.append(np.append(p, 1.0))
                position_list.append(np.append(p, -1.0))
                normal_list.append(np.append(normal, 0.0))
                normal_list.append(np.append(normal, 0.0))
                uv_list.append(p)
                uv_list.append(p)
            for _, ((i0, (ip0, _)), (i1, (ip1, _))) in zip(
                range(len(ip_normal_pairs)),
                it.pairwise(it.cycle(enumerate(ip_normal_pairs)))
            ):
                if ip0 == ip1:
                    continue
                index_list.extend(
                    index_offset + i
                    for i in (2 * i0, 2 * i0 + 1, 2 * i1, 2 * i1 + 1, 2 * i1, 2 * i0 + 1)
                )
            index_offset += 2 * len(ip_normal_pairs)

        # Assemble top and bottom faces
        shape_index, shape_coords = ShapeGeometry._get_shape_triangulation(shape)
        n_coords = len(shape_coords)
        for sign in (1.0, -1.0):
            position_list.extend(np.insert(shape_coords, 2, sign, axis=1))
            normal_list.extend(np.repeat(np.array((0.0, 0.0, sign))[None], n_coords, axis=0))
            uv_list.extend(shape_coords)
            index_list.extend(index_offset + shape_index)
            index_offset += n_coords

        super().__init__(
            index=np.array(index_list),
            position=np.array(position_list),
            normal=np.array(normal_list),
            uv=np.array(uv_list)
        )
