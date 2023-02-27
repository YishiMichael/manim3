__all__ = ["PrismoidGeometry"]


import itertools as it

import numpy as np

from ..custom_typing import (
    Vec2T,
    Vec3T
)
from ..geometries.geometry import (
    Geometry,
    GeometryData
)
from ..geometries.shape_geometry import ShapeGeometry
from ..utils.lazy import (
    LazyWrapper,
    lazy_object,
    lazy_property
)
from ..utils.shape import Shape
from ..utils.space import SpaceUtils


class PrismoidGeometry(Geometry):
    __slots__ = ()

    @lazy_object
    @classmethod
    def _shape_(cls) -> Shape:
        return Shape()

    @lazy_property
    @classmethod
    def _geometry_data_(
        cls,
        _shape_: Shape
    ) -> LazyWrapper[GeometryData]:
        position_list: list[Vec3T] = []
        normal_list: list[Vec3T] = []
        uv_list: list[Vec2T] = []
        index_list: list[int] = []
        index_offset = 0
        for line_string in _shape_._multi_line_string_._children_:
            coords = line_string._coords_.value
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

            ip_normal_pairs: list[tuple[int, Vec2T]] = []
            rotation_mat = np.array([[0.0, 1.0], [-1.0, 0.0]])
            for ip, (p_prev, p, p_next) in enumerate(zip(np.roll(points, 1), points, np.roll(points, -1))):
                n0 = rotation_mat @ SpaceUtils.normalize(p - p_prev)
                n1 = rotation_mat @ SpaceUtils.normalize(p_next - p)

                angle = abs(np.arccos(np.dot(n0, n1)))
                if angle <= np.pi / 16.0:
                    n_avg = n0 + n1
                    ip_normal_pairs.append((ip, SpaceUtils.normalize(n_avg)))
                else:
                    # Vertices shall be duplicated
                    # if its connected faces have significantly different normal vectors.
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
            for (i0, (ip0, _)), (i1, (ip1, _)) in it.islice(it.pairwise(it.cycle(enumerate(ip_normal_pairs))), len(ip_normal_pairs)):
                if ip0 == ip1:
                    continue
                index_list.extend(
                    index_offset + i
                    for i in (2 * i0, 2 * i0 + 1, 2 * i1, 2 * i1 + 1, 2 * i1, 2 * i0 + 1)
                )
            index_offset += 2 * len(ip_normal_pairs)

        # Assemble top and bottom faces
        shape_index, shape_coords = ShapeGeometry._get_shape_triangulation(_shape_)  # TODO
        n_coords = len(shape_coords)
        for sign in (1.0, -1.0):
            position_list.extend(SpaceUtils.increase_dimension(shape_coords, sign))
            normal_list.extend(np.repeat(np.array((0.0, 0.0, sign))[None], n_coords, axis=0))
            uv_list.extend(shape_coords)
            index_list.extend(index_offset + shape_index)
            index_offset += n_coords

        return LazyWrapper(GeometryData(
            index=np.array(index_list),
            position=np.array(position_list),
            normal=np.array(normal_list),
            uv=np.array(uv_list)
        ))
