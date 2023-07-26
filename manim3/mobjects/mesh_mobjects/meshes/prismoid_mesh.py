import itertools as it

import numpy as np

from ....constants.custom_typing import (
    NP_2f8,
    NP_3f8,
    NP_x3i4
)
from ....utils.iterables import IterUtils
from ....utils.space import SpaceUtils
from ...shape_mobjects.shapes.shape import Shape
from .mesh import Mesh


class PrismoidMesh(Mesh):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape
    ) -> None:
        position_list: list[NP_3f8] = []
        normal_list: list[NP_3f8] = []
        uv_list: list[NP_2f8] = []
        indices_list: list[NP_x3i4] = []
        index_offset = 0
        for line_string in shape._multi_line_string_._line_strings_:  # TODO
            points = SpaceUtils.decrease_dimension(line_string._points_)
            # TODO: Shall we normalize winding?

            # Assemble side faces.
            triplets: list[tuple[int, NP_2f8, NP_2f8]] = []
            rotation_mat = np.array(((0.0, 1.0), (-1.0, 0.0)))
            for ip, (p_prev, p, p_next) in enumerate(zip(
                np.roll(points, 1, axis=0),
                points,
                np.roll(points, -1, axis=0),
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
                indices_list.append(np.array((i0, i0 + l, i1)) + index_offset)
                indices_list.append(np.array((i1 + l, i1, i0 + l)) + index_offset)
            index_offset += 2 * l

        # Assemble top and bottom faces.
        shape_index, shape_points = shape._triangulation_
        for sign in (1.0, -1.0):
            position_list.extend(SpaceUtils.increase_dimension(shape_points, z_value=sign))
            normal_list.extend(SpaceUtils.increase_dimension(np.zeros_like(shape_points), z_value=sign))
            uv_list.extend(shape_points)
            indices_list.extend(index_offset + shape_index)
            index_offset += len(shape_points)

        super().__init__(
            positions=np.fromiter(position_list, dtype=np.dtype((np.float64, (3,)))),
            normals=np.fromiter(normal_list, dtype=np.dtype((np.float64, (3,)))),
            uvs=np.fromiter(uv_list, dtype=np.dtype((np.float64, (2,)))),
            indices=np.fromiter(indices_list, dtype=np.dtype((np.int32, (3,))))
        )
