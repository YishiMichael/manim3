from __future__ import annotations

from dataclasses import dataclass
import itertools as it

import numpy as np
import pathops

from path_utils import triangulate_path
from utils.arrays import Vec2, Vec3, Vec4


@dataclass
class GeometryAttributes:
    tangent: list[Vec3] | None = None
    color: list[Vec3] | list[Vec4] | None = None  # TODO
    position: list[Vec3] | None = None
    normal: list[Vec3] | None = None
    uv: list[Vec2] | None = None


@dataclass
class GeometryMorphAttributes:
    position: list[Vec3] | None = None
    normal: list[Vec3] | None = None
    color: list[Vec3] | None = None  # TODO


class Geometry:
    def __init__(self):
        self.index: list[int] | None = None
        self.attributes: GeometryAttributes = GeometryAttributes()

        self.morphAttributes: GeometryMorphAttributes = GeometryMorphAttributes()
        self.morphTargetsRelative: bool = False

        #self.groups = []

        #self.boundingBox = None
        #self.boundingSphere = None

        #self.drawRange = { start: 0, count: Infinity }

        #self._indices: list[int] = []
        #self._positions: list[Vec3] = []
        #self._normals: list[Vec3] = []
        #self._uvs: list[Vec2] = []

        #self.init_points()


class SphereGeometry(Geometry):
    def __init__(
        self,
        radius: float = 1.0,
        width_segments: int = 32,
        height_segments: int = 16,
        phi_start: float = 0.0,
        phi_length: float = np.pi * 2,
        theta_start: float = 0.0,
        theta_length = np.pi
    ):
        super().__init__()

        width_segments = max(3, width_segments)
        height_segments = max(2, height_segments)

        theta_end = min(theta_start + theta_length, np.pi)

        index = 0
        grid = []

        #vertex = new Vector3()
        #normal = new Vector3()

        # buffers

        indices = []
        vertices = []
        normals = []
        uvs = []

        # generate vertices, normals and uvs

        for iy in range(height_segments + 1):

            vertexRow = []

            v = iy / height_segments

            # special case for the poles

            uOffset = 0

            if iy == 0 and theta_start == 0:

                uOffset = 0.5 / width_segments

            elif iy == height_segments and theta_end == np.pi:

                uOffset = - 0.5 / width_segments

            for ix in range(width_segments + 1):

                u = ix / width_segments

                # vertex

                vertex = Vec3(
                    - radius * np.cos(phi_start + u * phi_length) * np.sin(theta_start + v * theta_length),
                    radius * np.cos(theta_start + v * theta_length),
                    radius * np.sin(phi_start + u * phi_length) * np.sin(theta_start + v * theta_length)
                )

                vertices.append(vertex)

                # normal

                #vertex.normalize()
                normals.append(vertex.normalize())

                # uv

                uvs.append(Vec2(u + uOffset, 1 - v))

                vertexRow.append(index)
                index += 1

            grid.append(vertexRow)

        # indices

        for iy in range(height_segments):

            for ix in range(width_segments):

                a = grid[ iy ][ ix + 1 ]
                b = grid[ iy ][ ix ]
                c = grid[ iy + 1 ][ ix ]
                d = grid[ iy + 1 ][ ix + 1 ]

                if iy != 0 or theta_start > 0:
                    indices.append((a, b, d))
                if iy != height_segments - 1 or theta_end < np.pi:
                    indices.append((b, c, d))

        # build geometry

        #self.index = indices
        self.index = list(it.chain(*indices))
        self.attributes = GeometryAttributes(
            position=vertices,
            normal=normals,
            uv=uvs
        )


class PlaneGeometry(Geometry):
    def __init__(
        self,
        width: float = 1.0,
        height: float = 1.0,
        widthSegments: int = 1,
        heightSegments: int = 1
    ):
        super().__init__()

        width_half = width / 2
        height_half = height / 2

        gridX = widthSegments
        gridY = heightSegments

        gridX1 = gridX + 1
        gridY1 = gridY + 1

        segment_width = width / gridX
        segment_height = height / gridY

        indices = []
        vertices = []
        normals = []
        uvs = []

        for iy in range(gridY1):

            y = iy * segment_height - height_half

            for ix in range(gridX1):

                x = ix * segment_width - width_half

                vertices.append( Vec3(x, - y, 0) )

                normals.append( Vec3(0, 0, 1) )

                uvs.append( Vec2(
                    ix / gridX,
                    1 - ( iy / gridY )
                ) )

        for iy in range(gridY):

            for ix in range(gridX):

                a = ix + gridX1 * iy
                b = ix + gridX1 * ( iy + 1 )
                c = ( ix + 1 ) + gridX1 * ( iy + 1 )
                d = ( ix + 1 ) + gridX1 * iy

                indices.append( (a, b, d) )
                indices.append( (b, c, d) )

        #self.index = indices
        self.index = list(it.chain(*indices))
        self.attributes = GeometryAttributes(
            position=vertices,
            normal=normals,
            uv=uvs
        )


class PathGeometry(Geometry):
    def __init__(self, path: pathops.Path):
        super().__init__()
        vertices, indices = triangulate_path(path)
        self.index = indices
        self.attributes = GeometryAttributes(
            position=[Vec3(vec.x, vec.y, 0.0) for vec in vertices],
            normal=[Vec3(0.0, 0.0, 1.0) for _ in vertices],
            uv=vertices
        )
