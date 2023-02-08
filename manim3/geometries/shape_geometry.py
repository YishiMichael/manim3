__all__ = ["ShapeGeometry"]


from typing import Generator

from mapbox_earcut import triangulate_float32
import numpy as np
import shapely.geometry

from ..custom_typing import (
    Vec2sT,
    VertexIndexType
)
from ..geometries.geometry import (
    Geometry,
    GeometryData
)
from ..utils.lazy import (
    NewData,
    lazy_basedata,
    lazy_property
)
from ..utils.shape import Shape
from ..utils.space import SpaceUtils


class ShapeGeometry(Geometry):
    __slots__ = ()

    def __new__(cls, shape: Shape | None = None):
        instance = super().__new__(cls)
        if shape is not None:
            instance._shape_ = NewData(shape)
        return instance

    @lazy_basedata
    @staticmethod
    def _shape_() -> Shape:
        return Shape()

    @lazy_property
    @staticmethod
    def _geometry_data_(shape: Shape) -> GeometryData:
        index, coords = ShapeGeometry._get_shape_triangulation(shape)
        position = SpaceUtils.increase_dimension(coords)
        normal = np.repeat(np.array((0.0, 0.0, 1.0))[None], len(position), axis=0)
        return GeometryData(
            index=index,
            position=position,
            normal=normal,
            uv=coords
        )

    @classmethod
    def _get_shape_triangulation(cls, shape: Shape) -> tuple[VertexIndexType, Vec2sT]:
        item_list: list[tuple[VertexIndexType, Vec2sT]] = []
        coords_len = 0
        for polygon in cls._get_shapely_polygons(shape._shapely_obj_):
            index, coords = cls._get_polygon_triangulation(polygon)
            item_list.append((index + coords_len, coords))
            coords_len += len(coords)

        if not item_list:
            return np.zeros((0,), dtype=np.uint32), np.zeros((0, 2))

        index_list, coords_list = zip(*item_list)
        return np.concatenate(index_list), np.concatenate(coords_list)

    @classmethod
    def _get_shapely_polygons(cls, shapely_obj: shapely.geometry.base.BaseGeometry) -> Generator[shapely.geometry.Polygon, None, None]:
        if isinstance(shapely_obj, shapely.geometry.Point | shapely.geometry.LineString):
            pass
        elif isinstance(shapely_obj, shapely.geometry.Polygon):
            yield shapely_obj
        elif isinstance(shapely_obj, shapely.geometry.base.BaseMultipartGeometry):
            for child in shapely_obj.geoms:
                yield from cls._get_shapely_polygons(child)
        else:
            raise TypeError

    @classmethod
    def _get_polygon_triangulation(cls, polygon: shapely.geometry.Polygon) -> tuple[VertexIndexType, Vec2sT]:
        ring_coords_list = [
            np.array(boundary.coords, dtype=np.float32)
            for boundary in [polygon.exterior, *polygon.interiors]
        ]
        coords = np.concatenate(ring_coords_list)
        if not len(coords):
            return np.zeros((0,), dtype=np.uint32), np.zeros((0, 2))

        ring_ends = np.cumsum([len(ring_coords) for ring_coords in ring_coords_list], dtype=np.uint32)
        return triangulate_float32(coords, ring_ends), coords
