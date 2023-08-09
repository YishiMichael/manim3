from dataclasses import dataclass
import pathlib
from typing import (
    Iterator,
    TypedDict
)

import numpy as np
import svgelements as se
from scipy.interpolate import BSpline

from ..constants.custom_typing import (
    NP_2f8,
    NP_x2f8
)
from ..utils.color_utils import ColorUtils
from ..utils.space_utils import SpaceUtils
from .graph_mobjects.graphs.graph import Graph
from .mobject.mobject_io import MobjectIO
from .shape_mobjects.shapes.shape import Shape
from .shape_mobjects.shape_mobject import ShapeMobject


#class BezierCurve(BSpline):
#    __slots__ = ()
#
#    def __init__(
#        self,
#        control_positions: NP_x2f8
#    ) -> None:
#        degree = len(control_positions) - 1
#        assert degree >= 0
#        super().__init__(
#            t=np.append(np.zeros(degree + 1), np.ones(degree + 1)),
#            c=control_positions,
#            k=degree
#        )

    #@overload
    #def gamma(
    #    self,
    #    sample: float
    #) -> NP_2f8: ...

    #@overload
    #def gamma(
    #    self,
    #    sample: NP_xf8
    #) -> NP_x2f8: ...

    #def gamma(
    #    self,
    #    sample: float | NP_xf8
    #) -> NP_2f8 | NP_x2f8:
    #    return self.__call__(sample)

    #def get_sample_positions(self) -> NP_x2f8:
    #    return self(np.linspace(0.0, 1.0, 9))

        #def smoothen_samples(
        #    gamma: Callable[[NP_xf8], NP_x2f8],
        #    samples: NP_xf8,
        #    bisect_depth: int
        #) -> NP_xf8:
        #    # Bisect a segment if one of its endpositions has a turning angle above the threshold.
        #    # Bisect for no more than 4 times, so each curve will be split into no more than 16 segments.
        #    if bisect_depth == 4:
        #        return samples
        #    positions = gamma(samples)
        #    directions = SpaceUtils.normalize(np.diff(positions, axis=0))
        #    angle_cosines = (directions[1:] * directions[:-1]).sum(axis=1)
        #    large_angle_indices = np.flatnonzero(angle_cosines < np.cos(np.pi / 16.0))
        #    if not len(large_angle_indices):
        #        return samples
        #    insertion_indices = np.unique(np.concatenate(
        #        (large_angle_indices, large_angle_indices + 1)
        #    ))
        #    new_samples = np.insert(
        #        samples,
        #        insertion_indices + 1,
        #        (samples[insertion_indices] + samples[insertion_indices + 1]) / 2.0
        #    )
        #    return smoothen_samples(gamma, new_samples, bisect_depth + 1)

        #if self._degree <= 1:
        #    start_position = self.gamma(0.0)
        #    stop_position = self.gamma(1.0)
        #    if np.isclose(SpaceUtils.norm(stop_position - start_position), 0.0):
        #        return np.array((start_position,))
        #    return np.array((start_position, stop_position))
        #samples = smoothen_samples(self.gamma, np.linspace(0.0, 1.0, 3), 1)
        #return self.gamma(samples)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class SVGMobjectInputData:
    svg_path: str | pathlib.Path
    svg_content: str
    width: float | None
    height: float | None
    frame_scale: float | None


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class SVGMobjectOutputData:
    shape_mobjects: list[ShapeMobject]


class ShapeMobjectJSON(TypedDict):
    positions: list[float]  # flattened
    edges: list[int]  # flattened
    color: str
    opacity: float


class SVGMobjectJSON(TypedDict):
    shape_mobjects: list[ShapeMobjectJSON]


class SVGMobjectIO(MobjectIO[SVGMobjectInputData, SVGMobjectOutputData, SVGMobjectJSON]):
    __slots__ = ()

    #_dir_name: ClassVar[str] = "svg_mobject"

    @property
    @classmethod
    def _dir_name(cls) -> str:
        return "svg_mobject"

    @classmethod
    def generate(
        cls,
        input_data: SVGMobjectInputData,
        temp_path: pathlib.Path
    ) -> SVGMobjectOutputData:
        return SVGMobjectOutputData(
            shape_mobjects=list(cls.iter_shape_mobject_from_svg(
                svg_path=input_data.svg_path,
                width=input_data.width,
                height=input_data.height,
                frame_scale=input_data.frame_scale
            ))
        )

    @classmethod
    def dump_json(
        cls,
        output_data: SVGMobjectOutputData
    ) -> SVGMobjectJSON:
        return SVGMobjectJSON(
            shape_mobjects=[
                cls.shape_mobject_to_json(shape_mobject)
                for shape_mobject in output_data.shape_mobjects
            ]
        )

    @classmethod
    def load_json(
        cls,
        json_data: SVGMobjectJSON
    ) -> SVGMobjectOutputData:
        return SVGMobjectOutputData(
            shape_mobjects=[
                cls.json_to_shape_mobject(shape_mobject_json)
                for shape_mobject_json in json_data["shape_mobjects"]
            ]
        )

    @classmethod
    def iter_shape_mobject_from_svg(
        cls,
        svg_path: str | pathlib.Path,
        *,
        width: float | None = None,
        height: float | None = None,
        frame_scale: float | None = None
    ) -> Iterator[ShapeMobject]:

        def perspective(
            origin_x: float,
            origin_y: float,
            radius_x: float,
            radius_y: float
        ) -> se.Matrix:
            # `(origin=(0.0, 0.0), radius=(1.0, 1.0))` ->
            # `(origin=(origin_x, origin_y), radius=(radius_x, radius_y))`
            return se.Matrix(
                radius_x,
                0.0,
                0.0,
                radius_y,
                origin_x,
                origin_y
            )

        def get_transform(
            bbox: tuple[float, float, float, float],
            width: float | None,
            height: float | None,
            frame_scale: float | None
        ) -> se.Matrix:

            min_x, min_y, max_x, max_y = bbox
            origin_x = (min_x + max_x) / 2.0
            origin_y = (min_y + max_y) / 2.0
            radius_x = (max_x - min_x) / 2.0
            radius_y = (max_y - min_y) / 2.0
            transform = ~perspective(
                origin_x=origin_x,
                origin_y=origin_y,
                radius_x=radius_x,
                radius_y=radius_y
            )
            scale_x, scale_y = SpaceUtils._get_frame_scale_vector(
                original_width=radius_x * 2.0,
                original_height=radius_y * 2.0,
                specified_width=width,
                specified_height=height,
                specified_frame_scale=frame_scale
            )
            transform *= perspective(
                origin_x=0.0,
                origin_y=0.0,
                radius_x=scale_x * radius_x,
                radius_y=-scale_y * radius_y  # Flip y.
            )
            return transform

        def iter_paths_from_se_shape(
            se_shape: se.Shape
        ) -> Iterator[tuple[NP_x2f8, bool]]:
            se_path = se.Path(se_shape.segments(transformed=True))
            se_path.approximate_arcs_with_cubics()
            positions_list: list[NP_2f8] = []
            is_ring: bool = False
            positions_dtype = np.dtype((np.float64, (2,)))
            for segment in se_path.segments(transformed=True):
                match segment:
                    case se.Move(end=end):
                        yield np.fromiter(positions_list, dtype=positions_dtype), is_ring
                        positions_list = [np.array(end)]
                        is_ring = False
                    case se.Close():
                        is_ring = True
                    case se.Line(end=end):
                        positions_list.append(np.array(end))
                    case se.QuadraticBezier() | se.CubicBezier():
                        # Approximate the bezier curve with a polyline.
                        control_positions = np.array(segment)
                        degree = len(control_positions) - 1
                        curve = BSpline(
                            t=np.append(np.zeros(degree + 1), np.ones(degree + 1)),
                            c=control_positions,
                            k=degree
                        )
                        positions_list.extend(curve(np.linspace(0.0, 1.0, 9)[1:]))
                    case _:
                        raise ValueError(f"Cannot handle path segment type: {type(segment)}")
            yield np.fromiter(positions_list, dtype=positions_dtype), is_ring

        svg: se.SVG = se.SVG.parse(svg_path)
        bbox: tuple[float, float, float, float] | None = svg.bbox()
        if bbox is None:
            return []

        # Handle transform before constructing `Shape`s
        # so that the center of the entire shape falls on the origin.
        transform = get_transform(
            bbox=bbox,
            width=width,
            height=height,
            frame_scale=frame_scale
        )

        for se_shape in svg.elements():
            if not isinstance(se_shape, se.Shape):
                continue
            shape = Shape.from_paths(iter_paths_from_se_shape(se_shape * transform))
            if not len(shape._graph_._edges_):
                # Filter out empty shapes.
                continue
            yield ShapeMobject(shape).set_style(
                color=se_shape.fill.hexrgb if se_shape.fill is not None else None,
                opacity=se_shape.fill.opacity if se_shape.fill is not None else None
            )
            #yield ShapeMobjectJSON(
            #    positions=[round(float(value), 6) for value in graph._positions_[:, :2].flatten()],
            #    edges=[int(value) for value in graph._edges_.flatten()],
            #    color=se_shape.fill.hexrgb if se_shape.fill is not None else None,
            #    opacity=se_shape.fill.opacity if se_shape.fill is not None else None
            #)

    @classmethod
    def shape_mobject_to_json(
        cls,
        shape_mobject: ShapeMobject
    ) -> ShapeMobjectJSON:
        graph = shape_mobject._shape_._graph_
        return ShapeMobjectJSON(
            positions=[round(float(value), 6) for value in graph._positions_[:, :2].flatten()],
            edges=[int(value) for value in graph._edges_.flatten()],
            color=ColorUtils.color_to_hex(shape_mobject._color_),
            opacity=round(float(shape_mobject._opacity_), 6)
        )

    @classmethod
    def json_to_shape_mobject(
        cls,
        shape_mobject_json: ShapeMobjectJSON
    ) -> ShapeMobject:
        positions = shape_mobject_json["positions"]
        edges = shape_mobject_json["edges"]
        color = shape_mobject_json["color"]
        opacity = shape_mobject_json["opacity"]

        return ShapeMobject(Shape(Graph(
            positions=SpaceUtils.increase_dimension(
                np.fromiter(positions, dtype=np.float64).reshape(-1, 2)
            ),
            edges=np.fromiter(edges, dtype=np.int32).reshape(-1, 2)
        ))).set_style(
            color=color,
            opacity=opacity
        )


class SVGMobject(ShapeMobject):
    __slots__ = ("_shape_mobjects")

    def __init__(
        self,
        svg_path: str | pathlib.Path,
        *,
        width: float | None = None,
        height: float | None = None,
        frame_scale: float | None = None
    ) -> None:
        super().__init__()
        with open(svg_path, encoding="utf-8") as svg_file:
            svg_content = svg_file.read()
        output_data = SVGMobjectIO.get(SVGMobjectInputData(
            svg_path=svg_path,
            svg_content=svg_content,
            width=width,
            height=height,
            frame_scale=frame_scale
        ))

        shape_mobjects = output_data.shape_mobjects
        self._shape_mobjects: list[ShapeMobject] = shape_mobjects
        self.add(*shape_mobjects)
