from __future__ import annotations


import pathlib
from typing import (
    Iterator,
    Self,
    TypedDict
)

import attrs
import numpy as np
import svgelements as se

from ..animatables.shape import Shape
from ..constants.custom_typing import (
    NP_2f8,
    NP_x2f8
)
from ..utils.color_utils import ColorUtils
from ..utils.space_utils import SpaceUtils
from .shape_mobjects.shape_mobject import ShapeMobject
from .mobject_io import (
    MobjectIO,
    MobjectInput,
    MobjectJSON,
    MobjectOutput
)


@attrs.frozen(kw_only=True)
class SVGMobjectInput(MobjectInput):
    svg_path: pathlib.Path
    svg_text: str
    width: float | None
    height: float | None
    frame_scale: float | None


@attrs.frozen(kw_only=True)
class SVGMobjectOutput(MobjectOutput):
    shape_mobjects: tuple[ShapeMobject, ...]


class ShapeMobjectJSON(TypedDict):
    coordinates: tuple[float, ...]  # flattened
    counts: tuple[int, ...]
    color: str
    opacity: float


class SVGMobjectJSON(MobjectJSON):
    shape_mobjects: tuple[ShapeMobjectJSON, ...]


class SVGMobjectIO(MobjectIO[SVGMobjectInput, SVGMobjectOutput, SVGMobjectJSON]):
    __slots__ = ()

    @property
    @classmethod
    def _dir_name(
        cls: type[Self]
    ) -> str:
        return "svg_mobject"

    @classmethod
    def generate(
        cls: type[Self],
        input_data: SVGMobjectInput,
        temp_path: pathlib.Path
    ) -> SVGMobjectOutput:
        return SVGMobjectOutput(
            shape_mobjects=cls._get_shape_mobjects_from_svg_path(
                svg_path=input_data.svg_path,
                width=input_data.width,
                height=input_data.height,
                frame_scale=input_data.frame_scale
            )
        )

    @classmethod
    def dump_json(
        cls: type[Self],
        output_data: SVGMobjectOutput
    ) -> SVGMobjectJSON:
        return SVGMobjectJSON(
            shape_mobjects=tuple(
                cls._shape_mobject_to_json(shape_mobject)
                for shape_mobject in output_data.shape_mobjects
            )
        )

    @classmethod
    def load_json(
        cls: type[Self],
        json_data: SVGMobjectJSON
    ) -> SVGMobjectOutput:
        return SVGMobjectOutput(
            shape_mobjects=tuple(
                cls._json_to_shape_mobject(shape_mobject_json)
                for shape_mobject_json in json_data["shape_mobjects"]
            )
        )

    @classmethod
    def _get_shape_mobjects_from_svg_path(
        cls: type[Self],
        svg_path: str | pathlib.Path,
        *,
        width: float | None = None,
        height: float | None = None,
        frame_scale: float | None = None
    ) -> tuple[ShapeMobject, ...]:

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
            scale_x, scale_y = SpaceUtils.get_frame_scale_vector(
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
        ) -> Iterator[NP_x2f8]:
            se_path = se.Path(se_shape.segments(transformed=True))
            se_path.approximate_arcs_with_cubics()
            coordinates_list: list[NP_2f8] = []
            is_ring: bool = False
            coordinates_dtype = np.dtype((np.float64, (2,)))
            for segment in se_path.segments(transformed=True):
                match segment:
                    case se.Move(end=end):
                        if is_ring:
                            yield np.fromiter(coordinates_list, dtype=coordinates_dtype)
                        coordinates_list = [np.array(end)]
                        is_ring = False
                    case se.Close():
                        is_ring = True
                    case se.Line(end=end):
                        coordinates_list.append(np.array(end))
                    case se.QuadraticBezier() | se.CubicBezier():
                        # Approximate the bezier curve with a polyline.
                        curve = SpaceUtils.bezier(np.array(segment))
                        coordinates_list.extend(curve(np.linspace(0.0, 1.0, 9)[1:]))
                    case _:
                        raise ValueError(f"Cannot handle path segment type: {type(segment)}")
            if is_ring:
                yield np.fromiter(coordinates_list, dtype=coordinates_dtype)

        def iter_shape_mobjects_from_svg(
            svg: se.SVG
        ) -> Iterator[ShapeMobject]:
            bbox: tuple[float, float, float, float] | None = svg.bbox()
            if bbox is None:
                return

            # Handle transform before constructing `Shape`s,
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
                shape = Shape().as_paths(iter_paths_from_se_shape(se_shape * transform))
                if not len(shape._coordinates_):
                    # Filter out empty shapes.
                    continue
                style_dict = {}
                if se_shape.fill is not None:
                    if (color := se_shape.fill.hexrgb) is not None:
                        style_dict["color"] = color
                    if (opacity := se_shape.fill.opacity) is not None:
                        style_dict["opacity"] = opacity
                yield ShapeMobject(shape).set(**style_dict)

        svg: se.SVG = se.SVG.parse(svg_path)
        return tuple(iter_shape_mobjects_from_svg(svg))

    @classmethod
    def _shape_mobject_to_json(
        cls: type[Self],
        shape_mobject: ShapeMobject
    ) -> ShapeMobjectJSON:
        shape = shape_mobject._shape_
        return ShapeMobjectJSON(
            coordinates=tuple(round(float(value), 6) for value in shape._coordinates_.flatten()),
            counts=tuple(int(value) for value in shape._counts_),
            color=ColorUtils.color_to_hex(shape_mobject._color_._array_),
            opacity=round(float(shape_mobject._opacity_._array_), 6)
        )

    @classmethod
    def _json_to_shape_mobject(
        cls: type[Self],
        shape_mobject_json: ShapeMobjectJSON
    ) -> ShapeMobject:
        coordinates = shape_mobject_json["coordinates"]
        counts = shape_mobject_json["counts"]
        color = shape_mobject_json["color"]
        opacity = shape_mobject_json["opacity"]

        return ShapeMobject(Shape(
            coordinates=np.fromiter(coordinates, dtype=np.float64).reshape(-1, 2),
            counts=np.fromiter(counts, dtype=np.int32)
        )).set(
            color=color,
            opacity=opacity
        )


class SVGMobject(ShapeMobject):
    __slots__ = ("_shape_mobjects",)

    def __init__(
        self: Self,
        svg_path: str | pathlib.Path,
        *,
        width: float | None = None,
        height: float | None = None,
        frame_scale: float | None = None
    ) -> None:
        super().__init__()
        svg_path = pathlib.Path(svg_path)
        output_data = SVGMobjectIO.get(SVGMobjectInput(
            svg_path=svg_path,
            svg_text=svg_path.read_text(encoding="utf-8"),
            width=width,
            height=height,
            frame_scale=frame_scale
        ))

        shape_mobjects = output_data.shape_mobjects
        self._shape_mobjects: tuple[ShapeMobject, ...] = shape_mobjects
        self.add(*shape_mobjects)
