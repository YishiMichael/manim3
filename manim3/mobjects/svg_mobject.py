from __future__ import annotations


import math
import pathlib
from typing import (
    Iterator,
    Self
)

import attrs
import numpy as np
import svgelements as se

from ..animatables.shape import Shape
from ..constants.custom_typing import (
    NP_2f8,
    NP_x2f8
)
from .shape_mobjects.shape_mobject import ShapeMobject
from .cached_mobject import (
    CachedMobject,
    CachedMobjectInputs
)
from .image_mobject import ImageMobject


@attrs.frozen(kw_only=True)
class SVGMobjectInputs(CachedMobjectInputs):
    svg_path: pathlib.Path
    svg_text: str


class SVGMobject(CachedMobject[SVGMobjectInputs]):
    __slots__ = ("_shape_mobjects",)

    def __init__(
        self: Self,
        svg_path: str | pathlib.Path,
        *,
        width: float | None = None,
        height: float | None = None,
        scale: float | None = None
    ) -> None:
        svg_path = pathlib.Path(svg_path)
        super().__init__(SVGMobjectInputs(
            svg_path=svg_path,
            svg_text=svg_path.read_text(encoding="utf-8")
        ))

        radii = self.box.get_radii()
        scale_x, scale_y = ImageMobject._get_scale_vector(
            original_width=float(radii[0]),
            original_height=float(radii[1]),
            specified_width=width,
            specified_height=height,
            specified_scale=scale
        )
        self.scale(np.array((
            scale_x,
            scale_y,
            1.0
        )))

    @classmethod
    def _generate_shape_mobjects(
        cls: type[Self],
        inputs: SVGMobjectInputs,
        temp_path: pathlib.Path
    ) -> tuple[ShapeMobject, ...]:
        return cls._generate_shape_mobjects_from_svg(inputs.svg_path)

    @classmethod
    def _generate_shape_mobjects_from_svg(
        cls: type[Self],
        svg_path: pathlib.Path
    ) -> tuple[ShapeMobject, ...]:

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
                        control_positions = np.array(segment)
                        degree = len(control_positions) - 1
                        coordinates_list.extend(
                            np.fromiter((
                                math.comb(degree, k) * pow(1.0 - alpha, degree - k) * pow(alpha, k) * control_position
                                for k, control_position in enumerate(control_positions)
                            ), dtype=np.dtype((np.float64, (2,)))).sum(axis=0)
                            for alpha in np.linspace(0.0, 1.0, 9)[1:]
                        )
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
            min_x, min_y, max_x, max_y = bbox
            transform = se.Matrix(
                1.0,
                0.0,
                0.0,
                -1.0,  # Flip y.
                -(min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0
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
