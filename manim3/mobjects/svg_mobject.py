from __future__ import annotations


import math
import pathlib
from typing import (
    Callable,
    Iterator,
    Self
)

import attrs
import numpy as np
import skia
import svgelements as se

from ..animatables.shape import Shape
from ..constants.custom_typing import NP_2f8
from ..toplevel.toplevel import Toplevel
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
        if isinstance(svg_path, str):
            for image_dir in Toplevel._get_config().image_search_dirs:
                if image_dir.joinpath(svg_path).exists():
                    svg_path = image_dir.joinpath(svg_path)
                    break
            else:
                raise FileNotFoundError(svg_path)
        else:
            if not svg_path.exists():
                raise FileNotFoundError(svg_path)

        super().__init__(SVGMobjectInputs(
            svg_path=svg_path,
            svg_text=svg_path.read_text(encoding="utf-8")
        ))

        size = self.box.get_size()
        scale_x, scale_y = ImageMobject._get_scale_vector(
            original_width=float(size[0]),
            original_height=float(size[1]),
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

        def iter_skia_paths_from_se_shape(
            se_shape: se.Shape
        ) -> Iterator[tuple[skia.Path, str | None, float | None]]:

            def convert_point(
                point: se.Point
            ) -> skia.Point:
                return skia.Point(point.x, point.y)

            def underscore_to_camelcase(
                name: str
            ) -> str:
                return "".join(s.capitalize() for s in name.split("_"))

            path = skia.Path()
            for segment in se_shape.segments():
                match segment:
                    case se.Move():
                        path.moveTo(convert_point(segment.end))
                    case se.Close():
                        path.close()
                    case se.Line():
                        path.lineTo(convert_point(segment.end))
                    case se.QuadraticBezier():
                        path.quadTo(convert_point(segment.control), convert_point(segment.end))
                    case se.CubicBezier():
                        path.cubicTo(convert_point(segment.control1), convert_point(segment.control2), convert_point(segment.end))
                    case se.Arc():
                        assert isinstance(segment.sweep, float)
                        path.arcTo(
                            convert_point(segment.radius),
                            segment.get_rotation().as_degrees,
                            skia.Path.ArcSize.kSmall_ArcSize if abs(segment.sweep) < math.pi else skia.Path.ArcSize.kLarge_ArcSize,
                            skia.PathDirection.kCW if segment.sweep < 0 else skia.PathDirection.kCCW,
                            convert_point(segment.end)
                        )
                    case _:
                        raise ValueError(f"Cannot handle svgelements path segment type: {type(segment)}")

            if se_shape.fill.value is not None:
                yield (path, se_shape.fill.hexrgb, se_shape.fill.opacity)
            if se_shape.stroke.value is not None:
                assert se_shape.values is not None
                paint = skia.Paint()
                paint.setStyle(skia.Paint.kStroke_Style)
                if (stroke_cap := se_shape.values.get("stroke-linecap")) is not None:
                    paint.setStrokeCap(getattr(skia.Paint.Cap, f"k{underscore_to_camelcase(stroke_cap)}_Cap"))
                if (stroke_join := se_shape.values.get("stroke-linejoin")) is not None:
                    paint.setStrokeJoin(getattr(skia.Paint.Join, f"k{underscore_to_camelcase(stroke_join)}_Join"))
                if (stroke_miter := se_shape.values.get("stroke-miterlimit")) is not None:
                    paint.setStrokeMiter(float(stroke_miter))
                if (stroke_width := se_shape.values.get("stroke-width")) is not None:
                    paint.setStrokeWidth(float(stroke_width))

                stroke_path = skia.Path()
                if paint.getFillPath(path, stroke_path):
                    yield (stroke_path, se_shape.stroke.hexrgb, se_shape.stroke.opacity)

        def get_shape_from_skia_path(
            path: skia.Path
        ) -> Shape:
            
            def convert_point(
                point: skia.Point
            ) -> NP_2f8:
                return np.array((point.x(), point.y()))

            def sample_positions(
                f: Callable[[float], NP_2f8]
            ) -> Iterator[NP_2f8]:
                for t in np.linspace(0.0, 1.0, 9)[1:]:
                    yield f(float(t))

            coordinates_list: list[NP_2f8] = []
            counts_list: list[int] = []
            prev_count = 0
            it = iter(path)
            verb, points = it.next()
            while verb != skia.Path.Verb.kDone_Verb:
                match (verb, points):
                    case (skia.Path.Verb.kMove_Verb, [_]):
                        pass
                    case (skia.Path.Verb.kLine_Verb, [_, end]):
                        end = convert_point(end)
                        coordinates_list.append(end)
                    case (skia.Path.Verb.kQuad_Verb, [start, control, end]):
                        start = convert_point(start)
                        control = convert_point(control)
                        end = convert_point(end)
                        coordinates_list.extend(sample_positions(
                            lambda t: (1.0 - t) * (1.0 - t) * start
                                + 2.0 * (1.0 - t) * t * control
                                + t * t * end
                        ))
                    case (skia.Path.Verb.kCubic_Verb, [start, control1, control2, end]):
                        start = convert_point(start)
                        control1 = convert_point(control1)
                        control2 = convert_point(control2)
                        end = convert_point(end)
                        coordinates_list.extend(sample_positions(
                            lambda t: (1.0 - t) * (1.0 - t) * (1.0 - t) * start
                                + 3.0 * (1.0 - t) * (1.0 - t) * t * control1
                                + 3.0 * (1.0 - t) * t * t * control2
                                + t * t * t * end
                        ))
                    case (skia.Path.Verb.kConic_Verb, [start, control, end]):
                        start = convert_point(start)
                        control = convert_point(control)
                        end = convert_point(end)
                        w = it.conicWeight()
                        coordinates_list.extend(sample_positions(
                            lambda t: ((1.0 - t) * (1.0 - t) * start
                                + 2.0 * (1.0 - t) * t * w * control
                                + t * t * end
                            ) / ((1.0 - t) * (1.0 - t) + 2.0 * (1.0 - t) * t * w + t * t)
                        ))
                    case (skia.Path.Verb.kClose_Verb, [_]):
                        counts_list.append(len(coordinates_list) - prev_count)
                        prev_count = len(coordinates_list)
                    case _:
                        raise ValueError(f"Cannot handle skia path segment: {(verb, points)}")
                verb, points = it.next()
            return Shape(np.array(coordinates_list), np.array(counts_list))

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
                for skia_path, color, opacity in iter_skia_paths_from_se_shape(se_shape * transform):
                    shape = get_shape_from_skia_path(skia_path)
                    if len(shape._coordinates_) == 0:
                        continue
                    style_dict = {}
                    if se_shape.fill is not None:
                        if color is not None:
                            style_dict["color"] = color
                        if opacity is not None:
                            style_dict["opacity"] = opacity
                    yield ShapeMobject(shape).set(**style_dict)

        svg: se.SVG = se.SVG.parse(svg_path)
        return tuple(iter_shape_mobjects_from_svg(svg))
