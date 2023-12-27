from __future__ import annotations


import pathlib
import subprocess
from abc import abstractmethod
from typing import (
    Self,
    TypedDict
)

import attrs

from ...animatables.arrays.animatable_color import AnimatableColor
from ...animatables.shape import Shape
from ...constants.custom_typing import (
    ColorType,
    SelectorType
)
from ...toplevel.toplevel import Toplevel
from ..shape_mobjects.shape_mobject import ShapeMobject
from ..cached_mobject import (
    CachedMobject,
    CachedMobjectInputs
)
from ..mobject import Mobject
from ..svg_mobject import SVGMobject


class TypstMobjectKwargs(TypedDict, total=False):
    preamble: str
    concatenate: bool
    align: str | None
    font: str | tuple[str, ...] | None
    color: ColorType | None


@attrs.frozen(kw_only=True)
class TypstMobjectInputs(CachedMobjectInputs):

    @staticmethod
    def _docstring_trim(
        string: str
    ) -> str:
        # Borrowed from `https://peps.python.org/pep-0257/`.
        if not string:
            return ""
        lines = string.splitlines()
        indents = tuple(
            len(line) - len(stripped)
            for line in lines[1:]
            if (stripped := line.lstrip())
        )
        trimmed = [lines[0].strip()]
        if indents:
            indent = min(indents)
            trimmed.extend(
                line[indent:].rstrip()
                for line in lines[1:]
            )
        while trimmed and not trimmed[-1]:
            trimmed.pop()
        while trimmed and not trimmed[0]:
            trimmed.pop(0)
        return "\n".join(trimmed)

    string: str = attrs.field(
        converter=_docstring_trim
    )
    preamble: str = attrs.field(
        factory=lambda: Toplevel._get_config().typst_preamble,
        converter=_docstring_trim
    )
    concatenate: bool = False
    align: str | None = attrs.field(
        factory=lambda: Toplevel._get_config().typst_align
    )
    font: str | tuple[str, ...] | None = attrs.field(
        factory=lambda: Toplevel._get_config().typst_font
    )
    color: ColorType | None = attrs.field(
        factory=lambda: Toplevel._get_config().default_color
    )


class TypstMobject[TypstMobjectInputsT: TypstMobjectInputs](CachedMobject[TypstMobjectInputsT]):
    __slots__ = (
        "_inputs",
        "_selector_to_indices_dict"
    )

    def __init__(
        self: Self,
        inputs: TypstMobjectInputsT
    ) -> None:
        super().__init__(inputs)
        self._inputs: TypstMobjectInputsT = inputs
        self._selector_to_indices_dict: dict[SelectorType, list[int]] = {}
        self.scale(1.0 / 32.0)

    @classmethod
    def _generate_shape_mobjects(
        cls: type[Self],
        inputs: TypstMobjectInputsT,
        temp_path: pathlib.Path
    ) -> tuple[ShapeMobject, ...]:
        preamble = cls._get_preamble_from_inputs(inputs, temp_path)
        environment_begin, environment_end = cls._get_environment_pair_from_inputs(inputs, temp_path)
        content = "\n".join(filter(None, (
            preamble,
            inputs.preamble,
            f"{environment_begin}{inputs.string}{environment_end}"
        )))

        svg_path = temp_path.with_suffix(".svg")
        typst_path = temp_path.with_suffix(".typ")
        typst_path.write_text(content, encoding="utf-8")

        completed_process = subprocess.run((
            "typst",
            "compile",
            "--root", pathlib.Path(),
            typst_path,
            svg_path
        ), capture_output=True)
        try:
            if completed_process.returncode:
                raise OSError(completed_process.stderr.decode())
            shape_mobjects = SVGMobject._generate_shape_mobjects_from_svg(svg_path)
        finally:
            for path in (svg_path, typst_path):
                path.unlink(missing_ok=True)

        if inputs.concatenate:
            shape_mobjects = (ShapeMobject(Shape().concatenate(tuple(
                shape_mobject._shape_ for shape_mobject in shape_mobjects
            ))),)
        return shape_mobjects

    @classmethod
    def _get_preamble_from_inputs(
        cls: type[Self],
        inputs: TypstMobjectInputsT,
        temp_path: pathlib.Path
    ) -> str:
        return "\n".join(filter(None, (
            f"""#set align({
                inputs.align
            })""" if inputs.align is not None else "",
            f"""#set text(font: {
                f"\"{inputs.font.replace("\"", "\\\"")}\""
                if isinstance(inputs.font, str)
                else f"({", ".join(f"\"{font.replace("\"", "\\\"")}\"" for font in inputs.font)})"
            })""" if inputs.font is not None else "",
            f"""#set text(fill: rgb({
                ", ".join(f"{component * 100.0}%" for component in AnimatableColor._color_to_array(inputs.color))
            }))""" if inputs.color is not None else ""
        )))

    @classmethod
    def _get_environment_pair_from_inputs(
        cls: type[Self],
        inputs: TypstMobjectInputsT,
        temp_path: pathlib.Path
    ) -> tuple[str, str]:
        return "", ""

    @classmethod
    @abstractmethod
    def _get_labelled_inputs(
        cls: type[Self],
        inputs: TypstMobjectInputsT,
        label_to_selector_dict: dict[int, SelectorType]
    ) -> TypstMobjectInputsT:
        pass

    def _probe_indices_from_selectors(
        self: Self,
        selectors: tuple[SelectorType, ...]
    ) -> None:
        selectors = tuple(
            selector for selector in selectors
            if selector not in self._selector_to_indices_dict
        )
        if not selectors:
            return

        # Label by 255 opacity values (opacity 0xFF is reserved).
        assert len(selectors) <= 255
        cls = type(self)
        label_to_selector_dict = {
            label: selector
            for label, selector in enumerate(selectors)
        }
        labelled_inputs = cls._get_labelled_inputs(
            inputs=self._inputs,
            label_to_selector_dict=label_to_selector_dict
        )
        labelled_shape_mobjects = cls._get_shape_mobjects(labelled_inputs)
        assert len(self._shape_mobjects) == len(labelled_shape_mobjects)

        self._selector_to_indices_dict.update((selector, []) for selector in selectors)
        for index, labelled_shape_mobject in enumerate(labelled_shape_mobjects):
            label = int(labelled_shape_mobject._opacity_._array_ * 255.0)
            if label == 255:
                continue
            self._selector_to_indices_dict[label_to_selector_dict[label]].append(index)

    def _build_from_selector(
        self: Self,
        selector: SelectorType
    ) -> Mobject:
        return Mobject().add(*(
            self._shape_mobjects[index]
            for index in self._selector_to_indices_dict[selector]
        ))

    def select_multiple(
        self: Self,
        selectors: tuple[SelectorType, ...]
    ) -> Mobject:
        self._probe_indices_from_selectors(selectors)
        return Mobject().add(*(
            self._build_from_selector(selector)
            for selector in selectors
        ))

    def select(
        self: Self,
        selector: SelectorType
    ) -> Mobject:
        self._probe_indices_from_selectors((selector,))
        return self._build_from_selector(selector)

    def set_local_colors(
        self: Self,
        selector_to_color_dict: dict[SelectorType, ColorType]
    ) -> Self:
        for mobject, color in zip(
            self.select_multiple(tuple(selector_to_color_dict)),
            selector_to_color_dict.values(),
            strict=True
        ):
            mobject.set(color=color)
        return self
