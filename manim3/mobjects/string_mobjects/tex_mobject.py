from abc import abstractmethod
import os
import pathlib
import re
from dataclasses import dataclass
from typing import (
    Iterable,
    Iterator,
    TypeVar
)

from ...constants.custom_typing import (
    AlignmentT,
    ColorT,
    SelectorT
)
from ...toplevel.toplevel import Toplevel
from .string_mobject import (
    CommandFlag,
    EdgeFlag,
    Span,
    StringMobject,
    StringMobjectIO,
    StringMobjectInputData
)


_LatexBaseInputDataT = TypeVar("_LatexBaseInputDataT", bound="LatexBaseInputData")


class LatexError(ValueError):
    def __init__(
        self,
        error_message: str | None
    ) -> None:
        message = "LaTeX Error! Not a worry, it happens to the best of us."
        if error_message is not None:
            message += f" The error could be: `{error_message}`"
        super().__init__(message)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class LatexBaseInputData(StringMobjectInputData):
    font_size: float
    local_spans: list[Span]


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TexInputData(LatexBaseInputData):
    #font_size: float
    compiler: str
    preamble: str
    alignment: AlignmentT
    environment: str
    #local_spans: list[Span]


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class MathJaxInputData(LatexBaseInputData):
    pass
    #font_size: float
    #local_spans: list[Span]


class LatexBaseIO(StringMobjectIO[_LatexBaseInputDataT]):
    __slots__ = ()

    #_dir_name: ClassVar[str] = "tex"
    #_parser_cls: ClassVar[type[StringParser]] = TexParser
    # Through the convension, `font_size=30` would make the height of "x" become roughly 0.30.
    #_scale_factor_per_font_point: ClassVar[float]
    #_TEX_SCALE_FACTOR_PER_FONT_POINT: ClassVar[float] = 0.001577

    @classmethod
    def _get_global_attrs(
        cls,
        input_data: _LatexBaseInputDataT,
        temp_path: pathlib.Path
    ) -> dict[str, str]:
        return {}

    @classmethod
    def _get_local_attrs(
        cls,
        input_data: _LatexBaseInputDataT,
        temp_path: pathlib.Path
    ) -> dict[Span, dict[str, str]]:
        local_spans = input_data.local_spans
        return {span: {} for span in local_spans}

    @classmethod
    def _get_svg_frame_scale(
        cls,
        input_data: _LatexBaseInputDataT
    ) -> float:
        # Through the convension, `font_size=30` would make the height of "x" become roughly 0.30.
        return cls._scale_factor_per_font_point * input_data.font_size

    #@classmethod
    #def _get_shape_mobjects(
    #    cls,
    #    content: str,
    #    input_data: _LatexBaseInputDataT,
    #    temp_path: pathlib.Path
    #) -> list[ShapeMobject]:
    #    font_size = input_data.font_size
    #    

    #        shape_mobjects = list(SVGMobjectIO.iter_shape_mobject_from_svg(
    #            svg_path=svg_path,
    #            frame_scale=cls._scale_factor_per_font_point * font_size
    #        ))

    #    finally:
    #        # Cleanup superfluous documents.
    #        for ext in (".tex", dvi_ext, ".log", ".aux", ".svg"):
    #            temp_path.with_suffix(ext).unlink(missing_ok=True)

    #    return shape_mobjects

    @classmethod
    @property
    @abstractmethod
    def _scale_factor_per_font_point(cls) -> float:
        pass

    @classmethod
    def _iter_command_matches(
        cls,
        string: str
    ) -> Iterator[re.Match[str]]:
        # Lump together adjacent brace pairs.
        pattern = re.compile(r"""
            (?P<command>\\(?:[a-zA-Z]+|.))
            |(?P<open>{+)
            |(?P<close>}+)
        """, flags=re.VERBOSE | re.DOTALL)
        open_stack: list[tuple[int, int]] = []
        for match_obj in pattern.finditer(string):
            if not match_obj.group("close"):
                if not match_obj.group("open"):
                    yield match_obj
                    continue
                open_stack.append(match_obj.span())
                continue
            close_start, close_stop = match_obj.span()
            while True:
                if not open_stack:
                    raise ValueError("Missing '{' inserted")
                open_start, open_stop = open_stack.pop()
                n = min(open_stop - open_start, close_stop - close_start)
                assert (open_match_obj := pattern.fullmatch(
                    string, pos=open_stop - n, endpos=open_stop
                )) is not None
                yield open_match_obj
                assert (close_match_obj := pattern.fullmatch(
                    string, pos=close_start, endpos=close_start + n
                )) is not None
                yield close_match_obj
                close_start += n
                if close_start < close_stop:
                    continue
                open_stop -= n
                if open_start < open_stop:
                    open_stack.append((open_start, open_stop))
                break
        if open_stack:
            raise ValueError("Missing '}' inserted")

    @classmethod
    def _get_command_flag(
        cls,
        match_obj: re.Match[str]
    ) -> CommandFlag:
        if match_obj.group("open"):
            return CommandFlag.OPEN
        if match_obj.group("close"):
            return CommandFlag.CLOSE
        return CommandFlag.OTHER

    @classmethod
    def _replace_for_content(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        return match_obj.group()

    @classmethod
    def _replace_for_matching(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        if match_obj.group("command"):
            return match_obj.group()
        return ""

    @classmethod
    def _get_attrs_from_command_pair(
        cls,
        open_command: re.Match[str],
        close_command: re.Match[str]
    ) -> dict[str, str] | None:
        if len(open_command.group()) >= 2:
            return {}
        return None

    @classmethod
    def _get_command_string(
        cls,
        label: int | None,
        edge_flag: EdgeFlag,
        attrs: dict[str, str]
    ) -> str:
        if label is None:
            return ""
        if edge_flag == EdgeFlag.STOP:
            return "}}"
        rg, b = divmod(label, 256)
        r, g = divmod(rg, 256)
        color_command = f"\\color[RGB]{{{r}, {g}, {b}}}"
        return "{{" + color_command


class TexIO(LatexBaseIO[TexInputData]):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(cls) -> str:
        return "tex"

    @classmethod
    @property
    def _temp_extensions(cls) -> list[str]:
        return [".tex", ".dvi", ".xdv", ".log", ".aux", ".svg"]

    @classmethod
    def _create_svg(
        cls,
        content: str,
        input_data: TexInputData,
        svg_path: pathlib.Path
    ) -> None:
        compiler = input_data.compiler
        preamble = input_data.preamble
        alignment = input_data.alignment
        environment = input_data.environment

        if compiler == "latex":
            program = "latex"
            dvi_extension = ".dvi"
        elif compiler == "xelatex":
            program = "xelatex -no-pdf"
            dvi_extension = ".xdv"
        else:
            raise ValueError(
                f"Compiler '{compiler}' is not implemented"
            )

        try:
            tex_path = svg_path.with_suffix(".tex")
            with tex_path.open(mode="w", encoding="utf-8") as tex_file:
                match alignment:
                    case "left":
                        alignment_command = "\\flushleft"
                    case "center":
                        alignment_command = "\\centering"
                    case "right":
                        alignment_command = "\\flushright"
                if environment:
                    begin_environment = f"\\begin{{{environment}}}"
                    end_environment = f"\\end{{{environment}}}"
                else:
                    begin_environment = ""
                    end_environment = ""
                full_content = "\n".join((
                    preamble,
                    "\\begin{document}",
                    alignment_command,
                    begin_environment,
                    content,
                    end_environment,
                    "\\end{document}"
                )) + "\n"
                tex_file.write(full_content)

            # tex to dvi
            if os.system(" ".join((
                program,
                "-interaction=batchmode",
                "-halt-on-error",
                f"-output-directory=\"{svg_path.parent}\"",
                f"\"{tex_path}\"",
                ">",
                os.devnull
            ))):
                error_message: str | None = None
                with svg_path.with_suffix(".log").open(encoding="utf-8") as log_file:
                    if (error_match_obj := re.search(r"(?<=\n! ).*", log_file.read())) is not None:
                        error_message = error_match_obj.group()
                raise LatexError(error_message)

            # dvi to svg
            os.system(" ".join((
                "dvisvgm",
                f"\"{svg_path.with_suffix(dvi_extension)}\"",
                "-n",
                "-v",
                "0",
                "-o",
                f"\"{svg_path}\"",
                ">",
                os.devnull
            )))

        finally:
            for extension in (".tex", dvi_extension, ".log", ".aux"):
                svg_path.with_suffix(extension).unlink(missing_ok=True)

    @classmethod
    @property
    def _scale_factor_per_font_point(cls) -> float:
        return 0.001577


class MathJaxIO(LatexBaseIO[MathJaxInputData]):
    __slots__ = ()

    #_dir_name: ClassVar[str] = "mathjax"
    #_parser_cls: ClassVar[type[StringParser]] = TexParser
    #_MATHJAX_SCALE_FACTOR_PER_FONT_POINT: ClassVar[float] = 0.009758

    @classmethod
    @property
    def _dir_name(cls) -> str:
        return "mathjax"

    @classmethod
    def _create_svg(
        cls,
        content: str,
        input_data: MathJaxInputData,
        svg_path: pathlib.Path
    ) -> None:
        import manimgl_mathjax  # TODO
        mathjax_program_path = pathlib.Path(manimgl_mathjax.__file__).absolute().with_name("index.js")
        full_content = content.replace("\n", " ")

        os.system(" ".join((
            "node",
            f"\"{mathjax_program_path}\"",
            f"\"{svg_path}\"",
            f"\"{full_content}\"",
            ">",
            os.devnull
        )))
        with svg_path.open(encoding="utf-8") as svg_file:
            if (error_match_obj := re.search(r"(?<=data\-mjx\-error\=\")(.*?)(?=\")", svg_file.read())) is not None:
                raise LatexError(error_match_obj.group())

    @property
    @classmethod
    def _scale_factor_per_font_point(cls) -> float:
        return 0.009758


class LatexBaseStringMobject(StringMobject):
    __slots__ = ()

    #_parser_cls: ClassVar[type[StringParser]] = TexParser
    #_dir_name: ClassVar[str] = "_mathjax"
    #_io_cls: ClassVar[type[StringMobjectIO]] = MathJaxIO

    def __init__(
        self,
        string: str,
        *,
        isolate: Iterable[SelectorT] = (),
        protect: Iterable[SelectorT] = (),
        color: ColorT | None = None,
        font_size: float | None = None,
        local_colors: dict[SelectorT, ColorT] | None = None,
        **kwargs
    ) -> None:
        config = Toplevel.config
        if color is None:
            color = config.tex_color
        if font_size is None:
            font_size = config.tex_font_size
        if local_colors is None:
            local_colors = {}

        #super().__init__(
        #    string=string,
        #    isolate=isolate,
        #    protect=protect,
        #    global_attrs={},
        #    local_attrs={
        #        selector: {}
        #        for selector in local_colors
        #    },
        #    writing_config=MathJaxWritingConfig(
        #        font_size=font_size
        #    )
        #)
        cls = type(self)
        super().__init__(
            string=string,
            isolate=cls._get_spans_by_selectors(isolate, string),
            protect=cls._get_spans_by_selectors(protect, string),
            font_size=font_size,
            local_spans=cls._get_spans_by_selectors(local_colors, string),
            **kwargs
        )

        self.set_style(color=color)
        for selector, color in local_colors.items():
            self.select_parts(selector).set_style(color=color)


class Tex(LatexBaseStringMobject):
    __slots__ = ()

    #_parser_cls: ClassVar[type[StringParser]] = TexParser
    #_dir_name: ClassVar[str] = "_tex"
    #_io_cls: ClassVar[type[StringMobjectIO]] = TexIO
    #_environment: ClassVar[str] = ""

    def __init__(
        self,
        string: str,
        *,
        alignment: AlignmentT | None = None,
        compiler: str | None = None,
        preamble: str | None = None,
        **kwargs
    ) -> None:
        config = Toplevel.config
        if alignment is None:
            alignment = config.tex_alignment
        if compiler is None:
            compiler = config.tex_compiler
        if preamble is None:
            preamble = config.tex_preamble

        cls = type(self)
        super().__init__(
            string=string,
            alignment=alignment,
            compiler=compiler,
            preamble=preamble,
            environment=cls._environment,
            **kwargs
        )
        #    string=string,
        #    isolate=isolate,
        #    protect=protect,
        #    global_attrs={},
        #    local_attrs={
        #        selector: {}
        #        for selector in local_colors
        #    },
        #    writing_config=TexWritingConfig(
        #        font_size=font_size,
        #        compiler=compiler,
        #        preamble=preamble,
        #        alignment=alignment,
        #        environment=cls._environment
        #        #frame_scale=cls._TEX_SCALE_FACTOR_PER_FONT_POINT * font_size
        #    )

        #self.set_style(color=color)
        #for selector, local_color in local_colors.items():
        #    self.select_parts(selector).set_style(color=local_color)

    @classmethod
    @property
    def _io_cls(cls) -> type[StringMobjectIO[TexInputData]]:
        return TexIO

    @classmethod
    @property
    def _input_data_cls(cls) -> type[TexInputData]:
        return TexInputData

    @classmethod
    @property
    def _environment(cls) -> str:
        return ""


class MathTex(Tex):
    __slots__ = ()

    #_environment: ClassVar[str] = "align*"

    @classmethod
    @property
    def _environment(cls) -> str:
        return "align*"


class MathJax(LatexBaseStringMobject):
    __slots__ = ()

    #_parser_cls: ClassVar[type[StringParser]] = TexParser
    #_dir_name: ClassVar[str] = "_mathjax"
    #_io_cls: ClassVar[type[StringMobjectIO]] = MathJaxIO

    #def __init__(
    #    self,
    #    string: str,
    #    **kwargs
    #) -> None:
    #    #config = Toplevel.config
    #    #if color is None:
    #    #    color = config.tex_color
    #    #if font_size is None:
    #    #    font_size = config.tex_font_size
    #    #if local_colors is None:
    #    #    local_colors = {}

    #    #super().__init__(
    #    #    string=string,
    #    #    isolate=isolate,
    #    #    protect=protect,
    #    #    global_attrs={},
    #    #    local_attrs={
    #    #        selector: {}
    #    #        for selector in local_colors
    #    #    },
    #    #    writing_config=MathJaxWritingConfig(
    #    #        font_size=font_size
    #    #    )
    #    #)
    #    #cls = type(self)
    #    super().__init__(
    #        string=string,
    #        **kwargs
    #        #isolate=cls._get_spans_by_selectors(isolate, string),
    #        #protect=cls._get_spans_by_selectors(protect, string),
    #        #font_size=font_size,
    #        #local_spans=cls._get_spans_by_selectors(local_colors, string)
    #    )

    #    #self.set_style(color=color)
    #    #for selector, color in local_colors.items():
    #    #    self.select_parts(selector).set_style(color=color)

    @classmethod
    @property
    def _io_cls(cls) -> type[StringMobjectIO[MathJaxInputData]]:
        return MathJaxIO

    @classmethod
    @property
    @abstractmethod
    def _input_data_cls(cls) -> type[MathJaxInputData]:
        return MathJaxInputData
