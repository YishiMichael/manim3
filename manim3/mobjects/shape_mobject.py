from ..geometries.shape_geometry import ShapeGeometry
from ..lazy.lazy import Lazy
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.mobject import MobjectStyleMeta
from ..mobjects.stroke_mobject import StrokeMobject
from ..shape.shape import Shape


class ShapeMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self._shape_ = shape
        self._enable_phong_lighting_ = False

    @MobjectStyleMeta.register(
        partial_method=Shape.partial,
        interpolate_method=Shape.interpolate,
        concatenate_method=Shape.concatenate
    )
    @Lazy.variable
    @classmethod
    def _shape_(cls) -> Shape:
        return Shape()

    @Lazy.property
    @classmethod
    def _geometry_(
        cls,
        shape: Shape
    ) -> ShapeGeometry:
        return ShapeGeometry(shape)

    def build_stroke(self) -> StrokeMobject:
        stroke = StrokeMobject()
        stroke.match_style(
            self,
            model_matrix=True,
            default=False
        )  # Copy `model_matrix` only.
        #stroke._model_matrix_ = self._model_matrix_  # TODO
        stroke._multi_line_string_ = self._shape_._multi_line_string_
        return stroke

    #def add_stroke(
    #    self,
    #    **kwargs
    #):
    #    self.add(self.build_stroke(**kwargs))
    #    return self

    #@property
    #def shape(self) -> Shape:
    #    return self._shape_
