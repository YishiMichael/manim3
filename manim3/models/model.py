from abc import (
    ABC,
    abstractmethod
)
from dataclasses import dataclass
import itertools as it
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterator,
    TypeVar,
    ParamSpec
)
import warnings

import moderngl
import numpy as np

from ..constants import (
    ORIGIN,
    PI
)
from ..custom_typing import (
    ColorT,
    NP_3f8,
    NP_44f8,
    NP_x3f8
)
from ..geometries.geometry import Geometry
from ..lazy.lazy import (
    Lazy,
    LazyCollectionConverter,
    LazyContainer,
    LazyObject,
    LazyVariableDescriptor
)
from ..rendering.gl_buffer import UniformBlockBuffer
from ..shape.line_string import MultiLineString
from ..shape.shape import Shape
from ..utils.color import ColorUtils
from ..utils.space import SpaceUtils

if TYPE_CHECKING:
    from .cameras.camera import Camera
    from .lights.lighting import Lighting


_ContainerT = TypeVar("_ContainerT", bound="LazyContainer")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_DataT = TypeVar("_DataT")
_DataRawT = TypeVar("_DataRawT")
_MethodParams = ParamSpec("_MethodParams")


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class BoundingBox:
    maximum: NP_3f8
    minimum: NP_3f8

    @property
    def center(self) -> NP_3f8:
        return (self.maximum + self.minimum) / 2.0

    @property
    def radii(self) -> NP_3f8:
        radii = (self.maximum - self.minimum) / 2.0
        # For zero-width dimensions of radii, thicken a little bit to avoid zero division.
        radii[np.isclose(radii, 0.0)] = 1e-8
        return radii


class AboutABC(ABC):
    __slots__ = ()

    @abstractmethod
    def _get_about_point(
        self,
        model: "Model"
    ) -> NP_3f8:
        pass


class AlignABC(ABC):
    __slots__ = (
        "_about",
        "_direction",
        "_buff"
    )

    def __init__(
        self,
        about: AboutABC,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__()
        self._about: AboutABC = about
        self._direction: NP_3f8 = direction
        self._buff: float | NP_3f8 = buff

    #@abstractmethod
    #def _get_target_point(self) -> NP_3f8:
    #    pass

    def _get_shift_vector(
        self,
        model: "Model",
        direction_sign: float
    ) -> NP_3f8:
        target_point = self._about._get_about_point(model)
        direction = direction_sign * self._direction
        point_to_align = model.get_bounding_box_point(direction) + self._buff * direction
        return target_point - point_to_align


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class StyleDescriptorInfo(Generic[_InstanceT, _ContainerT, _DataT, _DataRawT]):
    descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
    partial_method: Callable[[_ContainerT], Callable[[float, float], _ContainerT] | None]
    interpolate_method: Callable[[_ContainerT, _ContainerT], Callable[[float], _ContainerT] | None]
    concatenate_method: Callable[..., Callable[[], _ContainerT] | None]


class StyleMeta:
    __slots__ = ()

    #_name_to_descriptors_dict: ClassVar[dict[str, list[LazyVariableDescriptor]]] = {}
    _style_descriptor_infos: ClassVar[list[StyleDescriptorInfo]] = []

    def __new__(cls):
        raise TypeError

    @classmethod
    def register(
        cls,
        *,
        partial_method: Callable[[_DataRawT], Callable[[float, float], _DataRawT]] | None = None,
        interpolate_method: Callable[[_DataRawT, _DataRawT], Callable[[float], _DataRawT]] | None = None,
        concatenate_method: Callable[..., Callable[[], _DataRawT]] | None = None
        #matcher: Callable[[_DataRawT, _DataRawT], bool] | None = None
    ) -> Callable[
        [LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]],
        LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
    ]:

        def callback(
            descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
        ) -> LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]:
            assert not isinstance(descriptor.converter, LazyCollectionConverter)
            #cls._name_to_descriptors_dict.setdefault(descriptor.method.__name__, []).append(descriptor)
            cls._style_descriptor_infos.append(StyleDescriptorInfo(
                descriptor=descriptor,
                partial_method=cls._partial_method_decorator(descriptor, partial_method),
                interpolate_method=cls._interpolate_method_decorator(descriptor, interpolate_method),
                concatenate_method=cls._concatenate_method_decorator(descriptor, concatenate_method)
            ))
            return descriptor

        return callback

    @classmethod
    def _get_callback_from(
        cls,
        method_dict: dict[LazyVariableDescriptor, Callable[..., Callable[_MethodParams, Any] | None]]
    ) -> "Callable[..., Callable[[Model], Callable[_MethodParams, None]]]":

        def get_descriptor_callback(
            descriptor: LazyVariableDescriptor,
            method: Callable[..., Callable[_MethodParams, Any] | None],
            srcs: tuple[Model, ...]
        ) -> Callable[[Model], Callable[_MethodParams, None]] | None:
            if not all(
                descriptor in type(src)._lazy_variable_descriptors
                for src in srcs
            ):
                return None
            src_containers = tuple(
                descriptor.get_container(src)
                for src in srcs
            )
            method_callback = method(*src_containers)
            if method_callback is None:
                return None

            def descriptor_callback(
                dst: Model
            ) -> Callable[_MethodParams, None]:

                def descriptor_dst_callback(
                    *args: _MethodParams.args,
                    **kwargs: _MethodParams.kwargs
                ) -> None:
                    if descriptor not in type(dst)._lazy_variable_descriptors:
                        return
                    new_container = method_callback(*args, **kwargs)
                    descriptor.set_container(dst, new_container)

                return descriptor_dst_callback

            return descriptor_callback

        def callback(
            *srcs: Model
        ) -> Callable[[Model], Callable[_MethodParams, None]]:
            descriptor_callbacks = [
                descriptor_callback
                for descriptor, method in method_dict.items()
                if (descriptor_callback := get_descriptor_callback(descriptor, method, srcs)) is not None
            ]

            def src_callback(
                dst: Model
            ) -> Callable[_MethodParams, None]:
                descriptor_dst_callbacks = [
                    descriptor_callback(dst)
                    for descriptor_callback in descriptor_callbacks
                ]

                def dst_callback(
                    *args: _MethodParams.args,
                    **kwargs: _MethodParams.kwargs
                ) -> None:
                    for descriptor_dst_callback in descriptor_dst_callbacks:
                        descriptor_dst_callback(*args, **kwargs)

                return dst_callback

            return src_callback

        return callback

    @classmethod
    def _get_dst_callback(
        cls,
        descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, Any, _DataRawT],
        method: Callable[..., Callable[_MethodParams, _DataRawT]],
        *src_containers: _ContainerT
    ) -> Callable[_MethodParams, _ContainerT]:
        method_callback = method(*(
            descriptor.converter.c2r(src_container)
            for src_container in src_containers
        ))

        def dst_callback(
            *args: _MethodParams.args,
            **kwargs: _MethodParams.kwargs
        ) -> _ContainerT:
            return descriptor.converter.r2c(method_callback(*args, **kwargs))

        return dst_callback

    @classmethod
    def _partial_method_decorator(
        cls,
        descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, Any, _DataRawT],
        method: Callable[[_DataRawT], Callable[_MethodParams, _DataRawT]] | None
    ) -> Callable[[_ContainerT], Callable[_MethodParams, _ContainerT] | None]:

        def new_method(
            src_container: _ContainerT
        ) -> Callable[_MethodParams, _ContainerT] | None:
            if method is None:
                # Do not make into callback if the method is not provided.
                return None

            return cls._get_dst_callback(descriptor, method, src_container)

        return new_method

    @classmethod
    def _interpolate_method_decorator(
        cls,
        descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, Any, _DataRawT],
        method: Callable[[_DataRawT, _DataRawT], Callable[_MethodParams, _DataRawT]] | None
    ) -> Callable[[_ContainerT, _ContainerT], Callable[_MethodParams, _ContainerT] | None]:

        def new_method(
            src_container_0: _ContainerT,
            src_container_1: _ContainerT
        ) -> Callable[_MethodParams, _ContainerT] | None:
            if src_container_0._match_elements(src_container_1):
                # Do not make into callback if interpolated variables match.
                # This is a feature used by compositing animations played on the same model
                # which interpolate different variables.
                return None
            if method is None:
                raise ValueError(f"Uninterpolable variables of `{descriptor.method.__name__}` don't match")

            return cls._get_dst_callback(descriptor, method, src_container_0, src_container_1)

        return new_method

    @classmethod
    def _concatenate_method_decorator(
        cls,
        descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, Any, _DataRawT],
        method: Callable[..., Callable[_MethodParams, _DataRawT]] | None
    ) -> Callable[..., Callable[_MethodParams, _ContainerT] | None]:

        def new_method(
            *src_containers: _ContainerT
        ) -> Callable[_MethodParams, _ContainerT] | None:

            def return_common_container(
                common_container: _ContainerT
            ) -> Callable[_MethodParams, _ContainerT]:

                def dst_callback(
                    *args: _MethodParams.args,
                    **kwargs: _MethodParams.kwargs
                ) -> _ContainerT:
                    return common_container._copy_container()

                return dst_callback

            if not src_containers:
                return None
            src_container_0 = src_containers[0]
            if all(
                src_container_0._match_elements(src_container)
                for src_container in src_containers
            ):
                # If interpolated variables match, do copying in callback directly.
                # This is a feature used by children concatenation, which tries
                # copying all information from children.
                return return_common_container(src_containers[0])
            elif method is None:
                raise ValueError(f"Uncatenatable variables of `{descriptor.method.__name__}` don't match")

            return cls._get_dst_callback(descriptor, method, *src_containers)

        return new_method

    @classmethod
    @property
    def _partial(cls) -> "Callable[[Model], Callable[[Model], Callable[[float, float], None]]]":
        return cls._get_callback_from({
            info.descriptor: info.partial_method
            for info in cls._style_descriptor_infos
        })

    @classmethod
    @property
    def _interpolate(cls) -> "Callable[[Model, Model], Callable[[Model], Callable[[float], None]]]":
        return cls._get_callback_from({
            info.descriptor: info.interpolate_method
            for info in cls._style_descriptor_infos
        })

    @classmethod
    @property
    def _concatenate(cls) -> "Callable[..., Callable[[Model], Callable[[], None]]]":
        return cls._get_callback_from({
            info.descriptor: info.concatenate_method
            for info in cls._style_descriptor_infos
        })


#class AlignBorder(AlignABC):
#    __slots__ = ()

#    def _get_target_point(self) -> NP_3f8:
#        return self._direction * np.append(Config().size.frame_radii, 0.0)


#class CameraABC(LazyObject):
#    __slots__ = ()

#    @Lazy.variable_array
#    @classmethod
#    def _model_matrix_(cls) -> NP_44f8:
#        return NotImplemented

#    @Lazy.property
#    @classmethod
#    def _camera_uniform_block_buffer_(cls) -> UniformBlockBuffer:
#        return NotImplemented


class Model(LazyObject):
    __slots__ = ()

    @Lazy.property_collection
    @classmethod
    def _members_(cls) -> "list[Model]":
        return []

    def _yield_members(
        self,
        *,
        broadcast: bool = True
    ) -> "Iterator[Model]":
        yield self
        if broadcast:
            yield from self._members_

    @StyleMeta.register(
        interpolate_method=SpaceUtils.lerp_44f8
    )
    @Lazy.variable_array
    @classmethod
    def _model_matrix_(cls) -> NP_44f8:
        return np.identity(4)

    @Lazy.property
    @classmethod
    def _model_uniform_block_buffer_(
        cls,
        model_matrix: NP_44f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_model",
            fields=[
                "mat4 u_model_matrix"
            ],
            data={
                "u_model_matrix": model_matrix.T
            }
        )

    @Lazy.property_array
    @classmethod
    def _local_sample_points_(cls) -> NP_x3f8:
        # Implemented in subclasses.
        return np.zeros((0, 3))

    #@Lazy.property_hashable
    #@classmethod
    #def _has_local_sample_points_(
    #    cls,
    #    local_sample_points: NP_x3f8
    #) -> bool:
    #    return bool(len(local_sample_points))

    @Lazy.property_external
    @classmethod
    def _local_bounding_box_(
        cls,
        model_matrix: NP_44f8,
        local_sample_points: NP_x3f8
    ) -> BoundingBox | None:
        if not len(local_sample_points):
            return None
        world_sample_points = SpaceUtils.apply_affine(model_matrix, local_sample_points)
        return BoundingBox(
            maximum=world_sample_points.max(axis=0),
            minimum=world_sample_points.min(axis=0)
        )

    @Lazy.property_external
    @classmethod
    def _bounding_box_(
        cls,
        local_bounding_box: BoundingBox | None,
        members__local_bounding_box: list[BoundingBox | None]
    ) -> BoundingBox | None:
        points_array = np.array(list(it.chain.from_iterable(
            (bounding_box.maximum, bounding_box.minimum)
            for bounding_box in (
                local_bounding_box,
                *members__local_bounding_box
            )
            if bounding_box is not None
        )))
        if not len(points_array):
            return None
        return BoundingBox(
            maximum=points_array.max(axis=0),
            minimum=points_array.min(axis=0)
        )

    def get_bounding_box(self) -> BoundingBox:
        #if broadcast:
        result = self._bounding_box_
        #else:
        #    result = self._bounding_box_without_descendants_
        assert result is not None, "Trying to calculate the bounding box of some model with no points"
        return result

    def get_bounding_box_size(self) -> NP_3f8:
        bounding_box = self.get_bounding_box()
        return bounding_box.radii * 2.0

    def get_bounding_box_point(
        self,
        direction: NP_3f8
    ) -> NP_3f8:
        bounding_box = self.get_bounding_box()
        return bounding_box.center + direction * bounding_box.radii

    def get_center(self) -> NP_3f8:
        return self.get_bounding_box_point(ORIGIN)

    # transform

    def _make_callback_relative(
        self,
        matrix_callback: Callable[[float | NP_3f8], NP_44f8],
        about: AboutABC | None
    ) -> Callable[[float | NP_3f8], NP_44f8]:
        if about is None:
            return matrix_callback
        about_point = about._get_about_point(model=self)
        pre_transform = SpaceUtils.matrix_from_translation(-about_point)
        post_transform = SpaceUtils.matrix_from_translation(about_point)

        def callback(
            alpha: float | NP_3f8
        ) -> NP_44f8:
            return post_transform @ matrix_callback(alpha) @ pre_transform

        return callback

    def _apply_transform_callback(
        self,
        matrix_callback: Callable[[float], NP_44f8]
    ) -> Callable[[float], None]:

        model_to_model_matrix = {
            model: model._model_matrix_
            for model in self._yield_members()
        }

        def callback(
            alpha: float
        ) -> None:
            for model, model_matrix in model_to_model_matrix.items():
                model._model_matrix_ = matrix_callback(alpha) @ model_matrix

        return callback

    def apply_transform(
        self,
        matrix: NP_44f8,
    ):
        if np.isclose(np.linalg.det(matrix), 0.0):
            warnings.warn("Applying a singular matrix transform")
        self._apply_transform_callback(lambda alpha: matrix)(1.0)
        return self

    # shift relatives

    def _shift_callback(
        self,
        vector: NP_3f8
        # `about` is meaningless for shifting.
    ) -> Callable[[float | NP_3f8], NP_44f8]:
        return SpaceUtils.matrix_callback_from_translation(vector)

    def shift(
        self,
        vector: NP_3f8,
        *,
        alpha: float | NP_3f8 = 1.0
    ):
        matrix = self._shift_callback(vector)(alpha)
        self.apply_transform(matrix)
        return self

    def move_to(
        self,
        align: AlignABC,
        *,
        alpha: float | NP_3f8 = 1.0
    ):
        self.shift(
            vector=align._get_shift_vector(model=self, direction_sign=1.0),
            alpha=alpha
        )
        return self

    def next_to(
        self,
        align: AlignABC,
        *,
        alpha: float | NP_3f8 = 1.0
    ):
        self.shift(
            vector=align._get_shift_vector(model=self, direction_sign=-1.0),
            alpha=alpha
        )
        return self

    # scale relatives

    def _scale_callback(
        self,
        factor: float | NP_3f8,
        about: AboutABC | None = None
    ) -> Callable[[float | NP_3f8], NP_44f8]:
        return self._make_callback_relative(
            matrix_callback=SpaceUtils.matrix_callback_from_scale(factor),
            about=about
        )

    def scale(
        self,
        factor: float | NP_3f8,
        about: AboutABC | None = None,
        *,
        alpha: float | NP_3f8 = 1.0
    ):
        matrix = self._scale_callback(factor, about)(alpha)
        self.apply_transform(matrix)
        return self

    def scale_to(
        self,
        target: float | NP_3f8,
        about: AboutABC | None = None,
        alpha: float | NP_3f8 = 1.0
    ):
        factor = target / self.get_bounding_box_size()
        self.scale(
            factor=factor,
            about=about,
            alpha=alpha
        )
        return self

    def match_bounding_box(
        self,
        model: "Model"
    ):
        self.shift(model.get_center() - self.get_center()).scale_to(model.get_bounding_box_size())
        return self

    # rotate relatives

    def _rotate_callback(
        self,
        rotvec: NP_3f8,
        about: AboutABC | None = None
    ) -> Callable[[float | NP_3f8], NP_44f8]:
        return self._make_callback_relative(
            matrix_callback=SpaceUtils.matrix_callback_from_rotation(rotvec),
            about=about
        )

    def rotate(
        self,
        rotvec: NP_3f8,
        about: AboutABC | None = None,
        *,
        alpha: float | NP_3f8 = 1.0
    ):
        matrix = self._rotate_callback(rotvec, about)(alpha)
        self.apply_transform(matrix)
        return self

    def flip(
        self,
        axis: NP_3f8,
        about: AboutABC | None = None
    ):
        self.rotate(
            rotvec=SpaceUtils.normalize(axis) * PI,
            about=about
        )
        return self

    # meta methods

    def set_style(
        self,
        *,
        # polymorphism variables
        color: ColorT | None = None,
        opacity: float | None = None,
        weight: float | None = None,

        # Mobject
        model_matrix: NP_44f8 | None = None,
        camera: "Camera | None" = None,

        ## RenderableMobject
        #is_transparent: bool | None = None,

        # MeshMobject
        geometry: Geometry | None = None,
        color_maps: list[moderngl.Texture] | None = None,
        #enable_phong_lighting: bool | None = None,
        lighting: "Lighting | None" = None,
        ambient_strength: float | None = None,
        specular_strength: float | None = None,
        shininess: float | None = None,

        # ShapeMobject
        shape: Shape | None = None,

        # StrokeMobject
        multi_line_string: MultiLineString | None = None,
        width: float | None = None,
        #single_sided: bool | None = None,
        #dilate: float | None = None,

        # setting configs
        broadcast: bool = True,
        type_filter: "type[Model] | None" = None
    ):

        def standardize_input(
            value: Any
        ) -> np.ndarray:
            if not isinstance(value, float | int | np.ndarray):
                return value
            return (value * np.ones(())).astype(np.float64)

        if color is not None:
            color = ColorUtils.standardize_color(color)
        style = {
            f"_{key}_": standardize_input(value)
            for key, value in {
                "color": color,
                "opacity": opacity,
                "weight": weight,
                "model_matrix": model_matrix,
                "camera": camera,
                #"is_transparent": is_transparent,
                "geometry": geometry,
                "color_maps": color_maps,
                "lighting": lighting,
                #"enable_phong_lighting": enable_phong_lighting,
                "ambient_strength": ambient_strength,
                "specular_strength": specular_strength,
                "shininess": shininess,
                "shape": shape,
                "multi_line_string": multi_line_string,
                "width": width
                #"single_sided": single_sided,
                #"dilate": dilate
            }.items() if value is not None
        }

        #if type_filter is None:
        #    type_filter = Model

        for model in self._yield_members(broadcast=broadcast):
            if type_filter is not None and not isinstance(model, type_filter):
                continue
            for key, value in style.items():
                if (descriptor := type(model)._lazy_descriptor_dict.get(key)) is None:
                    continue
                if not isinstance(descriptor, LazyVariableDescriptor):
                    continue
                descriptor.__set__(model, value)
        return self
