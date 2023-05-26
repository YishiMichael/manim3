from abc import (
    ABC,
    abstractmethod
)
from dataclasses import dataclass
import itertools as it
#import operator as op
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Iterator,
    TypeVar,
    ParamSpec,
    overload
)
import warnings
import weakref

import moderngl
import numpy as np

from ..constants import (
    ORIGIN,
    PI
)
from ..config import ConfigSingleton
from ..custom_typing import (
    ColorT,
    NP_3f8,
    NP_44f8,
    NP_x3f8
)
from ..geometries.geometry import Geometry
from ..lazy.lazy import (
    Lazy,
    #LazyArrayConverter,
    LazyCollectionConverter,
    LazyContainer,
    #LazyExternalConverter,
    #LazyIndividualConverter,
    LazyObject,
    #LazyUnitaryContainer,
    LazyVariableDescriptor
    #LazyWrapper
)
from ..rendering.gl_buffer import UniformBlockBuffer
from ..shape.line_string import MultiLineString
from ..shape.shape import Shape
from ..utils.color import ColorUtils
#from ..utils.iterables import IterUtils
from ..utils.space import SpaceUtils


_MobjectT = TypeVar("_MobjectT", bound="Mobject")
_ContainerT = TypeVar("_ContainerT", bound="LazyContainer")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
#_DescriptorGetT = TypeVar("_DescriptorGetT")
#_DescriptorSetT = TypeVar("_DescriptorSetT")
#_DescriptorRawT = TypeVar("_DescriptorRawT")
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
        mobject: "Mobject"
    ) -> NP_3f8:
        pass


class AboutPoint(AboutABC):
    __slots__ = ("_point",)

    def __init__(
        self,
        point: NP_3f8
    ) -> None:
        super().__init__()
        self._point: NP_3f8 = point

    def _get_about_point(
        self,
        mobject: "Mobject"
    ) -> NP_3f8:
        return self._point


class AboutEdge(AboutABC):
    __slots__ = ("_edge",)

    def __init__(
        self,
        edge: NP_3f8
    ) -> None:
        super().__init__()
        self._edge: NP_3f8 = edge

    def _get_about_point(
        self,
        mobject: "Mobject"
    ) -> NP_3f8:
        return mobject.get_bounding_box_point(self._edge)


class AboutCenter(AboutEdge):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            edge=ORIGIN
        )


class AlignABC(ABC):
    __slots__ = (
        "_direction",
        "_buff"
    )

    def __init__(
        self,
        direction: NP_3f8,
        buff: float | NP_3f8
    ) -> None:
        super().__init__()
        self._direction: NP_3f8 = direction
        self._buff: float | NP_3f8 = buff

    @abstractmethod
    def _get_target_point(self) -> NP_3f8:
        pass

    def _get_shift_vector(
        self,
        mobject: "Mobject",
        direction_sign: float
    ) -> NP_3f8:
        target_point = self._get_target_point()
        direction = direction_sign * self._direction
        point_to_align = mobject.get_bounding_box_point(direction) + self._buff * direction
        return target_point - point_to_align


class AlignPoint(AlignABC):
    __slots__ = ("_point",)

    def __init__(
        self,
        point: NP_3f8,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__(
            direction=direction,
            buff=buff
        )
        self._point: NP_3f8 = point

    def _get_target_point(self) -> NP_3f8:
        return self._point


class AlignMobject(AlignABC):
    __slots__ = ("_mobject",)

    def __init__(
        self,
        mobject: "Mobject",
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__(
            direction=direction,
            buff=buff
        )
        self._mobject: "Mobject" = mobject

    def _get_target_point(self) -> NP_3f8:
        return self._mobject.get_bounding_box_point(self._direction)


class AlignBorder(AlignABC):
    __slots__ = ()

    def _get_target_point(self) -> NP_3f8:
        return self._direction * np.append(ConfigSingleton().size.frame_radii, 0.0)


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


class MobjectStyleMeta:
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
    ) -> "Callable[..., Callable[[Mobject], Callable[_MethodParams, None]]]":

        def get_descriptor_callback(
            descriptor: LazyVariableDescriptor,
            method: Callable[..., Callable[_MethodParams, Any] | None],
            srcs: tuple[Mobject, ...]
        ) -> Callable[[Mobject], Callable[_MethodParams, None]] | None:
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
                dst: Mobject
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
            *srcs: Mobject
        ) -> Callable[[Mobject], Callable[_MethodParams, None]]:
            descriptor_callbacks = [
                descriptor_callback
                for descriptor, method in method_dict.items()
                if (descriptor_callback := get_descriptor_callback(descriptor, method, srcs)) is not None
            ]

            def src_callback(
                dst: Mobject
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
            #method_callback = method(descriptor.converter.c2r(src_container))

            #def dst_callback(
            #    *args: _MethodParams.args,
            #    **kwargs: _MethodParams.kwargs
            #) -> _ContainerT:
            #    return descriptor.converter.r2c(method_callback(*args, **kwargs))

            #return dst_callback

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
            #src_value_0 = descriptor.converter.c2r(src_container_0)
            #src_value_1 = descriptor.converter.c2r(src_container_1)
            if src_container_0._match_elements(src_container_1):
                # Do not make into callback if interpolated variables match.
                # This is a feature used by compositing animations played on the same mobject
                # which interpolate different variables.
                return None
            if method is None:
                raise ValueError(f"Uninterpolable variables of `{descriptor.method.__name__}` don't match")

            return cls._get_dst_callback(descriptor, method, src_container_0, src_container_1)
            #method_callback = method(src_value_0, src_value_1)

            #def dst_callback(
            #    *args: _MethodParams.args,
            #    **kwargs: _MethodParams.kwargs
            #) -> _ContainerT:
            #    return descriptor.converter.r2c(method_callback(*args, **kwargs))

            #return dst_callback

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
            #method_callback = method(*src_values)

            #def dst_callback(
            #    *args: _MethodParams.args,
            #    **kwargs: _MethodParams.kwargs
            #) -> _ContainerT:
            #    return descriptor.converter.r2c(method_callback(*args, **kwargs))

            #return dst_callback

        return new_method

    @classmethod
    @property
    def _partial(cls) -> "Callable[[Mobject], Callable[[Mobject], Callable[[float, float], None]]]":
        return cls._get_callback_from({
            info.descriptor: info.partial_method
            for info in cls._style_descriptor_infos
        })

    @classmethod
    @property
    def _interpolate(cls) -> "Callable[[Mobject, Mobject], Callable[[Mobject], Callable[[float], None]]]":
        return cls._get_callback_from({
            info.descriptor: info.interpolate_method
            for info in cls._style_descriptor_infos
        })

    @classmethod
    @property
    def _concatenate(cls) -> "Callable[..., Callable[[Mobject], Callable[[], None]]]":
        return cls._get_callback_from({
            info.descriptor: info.concatenate_method
            for info in cls._style_descriptor_infos
        })

    #@classmethod
    #def _set_style(
    #    cls,
    #    mobjects: "Iterable[Mobject]",
    #    style: dict[str, Any]
    #) -> None:
    #    for mobject in mobjects:
    #        for key, value in style.items():
    #            if (descriptor := type(mobject)._lazy_descriptor_dict.get(key)) is None:
    #                continue
    #            if not isinstance(descriptor, LazyVariableDescriptor):
    #                continue
    #            descriptor.__set__(mobject, value)
    #    type_to_mobjects = dict(IterUtils.categorize(
    #        (type(mobject), mobject)
    #        for mobject in mobjects
    #    ))
    #    for key, value in style.items():
    #        for descriptor in cls._name_to_descriptors_dict[f"_{key}_"]:
    #            #if isinstance(value, Mobject):
    #            #    if descriptor not in type(value)._lazy_variable_descriptors:
    #            #        continue
    #            #    style_container = descriptor.get_container(value)
    #            #else:
    #            if isinstance(descriptor.converter, LazyIndividualConverter):
    #                assert isinstance(value, LazyObject)
    #                style_container = LazyUnitaryContainer(element=value)
    #            elif isinstance(descriptor.converter, LazyExternalConverter):
    #                assert not isinstance(value, LazyObject)
    #                style_container = LazyUnitaryContainer(element=LazyWrapper(value))
    #            elif isinstance(descriptor.converter, LazyArrayConverter):
    #                if isinstance(value, int | float):
    #                    value *= np.ones(())
    #                assert isinstance(value, np.ndarray)
    #                style_container = LazyUnitaryContainer(element=LazyWrapper(value.astype(np.float64)))
    #            else:
    #                raise TypeError
    #            for mobject_type, typed_mobjects in type_to_mobjects.items():
    #                if descriptor not in mobject_type._lazy_variable_descriptors:
    #                    continue
    #                for mobject in typed_mobjects:
    #                    descriptor.set_container(mobject, style_container)

    #@classmethod
    #def _match_style(
    #    cls,
    #    mobjects: "Iterable[Mobject]",
    #    target: "Mobject",
    #    style_names: list[str]
    #) -> None:
    #    cls._set_style(
    #        mobjects=mobjects,
    #        style={
    #            name: target
    #            for name in style_names
    #        }
    #    )


class Mobject(LazyObject):
    __slots__ = (
        "_parents",
        "_real_ancestors"
    )

    def __init__(self) -> None:
        super().__init__()
        # If `_parents` and `_real_ancestors` are implemented with `LazyDynamicContainer` also,
        # loops will pop up in the DAG of the lazy system.
        self._parents: weakref.WeakSet[Mobject] = weakref.WeakSet()
        self._real_ancestors: weakref.WeakSet[Mobject] = weakref.WeakSet()

    def __iter__(self) -> "Iterator[Mobject]":
        return iter(self._children_)

    @overload
    def __getitem__(
        self,
        index: int
    ) -> "Mobject": ...

    @overload
    def __getitem__(
        self,
        index: slice
    ) -> "list[Mobject]": ...

    def __getitem__(
        self,
        index: int | slice
    ) -> "Mobject | list[Mobject]":
        return self._children_.__getitem__(index)

    # family matters
    # These methods implement a DAG (directed acyclic graph).

    @Lazy.variable_collection
    @classmethod
    def _children_(cls) -> "list[Mobject]":
        return []

    @Lazy.variable_collection
    @classmethod
    def _real_descendants_(cls) -> "list[Mobject]":
        return []

    @classmethod
    def _refresh_families(
        cls,
        *mobjects: "Mobject"
    ) -> None:

        def iter_descendants_by_children(
            mobject: Mobject
        ) -> Iterator[Mobject]:
            yield mobject
            for child in mobject._children_:
                yield from iter_descendants_by_children(child)

        def iter_ancestors_by_parents(
            mobject: Mobject
        ) -> Iterator[Mobject]:
            yield mobject
            for parent in mobject._parents:
                yield from iter_ancestors_by_parents(parent)

        for ancestor in dict.fromkeys(it.chain.from_iterable(
            iter_ancestors_by_parents(mobject)
            for mobject in mobjects
        )):
            ancestor._real_descendants_.reset(dict.fromkeys(it.chain.from_iterable(
                iter_descendants_by_children(child)
                for child in ancestor._children_
            )))
        for descendant in dict.fromkeys(it.chain.from_iterable(
            iter_descendants_by_children(mobject)
            for mobject in mobjects
        )):
            descendant._real_ancestors.clear()
            descendant._real_ancestors.update(dict.fromkeys(it.chain.from_iterable(
                iter_ancestors_by_parents(parent)
                for parent in descendant._parents
            )))

    def iter_children(self) -> "Iterator[Mobject]":
        yield from self._children_

    def iter_parents(self) -> "Iterator[Mobject]":
        yield from self._parents

    def iter_descendants(
        self,
        *,
        broadcast: bool = True
    ) -> "Iterator[Mobject]":
        yield self
        if broadcast:
            yield from self._real_descendants_

    def iter_ancestors(
        self,
        *,
        broadcast: bool = True
    ) -> "Iterator[Mobject]":
        yield self
        if broadcast:
            yield from self._real_ancestors

    def iter_children_by_type(
        self,
        mobject_type: type[_MobjectT]
    ) -> Iterator[_MobjectT]:
        for mobject in self.iter_children():
            if isinstance(mobject, mobject_type):
                yield mobject

    def iter_descendants_by_type(
        self,
        mobject_type: type[_MobjectT],
        broadcast: bool = True
    ) -> Iterator[_MobjectT]:
        for mobject in self.iter_descendants(broadcast=broadcast):
            if isinstance(mobject, mobject_type):
                yield mobject

    def add(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject not in self._children_
        ]
        if (invalid_mobjects := [
            mobject for mobject in filtered_mobjects
            if mobject in self.iter_ancestors()
        ]):
            raise ValueError(f"Circular relationship occurred when adding {invalid_mobjects} to {self}")
        self._children_.extend(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._parents.add(self)
        self._refresh_families(self)
        return self

    def discard(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject in self._children_
        ]
        self._children_.eliminate(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._parents.remove(self)
        self._refresh_families(self, *filtered_mobjects)
        return self

    def added_by(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject not in self._parents
        ]
        if (invalid_mobjects := [
            mobject for mobject in filtered_mobjects
            if mobject in self.iter_descendants()
        ]):
            raise ValueError(f"Circular relationship occurred when adding {self} to {invalid_mobjects}")
        self._parents.update(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._children_.append(self)
        self._refresh_families(self)
        return self

    def discarded_by(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject in self._parents
        ]
        self._parents.difference_update(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._children_.remove(self)
        self._refresh_families(self, *filtered_mobjects)
        return self

    def clear(self):
        self.discard(*self.iter_children())
        return self

    def copy(self):
        # Copy all descendants. The result is not bound to any mobject.
        result = self._copy()
        real_descendants = list(self._real_descendants_)
        real_descendants_copy = [
            descendant._copy()
            for descendant in real_descendants
        ]
        descendants = [self, *real_descendants]
        descendants_copy = [result, *real_descendants_copy]

        def match_copies(
            mobjects: Iterable[Mobject]
        ) -> Iterator[Mobject]:
            return (
                descendants_copy[descendants.index(mobject)] if mobject in descendants else mobject
                for mobject in mobjects
            )

        result._children_.reset(match_copies(self._children_))
        result._parents.clear()
        for real_descendant, real_descendant_copy in zip(real_descendants, real_descendants_copy, strict=True):
            real_descendant_copy._children_.reset(match_copies(real_descendant._children_))
            real_descendant_copy._parents.clear()
            real_descendant_copy._parents.update(match_copies(real_descendant._parents))

        self._refresh_families(*descendants_copy)
        return result

    # bounding box

    @MobjectStyleMeta.register(
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

    @Lazy.property_hashable
    @classmethod
    def _has_local_sample_points_(
        cls,
        local_sample_points: NP_x3f8
    ) -> bool:
        return bool(len(local_sample_points))

    @Lazy.property_external
    @classmethod
    def _bounding_box_without_descendants_(
        cls,
        model_matrix: NP_44f8,
        local_sample_points: NP_x3f8,
        has_local_sample_points: bool
    ) -> BoundingBox | None:
        if not has_local_sample_points:
            return None
        world_sample_points = SpaceUtils.apply_affine(model_matrix, local_sample_points)
        return BoundingBox(
            maximum=world_sample_points.max(axis=0),
            minimum=world_sample_points.min(axis=0)
        )

    @Lazy.property_external
    @classmethod
    def _bounding_box_with_descendants_(
        cls,
        bounding_box_without_descendants: BoundingBox | None,
        real_descendants__bounding_box_without_descendants: list[BoundingBox | None]
    ) -> BoundingBox | None:
        points_array = np.array(list(it.chain.from_iterable(
            (bounding_box.maximum, bounding_box.minimum)
            for bounding_box in (
                bounding_box_without_descendants,
                *real_descendants__bounding_box_without_descendants
            )
            if bounding_box is not None
        )))
        if not len(points_array):
            return None
        return BoundingBox(
            maximum=points_array.max(axis=0),
            minimum=points_array.min(axis=0)
        )

    def get_bounding_box(
        self,
        *,
        broadcast: bool = True
    ) -> BoundingBox:
        if broadcast:
            result = self._bounding_box_with_descendants_
        else:
            result = self._bounding_box_without_descendants_
        assert result is not None, "Trying to calculate the bounding box of some mobject with no points"
        return result

    def get_bounding_box_size(
        self,
        *,
        broadcast: bool = True
    ) -> NP_3f8:
        bounding_box = self.get_bounding_box(broadcast=broadcast)
        return bounding_box.radii * 2.0

    def get_bounding_box_point(
        self,
        direction: NP_3f8,
        *,
        broadcast: bool = True
    ) -> NP_3f8:
        bounding_box = self.get_bounding_box(broadcast=broadcast)
        return bounding_box.center + direction * bounding_box.radii

    def get_center(
        self,
        *,
        broadcast: bool = True
    ) -> NP_3f8:
        return self.get_bounding_box_point(ORIGIN, broadcast=broadcast)

    # transform

    def _make_callback_relative(
        self,
        matrix_callback: Callable[[float | NP_3f8], NP_44f8],
        about: AboutABC | None
    ) -> Callable[[float | NP_3f8], NP_44f8]:
        if about is None:
            return matrix_callback
        about_point = about._get_about_point(mobject=self)
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
        # Keep shared model matrices shared during transform.
        #model_matrix_to_mobjects = dict(IterUtils.categorize(
        #    (Mobject._model_matrix_.get_container(mobject)._element, mobject)
        #    for mobject in self.iter_descendants()
        #))
        mobject_to_model_matrix = {
            mobject: mobject._model_matrix_
            for mobject in self.iter_descendants()
        }

        def callback(
            alpha: float
        ) -> None:
            for mobject, model_matrix in mobject_to_model_matrix.items():
                mobject._model_matrix_ = matrix_callback(alpha) @ model_matrix
            #for model_matrix, mobjects in model_matrix_to_mobjects.items():
            #    transformed_matrix = LazyWrapper(matrix_callback(alpha) @ model_matrix._value)
            #    for mobject in mobjects:
            #        Mobject._model_matrix_.set_container(mobject, LazyUnitaryContainer(
            #            element=transformed_matrix
            #        ))

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
            vector=align._get_shift_vector(mobject=self, direction_sign=1.0),
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
            vector=align._get_shift_vector(mobject=self, direction_sign=-1.0),
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
        mobject: "Mobject"
    ):
        self.move_to(AlignMobject(mobject)).scale_to(mobject.get_bounding_box_size())
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

    def concatenate(self):
        MobjectStyleMeta._concatenate(*self.iter_children())(self)()
        self.clear()
        return self

    def set_style(
        self,
        *,
        # polymorphism variables
        color: ColorT = ...,
        opacity: float = ...,

        # Mobject
        model_matrix: NP_44f8 = ...,

        # RenderableMobject
        is_transparent: bool = ...,

        # MeshMobject
        geometry: Geometry = ...,
        color_map: moderngl.Texture | None = ...,
        enable_phong_lighting: bool = ...,
        ambient_strength: float = ...,
        specular_strength: float = ...,
        shininess: float = ...,

        # ShapeMobject
        shape: Shape = ...,

        # StrokeMobject
        multi_line_string: MultiLineString = ...,
        width: float = ...,
        single_sided: bool = ...,
        has_linecap: bool = ...,
        dilate: float = ...,

        # setting configs
        broadcast: bool = True,
        type_filter: "type[Mobject] | None" = None
    ):

        def standardize_input(
            value: Any
        ) -> np.ndarray:
            if not isinstance(value, float | int | np.ndarray):
                return value
            return (value * np.ones(())).astype(np.float64)

        if color is not ...:
            color = ColorUtils.standardize_color(color)
        #style = {
        #    key: value
        #    for key, value in {
        #        "color": color_component,
        #        "opacity": opacity_component
        #    }.items() if value is not None
        #}
        style = {
            f"_{key}_": standardize_input(value)
            for key, value in {
                "color": color,
                "opacity": opacity,
                "model_matrix": model_matrix,
                "is_transparent": is_transparent,
                "geometry": geometry,
                "color_map": color_map,
                "enable_phong_lighting": enable_phong_lighting,
                "ambient_strength": ambient_strength,
                "specular_strength": specular_strength,
                "shininess": shininess,
                "shape": shape,
                "multi_line_string": multi_line_string,
                "width": width,
                "single_sided": single_sided,
                "has_linecap": has_linecap,
                "dilate": dilate
            }.items() if value is not ...
        }

        if type_filter is None:
            type_filter = Mobject

        for mobject in self.iter_descendants_by_type(mobject_type=type_filter, broadcast=broadcast):
            for key, value in style.items():
                if (descriptor := type(mobject)._lazy_descriptor_dict.get(key)) is None:
                    continue
                if not isinstance(descriptor, LazyVariableDescriptor):
                    continue
                descriptor.__set__(mobject, value)
        return self

    # TODO: remove
    #def match_style(
    #    self,
    #    target: "Mobject",
    #    *,

    #    color: bool = ...,
    #    opacity: bool = ...,
    #    model_matrix: bool = ...,
    #    is_transparent: bool = ...,
    #    geometry: bool = ...,
    #    color_map: bool = ...,
    #    enable_phong_lighting: bool = ...,
    #    ambient_strength: bool = ...,
    #    specular_strength: bool = ...,
    #    shininess: bool = ...,
    #    shape: bool = ...,
    #    multi_line_string: bool = ...,
    #    width: bool = ...,
    #    single_sided: bool = ...,
    #    has_linecap: bool = ...,
    #    dilate: bool = ...,

    #    default: bool = True,
    #    broadcast: bool = True,
    #    type_filter: "type[Mobject] | None" = None
    #):
    #    style_names = [
    #        name
    #        for name, checked in {
    #            "color": color,
    #            "opacity": opacity,
    #            "model_matrix": model_matrix,
    #            "is_transparent": is_transparent,
    #            "geometry": geometry,
    #            "color_map": color_map,
    #            "enable_phong_lighting": enable_phong_lighting,
    #            "ambient_strength": ambient_strength,
    #            "specular_strength": specular_strength,
    #            "shininess": shininess,
    #            "shape": shape,
    #            "multi_line_string": multi_line_string,
    #            "width": width,
    #            "single_sided": single_sided,
    #            "has_linecap": has_linecap,
    #            "dilate": dilate
    #        }.items()
    #        if checked is ... and default or checked is not ... and checked
    #    ]

    #    if type_filter is None:
    #        type_filter = Mobject
    #    MobjectStyleMeta._match_style(
    #        mobjects=self.iter_descendants_by_type(mobject_type=type_filter, broadcast=broadcast),
    #        target=target,
    #        style_names=style_names
    #    )
