from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
    ParamSpec
)

from ...lazy.lazy import (
    LazyCollectionConverter,
    LazyContainer,
    LazyObject,
    LazyVariableDescriptor
)

if TYPE_CHECKING:
    from .mobject import Mobject


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
class StyleDescriptorInfo(Generic[_InstanceT, _ContainerT, _DataT, _DataRawT]):
    descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
    partial_method: Callable[[_ContainerT], Callable[[float, float], _ContainerT] | None]
    interpolate_method: Callable[[_ContainerT, _ContainerT], Callable[[float], _ContainerT] | None]
    concatenate_method: Callable[..., Callable[[], _ContainerT] | None]


class MobjectStyleMeta:
    __slots__ = ()

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
    ) -> Callable[
        [LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]],
        LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
    ]:

        def callback(
            descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]
        ) -> LazyVariableDescriptor[_InstanceT, _ContainerT, _DataT, _DataRawT]:
            assert not isinstance(descriptor.converter, LazyCollectionConverter)
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
            srcs: "tuple[Mobject, ...]"
        ) -> "Callable[[Mobject], Callable[_MethodParams, None]] | None":
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
                dst: "Mobject"
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
            *srcs: "Mobject"
        ) -> "Callable[[Mobject], Callable[_MethodParams, None]]":
            descriptor_callbacks = [
                descriptor_callback
                for descriptor, method in method_dict.items()
                if (descriptor_callback := get_descriptor_callback(descriptor, method, srcs)) is not None
            ]

            def src_callback(
                dst: "Mobject"
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
                # This is a feature used by compositing animations played on the same mobject
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
