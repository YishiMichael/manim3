from typing import TYPE_CHECKING

from ....lazy.lazy import LazyVariableDescriptor
from ..style_meta import StyleMeta
from .interpolate_handler import InterpolateHandler

if TYPE_CHECKING:
    from ..mobject import Mobject


class InterpolateBoundHandler:
    __slots__ = (
        "_dst",
        "_interpolate_handler_dict"
    )

    def __init__(
        self,
        dst: "Mobject",
        src_0: "Mobject",
        src_1: "Mobject"
    ) -> None:
        super().__init__()
        interpolate_handler_dict: dict[LazyVariableDescriptor, InterpolateHandler] = {}
        for info in StyleMeta._operation_infos:
            descriptor = info.descriptor
            interpolate_handler_cls = info.interpolate_handler_cls
            if not all(
                descriptor in type(mobject)._lazy_variable_descriptors
                for mobject in (dst, src_0, src_1)
            ):
                continue
            src_container_0 = descriptor.get_container(src_0)
            src_container_1 = descriptor.get_container(src_1)
            if src_container_0._match_elements(src_container_1):
                # Do not construct the handler if interpolated variables match.
                # This is a feature used by compositing animations played on the same mobject
                # which interpolate different variables.
                continue
            if interpolate_handler_cls is None:
                raise ValueError(f"Uninterpolable variables of `{descriptor.method.__name__}` don't match")
            interpolate_handler_dict[descriptor] = interpolate_handler_cls(
                descriptor.converter.c2r(src_container_0),
                descriptor.converter.c2r(src_container_1)
            )

        self._dst: "Mobject" = dst
        self._interpolate_handler_dict: dict[LazyVariableDescriptor, InterpolateHandler] = interpolate_handler_dict

    def interpolate(
        self,
        alpha: float
    ) -> None:
        dst = self._dst
        for descriptor, interpolate_handler in self._interpolate_handler_dict.items():
            dst_container = descriptor.converter.r2c(interpolate_handler.interpolate(alpha))
            descriptor.set_container(dst, dst_container)
