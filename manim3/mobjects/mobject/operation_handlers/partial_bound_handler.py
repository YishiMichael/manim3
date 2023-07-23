from typing import TYPE_CHECKING

from ....lazy.lazy import LazyVariableDescriptor
from .mobject_operation import MobjectOperation
from .partial_handler import PartialHandler

if TYPE_CHECKING:
    from ..mobject import Mobject


class PartialBoundHandler:
    __slots__ = (
        "_dst",
        "_partial_handler_dict"
    )

    def __init__(
        self,
        dst: "Mobject",
        src: "Mobject"
    ) -> None:
        super().__init__()
        partial_handler_dict: dict[LazyVariableDescriptor, PartialHandler] = {}
        for info in MobjectOperation._operation_infos:
            descriptor = info.descriptor
            partial_handler_cls = info.partial_handler_cls
            if not all(
                descriptor in type(mobject)._lazy_variable_descriptors
                for mobject in (dst, src)
            ):
                continue
            if partial_handler_cls is None:
                # Do not construct the handler if the method is not provided.
                continue
            src_container = descriptor.get_container(src)
            partial_handler_dict[descriptor] = partial_handler_cls(
                descriptor.converter.c2r(src_container)
            )

        self._dst: "Mobject" = dst
        self._partial_handler_dict: dict[LazyVariableDescriptor, PartialHandler] = partial_handler_dict

    def partial(
        self,
        alpha_0: float,
        alpha_1: float
    ) -> None:
        dst = self._dst
        for descriptor, partial_handler in self._partial_handler_dict.items():
            dst_container = descriptor.converter.r2c(partial_handler.partial(alpha_0, alpha_1))
            descriptor.set_container(dst, dst_container)
