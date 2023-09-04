#from typing import TYPE_CHECKING

#from ....constants.custom_typing import NP_xf8
#from ....lazy.lazy import LazyVariableDescriptor
#from ..style_meta import StyleMeta
#from .split_handler import SplitHandler

#if TYPE_CHECKING:
#    from ..mobject import Mobject


#class SplitBoundHandler:
#    __slots__ = (
#        "_dst",
#        "_split_handler_dict"
#    )

#    def __init__(
#        self,
#        dst: "Mobject",
#        src: "Mobject"
#    ) -> None:
#        super().__init__()
#        split_handler_dict: dict[LazyVariableDescriptor, SplitHandler] = {}
#        for info in StyleMeta._operation_infos:
#            descriptor = info.descriptor
#            split_handler_cls = info.split_handler_cls
#            if not all(
#                descriptor in type(mobject)._lazy_variable_descriptors
#                for mobject in (dst, src)
#            ):
#                continue
#            if split_handler_cls is None:
#                # Do not construct the handler if the method is not provided.
#                continue
#            src_container = descriptor.get_container(src)
#            split_handler_dict[descriptor] = split_handler_cls(
#                descriptor.converter.c2r(src_container)
#            )

#        self._dst: "Mobject" = dst
#        self._split_handler_dict: dict[LazyVariableDescriptor, SplitHandler] = split_handler_dict

#    def split(
#        self,
#        alpha_0: float,
#        alpha_1: float
#        #alphas: NP_xf8
#    ) -> None:
#        dst = self._dst
#        for descriptor, split_handler in self._split_handler_dict.items():
#            dst_container = descriptor.converter.r2c(split_handler.split(alpha_0, alpha_1))
#            descriptor.set_container(dst, dst_container)
