#from typing import TYPE_CHECKING

#from ....lazy.lazy import (
#    LazyContainer,
#    LazyVariableDescriptor
#)
#from ..style_meta import StyleMeta
#from .concatenate_handler import ConcatenateHandler

#if TYPE_CHECKING:
#    from ..mobject import Mobject


#class ConcatenateBoundHandler:
#    __slots__ = (
#        "_dst",
#        "_concatenate_handler_dict",
#        "_copied_container_dict"
#    )

#    def __init__(
#        self,
#        dst: "Mobject",
#        *srcs: "Mobject"
#    ) -> None:
#        super().__init__()
#        concatenate_handler_dict: dict[LazyVariableDescriptor, ConcatenateHandler] = {}
#        copied_container_dict: dict[LazyVariableDescriptor, LazyContainer] = {}
#        for info in StyleMeta._operation_infos:
#            descriptor = info.descriptor
#            concatenate_handler_cls = info.concatenate_handler_cls
#            if not all(
#                descriptor in type(mobject)._lazy_variable_descriptors
#                for mobject in (dst, *srcs)
#            ):
#                continue
#            src_containers = [
#                descriptor.get_container(src)
#                for src in srcs
#            ]
#            if concatenate_handler_cls is None:
#                src_container_0 = src_containers[0]
#                if all(
#                    src_container_0._match_elements(src_container)
#                    for src_container in src_containers
#                ):
#                    # If `concatenate_handler_cls` is not provided,
#                    # check if concatenated variables match, and do copying directly.
#                    # This is a feature used by children concatenation, which tries
#                    # copying all information from children.
#                    copied_container_dict[descriptor] = src_container_0
#                    continue
#                raise ValueError(f"Uncatenatable variables of `{descriptor.method.__name__}` don't match")
#            concatenate_handler_dict[descriptor] = concatenate_handler_cls(*(
#                descriptor.converter.c2r(src_container)
#                for src_container in src_containers
#            ))

#        self._dst: "Mobject" = dst
#        self._concatenate_handler_dict: dict[LazyVariableDescriptor, ConcatenateHandler] = concatenate_handler_dict
#        self._copied_container_dict: dict[LazyVariableDescriptor, LazyContainer] = copied_container_dict

#    def concatenate(self) -> None:
#        dst = self._dst
#        for descriptor, interpolate_handler in self._concatenate_handler_dict.items():
#            dst_container = descriptor.converter.r2c(interpolate_handler.concatenate())
#            descriptor.set_container(dst, dst_container)
#        for descriptor, copied_container in self._copied_container_dict.items():
#            dst_container = copied_container._copy_container()
#            descriptor.set_container(dst, dst_container)
