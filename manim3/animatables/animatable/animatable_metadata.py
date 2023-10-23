from __future__ import annotations


from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Self
)

import attrs

from ...lazy.lazy_descriptor import LazySingularDescriptor

if TYPE_CHECKING:
    from .animatable import Animatable


@attrs.frozen(kw_only=True)
class AnimatableMetadata[AnimatableT: Animatable]:
    _registration: ClassVar[dict[LazySingularDescriptor, AnimatableMetadata]] = {}

    converter: Callable[[Any], AnimatableT]
    interpolate: bool = False
    piecewise: bool = False

    def __call__(
        self: Self,
        descriptor: LazySingularDescriptor[AnimatableT]
    ) -> LazySingularDescriptor[AnimatableT]:
        type(self)._registration[descriptor] = self
        return descriptor

    @classmethod
    def get(
        cls: type[Self],
        descriptor: LazySingularDescriptor[AnimatableT]
    ) -> AnimatableMetadata[AnimatableT] | None:
        return cls._registration.get(descriptor)
