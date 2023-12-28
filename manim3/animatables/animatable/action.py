from __future__ import annotations


from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Iterator,
    Self,
    overload
)

import attrs

from ...lazy.lazy_descriptor import LazyDescriptor

if TYPE_CHECKING:
    from .animatable import (
        Animatable,
        AnimatableTimeline,
        Animation
    )


@attrs.frozen(kw_only=True)
class DescriptorParameters:
    pass


@attrs.frozen(kw_only=True)
class ConverterDescriptorParameters(DescriptorParameters):
    converter: Callable[[Any], Animatable]


class Action[AnimatableT: Animatable, **P]:
    __slots__ = ("_method",)

    def __init__(
        self: Self,
        method: Callable[Concatenate[type[AnimatableT], AnimatableT, P], Iterator[Animation]]
    ) -> None:
        super().__init__()
        self._method: Callable[Concatenate[type[AnimatableT], AnimatableT, P], Iterator[Animation]] = method

    @overload
    def __get__[InstanceT: Animatable](
        self: Self,
        instance: InstanceT,
        owner: type[InstanceT] | None = None
    ) -> Callable[P, InstanceT]: ...

    @overload
    def __get__[InstanceT: AnimatableTimeline[Animatable]](
        self: Self,
        instance: InstanceT,
        owner: type[InstanceT] | None = None
    ) -> Callable[P, InstanceT]: ...

    @overload
    def __get__(
        self: Self,
        instance: None,
        owner: type[Animatable | AnimatableTimeline[Animatable]] | None = None
    ) -> Self: ...

    def __get__(
        self: Self,
        instance: Any,
        owner: Any | None = None
    ) -> Any:

        def get_bound_method(
            action: Self,
            instance: Any
        ) -> Any:

            def result(
                *args: P.args,
                **kwargs: P.kwargs
            ) -> Any:
                for animation in action.iter_animations(instance, *args, **kwargs):
                    animation.update(1.0)
                return instance

            return result

        def get_timeline_bound_method(
            action: Self,
            instance: Any
        ) -> Any:

            def result(
                *args: P.args,
                **kwargs: P.kwargs
            ) -> Any:
                instance._animations.extend(action.iter_animations(instance._dst, *args, **kwargs))
                return instance

            return result

        if instance is None:
            return self

        from .animatable import (
            Animatable,
            AnimatableTimeline
        )

        if isinstance(instance, Animatable):
            return get_bound_method(self, instance)
        if isinstance(instance, AnimatableTimeline):
            return get_timeline_bound_method(self, instance)
        raise TypeError

    @classmethod
    def register(
        cls: type[Self]
    ) -> Callable[[Callable[Concatenate[type[AnimatableT], AnimatableT, P], Iterator[Animation]]], Self]:

        def result(
            method: Callable[Concatenate[type[AnimatableT], AnimatableT, P], Iterator[Animation]]
        ) -> Self:
            assert isinstance(method, classmethod)
            return cls(method.__func__)

        return result

    def iter_animations(
        self: Self,
        dst: AnimatableT,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> Iterator[Animation]:
        return self._method(type(dst), dst, *args, **kwargs)


class DescriptiveAction[AnimatableT: Animatable, DescriptorParametersT: DescriptorParameters, **P](Action[AnimatableT, P]):
    __slots__ = (
        "_descriptor_parameters_cls",
        "_descriptor_items"
    )

    def __init__(
        self: Self,
        descriptor_parameters_cls: type[DescriptorParametersT],
        method: Callable[Concatenate[type[AnimatableT], AnimatableT, P], Iterator[Animation]]
    ) -> None:
        super().__init__(method)
        self._descriptor_parameters_cls: type[DescriptorParametersT] = descriptor_parameters_cls
        self._descriptor_items: list[tuple[LazyDescriptor, DescriptorParametersT]] = []

    def iter_descriptor_items(
        self: Self
    ) -> Iterator[tuple[LazyDescriptor, DescriptorParametersT]]:
        yield from self._descriptor_items

    @classmethod
    def descriptive_register(
        cls: type[Self],
        descriptor_parameters_cls: type[DescriptorParametersT]
    ) -> Callable[[Callable[Concatenate[type[AnimatableT], AnimatableT, P], Iterator[Animation]]], Self]:

        def result(
            method: Callable[Concatenate[type[AnimatableT], AnimatableT, P], Iterator[Animation]]
        ) -> Self:
            assert isinstance(method, classmethod)
            return cls(descriptor_parameters_cls, method.__func__)

        return result

    def register_descriptor[LazyDescriptorT: LazyDescriptor](
        self: Self,
        **parameters: Any
    ) -> Callable[[LazyDescriptorT], LazyDescriptorT]:

        def result(
            descriptor: LazyDescriptorT
        ) -> LazyDescriptorT:
            assert not descriptor._freeze
            self._descriptor_items.append((descriptor, self._descriptor_parameters_cls(**parameters)))
            return descriptor

        return result
