from __future__ import annotations


from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Iterator,
    Never,
    Self,
    overload
)

import attrs

from ...lazy.lazy_descriptor import LazyDescriptor

if TYPE_CHECKING:
    from .animatable import (
        Animatable,
        Animation,
        DynamicAnimatable
    )


@attrs.frozen(kw_only=True)
class DescriptorParameters:
    pass


@attrs.frozen(kw_only=True)
class ConverterDescriptorParameters(DescriptorParameters):
    converter: Callable[[Any], Animatable] | None = None


class Actions:
    __slots__ = ()

    def __init_subclass__(
        cls: type[Self]
    ) -> None:
        super().__init_subclass__()
        for action in cls.__dict__.values():
            if not isinstance(action, Action):
                continue
            action._actions_cls = cls

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError


class Action[ActionsT: Actions, AnimatableT: Animatable, **P]:
    __slots__ = (
        "_method",
        "_actions_cls"
    )

    def __init__(
        self: Self,
        method: Callable[Concatenate[type[ActionsT], AnimatableT, P], Iterator[Animation]]
    ) -> None:
        super().__init__()
        self._method: Callable[Concatenate[type[ActionsT], AnimatableT, P], Iterator[Animation]] = method
        self._actions_cls: type[ActionsT] = NotImplemented

    def iter_animations(
        self: Self,
        dst: AnimatableT,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> Iterator[Animation]:
        return self._method(self._actions_cls, dst, *args, **kwargs)

    @classmethod
    def register(
        cls: type[Self]
    ) -> Callable[
        [Callable[Concatenate[type[ActionsT], AnimatableT, P], Iterator[Animation]]],
        Self
    ]:

        def result(
            method: Callable[Concatenate[type[ActionsT], AnimatableT, P], Iterator[Animation]]
        ) -> Self:
            assert isinstance(method, classmethod)
            return cls(method.__func__)

        return result

    def build_action_descriptor(
        self: Self
    ) -> ActionDescriptor[ActionsT, AnimatableT, P]:
        return ActionDescriptor(self)

    def build_dynamic_action_descriptor(
        self: Self
    ) -> DynamicActionDescriptor[ActionsT, AnimatableT, P]:
        return DynamicActionDescriptor(self)


class DescriptiveAction[ActionsT: Actions, AnimatableT: Animatable, DescriptorParametersT: DescriptorParameters, **P](
    Action[ActionsT, AnimatableT, P]
):
    __slots__ = (
        "_descriptor_parameters_cls",
        "_descriptor_items"
    )

    def __init__(
        self: Self,
        descriptor_parameters_cls: type[DescriptorParametersT],
        method: Callable[Concatenate[type[ActionsT], AnimatableT, P], Iterator[Animation]]
    ) -> None:
        super().__init__(method)
        self._descriptor_parameters_cls: type[DescriptorParametersT] = descriptor_parameters_cls
        self._descriptor_items: list[tuple[LazyDescriptor, DescriptorParametersT]] = []

    def iter_descriptor_items(
        self: Self
    ) -> Iterator[tuple[LazyDescriptor, DescriptorParametersT]]:
        yield from self._descriptor_items

    @classmethod
    def register(
        cls: type[Self],
        descriptor_parameters_cls: type[DescriptorParametersT]
    ) -> Callable[
        [Callable[Concatenate[type[ActionsT], AnimatableT, P], Iterator[Animation]]],
        Self
    ]:

        def result(
            method: Callable[Concatenate[type[ActionsT], AnimatableT, P], Iterator[Animation]]
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


class ActionDescriptor[ActionsT: Actions, AnimatableT: Animatable, **P]:
    __slots__ = ("_action",)

    def __init__(
        self: Self,
        action: Action[ActionsT, AnimatableT, P]
    ) -> None:
        super().__init__()
        self._action: Action[ActionsT, AnimatableT, P] = action

    @overload
    def __get__[InstanceT: Animatable](
        self: Self,
        instance: InstanceT,
        owner: type[InstanceT] | None = None
    ) -> Callable[P, InstanceT]: ...

    @overload
    def __get__[InstanceT: Animatable](
        self: Self,
        instance: None,
        owner: type[InstanceT] | None = None
    ) -> Self: ...

    def __get__[InstanceT: Animatable](
        self: Self,
        instance: InstanceT | None,
        owner: type[InstanceT] | None = None
    ) -> Self | Callable[P, InstanceT]:
        if instance is None:
            return self

        def bound_method(
            *args: P.args,
            **kwargs: P.kwargs
        ) -> InstanceT:
            for animation in self._action.iter_animations(instance, *args, **kwargs):
                animation.update(1.0)
            return instance

        return bound_method


class DynamicActionDescriptor[ActionsT: Actions, AnimatableT: Animatable, **P](ActionDescriptor[ActionsT, AnimatableT, P]):
    __slots__ = ()

    @overload
    def __get__[InstanceT: DynamicAnimatable[Animatable]](
        self: Self,
        instance: InstanceT,
        owner: type[InstanceT] | None = None
    ) -> Callable[P, InstanceT]: ...

    @overload
    def __get__[InstanceT: DynamicAnimatable[Animatable]](
        self: Self,
        instance: None,
        owner: type[InstanceT] | None = None
    ) -> Self: ...

    def __get__[InstanceT: DynamicAnimatable[Animatable]](
        self: Self,
        instance: InstanceT | None,
        owner: type[InstanceT] | None = None
    ) -> Self | Callable[P, InstanceT]:
        if instance is None:
            return self

        def bound_method(
            *args: P.args,
            **kwargs: P.kwargs
        ) -> InstanceT:
            instance._animations.extend(self._action.iter_animations(instance._dst, *args, **kwargs))
            return instance

        return bound_method
