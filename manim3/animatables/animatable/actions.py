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


class Action[ActionsT: Actions, AnimatableT: Animatable, DescriptorParametersT: DescriptorParameters, **P]:
    __slots__ = (
        "_descriptor_parameters_cls",
        "_descriptor_dict",
        "_method",
        "_actions_cls"
    )

    def __init__(
        self: Self,
        descriptor_parameters_cls: type[DescriptorParametersT],
        method: Callable[Concatenate[type[ActionsT], AnimatableT, P], Iterator[Animation]]
    ) -> None:
        super().__init__()
        self._descriptor_parameters_cls: type[DescriptorParametersT] = descriptor_parameters_cls
        self._descriptor_dict: dict[LazyDescriptor, DescriptorParametersT] = {}
        self._method: Callable[Concatenate[type[ActionsT], AnimatableT, P], Iterator[Animation]] = method
        self._actions_cls: type[ActionsT] = NotImplemented

    def __call__(
        self: Self,
        dst: AnimatableT,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> Iterator[Animation]:
        return self._method(self._actions_cls, dst, *args, **kwargs)

    @classmethod
    def register[ActionsT_: Actions, AnimatableT_: Animatable, DescriptorParametersT_: DescriptorParameters, **P_](
        cls: type[Self],
        descriptor_parameters_cls: type[DescriptorParametersT_] = DescriptorParameters
    ) -> Callable[
        [Callable[Concatenate[type[ActionsT_], AnimatableT_, P_], Iterator[Animation]]],
        Action[ActionsT_, AnimatableT_, DescriptorParametersT_, P_]
    ]:

        def result(
            method: Callable[Concatenate[type[ActionsT_], AnimatableT_, P_], Iterator[Animation]]
        ) -> Action[ActionsT_, AnimatableT_, DescriptorParametersT_, P_]:
            assert isinstance(method, classmethod)
            return Action(descriptor_parameters_cls, method.__func__)

        return result

    def register_descriptor[LazyDescriptorT: LazyDescriptor](
        self: Self,
        **parameters: Any
    ) -> Callable[[LazyDescriptorT], LazyDescriptorT]:

        def result(
            descriptor: LazyDescriptorT
        ) -> LazyDescriptorT:
            assert not descriptor._freeze
            self._descriptor_dict[descriptor] = self._descriptor_parameters_cls(**parameters)
            return descriptor

        return result

    def build_action_descriptor(
        self: Self
    ) -> ActionDescriptor[ActionsT, AnimatableT, DescriptorParametersT, P]:
        return ActionDescriptor(self)

    def build_dynamic_action_descriptor(
        self: Self
    ) -> DynamicActionDescriptor[ActionsT, AnimatableT, DescriptorParametersT, P]:
        return DynamicActionDescriptor(self)


class ActionDescriptor[ActionsT: Actions, AnimatableT: Animatable, DescriptorParametersT: DescriptorParameters, **P]:
    __slots__ = ("_action",)

    def __init__(
        self: Self,
        action: Action[ActionsT, AnimatableT, DescriptorParametersT, P]
    ) -> None:
        super().__init__()
        self._action: Action[ActionsT, AnimatableT, DescriptorParametersT, P] = action

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
            for animation in self._action(instance, *args, **kwargs):
                animation.update_boundary(1)
            return instance

        return bound_method


class DynamicActionDescriptor[ActionsT: Actions, AnimatableT: Animatable, DescriptorParametersT: DescriptorParameters, **P](
    ActionDescriptor[ActionsT, AnimatableT, DescriptorParametersT, P]
):
    __slots__ = ()

    @overload
    def __get__[InstanceT: DynamicAnimatable](
        self: Self,
        instance: InstanceT,
        owner: type[InstanceT] | None = None
    ) -> Callable[P, InstanceT]: ...

    @overload
    def __get__[InstanceT: DynamicAnimatable](
        self: Self,
        instance: None,
        owner: type[InstanceT] | None = None
    ) -> Self: ...

    def __get__[InstanceT: DynamicAnimatable](
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
            instance._animations.extend(self._action(instance._dst, *args, **kwargs))
            return instance

        return bound_method
