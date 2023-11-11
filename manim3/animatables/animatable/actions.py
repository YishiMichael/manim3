from __future__ import annotations


from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Iterator,
    Never,
    #Protocol,
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

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for action in cls.__dict__.values():
            if not isinstance(action, Action):
                continue
            action._actions_cls = cls

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError


#class ActionMeta:
#    __slots__ = ()
#
#    def __new__(
#        cls: type[Self]
#    ) -> Never:
#        raise TypeError
        #def result(
        #    method: ActionMethodProtocol[Self, AnimatableT, P]
        #) -> Action[Self, AnimatableT, P]:
        #    return Action(method)

        #return result

    #@classmethod
    #def register_action[AnimatableT: Animatable, TypedDictT: TypedDict, **P](
    #    cls: type[Self],
    #    descriptive: Literal[True]
    #) -> Callable[
    #    [DescriptiveActionMethodProtocol[Self, AnimatableT, TypedDictT, P]],
    #    DescriptiveActionDescriptor[Self, AnimatableT, TypedDictT, P]
    #]: ...


#class ActionMeta:
#    __slots__ = ()

#    def __new__(
#        cls: type[Self]
#    ) -> Never:
#        raise TypeError

#    @overload
#    @classmethod
#    def register[ActionsT: Actions, AnimatableT: Animatable, **P](
#        cls: type[Self],
#        descriptive: Literal[False] = False
#    ) -> Callable[[ActionMethodProtocol[ActionsT, AnimatableT, P]], ActionDescriptor[ActionsT, AnimatableT, P]]: ...


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
        #self._bound_classmethod: Callable[Concatenate[AnimatableT, P], Iterator[Animation]] = NotImplemented
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

    #@property
    #def descriptor_dict(
    #    self: Self
    #) -> dict[LazyDescriptor, DescriptorParameters[AnimatableT]]:
    #    return self._descriptor_dict

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

    def build_animatable_method_descriptor(
        self: Self
        #method: AnimatableMethodProtocol[StaticT, P]
    ) -> AnimatableMethodDescriptor[ActionsT, AnimatableT, DescriptorParametersT, P]:
        return AnimatableMethodDescriptor(self)

    def build_dynamic_animatable_method_descriptor(
        self: Self
        #method: AnimatableMethodProtocol[StaticT, P]
    ) -> DynamicAnimatableMethodDescriptor[ActionsT, AnimatableT, DescriptorParametersT, P]:
        return DynamicAnimatableMethodDescriptor(self)

        #def result(
        #    instance: AnimatableT,
        #    *args: P.args,
        #    **kwargs: P.kwargs
        #) -> AnimatableT:
        #    for animation in self(instance, *args, **kwargs):
        #        animation.update_boundary(1)
        #    return instance

        #return result

    #def build_dynamic_animatable_method(
    #    self: Self
    #) -> DynamicAnimatableMethodProtocol[AnimatableT, P]:

    #    def dynamic_animatable_method(
    #        instance: DynamicAnimatable[AnimatableT],
    #        *args: P.args,
    #        **kwargs: P.kwargs
    #    ) -> DynamicAnimatable[AnimatableT]:
    #        instance._animations.extend(self(instance._dst, *args, **kwargs))
    #        return instance

    #    return dynamic_animatable_method

    #@overload
    #def __get__[StaticInstanceT: Animatable](
    #    self: Self,
    #    instance: StaticInstanceT,
    #    owner: type[ActionsT] | None
    #) -> StaticInstanceBoundMethodProtocol[StaticInstanceT, P]: ...

    #@overload
    #def __get__[DynamicInstanceT: DynamicAnimatable](
    #    self: Self,
    #    instance: DynamicInstanceT,
    #    owner: type[ActionsT] | None
    #) -> DynamicInstanceBoundMethodProtocol[DynamicInstanceT, P]: ...

    #@overload
    #def __get__(
    #    self: Self,
    #    instance: Any,
    #    owner: Any
    #) -> ActionsClassBoundMethodProtocol[AnimatableT, P]: ...

    #def __get__(
    #    self: Self,
    #    instance: Any,
    #    owner: Any = None
    #) -> Any:
    #    from .animatable import (
    #        Animatable,
    #        DynamicAnimatable
    #    )

    #    if owner is None:
    #        owner = type(instance)
    #    match instance:
    #        case Animatable():
    #            return self._make_static_instance_method(instance, owner)
    #        case DynamicAnimatable():
    #            return self._make_dynamic_instance_method(instance, owner)
    #        case _:
    #            return partial(self._method, owner)

    #def _make_static_instance_method[StaticInstanceT: Animatable](
    #    self: Self,
    #    instance: StaticInstanceT,
    #    owner: type[ActionsT]
    #) -> StaticInstanceBoundMethodProtocol[StaticInstanceT, P]:
    #    method = self._method

    #    def static_action(
    #        *args: P.args,
    #        **kwargs: P.kwargs
    #    ) -> StaticInstanceT:
    #        for animation in method(owner, instance, *args, **kwargs):
    #            animation.update_boundary(1)
    #        return instance

    #    return static_action

    #def _make_dynamic_instance_method[DynamicInstanceT: DynamicAnimatable](
    #    self: Self,
    #    instance: DynamicInstanceT,
    #    owner: type[ActionsT]
    #) -> DynamicInstanceBoundMethodProtocol[DynamicInstanceT, P]:
    #    method = self._method

    #    def dynamic_action(
    #        *args: P.args,
    #        **kwargs: P.kwargs
    #    ) -> DynamicInstanceT:
    #        instance._animations.extend(method(owner, instance._dst, *args, **kwargs))
    #        return instance

    #    return dynamic_action


class AnimatableMethodDescriptor[ActionsT: Actions, AnimatableT: Animatable, DescriptorParametersT: DescriptorParameters, **P]:
    __slots__ = ("_action",)

    def __init__(
        self: Self,
        action: Action[ActionsT, AnimatableT, DescriptorParametersT, P]
    ) -> None:
        super().__init__()
        self._action: Action[ActionsT, AnimatableT, DescriptorParametersT, P] = action

    #def __call__(
    #    self: Self,
    #    dst: AnimatableT,
    #    *args: P.args,
    #    **kwargs: P.kwargs
    #) -> Iterator[Animation]:
    #    return self._action(dst=dst, *args, **kwargs)

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


class DynamicAnimatableMethodDescriptor[ActionsT: Actions, AnimatableT: Animatable, DescriptorParametersT: DescriptorParameters, **P](
    AnimatableMethodDescriptor[ActionsT, AnimatableT, DescriptorParametersT, P]
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


#class ActionMethodProtocol[AnimatableT: Animatable, **P](Protocol):
#    def __call__(
#        self,
#        cls: Any,
#        dst: AnimatableT,
#        *args: P.args,
#        **kwargs: P.kwargs
#    ) -> Iterator[Animation]: ...


#class DescriptiveActionMethodProtocol[ActionsT: Actions, AnimatableT: Animatable, TypedDictT: TypedDict, **P](Protocol):
#    def __call__(
#        self,
#        cls: type[ActionsT],
#        descriptors: dict[LazyDescriptor, TypedDictT],
#        dst: AnimatableT,
#        *args: P.args,
#        **kwargs: P.kwargs
#    ) -> Iterator[Animation]:
#        ...


#class ActionsClassBoundMethodProtocol[AnimatableT: Animatable, **P](Protocol):
#    def __call__(
#        self,
#        dst: AnimatableT,
#        *args: P.args,
#        **kwargs: P.kwargs
#    ) -> Iterator[Animation]:
#        ...


#class StaticActionMethodProtocol[AnimatableT: Animatable, **P](Protocol):
#    def __call__(
#        self,
#        dst: AnimatableT,
#        *args: P.args,
#        **kwargs: P.kwargs
#    ) -> AnimatableT:
#        ...


#class AnimatableMethodProtocol[AnimatableT: Animatable, **P](Protocol):
#    def __call__(
#        self,
#        instance: AnimatableT,
#        *args: P.args,
#        **kwargs: P.kwargs
#    ) -> AnimatableT: ...


#class DynamicActionMethodProtocol[DynamicT: DynamicAnimatable, **P](Protocol):
#    def __call__(
#        self,
#        dst_animations_timeline: DynamicT,
#        *args: P.args,
#        **kwargs: P.kwargs
#    ) -> DynamicT:
#        ...


#class DynamicAnimatableMethodProtocol[AnimatableT: Animatable, **P](Protocol):
#    def __call__(
#        self,
#        instance: DynamicAnimatable[AnimatableT],
#        *args: P.args,
#        **kwargs: P.kwargs
#    ) -> DynamicAnimatable[AnimatableT]: ...
