from __future__ import annotations


from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    #ClassVar,
    Iterator,
    Never,
    Protocol,
    Self,
    overload
)

if TYPE_CHECKING:
    from .animatable import (
        Animatable,
        Animation,
        DynamicAnimatable
    )


class Actions:
    __slots__ = ()

    #def __new__(
    #    cls: type[Self]
    #) -> Never:
    #    raise TypeError

    #_action_descriptor_cls_dict: ClassVar[dict[ActionDescriptor, type[Actions]]] = {}

    #def __init_subclass__(
    #    cls: type[Self]
    #) -> None:
    #    super().__init_subclass__()
    #    cls._action_descriptor_cls_dict.update({
    #        descriptor: cls
    #        for descriptor in cls.__dict__.items()
    #        if isinstance(descriptor, ActionDescriptor)
    #    })


class ActionMeta:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def register[ActionsT: Actions, StaticT: Animatable, **P](
        cls: type[Self],
        method: ActionMethodProtocol[ActionsT, StaticT, P]
    ) -> ActionDescriptor[ActionsT, StaticT, P]:
        assert isinstance(method, classmethod)
        return ActionDescriptor(method.__func__)


class ActionDescriptor[ActionsT: Actions, StaticT: Animatable, **P]:
    __slots__ = (
        "_method_cls",
        "_method"
    )

    def __init__(
        self: Self,
        method: ActionMethodProtocol[ActionsT, StaticT, P]
    ) -> None:

        #def static_action_method(
        #    dst: StaticT,
        #    *args: P.args,
        #    **kwargs: P.kwargs
        #) -> StaticT:
        #    for animation in method(type(dst), dst, *args, **kwargs):
        #        animation.update_boundary(1)
        #    return dst

        #def dynamic_action_method(
        #    dst: DynamicT,
        #    *args: P.args,
        #    **kwargs: P.kwargs
        #) -> DynamicT:
        #    dst._animations.extend(method(type(dst._animatable), dst, *args, **kwargs))
        #    return dst

        super().__init__()
        self._method: ActionMethodProtocol[ActionsT, StaticT, P] = method
        #self._static_action_method: StaticActionMethodProtocol[StaticT, P] = static_action_method
        #self._dynamic_action_method: DynamicActionMethodProtocol[DynamicT, P] = dynamic_action_method

    @overload
    def __get__(
        self: Self,
        instance: None,
        owner: type[ActionsT] | None
    ) -> ActionBoundMethodProtocol[StaticT, P]: ...

    @overload
    def __get__[InstanceStaticT: Animatable](
        self: Self,
        instance: InstanceStaticT,
        owner: type[ActionsT] | None
    ) -> StaticActionBoundMethodProtocol[InstanceStaticT, P]: ...

    @overload
    def __get__[InstanceDynamicT: DynamicAnimatable](
        self: Self,
        instance: InstanceDynamicT,
        owner: type[ActionsT] | None
    ) -> DynamicActionBoundMethodProtocol[InstanceDynamicT, P]: ...

    def __get__(
        self: Self,
        instance: Any,
        owner: Any = None
    ) -> Any:
        from .animatable import (
            Animatable,
            DynamicAnimatable
        )

        match instance:
            case Animatable():
                return self._make_static_action(instance, owner)
            case DynamicAnimatable():
                return self._make_dynamic_action(instance, owner)
            case _:
                return partial(self._method, owner)

    def _make_static_action[InstanceStaticT: Animatable](
        self: Self,
        instance: InstanceStaticT,
        owner: type[ActionsT]
    ) -> StaticActionBoundMethodProtocol[InstanceStaticT, P]:
        method = self._method

        def static_action(
            *args: P.args,
            **kwargs: P.kwargs
        ) -> InstanceStaticT:
            for animation in method(owner, instance, *args, **kwargs):
                animation.update_boundary(1)
            return instance

        return static_action

    def _make_dynamic_action[InstanceDynamicT: DynamicAnimatable](
        self: Self,
        instance: InstanceDynamicT,
        owner: type[ActionsT]
    ) -> DynamicActionBoundMethodProtocol[InstanceDynamicT, P]:
        method = self._method

        def dynamic_action(
            *args: P.args,
            **kwargs: P.kwargs
        ) -> InstanceDynamicT:
            instance._animations.extend(method(owner, instance._dst, *args, **kwargs))
            return instance

        return dynamic_action


class ActionMethodProtocol[ActionsT: Actions, StaticT: Animatable, **P](Protocol):
    def __call__(
        self,
        cls: type[ActionsT],
        dst: StaticT,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> Iterator[Animation]:
        ...


class ActionBoundMethodProtocol[StaticT: Animatable, **P](Protocol):
    def __call__(
        self,
        dst: StaticT,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> Iterator[Animation]:
        ...


class StaticActionMethodProtocol[StaticT: Animatable, **P](Protocol):
    def __call__(
        self,
        dst: StaticT,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> StaticT:
        ...


class StaticActionBoundMethodProtocol[StaticT: Animatable, **P](Protocol):
    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> StaticT:
        ...


class DynamicActionMethodProtocol[DynamicT: DynamicAnimatable, **P](Protocol):
    def __call__(
        self,
        dst_animations_timeline: DynamicT,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> DynamicT:
        ...


class DynamicActionBoundMethodProtocol[DynamicT: DynamicAnimatable, **P](Protocol):
    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> DynamicT:
        ...
