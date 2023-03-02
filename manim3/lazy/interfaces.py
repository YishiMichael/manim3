__all__ = [
    "LazyWrapper",
    "lazy_collection",
    "lazy_object",
    "lazy_object_shared",
    "lazy_object_unwrapped",
    "lazy_property",
    "lazy_property_shared",
    "lazy_property_unwrapped"
]


import inspect
import re
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    TypeVar
)

from bidict import bidict

from ..lazy.core import (
    LazyCollection,
    LazyCollectionDescriptor,
    LazyEntity,
    LazyObject,
    LazyObjectDescriptor,
    LazyPropertyDescriptor
)


_T = TypeVar("_T")
_HashableT = TypeVar("_HashableT", bound=Hashable)
_LazyEntityT = TypeVar("_LazyEntityT", bound="LazyEntity")
_LazyObjectT = TypeVar("_LazyObjectT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")


class LazyWrapper(Generic[_T], LazyObject):
    __slots__ = ("__value",)

    def __init__(
        self,
        value: _T
    ) -> None:
        super().__init__()
        self.__value: _T = value

    @property
    def value(self) -> _T:
        return self.__value


class lazy_object(LazyObjectDescriptor[_InstanceT, _LazyObjectT]):
    __slots__ = ()

    def __init__(
        self,
        cls_method: Callable[[type[_InstanceT]], _LazyObjectT]
    ) -> None:
        super().__init__(
            method=cls_method.__func__
        )


class lazy_object_unwrapped(Generic[_InstanceT, _T], LazyObjectDescriptor[_InstanceT, LazyWrapper[_T]]):
    __slots__ = ()

    def __init__(
        self,
        cls_method: Callable[[type[_InstanceT]], _T]
    ) -> None:
        method = cls_method.__func__

        def new_method(
            cls: type[_InstanceT]
        ) -> LazyWrapper[_T]:
            return LazyWrapper(method(cls))

        super().__init__(
            method=new_method
        )

    def __set__(
        self,
        instance: _InstanceT,
        obj: _T | LazyWrapper[_T]
    ) -> None:
        if not isinstance(obj, LazyWrapper):
            obj = LazyWrapper(obj)
        super().__set__(instance, obj)


class lazy_object_shared(Generic[_InstanceT, _HashableT], LazyObjectDescriptor[_InstanceT, LazyWrapper[_HashableT]]):
    __slots__ = ("content_to_object_bidict",)

    def __init__(
        self,
        cls_method: Callable[[type[_InstanceT]], _HashableT]
    ) -> None:
        self.content_to_object_bidict: bidict[_HashableT, LazyWrapper[_HashableT]] = bidict()
        method = cls_method.__func__

        def new_method(
            cls: type[_InstanceT]
        ) -> LazyWrapper[_HashableT]:
            return LazyWrapper(method(cls))

        super().__init__(
            method=new_method
        )

    def __set__(
        self,
        instance: _InstanceT,
        obj: _HashableT
    ) -> None:
        if (cached_object := self.content_to_object_bidict.get(obj)) is None:
            cached_object = LazyWrapper(obj)
            self.content_to_object_bidict[obj] = cached_object
        super().__set__(instance, cached_object)

    def restock(
        self,
        instance: _InstanceT
    ) -> None:
        self.content_to_object_bidict.inverse.pop(self.get_object(instance))
        super().restock(instance)


class lazy_collection(LazyCollectionDescriptor[_InstanceT, _LazyObjectT]):
    __slots__ = ()

    def __init__(
        self,
        cls_method: Callable[[type[_InstanceT]], LazyCollection[_LazyObjectT]]
    ) -> None:
        super().__init__(
            method=cls_method.__func__
        )


class lazy_property(LazyPropertyDescriptor[_InstanceT, _LazyEntityT]):
    __slots__ = ()

    def __init__(
        self,
        cls_method: Callable[..., _LazyEntityT]
    ) -> None:
        new_method, parameter_chains = self.wrap_parameters(cls_method.__func__)
        super().__init__(
            method=new_method,
            parameter_chains=parameter_chains
        )

    @classmethod
    def wrap_parameters(
        cls,
        method: Callable[..., _LazyEntityT]
    ) -> tuple[Callable[..., _LazyEntityT], tuple[tuple[str, ...], ...]]:
        parameter_items = tuple(
            (name, False) if re.fullmatch(r"_\w+_", name) else (f"_{name}_", True)
            for name in tuple(inspect.signature(method).parameters)[1:]  # remove `cls`
        )
        parameter_chains = tuple(
            tuple(re.findall(r"_\w+?_(?=_|$)", parameter_name))
            for parameter_name, _ in parameter_items
        )
        assert all(
            "".join(parameter_chain) == parameter_name
            for parameter_chain, (parameter_name, _) in zip(parameter_chains, parameter_items, strict=True)
        )
        requires_unwrapping_tuple = tuple(requires_unwrapping for _, requires_unwrapping in parameter_items)

        def parameters_wrapped_method(
            kls: type[_InstanceT],
            *args: Any,
            **kwargs: Any
        ) -> _LazyEntityT:
            return method(kls, *(
                arg if not requires_unwrapping else cls.apply_deepest(lambda obj: obj.value, arg)
                for arg, requires_unwrapping in zip(args, requires_unwrapping_tuple, strict=True)
            ), **kwargs)
        return parameters_wrapped_method, parameter_chains


class lazy_property_unwrapped(LazyPropertyDescriptor[_InstanceT, LazyWrapper[_T]]):
    __slots__ = ("restock_method",)

    def __init__(
        self,
        cls_method: Callable[..., _T]
    ) -> None:
        self.restock_method: Callable[[_T], None] | None = None
        parameters_wrapped_method, parameter_chains = lazy_property.wrap_parameters(cls_method.__func__)

        def new_method(
            cls: type[_InstanceT],
            *args: Any,
            **kwargs: Any
        ) -> LazyWrapper[_T]:
            return LazyWrapper(parameters_wrapped_method(cls, *args, **kwargs))

        super().__init__(
            method=new_method,
            parameter_chains=parameter_chains
        )

    def restock(
        self,
        instance: _InstanceT
    ) -> None:
        if (obj := self.get_property(instance)._get()) is not None:
            if self.restock_method is not None:
                self.restock_method(obj.value)
        super().restock(instance)

    def restocker(
        self,
        restock_method: Callable[[_T], None]
    ) -> Callable[[_T], None]:
        self.restock_method = restock_method
        return restock_method


class lazy_property_shared(LazyPropertyDescriptor[_InstanceT, LazyWrapper[_HashableT]]):
    __slots__ = (
        "restock_method",
        "content_to_object_bidict"
    )

    def __init__(
        self,
        cls_method: Callable[..., _HashableT]
    ) -> None:
        self.restock_method: Callable[[_HashableT], None] | None = None
        self.content_to_object_bidict: bidict[_HashableT, LazyWrapper[_HashableT]] = bidict()
        parameters_wrapped_method, parameter_chains = lazy_property.wrap_parameters(cls_method.__func__)

        def new_method(
            cls: type[_InstanceT],
            *args: Any,
            **kwargs: Any
        ) -> LazyWrapper[_HashableT]:
            content = parameters_wrapped_method(cls, *args, **kwargs)
            if (cached_object := self.content_to_object_bidict.get(content)) is None:
                cached_object = LazyWrapper(content)
                self.content_to_object_bidict[content] = cached_object
            return cached_object

        super().__init__(
            method=new_method,
            parameter_chains=parameter_chains
        )
        self.restock_methods: list[Callable[[_HashableT], None]] = []

    def restock(
        self,
        instance: _InstanceT
    ) -> None:
        if (obj := self.get_property(instance)._get()) is not None:
            self.content_to_object_bidict.inverse.pop(obj)
            if self.restock_method is not None:
                self.restock_method(obj.value)
        super().restock(instance)

    def restocker(
        self,
        restock_method: Callable[[_HashableT], None]
    ) -> Callable[[_HashableT], None]:
        self.restock_method = restock_method
        return restock_method
