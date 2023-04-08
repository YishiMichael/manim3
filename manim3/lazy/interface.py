__all__ = [
    "Lazy",
    "LazyMode"
]


from abc import (
    ABC,
    abstractmethod
)
from enum import Enum
import inspect
import re
from typing import (
    Any,
    Callable,
    Concatenate,
    Hashable,
    Iterable,
    Literal,
    ParamSpec,
    TypeVar,
    final,
    overload
)
import weakref

from ..lazy.core import (
    LazyDynamicContainer,
    LazyDynamicPropertyDescriptor,
    LazyDynamicVariableDescriptor,
    LazyObject,
    LazyUnitaryContainer,
    LazyUnitaryPropertyDescriptor,
    LazyUnitaryVariableDescriptor,
    LazyWrapper
)


_T = TypeVar("_T")
_HashableT = TypeVar("_HashableT", bound=Hashable)
_ElementT = TypeVar("_ElementT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_PropertyParameters = ParamSpec("_PropertyParameters")


class AnnotationUtils(ABC):
    __slots__ = ()

    @abstractmethod
    def __new__(cls) -> None:
        pass

    @classmethod
    def get_return_type(
        cls,
        method: Callable
    ) -> Any:
        if isinstance(return_type := inspect.signature(method).return_annotation, str):
            return NotImplemented
        if isinstance(return_type, type):
            return return_type
        return return_type.__origin__

    @classmethod
    def get_element_return_type(
        cls,
        method: Callable
    ) -> Any:
        if isinstance(collection_type := inspect.signature(method).return_annotation, str):
            return NotImplemented
        assert issubclass(collection_type.__origin__, Iterable)
        return_type = collection_type.__args__[0]
        if isinstance(return_type, type):
            return return_type
        return return_type.__origin__

    @classmethod
    def get_parameter_items(
        cls,
        method: Callable
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[bool, ...]]:
        parameter_items = tuple(
            (name, False) if re.fullmatch(r"_\w+_", name) else (f"_{name}_", True)
            for name in tuple(inspect.signature(method).parameters)[1:]  # remove `cls`
        )
        parameter_name_chains = tuple(
            tuple(re.findall(r"_\w+?_(?=_|$)", parameter_name))
            for parameter_name, _ in parameter_items
        )
        assert all(
            "".join(parameter_name_chain) == parameter_name
            for parameter_name_chain, (parameter_name, _) in zip(parameter_name_chains, parameter_items, strict=True)
        )
        requires_unwrapping_tuple = tuple(
            requires_unwrapping
            for _, requires_unwrapping in parameter_items
        )
        return parameter_name_chains, requires_unwrapping_tuple


@final
class LazyUnitaryVariableDecorator(LazyUnitaryVariableDescriptor[_InstanceT, _ElementT, _ElementT]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], _ElementT]
    ) -> None:
        super().__init__(
            element_type=AnnotationUtils.get_return_type(method),
            method=method
        )

    def convert_set(
        self,
        new_value: _ElementT
    ) -> LazyUnitaryContainer[_ElementT]:
        return LazyUnitaryContainer(
            element=new_value
        )


@final
class LazyUnitaryVariableUnwrappedDecorator(LazyUnitaryVariableDescriptor[_InstanceT, LazyWrapper[_T], _T | LazyWrapper[_T]]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], _T]
    ) -> None:
        super().__init__(
            element_type=LazyWrapper,
            method=method
        )

    def convert_set(
        self,
        new_value: _T | LazyWrapper[_T]
    ) -> LazyUnitaryContainer[LazyWrapper[_T]]:
        if not isinstance(new_value, LazyWrapper):
            new_value = LazyWrapper(new_value)
        return LazyUnitaryContainer(
            element=new_value
        )


@final
class LazyUnitaryVariableSharedDecorator(LazyUnitaryVariableDescriptor[_InstanceT, LazyWrapper[_HashableT], _HashableT]):
    __slots__ = ("content_to_element_dict",)

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], _HashableT]
    ) -> None:
        self.content_to_element_dict: weakref.WeakValueDictionary[_HashableT, LazyWrapper[_HashableT]] = weakref.WeakValueDictionary()
        super().__init__(
            element_type=LazyWrapper,
            method=method
        )

    def convert_set(
        self,
        new_value: _HashableT
    ) -> LazyUnitaryContainer[LazyWrapper[_HashableT]]:
        if (cached_element := self.content_to_element_dict.get(new_value)) is None:
            cached_element = LazyWrapper(new_value)
            self.content_to_element_dict[new_value] = cached_element
        return LazyUnitaryContainer(
            element=cached_element
        )


@final
class LazyDynamicVariableDecorator(LazyDynamicVariableDescriptor[_InstanceT, _ElementT, Iterable[_ElementT]]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], Iterable[_ElementT]]
    ) -> None:
        super().__init__(
            element_type=AnnotationUtils.get_element_return_type(method),
            method=method
        )

    def convert_set(
        self,
        new_value: Iterable[_ElementT]
    ) -> LazyDynamicContainer[_ElementT]:
        return LazyDynamicContainer(
            elements=new_value
        )


@final
class LazyUnitaryPropertyDecorator(LazyUnitaryPropertyDescriptor[_InstanceT, _ElementT, _ElementT]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _ElementT]
    ) -> None:
        parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=AnnotationUtils.get_return_type(method),
            method=method,
            parameter_name_chains=parameter_name_chains,
            requires_unwrapping_tuple=requires_unwrapping_tuple
        )

    def convert_set(
        self,
        new_value: _ElementT
    ) -> LazyUnitaryContainer[_ElementT]:
        return LazyUnitaryContainer(
            element=new_value
        )


@final
class LazyUnitaryPropertyUnwrappedDecorator(LazyUnitaryPropertyDescriptor[_InstanceT, LazyWrapper[_T], _T]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _T]
    ) -> None:
        parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=LazyWrapper,
            method=method,
            parameter_name_chains=parameter_name_chains,
            requires_unwrapping_tuple=requires_unwrapping_tuple
        )

    def convert_set(
        self,
        new_value: _T
    ) -> LazyUnitaryContainer[LazyWrapper[_T]]:
        return LazyUnitaryContainer(
            element=LazyWrapper(new_value)
        )

    def finalizer(
        self,
        finalize_method: Any
    ) -> Any:
        assert isinstance(finalize_method, classmethod)
        func = finalize_method.__func__

        def new_finalize_method(
            cls: type[_InstanceT],
            value: LazyWrapper[_T]
        ) -> None:
            func(cls, value.value)

        self.finalize_method = new_finalize_method
        return finalize_method


@final
class LazyUnitaryPropertySharedDecorator(LazyUnitaryPropertyDescriptor[_InstanceT, LazyWrapper[_HashableT], _HashableT]):
    __slots__ = ("content_to_element_dict",)

    def __init__(
        self,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _HashableT]
    ) -> None:
        self.content_to_element_dict: weakref.WeakValueDictionary[_HashableT, LazyWrapper[_HashableT]] = weakref.WeakValueDictionary()
        parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=LazyWrapper,
            method=method,
            parameter_name_chains=parameter_name_chains,
            requires_unwrapping_tuple=requires_unwrapping_tuple
        )

    def convert_set(
        self,
        new_value: _HashableT
    ) -> LazyUnitaryContainer[LazyWrapper[_HashableT]]:
        if (cached_element := self.content_to_element_dict.get(new_value)) is None:
            cached_element = LazyWrapper(new_value)
            self.content_to_element_dict[new_value] = cached_element
        return LazyUnitaryContainer(
            element=cached_element
        )


@final
class LazyDynamicPropertyDecorator(LazyDynamicPropertyDescriptor[_InstanceT, _ElementT, Iterable[_ElementT]]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], Iterable[_ElementT]]
    ) -> None:
        parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=AnnotationUtils.get_element_return_type(method),
            method=method,
            parameter_name_chains=parameter_name_chains,
            requires_unwrapping_tuple=requires_unwrapping_tuple
        )

    def convert_set(
        self,
        new_value: Iterable[_ElementT]
    ) -> LazyDynamicContainer[_ElementT]:
        return LazyDynamicContainer(
            elements=new_value
        )


@final
class LazyMode(Enum):
    OBJECT = 0
    UNWRAPPED = 1
    SHARED = 2
    COLLECTION = 3


@final
class Lazy(ABC):
    @overload
    @classmethod
    def variable(
        cls,
        mode: Literal[LazyMode.OBJECT]
    ) -> Callable[
        [Callable[[type[_InstanceT]], _ElementT]],
        LazyUnitaryVariableDecorator[_InstanceT, _ElementT]
    ]: ...

    @overload
    @classmethod
    def variable(
        cls,
        mode: Literal[LazyMode.UNWRAPPED]
    ) -> Callable[
        [Callable[[type[_InstanceT]], _T]],
        LazyUnitaryVariableUnwrappedDecorator[_InstanceT, _T]
    ]: ...

    @overload
    @classmethod
    def variable(
        cls,
        mode: Literal[LazyMode.SHARED]
    ) -> Callable[
        [Callable[[type[_InstanceT]], _HashableT]],
        LazyUnitaryVariableSharedDecorator[_InstanceT, _HashableT]
    ]: ...

    @overload
    @classmethod
    def variable(
        cls,
        mode: Literal[LazyMode.COLLECTION]
    ) -> Callable[
        [Callable[[type[_InstanceT]], Iterable[_ElementT]]],
        LazyDynamicVariableDecorator[_InstanceT, _ElementT]
    ]: ...

    @classmethod
    def variable(
        cls,
        mode: LazyMode
    ) -> Callable[[Callable], Any]:
        if mode is LazyMode.OBJECT:
            decorator_cls = LazyUnitaryVariableDecorator
        elif mode is LazyMode.UNWRAPPED:
            decorator_cls = LazyUnitaryVariableUnwrappedDecorator
        elif mode is LazyMode.SHARED:
            decorator_cls = LazyUnitaryVariableSharedDecorator
        elif mode is LazyMode.COLLECTION:
            decorator_cls = LazyDynamicVariableDecorator
        else:
            raise ValueError

        def result(
            cls_method: Callable
        ) -> Any:
            assert isinstance(cls_method, classmethod)
            return decorator_cls(cls_method.__func__)

        return result

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.OBJECT]
    ) -> Callable[
        [Callable[Concatenate[type[_InstanceT], _PropertyParameters], _ElementT]],
        LazyUnitaryPropertyDecorator[_InstanceT, _ElementT]
    ]: ...

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.UNWRAPPED]
    ) -> Callable[
        [Callable[Concatenate[type[_InstanceT], _PropertyParameters], _T]],
        LazyUnitaryPropertyUnwrappedDecorator[_InstanceT, _T]
    ]: ...

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.SHARED]
    ) -> Callable[
        [Callable[Concatenate[type[_InstanceT], _PropertyParameters], _HashableT]],
        LazyUnitaryPropertySharedDecorator[_InstanceT, _HashableT]
    ]: ...

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.COLLECTION]
    ) -> Callable[
        [Callable[Concatenate[type[_InstanceT], _PropertyParameters], Iterable[_ElementT]]],
        LazyDynamicPropertyDecorator[_InstanceT, _ElementT]
    ]: ...

    @classmethod
    def property(
        cls,
        mode: LazyMode
    ) -> Callable[[Callable], Any]:
        if mode is LazyMode.OBJECT:
            decorator_cls = LazyUnitaryPropertyDecorator
        elif mode is LazyMode.UNWRAPPED:
            decorator_cls = LazyUnitaryPropertyUnwrappedDecorator
        elif mode is LazyMode.SHARED:
            decorator_cls = LazyUnitaryPropertySharedDecorator
        elif mode is LazyMode.COLLECTION:
            decorator_cls = LazyDynamicPropertyDecorator
        else:
            raise ValueError

        def result(
            cls_method: Callable
        ) -> Any:
            assert isinstance(cls_method, classmethod)
            return decorator_cls(cls_method.__func__)

        return result
