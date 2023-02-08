"""
This module implements a basic class with lazy properties.

The functionality of `LazyBase` is to save resource as much as it can.
On one hand, all data of `lazy_basedata` and `lazy_property` are shared among
instances. On the other hand, these data will be restocked (recursively) if
they themselves are instances of `LazyBase`. One may also define custom
restockers for individual data.

Every child class of `LazyBase` shall be declared with an empty `__slots__`,
and all methods shall be sorted in the following way:
- magic methods
- lazy_basedata_cached
- lazy_basedata
- lazy_property
- lazy_slot
- private class methods
- private methods
- public methods

The instantiation process shall be done in `__new__` method. If one wants to
make use of the functionality of `_copy()`, a default construction shall be
provided (that is, to provide default values for all arguments).

All methods decorated by any of `lazy_basedata`, `lazy_basedata_cached`,
`lazy_property` and `lazy_slot` should be static methods, and so are their
restockers. Type annotation is strictly applied to reduce chances of running
into unexpected behaviors.

Methods decorated by `lazy_basedata` should be named with underscores appeared
on both sides, i.e. `_data_`. Each should not take any argument and return
the *initial* value for this data. `NotImplemented` may be an alternative for
the value returned, as long as the data is initialized in `__new__` method.
In principle, the data can be of any type including mutable ones, but one must
keep in mind that data *cannot be mutated* as they are shared. The only way to
change the value is to reset the data via `__set__`, and the new value shall
be wrapped up with `NewData`. This makes it possible to manually share data
which is not the initial value. Note, the `__get__` method will return the
unwrapped data. One shall use `instance.__class__._data_._get_data(instance)`
to obtain the wrapped data if one wishes to share it with other instances.

Methods decorated by `lazy_basedata_cached` are pretty much similar to ones
decorated by `lazy_basedata`, except that an argument `hasher` should be
additionally passed to the decorator. Data handled in these methods are
expected to be light-weighted and have much duplicated usage so that caching
can take effect. Data wrapping is not necessary when calling `__set__`.

Methods decorated by `lazy_property` should be named with the same style of
`lazy_basedata`. They should take *at least one* argument, and all names of
arguments should be matched with any `lazy_basedata` or other `lazy_property`
where underscores on edges are eliminated. Data is immutable, and calling
`__set__` method will trigger an exception. As the name `lazy` suggests, if
any correlated `lazy_basedata` is altered, as long as the calculation is never
done before, the recalculation will be executed when one calls `__get__`.

Methods decorated by `lazy_slot` should be named with an underscore inserted
at front, i.e. `_data`. They behave like a normal attribute of the class.
Again, each should not take any argument and return the *initial* value for
this data, with `NotImplemented` as an alternative if the data is set in
`__new__`. Data can be freely mutated because they are no longer shared
(as long as one does not do something like `b._data = a._data`, or calls the
`_copy` method). Data wrapping is not necessary when calling `__set__`.
"""


__all__ = [
    "LazyBase",
    "LazyBasedata",
    "LazyBasedataCached",
    "LazyDescriptor",
    "LazyProperty",
    "NewData",
    "lazy_basedata",
    "lazy_basedata_cached",
    "lazy_property",
    "lazy_slot"
]


from abc import ABC
import inspect
from types import (
    GenericAlias,
    UnionType
)
from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    Generic,
    Hashable,
    TypeVar,
    overload
)


_T = TypeVar("_T")
_R = TypeVar("_R", bound=Hashable)
_LazyBaseT = TypeVar("_LazyBaseT", bound="LazyBase")
_Annotation = Any


class NewData(Generic[_T]):
    def __init__(self, data: _T):
        self._data: _T = data

    #def __repr__(self) -> str:
    #    return f"<NewData: {self._data}>"

    @property
    def data(self) -> _T:
        return self._data


class LazyDescriptor(Generic[_LazyBaseT, _T]):
    def __init__(self, method: Callable[..., _T]):
        self.name: str = method.__name__
        self.method: Callable[..., _T] = method
        self.signature: inspect.Signature = inspect.signature(method)
        self.restock_method: Callable[[_T], None] | None = None

    @property
    def parameters(self) -> dict[str, _Annotation]:
        return {
            f"_{parameter.name}_": parameter.annotation
            for parameter in list(self.signature.parameters.values())
        }

    @property
    def return_annotation(self) -> _Annotation:
        return self.signature.return_annotation

    def _restock(self, data: _T) -> None:
        if self.restock_method is not None:
            self.restock_method(data)
        elif isinstance(data, LazyBase):
            data._restock()

    def restocker(self, restock_method: Callable[[_T], None]) -> Callable[[_T], None]:
        self.restock_method = restock_method
        return restock_method


class LazyBasedata(LazyDescriptor[_LazyBaseT, _T]):
    def __init__(self, method: Callable[[], _T]):
        super().__init__(method)
        assert not self.parameters
        self.instance_to_basedata_dict: dict[_LazyBaseT, NewData[_T]] = {}
        self.basedata_to_instances_dict: dict[NewData[_T], list[_LazyBaseT]] = {}
        self._default_basedata: NewData[_T] | None = None

    @overload
    def __get__(self, instance: None, owner: type[_LazyBaseT] | None = None) -> "LazyBasedata[_LazyBaseT, _T]": ...

    @overload
    def __get__(self, instance: _LazyBaseT, owner: type[_LazyBaseT] | None = None) -> _T: ...

    def __get__(self, instance: _LazyBaseT | None, owner: type[_LazyBaseT] | None = None) -> "LazyBasedata[_LazyBaseT, _T] | _T":
        if instance is None:
            return self
        return self._get_data(instance).data

    def __set__(self, instance: _LazyBaseT, basedata: NewData[_T]) -> None:
        assert isinstance(basedata, NewData)
        self._set_data(instance, basedata)

    @property
    def default_basedata(self) -> NewData[_T]:
        if self._default_basedata is None:
            self._default_basedata = NewData(self.method())
        return self._default_basedata

    def _get_data(self, instance: _LazyBaseT) -> NewData[_T]:
        return self.instance_to_basedata_dict.get(instance, self.default_basedata)

    def _set_data(self, instance: _LazyBaseT, basedata: NewData[_T] | None) -> None:
        self._clear_instance_basedata(instance)
        for property_descr in instance.__class__._BASEDATA_DESCR_TO_PROPERTY_DESCRS[self]:
            property_descr._clear_instance_basedata_tuple(instance)
        if basedata is None:
            return
        self.instance_to_basedata_dict[instance] = basedata
        self.basedata_to_instances_dict.setdefault(basedata, []).append(instance)

    def _clear_instance_basedata(self, instance: _LazyBaseT) -> None:
        if (basedata := self.instance_to_basedata_dict.pop(instance, None)) is None:
            return
        self.basedata_to_instances_dict[basedata].remove(instance)
        if self.basedata_to_instances_dict[basedata]:
            return
        self.basedata_to_instances_dict.pop(basedata)
        self._restock(basedata.data)


class LazyBasedataCached(Generic[_LazyBaseT, _T, _R], LazyBasedata[_LazyBaseT, _T]):
    def __init__(self, method: Callable[[], _T], hasher: Callable[[_T], _R]):
        super().__init__(method)
        self.hasher: Callable[[_T], _R] = hasher
        self.hash_to_basedata_dict: dict[_R, NewData[_T]] = {}

    def __set__(self, instance: _LazyBaseT, basedata: _T) -> None:
        hashed_value = self.hasher(basedata)
        if (cached_basedata := self.hash_to_basedata_dict.get(hashed_value)) is None:
            cached_basedata = NewData(basedata)
            self.hash_to_basedata_dict[hashed_value] = cached_basedata
        super().__set__(instance, cached_basedata)


class LazyProperty(LazyDescriptor[_LazyBaseT, _T]):
    def __init__(self, method: Callable[..., _T]):
        super().__init__(method)
        assert self.parameters
        self.instance_to_basedata_tuple_dict: dict[_LazyBaseT, tuple[NewData[Any], ...]] = {}
        self.basedata_tuple_to_instances_dict: dict[tuple[NewData[Any], ...], list[_LazyBaseT]] = {}
        self.basedata_tuple_to_property_dict: dict[tuple[NewData[Any], ...], _T] = {}

    @overload
    def __get__(self, instance: None, owner: type[_LazyBaseT] | None = None) -> "LazyProperty[_LazyBaseT, _T]": ...

    @overload
    def __get__(self, instance: _LazyBaseT, owner: type[_LazyBaseT] | None = None) -> _T: ...

    def __get__(self, instance: _LazyBaseT | None, owner: type[_LazyBaseT] | None = None) -> "LazyProperty[_LazyBaseT, _T] | _T":
        if instance is None:
            return self
        if (basedata_tuple := self.instance_to_basedata_tuple_dict.get(instance)) is None:
            basedata_tuple = tuple(
                basedata_descr._get_data(instance)
                for basedata_descr in instance.__class__._PROPERTY_DESCR_TO_BASEDATA_DESCRS[self]
            )
            self.instance_to_basedata_tuple_dict[instance] = basedata_tuple
            self.basedata_tuple_to_instances_dict.setdefault(basedata_tuple, []).append(instance)
        if (result := self.basedata_tuple_to_property_dict.get(basedata_tuple)) is None:
            result = self.method(*(
                param_descr.__get__(instance)
                for param_descr in instance.__class__._PROPERTY_DESCR_TO_PARAMETER_DESCRS[self]
            ))
            self.basedata_tuple_to_property_dict[basedata_tuple] = result
        return result

    def __set__(self, instance: _LazyBaseT, value: Any) -> None:
        raise RuntimeError("Attempting to set a readonly lazy property")

    def _clear_instance_basedata_tuple(self, instance: _LazyBaseT) -> None:
        if (basedata_tuple := self.instance_to_basedata_tuple_dict.pop(instance, None)) is None:
            return
        self.basedata_tuple_to_instances_dict[basedata_tuple].remove(instance)
        if self.basedata_tuple_to_instances_dict[basedata_tuple]:
            return
        self.basedata_tuple_to_instances_dict.pop(basedata_tuple)
        if (property_data := self.basedata_tuple_to_property_dict.pop(basedata_tuple, None)) is None:
            return
        self._restock(property_data)


class LazySlot(LazyDescriptor[_LazyBaseT, _T]):
    def __init__(self, method: Callable[[], _T]):
        super().__init__(method)
        assert not self.parameters
        self.instance_to_value_dict: dict[_LazyBaseT, _T] = {}

    @overload
    def __get__(self, instance: None, owner: type[_LazyBaseT] | None = None) -> "LazySlot[_LazyBaseT, _T]": ...

    @overload
    def __get__(self, instance: _LazyBaseT, owner: type[_LazyBaseT] | None = None) -> _T: ...

    def __get__(self, instance: _LazyBaseT | None, owner: type[_LazyBaseT] | None = None) -> "LazySlot[_LazyBaseT, _T] | _T":
        if instance is None:
            return self
        if (value := self.instance_to_value_dict.get(instance)) is None:
            value = self.method()
            self.instance_to_value_dict[instance] = value
        return value

    def __set__(self, instance: _LazyBaseT, value: _T) -> None:
        self.instance_to_value_dict[instance] = value

    def _copy_value(self, instance_src: _LazyBaseT, instance_dst: _LazyBaseT) -> None:
        if (value := self.instance_to_value_dict.get(instance_src)) is None:
            self.instance_to_value_dict.pop(instance_dst, None)
            return
        self.instance_to_value_dict[instance_dst] = value


class lazy_basedata(Generic[_LazyBaseT, _T]):
    def __new__(cls, method: Callable[[], _T]) -> LazyBasedata[_LazyBaseT, _T]:
        return LazyBasedata(method)


class lazy_basedata_cached(Generic[_LazyBaseT, _T, _R]):
    def __init__(self, hasher: Callable[[_T], _R]):
        self.hasher: Callable[[_T], _R] = hasher

    def __call__(self, method: Callable[[], _T]) -> LazyBasedataCached[_LazyBaseT, _T, _R]:
        return LazyBasedataCached(method, self.hasher)

class lazy_property(Generic[_LazyBaseT, _T]):
    def __new__(cls, method: Callable[..., _T]) -> LazyProperty[_LazyBaseT, _T]:
        return LazyProperty(method)


class lazy_slot(Generic[_LazyBaseT, _T]):
    def __new__(cls, method: Callable[[], _T]) -> LazySlot[_LazyBaseT, _T]:
        return LazySlot(method)


class LazyBase(ABC):
    __slots__ = ()

    _VACANT_INSTANCES: "ClassVar[list[LazyBase]]"
    _BASEDATA_DESCR_TO_PROPERTY_DESCRS: ClassVar[dict[LazyBasedata, tuple[LazyProperty, ...]]]
    _PROPERTY_DESCR_TO_BASEDATA_DESCRS: ClassVar[dict[LazyProperty, tuple[LazyBasedata, ...]]]
    _PROPERTY_DESCR_TO_PARAMETER_DESCRS: ClassVar[dict[LazyProperty, tuple[LazyBasedata | LazyProperty, ...]]]
    _SLOT_DESCRS: ClassVar[tuple[LazySlot, ...]] = ()

    def __init_subclass__(cls) -> None:
        descrs: dict[str, LazyBasedata | LazyProperty] = {}
        slots: dict[str, LazySlot] = {}
        for parent_cls in cls.__mro__[::-1]:
            for name, method in parent_cls.__dict__.items():
                if (covered_descr := descrs.get(name)) is not None:
                    assert isinstance(method, LazyBasedata | LazyProperty)
                    cls._check_annotation_matching(method.return_annotation, covered_descr.return_annotation)
                if isinstance(method, LazyBasedata | LazyProperty):
                    descrs[name] = method
                if (covered_slot := slots.get(name)) is not None:
                    assert isinstance(covered_slot, LazySlot)
                if isinstance(method, LazySlot):
                    slots[name] = method

        property_descr_to_parameter_descrs: dict[LazyProperty, tuple[LazyBasedata | LazyProperty, ...]] = {}
        for descr in descrs.values():
            if not isinstance(descr, LazyProperty):
                continue
            param_descrs: list[LazyBasedata | LazyProperty] = []
            for name, param_annotation in descr.parameters.items():
                param_descr = descrs[name]
                cls._check_annotation_matching(param_descr.return_annotation, param_annotation)
                param_descrs.append(param_descr)
            property_descr_to_parameter_descrs[descr] = tuple(param_descrs)

        def traverse(property_descr: LazyProperty, occurred: set[LazyBasedata]) -> Generator[LazyBasedata, None, None]:
            for name in property_descr.parameters:
                param_descr = descrs[name]
                if isinstance(param_descr, LazyBasedata):
                    yield param_descr
                    occurred.add(param_descr)
                else:
                    yield from traverse(param_descr, occurred)

        property_descr_to_basedata_descrs = {
            property_descr: tuple(traverse(property_descr, set()))
            for property_descr in descrs.values()
            if isinstance(property_descr, LazyProperty)
        }
        basedata_descr_to_property_descrs = {
            basedata_descr: tuple(
                property_descr
                for property_descr, basedata_descrs in property_descr_to_basedata_descrs.items()
                if basedata_descr in basedata_descrs
            )
            for basedata_descr in descrs.values()
            if isinstance(basedata_descr, LazyBasedata)
        }

        cls._VACANT_INSTANCES = []
        cls._BASEDATA_DESCR_TO_PROPERTY_DESCRS = basedata_descr_to_property_descrs
        cls._PROPERTY_DESCR_TO_BASEDATA_DESCRS = property_descr_to_basedata_descrs
        cls._PROPERTY_DESCR_TO_PARAMETER_DESCRS = property_descr_to_parameter_descrs
        cls._SLOT_DESCRS = tuple(slots.values())
        return super().__init_subclass__()

    def __new__(cls):
        if (instances := cls._VACANT_INSTANCES):
            instance = instances.pop()
            assert isinstance(instance, cls)
        else:
            instance = super().__new__(cls)
        return instance

    #def __delete__(self) -> None:
    #    self._restock()

    @classmethod
    def _check_annotation_matching(cls, child_annotation: _Annotation, parent_annotation: _Annotation) -> None:
        error_message = f"Type annotation mismatched: `{child_annotation}` is not compatible with `{parent_annotation}`"
        if isinstance(child_annotation, TypeVar) or isinstance(parent_annotation, TypeVar):
            if isinstance(child_annotation, TypeVar) and isinstance(parent_annotation, TypeVar):
                assert child_annotation == parent_annotation, error_message
            return

        def to_classes(annotation: _Annotation) -> tuple[type, ...]:
            return tuple(
                child.__origin__ if isinstance(child, GenericAlias) else
                Callable if isinstance(child, Callable) else child
                for child in (
                    annotation.__args__ if isinstance(annotation, UnionType) else (annotation,)
                )
            )

        assert all(
            any(
                issubclass(child_cls, parent_cls)
                for parent_cls in to_classes(parent_annotation)
            )
            for child_cls in to_classes(child_annotation)
        ), error_message

    def _copy(self):
        result = self.__new__(self.__class__)
        for basedata_descr in self._BASEDATA_DESCR_TO_PROPERTY_DESCRS:
            basedata = basedata_descr.instance_to_basedata_dict.get(self, None)
            basedata_descr._set_data(result, basedata)
        for slot_descr in self._SLOT_DESCRS:
            slot_descr._copy_value(self, result)
        return result

    def _restock(self) -> None:
        for basedata_descr in self._BASEDATA_DESCR_TO_PROPERTY_DESCRS:
            basedata_descr._set_data(self, None)
        for slot_descr in self._SLOT_DESCRS:
            slot_descr.instance_to_value_dict.pop(self, None)
        self._VACANT_INSTANCES.append(self)
