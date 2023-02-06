__all__ = [
    "LazyData",
    "LazyBase",
    "lazy_basedata",
    "lazy_property"
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
    TypeVar,
    overload
)


_T = TypeVar("_T")
_Annotation = Any
_LazyBaseT = TypeVar("_LazyBaseT", bound="LazyBase")


class LazyData(Generic[_T]):
    def __init__(self, data: _T):
        self._data: _T = data

    def __repr__(self) -> str:
        return f"<LazyData: {self._data}>"

    @property
    def data(self) -> _T:
        return self._data


class lazy_basedata(Generic[_LazyBaseT, _T]):
    def __init__(self, method: Callable[[], _T]):
        self.name: str = method.__name__
        self.method: Callable[..., _T] = method
        self._default_basedata: LazyData[_T] | None = None
        #self.default_basedata: LazyData[_T] = LazyData(method())
        self.annotation: _Annotation = inspect.signature(method).return_annotation
        #self.property_descrs: tuple[lazy_property[_LazyBaseT, Any], ...] = ()
        self.instance_to_basedata_dict: dict[_LazyBaseT, LazyData[_T]] = {}
        self.basedata_to_instances_dict: dict[LazyData[_T], list[_LazyBaseT]] = {}
        #self.basedata_refcnt_dict: dict[LazyData[_T], int] = {}

    def __repr__(self) -> str:
        return f"<lazy_basedata: {self.name}>"

    @overload
    def __get__(self, instance: None, owner: type[_LazyBaseT] | None = None) -> "lazy_basedata[_LazyBaseT, _T]": ...

    @overload
    def __get__(self, instance: _LazyBaseT, owner: type[_LazyBaseT] | None = None) -> _T: ...

    def __get__(self, instance: _LazyBaseT | None, owner: type[_LazyBaseT] | None = None) -> "lazy_basedata[_LazyBaseT, _T] | _T":
        if instance is None:
            return self
        return self._get_data(instance).data

    def __set__(self, instance: _LazyBaseT, basedata: LazyData[_T]) -> None:
        assert isinstance(basedata, LazyData)
        self._set_data(instance, basedata)

        #for property_descr in instance.__class__._BASEDATA_DESCR_TO_PROPERTY_DESCRS[self]:
        #    basedata_descrs = instance.__class__._PROPERTY_DESCR_TO_BASEDATA_DESCRS[property_descr]
        #    if (old_basedata_tuple := property_descr.instance_to_basedata_tuple_dict.pop(instance, None)) is not None:
        #        property_descr._decrement_basedata_tuple_refcnt(old_basedata_tuple)
        #    else:
        #        old_basedata_tuple = tuple(
        #            basedata_descr.default_basedata
        #            for basedata_descr in basedata_descrs
        #        )

        #    #index = basedata_descrs.index(self)
        #    #print(old_basedata, old_basedata_tuple, index)
        #    #assert old_basedata is old_basedata_tuple[index]
        #    basedata_list = list(old_basedata_tuple)
        #    basedata_list[basedata_descrs.index(self)] = basedata
        #    basedata_tuple = tuple(basedata_list)
        #    property_descr.instance_to_basedata_tuple_dict[instance] = basedata_tuple
        #    property_descr._increment_basedata_tuple_refcnt(basedata_tuple)

        #    #data_list = list(old_basedata_tuple)
        #    #data_list[index] = data
        #    #property_descr.instance_to_basedata_tuple_dict[instance] = tuple(data_list)
        #    #if old_basedata_tuple not in property_descr.basedata_tuple_to_property_dict:
        #    #    continue

    @property
    def default_basedata(self) -> LazyData[_T]:
        if self._default_basedata is None:
            self._default_basedata = LazyData(self.method())
        return self._default_basedata

    def _get_data(self, instance: _LazyBaseT) -> LazyData[_T]:
        return self.instance_to_basedata_dict.get(instance, self.default_basedata)

    def _set_data(self, instance: _LazyBaseT, basedata: LazyData[_T] | None) -> None:
        self._clear_instance_basedata(instance)
        for property_descr in instance.__class__._BASEDATA_DESCR_TO_PROPERTY_DESCRS[self]:
            property_descr._clear_instance_basedata_tuple(instance)
        if basedata is None:
            return
        self.instance_to_basedata_dict[instance] = basedata
        #self._increment_basedata_refcnt(basedata)
        self.basedata_to_instances_dict.setdefault(basedata, []).append(instance)

    def _clear_instance_basedata(self, instance: _LazyBaseT) -> None:
        if (basedata := self.instance_to_basedata_dict.pop(instance, None)) is None:
            return
        self.basedata_to_instances_dict[basedata].remove(instance)
        if self.basedata_to_instances_dict[basedata]:
            return
        self.basedata_to_instances_dict.pop(basedata)
        if isinstance(basedata.data, LazyBase):  # TODO
            basedata.data._restock()


class lazy_property(Generic[_LazyBaseT, _T]):
    def __init__(self, method: Callable[..., _T]):
        signature = inspect.signature(method)
        self.name: str = method.__name__
        self.method: Callable[..., _T] = method
        self.annotation: _Annotation = signature.return_annotation
        self.parameters: dict[str, _Annotation] = {
            f"_{parameter.name}_": parameter.annotation
            for parameter in list(signature.parameters.values())
        }
        #self.basedata_descrs: tuple[lazy_basedata[_LazyBaseT, Any], ...] = ()
        self.instance_to_basedata_tuple_dict: dict[_LazyBaseT, tuple[LazyData[Any], ...]] = {}
        self.basedata_tuple_to_instances_dict: dict[tuple[LazyData[Any], ...], list[_LazyBaseT]] = {}
        self.basedata_tuple_to_property_dict: dict[tuple[LazyData[Any], ...], _T] = {}
        #self.instance_to_property_dict: dict[_LazyBaseT, _T] = {}

    def __repr__(self) -> str:
        return f"<lazy_property: {self.name}>"

    @overload
    def __get__(self, instance: None, owner: type[_LazyBaseT] | None = None) -> "lazy_property[_LazyBaseT, _T]": ...

    @overload
    def __get__(self, instance: _LazyBaseT, owner: type[_LazyBaseT] | None = None) -> _T: ...

    def __get__(self, instance: _LazyBaseT | None, owner: type[_LazyBaseT] | None = None) -> "lazy_property[_LazyBaseT, _T] | _T":
        if instance is None:
            return self
        if (basedata_tuple := self.instance_to_basedata_tuple_dict.get(instance)) is None:
            basedata_tuple = tuple(
                basedata_descr._get_data(instance)
                for basedata_descr in instance.__class__._PROPERTY_DESCR_TO_BASEDATA_DESCRS[self]
            )
            self.instance_to_basedata_tuple_dict[instance] = basedata_tuple
            self.basedata_tuple_to_instances_dict.setdefault(basedata_tuple, []).append(instance)
        return self.basedata_tuple_to_property_dict.setdefault(
            basedata_tuple,
            self.method(*(
                param_descr.__get__(instance)
                for param_descr in instance.__class__._PROPERTY_DESCR_TO_PARAMETER_DESCRS[self]
            ))
        )
        #if (basedata_tuple := self.instance_to_basedata_tuple_dict.setdefault(instance, tuple(
        #    basedata_descr._get_basedata(instance) for basedata_descr in self.basedata_descrs
        #))) in self.basedata_tuple_to_property_dict:
        #result = self.basedata_tuple_to_property_dict.setdefault(
        #    basedata_tuple,
        #    LazyData(self.method(*(basedata.data for basedata in basedata_tuple)))
        #)
        #if basedata_tuple in self.basedata_tuple_to_property_dict:
        #    return self.basedata_tuple_to_property_dict[basedata_tuple].data
        ##self.basedata_tuple_refcnt_dict[basedata_tuple] = 1
        #result = LazyData(self.method(*(basedata.data for basedata in basedata_tuple)))
        #self.basedata_tuple_to_property_dict[basedata_tuple] = result
        #return result.data

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
        if isinstance(property_data, LazyBase):  # TODO
            property_data._restock()

    #def _increment_basedata_tuple_refcnt(self, basedata_tuple: tuple[LazyData[Any], ...]) -> None:
    #    if basedata_tuple not in self.basedata_tuple_refcnt_dict:
    #        self.basedata_tuple_refcnt_dict[basedata_tuple] = 0
    #    self.basedata_tuple_refcnt_dict[basedata_tuple] += 1

    #def _decrement_basedata_tuple_refcnt(self, basedata_tuple: tuple[LazyData[Any], ...]) -> None:
    #    self.basedata_tuple_refcnt_dict[basedata_tuple] -= 1
    #    if self.basedata_tuple_refcnt_dict[basedata_tuple] != 0:
    #        return
    #    self.basedata_tuple_refcnt_dict.pop(basedata_tuple)
    #    if (property_data := self.basedata_tuple_to_property_dict.pop(basedata_tuple, None)) is None:
    #        return
    #    if isinstance(property_data, LazyBase):  # TODO
    #        property_data._restock()


class LazyBase(ABC):
    __slots__ = ()

    _VACANT_INSTANCES: "ClassVar[list[LazyBase]]"
    _BASEDATA_DESCR_TO_PROPERTY_DESCRS: ClassVar[dict[lazy_basedata, tuple[lazy_property, ...]]]
    _PROPERTY_DESCR_TO_BASEDATA_DESCRS: ClassVar[dict[lazy_property, tuple[lazy_basedata, ...]]]
    _PROPERTY_DESCR_TO_PARAMETER_DESCRS: ClassVar[dict[lazy_property, tuple[lazy_basedata | lazy_property, ...]]]

    def __init_subclass__(cls) -> None:
        descrs: dict[str, lazy_basedata | lazy_property] = {}
        for parent_cls in cls.__mro__[::-1]:
            for name, method in parent_cls.__dict__.items():
                if name in descrs:
                    assert isinstance(method, lazy_basedata | lazy_property)
                    cls._check_annotation_matching(method.annotation, descrs[name].annotation)
                if isinstance(method, lazy_basedata | lazy_property):
                    descrs[name] = method

        property_descr_to_parameter_descrs: dict[lazy_property, tuple[lazy_basedata | lazy_property, ...]] = {}
        for descr in descrs.values():
            if not isinstance(descr, lazy_property):
                continue
            param_descrs: list[lazy_basedata | lazy_property] = []
            for name, param_annotation in descr.parameters.items():
                param_descr = descrs[name]
                cls._check_annotation_matching(param_descr.annotation, param_annotation)
                param_descrs.append(param_descr)
            property_descr_to_parameter_descrs[descr] = tuple(param_descrs)

        def traverse(property_descr: lazy_property, occurred: set[lazy_basedata]) -> Generator[lazy_basedata, None, None]:
            for name in property_descr.parameters:
                param_descr = descrs[name]
                if isinstance(param_descr, lazy_basedata):
                    yield param_descr
                    occurred.add(param_descr)
                else:
                    yield from traverse(param_descr, occurred)

        property_descr_to_basedata_descrs = {
            property_descr: tuple(traverse(property_descr, set()))
            for property_descr in descrs.values()
            if isinstance(property_descr, lazy_property)
        }
        basedata_descr_to_property_descrs = {
            basedata_descr: tuple(
                property_descr
                for property_descr, basedata_descrs in property_descr_to_basedata_descrs.items()
                if basedata_descr in basedata_descrs
            )
            for basedata_descr in descrs.values()
            if isinstance(basedata_descr, lazy_basedata)
        }

        cls.__slots__ = ()  # TODO
        cls._VACANT_INSTANCES = []
        cls._BASEDATA_DESCR_TO_PROPERTY_DESCRS = basedata_descr_to_property_descrs
        cls._PROPERTY_DESCR_TO_BASEDATA_DESCRS = property_descr_to_basedata_descrs
        cls._PROPERTY_DESCR_TO_PARAMETER_DESCRS = property_descr_to_parameter_descrs
        return super().__init_subclass__()

    def __new__(cls):
        if (instances := cls._VACANT_INSTANCES):
            instance = instances.pop()
            assert isinstance(instance, cls)
        else:
            instance = super().__new__(cls)
        return instance

    def __delete__(self) -> None:
        self._restock()

    #@classmethod
    #def _setup_descrs(cls, descrs: dict[str, lazy_basedata | lazy_property]) -> tuple[list[lazy_basedata], list[lazy_property]]:
    #    

    #    basedata_descr_list: list[lazy_basedata] = []
    #    for basedata_descr, property_descrs in basedata_descr_to_property_descrs.items():
    #        basedata_descr.property_descrs = property_descrs
    #        basedata_descr_list.append(basedata_descr)
    #    property_descr_list: list[lazy_property] = []
    #    for property_descr, basedata_descrs in property_descr_to_basedata_descrs.items():
    #        property_descr.basedata_descrs = basedata_descrs
    #        property_descr_list.append(property_descr)
    #    #import pprint
    #    #print(cls)
    #    #pprint.pprint(property_descr_to_basedata_descrs)
    #    #pprint.pprint(basedata_descr_to_property_descrs)
    #    #print()
    #    return basedata_descr_list, property_descr_list


        ##data_descrs = [descr for _, descr in descrs.items() if isinstance(descr, lazy_data)]
        ##property_descrs = [descr for _, descr in descrs.items() if isinstance(descr, lazy_property)]
        #property_descr_to_basedata_descrs: dict[lazy_property, list[lazy_basedata]] = {}
        #property_descr_to_property_descrs: dict[lazy_property, list[lazy_property]] = {}
        #for descr in descrs.values():
        #    if not isinstance(descr, lazy_property):
        #        continue
        #    basedata_descrs: list[lazy_basedata] = []
        #    property_descrs: list[lazy_property] = []

        #    #children: list[lazy_basedata | lazy_property] = []
        #    for name, param_annotation in descr.parameters.items():
        #        param_descr = descrs[name]
        #        cls._check_annotation_matching(param_descr.annotation, param_annotation)
        #        if isinstance(param_descr, lazy_property):
        #            if param_descr in property_descr_to_basedata_descrs:
        #                basedata_descrs.extend(
        #                    basedata_descr for basedata_descr in property_descr_to_basedata_descrs[param_descr]
        #                    if basedata_descr not in basedata_descrs
        #                )
        #                property_descrs.extend(
        #                    property_descr for property_descr in property_descr_to_property_descrs[param_descr]
        #                    if property_descr not in property_descrs
        #                )
        #                #children.extend(
        #                #    child for child in children_dict[param_descr]
        #                #    if child not in children
        #                #)
        #            else:
        #                if param_descr not in property_descrs:
        #                    property_descrs.append(param_descr)
        #        else:
        #            if param_descr not in basedata_descrs:
        #                basedata_descrs.append(param_descr)
        #        #children.update(children_dict.get(descr, {descr}))
        #        #prop.add(properties[param_name])
        #    for property_descrs in property_descr_to_property_descrs.values():
        #        if descr in property_descrs:
        #            property_descrs.remove(descr)
        #            property_descrs.extend(descr)
        #    for property_children in children_dict.values():
        #        if descr in property_children:
        #            property_children.remove(descr)
        #            property_children.extend(
        #                child for child in children
        #                if child not in property_children
        #            )
        #    #children_dict[descr] = children
        #    property_descr_to_basedata_descrs[descr] = basedata_descrs
        #    property_descr_to_property_descrs[descr] = property_descrs

        #property_descrs: list[lazy_property] = []
        #parents_dict: dict[lazy_basedata, list[lazy_property]] = {
        #    descr: [] for descr in descrs.values() if isinstance(descr, lazy_basedata)
        #}
        #print(cls.__name__)
        #for property_descr, children in children_dict.items():
        #    basedata_descrs: list[lazy_basedata] = []
        #    for basedata_descr in children:
        #        assert isinstance(basedata_descr, lazy_basedata)
        #        parents_dict[basedata_descr].append(property_descr)
        #        basedata_descrs.append(basedata_descr)
        #    property_descr.basedata_descrs = tuple(basedata_descrs)
        #    print(property_descr, basedata_descrs)
        #    property_descrs.append(property_descr)
        #import pprint
        #pprint.pprint(parents_dict)
        #print()
        #basedata_descrs: list[lazy_basedata] = []
        #for basedata_descr, parents in parents_dict.items():
        #    basedata_descr.property_descrs = tuple(parents)
        #    basedata_descrs.append(basedata_descr)

    def _copy(self):
        result = self.__new__(self.__class__)
        for basedata_descr in self._BASEDATA_DESCR_TO_PROPERTY_DESCRS:
            basedata = basedata_descr.instance_to_basedata_dict.get(self, None)
            basedata_descr._set_data(result, basedata)
                #basedata_descr.instance_to_basedata_dict[result] = basedata
                #basedata_descr._increment_basedata_refcnt(basedata)
        #for property_descr in self._PROPERTY_DESCR_TO_BASEDATA_DESCRS:
        #    if (basedata_tuple := property_descr.instance_to_basedata_tuple_dict.get(self, None)) is not None:
        #        property_descr.instance_to_basedata_tuple_dict[result] = basedata_tuple
        #        property_descr._increment_basedata_tuple_refcnt(basedata_tuple)
        return result

    #def _reinitialize_data(self) -> None:
    #    for basedata_descr in self._LAZY_BASEDATA_DESCRS:
    #        basedata_descr.instance_to_basedata_dict.pop(self, None)

    def _restock(self) -> None:
        for basedata_descr in self._BASEDATA_DESCR_TO_PROPERTY_DESCRS:
            basedata_descr._set_data(self, None)
                #basedata_descr._decrement_basedata_refcnt(basedata)
        #for property_descr in self._PROPERTY_DESCR_TO_BASEDATA_DESCRS:
        #    if (basedata_tuple := property_descr.instance_to_basedata_tuple_dict.pop(self, None)) is not None:
        #        property_descr._decrement_basedata_tuple_refcnt(basedata_tuple)
        self._VACANT_INSTANCES.append(self)

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




#class lazy_property(Generic[_LazyBaseT, _T], Node):
#    def __init__(self, static_method: Callable[..., _T]):
#        #assert isinstance(method, staticmethod)
#        method = static_method.__func__
#        self.method: Callable[..., _T] = method
#        signature = inspect.signature(method)
#        self.name: str = method.__name__
#        self.annotation: _Annotation = signature.return_annotation
#        self.parameters: dict[str, _Annotation] = {
#            f"_{parameter.name}_": parameter.annotation
#            for parameter in list(signature.parameters.values())
#        }
#        self.ancestors: list[lazy_property[_LazyBaseT, _T]] = []
#        self.value_dict: dict[_LazyBaseT, _T] = {}
#        self.requires_update: dict[_LazyBaseT, bool] = {}
#        #self.release_method: Callable[[_T], None] | None = None
#        super().__init__()

#    @overload
#    def __get__(self, instance: None, owner: type[_LazyBaseT] | None = None) -> "lazy_property[_LazyBaseT, _T]": ...

#    @overload
#    def __get__(self, instance: _LazyBaseT, owner: type[_LazyBaseT] | None = None) -> _T: ...

#    def __get__(self, instance: _LazyBaseT | None, owner: type[_LazyBaseT] | None = None) -> "lazy_property[_LazyBaseT, _T] | _T":
#        if instance is None:
#            return self
#        if not self.requires_update[instance]:
#            return self.value_dict[instance]
#        #if self.release_method is not None:
#        #if instance in self.value_dict:
#        #    del self.value_dict[instance]
#                #self.release_method(self.value_dict[instance])
#        value = self.method(*(
#            instance.__getattribute__(parameter)
#            for parameter in self.parameters
#        ))
#        self.value_dict[instance] = value
#        self.requires_update[instance] = False
#        return value

#    def __set__(self, instance: _LazyBaseT, value: _T) -> None:
#        raise ValueError("Attempting to set a readonly lazy property")

#    #@property
#    #def stripped_name(self) -> str:
#    #    return self.name.strip("_")

#    #def releaser(self, release_method: Callable[[_T], None]) -> Callable[[_T], None]:
#    #    self.release_method = release_method
#    #    return release_method

#    def add_instance(self, instance: _LazyBaseT) -> None:
#        self.requires_update[instance] = True

#    def update_ancestors_cache(self) -> None:
#        self.ancestors = list(self.iter_ancestors())

#    def expire_instance(self, instance: _LazyBaseT) -> None:
#        for expired_prop in self.ancestors:
#            expired_prop.requires_update[instance] = True


#class lazy_property_updatable(lazy_property[_LazyBaseT, _T]):
#    @overload
#    def __get__(self, instance: None, owner: type[_LazyBaseT] | None = None) -> "lazy_property_updatable[_LazyBaseT, _T]": ...

#    @overload
#    def __get__(self, instance: _LazyBaseT, owner: type[_LazyBaseT] | None = None) -> _T: ...

#    def __get__(self, instance: _LazyBaseT | None, owner: type[_LazyBaseT] | None = None) -> "lazy_property_updatable[_LazyBaseT, _T] | _T":
#        if instance is None:
#            return self
#        return self.value_dict[instance]

#    def add_instance(self, instance: _LazyBaseT) -> None:
#        self.value_dict[instance] = self.method()

#    def updater(self, update_method: Callable[Concatenate[_LazyBaseT, _P], _R]) -> Callable[Concatenate[_LazyBaseT, _P], _R]:
#        def new_update_method(instance: _LazyBaseT, *args: _P.args, **kwargs: _P.kwargs) -> _R:
#            self.expire_instance(instance)
#            return update_method(instance, *args, **kwargs)
#        return new_update_method


#class lazy_property_writable(lazy_property_updatable[_LazyBaseT, _T]):
#    def __set__(self, instance: _LazyBaseT, value: _T) -> None:
#        self.expire_instance(instance)
#        self.value_dict[instance] = value


#class LazyBase(ABC):
#    _PROPERTIES: ClassVar[list[lazy_property]]

#    def __init_subclass__(cls) -> None:
#        properties: dict[str, lazy_property] = {}
#        for parent_cls in cls.__mro__[::-1]:
#            for name, method in parent_cls.__dict__.items():
#                if name not in properties:
#                    if isinstance(method, lazy_property):
#                        properties[name] = method
#                    continue
#                assert isinstance(method, lazy_property)
#                cls._check_annotation_matching(method.annotation, properties[name].annotation)
#                properties[name] = method

#        for prop in properties.values():
#            if isinstance(prop, lazy_property_updatable):
#                assert not prop.parameters
#                continue
#            for param_name, param_annotation in prop.parameters.items():
#                cls._check_annotation_matching(properties[param_name].annotation, param_annotation)
#                prop.add(properties[param_name])
#        for prop in properties.values():
#            prop.update_ancestors_cache()

#        cls._PROPERTIES = list(properties.values())
#        return super().__init_subclass__()

#    def __new__(cls, *args, **kwargs):
#        instance = super().__new__(cls)
#        for prop in cls._PROPERTIES:
#            prop.add_instance(instance)
#        return instance

#    #def __init__(self) -> None:
#    #    for prop in self._PROPERTIES:
#    #        prop.add_instance(self)
#    #        #print(self.__class__.__name__, prop.name, len(prop.value_dict))
#    #    super().__init__()

#    #def __del__(self) -> None:
#    #    for prop in self._PROPERTIES:
#    #        print(prop.name, len(prop.value_dict))
#    #    super().__del__(self)


"""
class A(LazyBase):
    @lazy_property
    @staticmethod
    def _p_(q: str) -> int:
        return int(q)
    @lazy_basedata
    @staticmethod
    def _q_() -> str:
        return "2"

class B(A):
    pass


a = B()
s = a._p_ + 3
#a._q_ + "8"
print(s, a._p_)
"""
