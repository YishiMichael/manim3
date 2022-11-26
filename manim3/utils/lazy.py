from abc import ABCMeta
from typing import Any, Callable


__all__ = [
    "lazy_property",
    "lazy_property_initializer",
    "LazyMeta"
]


# TODO: typing disaster


class lazy_property(property):
    def __init__(self, class_method: Callable):
        self._class_method_: Callable = class_method
        super().__init__()


class lazy_property_initializer(property):
    def __init__(self, static_method: Callable[[], Any]):
        self._static_method_: Callable[[], Any] = static_method
        self._update_method_names_: list[str] = []
        super().__init__()

    def updater(self, update_method: Callable) -> Callable:
        self._update_method_names_.append(update_method.__name__)
        return update_method


class LazyMeta(ABCMeta):
    _PROPERTY_INITIALIZERS_DICTS_: dict[str, dict[str, Callable[[], Any]]] = {}
    _PROPERTY_PARAMETERS_DICTS_: dict[str, dict[str, list[str]]] = {}

    def __new__(cls, cls_name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        root_lazy_properties: dict[str, lazy_property_initializer] = {}
        leaf_lazy_properties: dict[str, lazy_property] = {}
        for name, attr in namespace.items():
            #if not name.startswith("_") and name.endswith("_"):
            #    continue
            if isinstance(attr, lazy_property_initializer):
                root_lazy_properties[name] = attr
            elif isinstance(attr, lazy_property):
                leaf_lazy_properties[name] = attr

        cls_initializers_dict: dict[str, Callable[[], Any]] = {
            name: root_property._static_method_
            for name, root_property in root_lazy_properties.items()
        }
        cls_parameters_dict: dict[str, list[str]] = {}
        for name, leaf_property in leaf_lazy_properties.items():
            code = leaf_property._class_method_.__code__
            cls_parameters_dict[name] = [
                f"_{varname}_"
                for varname in code.co_varnames[1:code.co_argcount]  # ignore cls
            ]

        cls._PROPERTY_INITIALIZERS_DICTS_[cls_name] = cls_initializers_dict
        cls._PROPERTY_PARAMETERS_DICTS_[cls_name] = cls_parameters_dict

        initializers_dict: dict[str, Callable[[], Any]] = {}
        parameters_dict: dict[str, list[str]] = {}
        mro = cls.get_mro_by_bases(bases)
        for base in reversed(mro):
            if not isinstance(base, LazyMeta):
                continue
            initializers_dict.update(cls._PROPERTY_INITIALIZERS_DICTS_[base.__name__])
            parameters_dict.update(cls._PROPERTY_PARAMETERS_DICTS_[base.__name__])
        initializers_dict.update(cls_initializers_dict)
        parameters_dict.update(cls_parameters_dict)

        parameters_affecting_dict: dict[str, set[str]] = {
            name: cls.search_for_affected_parameters(parameters_dict, name)
            for name in initializers_dict
        }

        for name, root_property in root_lazy_properties.items():
            namespace[name] = cls.setup_root_property(name)

            for update_method_name in root_property._update_method_names_:
                namespace[update_method_name] = cls.setup_root_property_updater(
                    name, namespace[update_method_name], parameters_affecting_dict
                )

        for name, leaf_property in leaf_lazy_properties.items():
            namespace[name] = cls.setup_leaf_property(
                name, leaf_property._class_method_, parameters_dict[name]
            )

        init_requires_update = {}
        for name in initializers_dict:
            init_requires_update[name] = False
        for name in parameters_dict:
            init_requires_update[name] = True

        namespace["__new__"] = cls.setup_instance_new(mro[0], initializers_dict, init_requires_update)

        return super().__new__(cls, cls_name, bases, namespace)

    @classmethod
    def get_mro_by_bases(cls, bases: tuple[type, ...]) -> tuple[type, ...]:
        if not bases:
            return (object,)
        result = []
        mro_lists = [base.__mro__ for base in bases] + [(base,) for base in bases]
        while mro_lists:
            for l in mro_lists:
                head = l[0]
                if all(head not in ll[1:] for ll in mro_lists):
                    result.append(head)
                    new_lists = []
                    for ll in mro_lists:
                        if ll[0] == head:
                            if len(ll) != 1:
                                new_lists.append(ll[1:])
                        else:
                            new_lists.append(ll)
                    mro_lists = new_lists
                    break
            else:
                break
        if mro_lists:
            raise TypeError
        return tuple(result)

    @classmethod
    def search_for_affected_parameters(cls, parameters_dict: dict[str, list[str]], name: str) -> set[str]:
        result = set()
        extended = {name}
        while result != extended:
            result = extended
            extended = result.union(
                set(
                    param for param, args in parameters_dict.items()
                    if any(arg in args for arg in result)
                )
            )
        return result

    @classmethod
    def setup_root_property(cls, name):
        #@wraps
        def f_get(self):
            return self._lazy_properties_[name]

        def f_set(self, value):
            self._lazy_properties_[name] = value
        return property(f_get, f_set)

    @classmethod
    def setup_root_property_updater(cls, name, update_method, parameters_affecting_dict):
        #@wraps
        def f_update(self, *args, **kwargs):
            #result = update_method(self, *args, **kwargs)
            #self._lazy_properties_[name] = result
            for supported_name in parameters_affecting_dict[name]:
                self._requires_update_[supported_name] = True
            return update_method(self, *args, **kwargs)
        return f_update

    @classmethod
    def setup_leaf_property(cls, name, class_method, varnames):
        #@wraps
        def f_get(self):
            if not self._requires_update_[name]:
                return self._lazy_properties_[name]
            value = class_method(self.__class__, *(
                self.__getattribute__(varname)
                for varname in varnames
            ))
            self._lazy_properties_[name] = value
            self._requires_update_[name] = False
            return value
        return property(f_get)

    @classmethod
    def setup_instance_new(cls, parent_cls, initializers_dict, init_requires_update):
        if parent_cls is object:
            __new__ = lambda kls, *args, **kwargs: object.__new__(kls)
        else:
            __new__ = lambda kls, *args, **kwargs: parent_cls.__new__(kls, *args, **kwargs)

        def instance_new(kls, *args, **kwargs):
            result = __new__(kls, *args, **kwargs)
            lazy_properties = {}
            for name, initializer in initializers_dict.items():
                try:
                    init_value = initializer()
                except NotImplementedError:
                    init_value = NotImplemented
                lazy_properties[name] = init_value
            result._lazy_properties_ = lazy_properties
            result._requires_update_ = init_requires_update.copy()
            return result
        return instance_new


"""
class A(metaclass=LazyMeta):
    @lazy_property_initializer
    def _foo_() -> list[int]:
        return []

    @lazy_property_initializer
    def _bar_() -> list[str]:
        return ["bar"]

    @lazy_property
    def _lazy_prop_(foo: list[int], bar: list[str]):
        return foo + bar

    #@_foo_.setter  # ???
    #def _foo_(self, i: int):
    #    self._foo_ = i
    #    return self

    @_foo_.updater
    def add_foo(self, i: int):
        self._foo_.append(i)
        return self


a = A()
a.add_foo("")
print(a._foo_)  # ['']
print(a._lazy_prop_)  # ['', 'bar']
a.add_foo("123")
print(a._lazy_prop_)  # ['', '123', 'bar']
a._foo_ = [""]
print(a._foo_)
"""
