#from typing import Callable, Generic, TypeVar


from abc import ABCMeta


__all__ = [
    "lazy_property",
    "LazyMeta",
    "expire_properties"
]


# TODO: typing disaster


#class LazyPropertyData:
#    def __init__(self, value):
#        self.value = value
#        self.requires_update: bool = False


class lazy_property(property):
    pass


class LazyMeta(ABCMeta):
    def __new__(cls, cls_name, bases, attrs):
        lazy_property_names = set()

        for name, attr in attrs.items():
            if not isinstance(attr, lazy_property):
                continue
            if attr.fget is None:
                continue

            lazy_property_names.add(name)
            getter = cls.setup_lazy_getter(name, attr.fget)
            if attr.fset is not None:
                setter = cls.setup_lazy_setter(name)
            else:
                setter = cls.setup_deleted_setter()
            deleter = cls.setup_deleted_deleter()
            attrs[name] = lazy_property(getter, setter, deleter)

        for base in bases:
            if isinstance(base, LazyMeta):
                lazy_property_names.update(base._supporting_dict_)

        attrs["_currently_initializing_"] = []
        attrs["_supporting_dict_"] = {
            name: set() for name in lazy_property_names
        }

        new = attrs.get("__new__", None)
        if new is None:
            for base in bases:
                new = base.__dict__.get("__new__", None)
                if new is not None:
                    break
            else:
                new = lambda kls, *args, **kwargs: object.__new__(kls)

        def instance_new(kls, *args, **kwargs):
            result = new(kls, *args, **kwargs)
            result._lazy_properties_ = {}
            result._requires_update_ = {
                name: True for name in lazy_property_names
            }
            return result
        attrs["__new__"] = instance_new

        return super().__new__(cls, cls_name, bases, attrs)

    @classmethod
    def setup_lazy_getter(cls, name, fget):
        def new_fget(self):
            if self.__class__._currently_initializing_:
                self.__class__._supporting_dict_[name].update(self.__class__._currently_initializing_)
            if name not in self._lazy_properties_:
                self.__class__._currently_initializing_.append(name)
                value = fget(self)
                self.__class__._currently_initializing_.remove(name)
                self._lazy_properties_[name] = value
            else:
                #print(self.__class__, self._requires_update_)
                if self._requires_update_[name]:
                    value = fget(self)
                    self._lazy_properties_[name] = value
                    self._requires_update_[name] = False
                else:
                    value = self._lazy_properties_[name]
            return value
        return new_fget

    @classmethod
    def setup_lazy_setter(cls, name):
        def new_fset(self, value):
            self._lazy_properties_[name] = value
            for expired_name in self.__class__._supporting_dict_[name]:
                self._requires_update_[expired_name] = True
            #fset(self, value)
        return new_fset

    @classmethod
    def setup_deleted_setter(cls):
        def new_fset(self, value):
            raise NotImplementedError
        return new_fset

    @classmethod
    def setup_deleted_deleter(cls):
        def new_fdel(self):
            raise NotImplementedError
        return new_fdel


def expire_properties(*property_names):
    def wrapped(method):
        def new_method(self, *args, **kwargs):
            expired_names = set(property_names).union(*(
                self.__class__._supporting_dict_[name]
                for name in property_names
            ))
            for expired_name in expired_names:
                self._requires_update_[expired_name] = True
            return method(self, *args, **kwargs)
        return new_method
    return wrapped
