from __future__ import annotations


"""
This module implements lazy evaluation based on weak reference. Meanwhile,
this also introduces functional programming into the project paradigm.

Every child class of `LazyObject` shall define `__slots__`, and all methods
shall be basically sorted in the following way:
- magic methods;
- lazy variables;
- lazy properties;
- private class methods;
- private methods;
- public methods.

All methods decorated by any decorator provided by `Lazy` should be static
methods and be named with underscores appeared on both sides, i.e. `_data_`.
Successive underscores shall not occur, due to the name convension handled by
lazy properties.

The constructor of `LazyDescriptor[T, DataT]` has the following notable
parameters:

- `method: Callable[..., DataT]`
  Defines how the data is calculated through parameters. The returned data
  should be a single element, or a tuple of elements, determined by
  the `plural` flag.

  When `is_property` is false, the method should not take any parameter, and
  returns the initial value for the variable slot.

  The name of each parameter should be concatenated by names of a descriptor
  chain started from the local class, with underscores stripped. For example,
  the name `a__b__c` under class `A` fetches data through the path
  `A._a_._b_._c_`. The fetched data will be an `n`-layer tuple tree, where `n`
  is the number of descriptors with their `plural` flags set to be true.

- `is_property: bool`
  Lazy.variable: false
  Lazy.volatile: false
  Lazy.property: true

  Determines whether the descriptor behaves as a variable or a property.

  One can call `__set__` of the descriptor on some instance only when:
  - `is_property` is false;
  - the instance is not frozen.

- `plural: bool`
  Lazy.variable: =false
  Lazy.volatile: =false
  Lazy.property: =false

  Determines whether data contains exactly one or arbitrarily many elements.
  When true, `DataT` is specialized as `tuple[T]`; when false, specialized
  as `T`.

- `freeze: bool`
  Lazy.variable: true
  Lazy.volatile: false
  Lazy.property: true

  Determines whether data should be frozen when binding.

  Note, freezing bound data does not block `__set__`. In other words, we are
  freezing the data itself, not the binding relation. However, Unbinding data
  by reassigning a new one does not unfreeze the data.

  In fact, the freezing procedure can not go beyond the lazy scope. It only
  prevents users from calling `__set__` of variable descriptors on descendant
  lazy objects, but does not prevent users from modifying data that is not of
  type `LazyObject`, e.g., `np.ndarray`.

- `deepcopy: bool`
  Lazy.variable: false
  Lazy.volatile: =true
  Lazy.property: false

  Determines how data in the descriptor is copied when calling
  `LazyObject.copy` from the parent object. Does not take effect when
  `is_property` is true, since only data in variable descriptors are copied.

- `cache_capacity: int`
  Lazy.variable: 1
  Lazy.volatile: 0
  Lazy.property: =128

  Determines the capacity of the lru cache of parameters-data pairs generated
  from `method`.

Descriptor overriding is allowed. The overriding descriptor should match the
overridden one in `plural`, `freeze` flags (a change in `freeze` from false to
true is allowed). Furthermore, the element type (the specialization of type
variable `T`) should be consistent between descriptors.
"""


from typing import (
    Any,
    Callable,
    Hashable,
    Literal,
    Never,
    Self,
    overload
)

import numpy as np

from .lazy_descriptor import LazyDescriptor
from .lazy_object import LazyObject


class Lazy:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @overload
    @classmethod
    def variable[T](
        cls: type[Self],
        *,
        plural: Literal[False] = False
        #is_variable: bool,
        #hasher: Callable[..., Hashable],
        #freeze: bool = True,
        #deepcopy: bool = True
        #cache_capacity: int
    ) -> Callable[[Callable[[], T]], LazyDescriptor[T, T]]: ...

    @overload
    @classmethod
    def variable[T](
        cls: type[Self],
        *,
        plural: Literal[True]
        #is_variable: bool,
        #hasher: Callable[..., Hashable],
        #freeze: bool = True,
        #deepcopy: bool = True
        #cache_capacity: int
    ) -> Callable[[Callable[[], tuple[T, ...]]], LazyDescriptor[T, tuple[T, ...]]]: ...

    @classmethod
    def variable(
        cls: type[Self],
        *,
        plural: bool = False
        #is_variable: bool,
        #hasher: Callable[..., Hashable],
        #freeze: bool = True,
        #deepcopy: bool = True
        #cache_capacity: int = 1
    ) -> Callable[[Callable], LazyDescriptor]:

        def result(
            method: Callable
        ) -> LazyDescriptor:
            assert isinstance(method, staticmethod)
            return LazyDescriptor(
                method=method.__func__,
                is_property=False,
                plural=plural,
                #hasher=hasher,
                #freezer=cls.lazy_freezer,
                freeze=True,
                deepcopy=False,
                cache_capacity=1
            )

        return result

    @overload
    @classmethod
    def volatile[T](
        cls: type[Self],
        *,
        plural: Literal[False] = False,
        #is_variable: bool,
        #hasher: Callable[..., Hashable],
        #freeze: bool = True,
        deepcopy: bool = True
        #cache_capacity: int
    ) -> Callable[[Callable[[], T]], LazyDescriptor[T, T]]: ...

    @overload
    @classmethod
    def volatile[T](
        cls: type[Self],
        *,
        plural: Literal[True],
        #is_variable: bool,
        #hasher: Callable[..., Hashable],
        #freeze: bool = True,
        deepcopy: bool = True
        #cache_capacity: int
    ) -> Callable[[Callable[[], tuple[T, ...]]], LazyDescriptor[T, tuple[T, ...]]]: ...

    @classmethod
    def volatile(
        cls: type[Self],
        *,
        plural: bool = False,
        #is_variable: bool,
        #hasher: Callable[..., Hashable],
        #freeze: bool = True,
        deepcopy: bool = True
        #cache_capacity: int = 1
    ) -> Callable[[Callable], LazyDescriptor]:

        def result(
            method: Callable
        ) -> LazyDescriptor:
            assert isinstance(method, staticmethod)
            return LazyDescriptor(
                method=method.__func__,
                is_property=False,
                plural=plural,
                #hasher=hasher,
                #freezer=cls.lazy_freezer,
                freeze=False,
                deepcopy=deepcopy,
                cache_capacity=0
            )

        return result

    @overload
    @classmethod
    def property[T](
        cls: type[Self],
        *,
        plural: Literal[False] = False,
        #is_variable: bool,
        #hasher: Callable[..., Hashable],
        #freeze: bool,
        #deepcopy: bool,
        cache_capacity: int = 128
    ) -> Callable[[Callable[..., T]], LazyDescriptor[T, T]]: ...

    @overload
    @classmethod
    def property[T](
        cls: type[Self],
        *,
        plural: Literal[True],
        #is_variable: bool,
        #hasher: Callable[..., Hashable],
        #freeze: bool,
        #deepcopy: bool,
        cache_capacity: int = 128
    ) -> Callable[[Callable[..., tuple[T, ...]]], LazyDescriptor[T, tuple[T, ...]]]: ...

    @classmethod
    def property(
        cls: type[Self],
        *,
        plural: bool = False,
        #is_variable: bool,
        #hasher: Callable[..., Hashable],
        #freeze: bool = True,
        #deepcopy: bool = False,
        cache_capacity: int = 128
    ) -> Callable[[Callable], LazyDescriptor]:

        def result(
            method: Callable
        ) -> LazyDescriptor:
            assert isinstance(method, staticmethod)
            return LazyDescriptor(
                method=method.__func__,
                is_property=True,
                plural=plural,
                #hasher=hasher,
                #freezer=cls.lazy_freezer,
                freeze=True,
                deepcopy=False,
                cache_capacity=cache_capacity
            )

        return result

    #@classmethod
    #def _singular_descriptor[T](
    #    cls: type[Self],
    #    is_variable: bool,
    #    #hasher: Callable[..., Hashable],
    #    freeze: bool,
    #    deepcopy: bool,
    #    cache_capacity: int
    #) -> Callable[[Callable[..., T]], LazySingularDescriptor[T]]:

    #    def result(
    #        method: Callable[[], T]
    #    ) -> LazySingularDescriptor[T]:
    #        assert isinstance(method, staticmethod)
    #        return LazySingularDescriptor(
    #            method=method.__func__,
    #            is_variable=is_variable,
    #            #hasher=hasher,
    #            #freezer=cls.lazy_freezer,
    #            freeze=freeze,
    #            deepcopy=deepcopy,
    #            cache_capacity=cache_capacity
    #        )

    #    return result

    #@classmethod
    #def _plural_descriptor[T](
    #    cls: type[Self],
    #    is_variable: bool,
    #    #hasher: Callable[..., Hashable],
    #    freeze: bool,
    #    deepcopy: bool,
    #    cache_capacity: int
    #) -> Callable[[Callable[..., tuple[T, ...]]], LazyPluralDescriptor[T]]:

    #    def result(
    #        method: Callable[[], tuple[T, ...]]
    #    ) -> LazyPluralDescriptor[T]:
    #        assert isinstance(method, staticmethod)
    #        return LazyPluralDescriptor(
    #            method=method.__func__,
    #            is_variable=is_variable,
    #            #hasher=hasher,
    #            #freezer=cls.lazy_freezer,
    #            freeze=freeze,
    #            deepcopy=deepcopy,
    #            cache_capacity=cache_capacity
    #        )

    #    return result

    #@classmethod
    #def variable[T](
    #    cls: type[Self],
    #    #hasher: Callable[..., Hashable] = id,
    #    freeze: bool = False,
    #    deepcopy: bool = True,
    #) -> Callable[[Callable[[], T]], LazyDescriptor[T, T]]:
    #    return cls._descriptor(
    #        plural=False,
    #        is_variable=True,
    #        #hasher=hasher,
    #        freeze=freeze,
    #        deepcopy=deepcopy,
    #        cache_capacity=1
    #    )

    #@classmethod
    #def variable_collection[T](
    #    cls: type[Self],
    #    #hasher: Callable[..., Hashable] = id,
    #    freeze: bool = False,
    #    deepcopy: bool = True
    #) -> Callable[[Callable[[], tuple[T, ...]]], LazyDescriptor[T, tuple[T, ...]]]:
    #    return cls._descriptor(
    #        plural=True,
    #        is_variable=True,
    #        #hasher=hasher,
    #        freeze=freeze,
    #        deepcopy=deepcopy,
    #        cache_capacity=1
    #    )

    #@classmethod
    #def property[T](
    #    cls: type[Self],
    #    #hasher: Callable[..., Hashable] = id,
    #    #deepcopy: bool = True,
    #    cache_capacity: int = 128
    #) -> Callable[[Callable[..., T]], LazyDescriptor[T, T]]:
    #    return cls._descriptor(
    #        plural=False,
    #        is_variable=False,
    #        #hasher=hasher,
    #        freeze=True,
    #        deepcopy=False,
    #        cache_capacity=cache_capacity
    #    )

    #@classmethod
    #def property_collection[T](
    #    cls: type[Self],
    #    #hasher: Callable[..., Hashable] = id,
    #    #deepcopy: bool = True,
    #    cache_capacity: int = 128
    #) -> Callable[[Callable[..., tuple[T, ...]]], LazyDescriptor[T, tuple[T, ...]]]:
    #    return cls._descriptor(
    #        plural=True,
    #        is_variable=False,
    #        #hasher=hasher,
    #        freeze=True,
    #        deepcopy=False,
    #        cache_capacity=cache_capacity
    #    )

    #@staticmethod
    #def naive_hasher(
    #    element: Hashable
    #) -> Hashable:
    #    return element

    #@staticmethod
    #def array_hasher(
    #    element: np.ndarray
    #) -> Hashable:
    #    return (element.shape, element.dtype, element.tobytes())

    #@classmethod
    #def lazy_freezer(
    #    cls: type[Self],
    #    element: Any
    #) -> None:
    #    if not isinstance(element, LazyObject):
    #        return
    #    if element._is_frozen:
    #        return
    #    element._is_frozen = True
    #    for descriptor in type(element)._lazy_descriptors:
    #        descriptor.get_slot(element).disable_writability()
    #        for child_element in descriptor.get_elements(element):
    #            cls.lazy_freezer(child_element)
