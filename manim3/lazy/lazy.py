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
  the `_is_plural` flag.

  When `is_variable` is true, the method should not take any parameter, and
  returns the initial value for the variable slot.

  The name of each parameter should be concatenated by names of a descriptor
  chain started from the local class, with underscores stripped. For example,
  the name `a__b__c` under class `A` fetches data through the path
  `A._a_._b_._c_`. The fetched data will be an `n`-layer tuple tree, where `n`
  is the number of descriptors with their `_is_plural` flags set to be true.

- `is_variable: bool`
  Determines whether the descriptor behaves as a variable or a property.

  One can call `__set__` of the descriptor on some instance only when:
  - the `is_variable` is true;
  - the instance is not frozen.

- `hasher: Callable[[T], Hashable]`
  Defines how elements are shared. Defaults to be `id`, meaning elements are
  never shared unless direct assignment is performed. Other options are also
  provided under `Lazy` namespace, named as `xxx_hasher`.

  Providing a hasher is encouraged whenever applicable, as this reduces
  redundant calculations. However, one cannot provide a hasher other than `id`
  when `freeze` is false.

- `freeze: bool`
  Determines whether data should be frozen when binding. Defaults to be true.
  Forced to be true when `is_variable` is false. When false, `hasher` is
  forced to be `id`.

  Note, freezing bound data does not block `__set__`. In other words, we are
  freezing the data itself, not the binding relation. However, Unbinding data
  by reassigning a new one does not unfreeze the data.

  In fact, the freezing procedure can not go beyond the lazy scope. It only
  prevents users from calling `__set__` of variable descriptors on descendant
  lazy objects, but does not prevent users from modifying data that is not of
  type `LazyObject`, e.g., `np.ndarray`.

- `cache_capacity: int`
  Determines the capacity of the lru cache of parameters-data pairs generated
  from `method`. Forced to be 1 when `is_variable` is true (the parameter list
  is always empty). Defaults to be 128 when `is_variable` is false. However,
  the cache is only used when `freeze` is true.

- `_is_plural: bool` (readonly property)
  Determines whether data contains exactly one or arbitrarily many elements.
  When true, `DataT` is specialized as `tuple[T]`; when false, specialized
  as `T`.

Descriptor overriding is allowed. The overriding descriptor should match the
overridden one in `_is_plural` and `hasher`. Furthermore, the type of element
(the specialization of type variable `T`) should be consistent between
descriptors.
"""


from typing import (
    Any,
    Callable,
    Hashable,
    Never,
    Self
)

import numpy as np

from .lazy_descriptor import (
    LazyDescriptor,
    LazyPluralDescriptor,
    LazySingularDescriptor
)
from .lazy_object import LazyObject


class Lazy:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def _singular_descriptor[T](
        cls: type[Self],
        is_variable: bool,
        freeze: bool,
        cache_capacity: int,
        hasher: Callable[..., Hashable]
    ) -> Callable[[Callable[..., T]], LazySingularDescriptor[T]]:

        def result(
            method: Callable[[], T]
        ) -> LazySingularDescriptor[T]:
            return LazySingularDescriptor(
                method=method,
                is_variable=is_variable,
                hasher=hasher,
                freezer=cls.lazy_freezer,
                freeze=freeze,
                cache_capacity=cache_capacity
            )

        return result

    @classmethod
    def _plural_descriptor[T](
        cls: type[Self],
        is_variable: bool,
        hasher: Callable[..., Hashable],
        freeze: bool,
        cache_capacity: int
    ) -> Callable[[Callable[..., tuple[T, ...]]], LazyPluralDescriptor[T]]:

        def result(
            method: Callable[[], tuple[T, ...]]
        ) -> LazyPluralDescriptor[T]:
            return LazyPluralDescriptor(
                method=method,
                is_variable=is_variable,
                hasher=hasher,
                freezer=cls.lazy_freezer,
                freeze=freeze,
                cache_capacity=cache_capacity
            )

        return result

    @classmethod
    def variable[T](
        cls: type[Self],
        hasher: Callable[..., Hashable] = id,
        freeze: bool = True
    ) -> Callable[[Callable[[], T]], LazyDescriptor[T, T]]:
        return cls._singular_descriptor(
            is_variable=True,
            hasher=hasher,
            freeze=freeze,
            cache_capacity=1
        )

    @classmethod
    def property[T](
        cls: type[Self],
        hasher: Callable[..., Hashable] = id,
        cache_capacity: int = 128
    ) -> Callable[[Callable[..., T]], LazyDescriptor[T, T]]:
        return cls._singular_descriptor(
            is_variable=False,
            hasher=hasher,
            freeze=True,
            cache_capacity=cache_capacity
        )

    @classmethod
    def variable_collection[T](
        cls: type[Self],
        hasher: Callable[..., Hashable] = id,
        freeze: bool = True
    ) -> Callable[[Callable[[], tuple[T, ...]]], LazyDescriptor[T, tuple[T, ...]]]:
        return cls._plural_descriptor(
            is_variable=True,
            hasher=hasher,
            freeze=freeze,
            cache_capacity=1
        )

    @classmethod
    def property_collection[T](
        cls: type[Self],
        hasher: Callable[..., Hashable] = id,
        cache_capacity: int = 128
    ) -> Callable[[Callable[..., tuple[T, ...]]], LazyDescriptor[T, tuple[T, ...]]]:
        return cls._plural_descriptor(
            is_variable=False,
            hasher=hasher,
            freeze=True,
            cache_capacity=cache_capacity
        )

    @staticmethod
    def naive_hasher(
        element: Hashable
    ) -> Hashable:
        return element

    @staticmethod
    def array_hasher(
        element: np.ndarray
    ) -> bytes:
        # In order to make the hasher work properly, all `np.ndarray`
        # instances bound to some descriptor shall share `dtype`, `ndim`,
        # and at least `ndim - 1` fixed entries of `shape`.
        return element.tobytes()

    @classmethod
    def lazy_freezer(
        cls: type[Self],
        element: Any
    ) -> None:
        if not isinstance(element, LazyObject):
            return
        if element._is_frozen:
            return
        element._is_frozen = True
        for descriptor in type(element)._lazy_descriptors:
            descriptor.get_slot(element).disable_writability()
            for child_element in descriptor.get_elements(element):
                cls.lazy_freezer(child_element)
