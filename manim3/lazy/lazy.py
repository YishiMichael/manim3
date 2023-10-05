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

The constructor of `LazyDescriptor[_DataT, _T]` has the following notable
parameters:

- `method: Callable[..., _DataT]`
  Defines how the data is calculated through parameters. The returned data
  should be a single element, or a tuple of elements, determined by
  the `is_multiple` flag.

  When `is_variable` is true, the method should not take any parameter, and
  returns the initial value for the variable slot.

  The name of each parameter should be concatenated by names of a descriptor
  chain started from the local class, with underscores stripped. For example,
  the name `a__b__c` under class `A` fetches data through the path
  `A._a_._b_._c_`. The fetched data will be an `n`-layer tuple tree, where `n`
  is the number of descriptors with their `is_multiple` flags set to be true.

- `is_multiple: bool`
  Determines whether data contains a singular element or multiple elements.
  When true, `_DataT` is specialized as `tuple[_T]`; when false, specialized
  as `_T`.

- `is_variable: bool`
  Determines whether the descriptor behaves as a variable or a property.

  One can call `__set__` of the descriptor on some instance only when:
  - the `is_variable` is true;
  - the instance is not frozen.

- `hasher: Callable[[_T], Hashable]`
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

Descriptor overriding is allowed. The overriding descriptor should match the
overridden one in `is_multiple` and `hasher`. Furthermore, the type of element
(the specialization of type variable `_T`) should be consistent between
descriptors.
"""


from typing import (
    Callable,
    Hashable,
    Never,
    Self
)

import numpy as np

from .lazy_descriptor import LazyDescriptor
#from .lazy_object import LazyObject


class Lazy:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def _descriptor_singular[T](
        cls: type[Self],
        is_variable: bool,
        freeze: bool,
        cache_capacity: int,
        hasher: Callable[..., Hashable]
    ) -> Callable[[Callable[..., T]], LazyDescriptor[T, T]]:

        def singular_decomposer(
            data: T
        ) -> tuple[T, ...]:
            return (data,)

        def singular_composer(
            elements: tuple[T, ...]
        ) -> T:
            (element,) = elements
            return element

        def result(
            method: Callable[[], T]
        ) -> LazyDescriptor[T, T]:
            return LazyDescriptor(
                method=method,
                is_multiple=False,
                decomposer=singular_decomposer,
                composer=singular_composer,
                is_variable=is_variable,
                hasher=hasher,
                freeze=freeze,
                cache_capacity=cache_capacity
            )

        return result

    @classmethod
    def _descriptor_multiple[T](
        cls: type[Self],
        is_variable: bool,
        hasher: Callable[..., Hashable],
        freeze: bool,
        cache_capacity: int
    ) -> Callable[[Callable[..., tuple[T, ...]]], LazyDescriptor[tuple[T, ...], T]]:

        def multiple_decomposer(
            data: tuple[T, ...]
        ) -> tuple[T, ...]:
            return data

        def multiple_composer(
            elements: tuple[T, ...]
        ) -> tuple[T, ...]:
            return elements

        def result(
            method: Callable[[], tuple[T, ...]]
        ) -> LazyDescriptor[tuple[T, ...], T]:
            return LazyDescriptor(
                method=method,
                is_multiple=True,
                decomposer=multiple_decomposer,
                composer=multiple_composer,
                is_variable=is_variable,
                hasher=hasher,
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
        return cls._descriptor_singular(
            is_variable=True,
            hasher=hasher,
            freeze=freeze,
            cache_capacity=1
        )

    @classmethod
    def variable_collection[T](
        cls: type[Self],
        hasher: Callable[..., Hashable] = id,
        freeze: bool = True
    ) -> Callable[[Callable[[], tuple[T, ...]]], LazyDescriptor[tuple[T, ...], T]]:
        return cls._descriptor_multiple(
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
        return cls._descriptor_singular(
            is_variable=False,
            hasher=hasher,
            freeze=True,
            cache_capacity=cache_capacity
        )

    @classmethod
    def property_collection[T](
        cls: type[Self],
        hasher: Callable[..., Hashable] = id,
        cache_capacity: int = 128
    ) -> Callable[[Callable[..., tuple[T, ...]]], LazyDescriptor[tuple[T, ...], T]]:
        return cls._descriptor_multiple(
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

    #@staticmethod
    #def branch_hasher(
    #    element: LazyObject
    #) -> Hashable:
    #    return (type(element), tuple(
    #        tuple(
    #            id(variable_element)
    #            for variable_element in descriptor._get_elements(element)
    #        )
    #        for descriptor in type(element)._lazy_descriptors.values()
    #        if descriptor._is_variable
    #    ))
