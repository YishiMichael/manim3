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
  Finally, the last descriptor should have `freeze` set true.

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

  Determines whether data can be internally modified.

  One shall note that actually no freezing procedure is taken in runtime for
  better performance. Internally modifying data bound to a descriptor with
  `freeze` set true may result in unexpected results.

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
    Callable,
    Literal,
    Never,
    Self,
    overload
)

from .lazy_descriptor import LazyDescriptor


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
    ) -> Callable[[Callable[[], T]], LazyDescriptor[T, T]]: ...

    @overload
    @classmethod
    def variable[T](
        cls: type[Self],
        *,
        plural: Literal[True]
    ) -> Callable[[Callable[[], tuple[T, ...]]], LazyDescriptor[T, tuple[T, ...]]]: ...

    @classmethod
    def variable(
        cls: type[Self],
        *,
        plural: bool = False
    ) -> Callable[[Callable], LazyDescriptor]:

        def result(
            method: Callable
        ) -> LazyDescriptor:
            assert isinstance(method, staticmethod)
            return LazyDescriptor(
                method=method.__func__,
                is_property=False,
                plural=plural,
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
        deepcopy: bool = True
    ) -> Callable[[Callable[[], T]], LazyDescriptor[T, T]]: ...

    @overload
    @classmethod
    def volatile[T](
        cls: type[Self],
        *,
        plural: Literal[True],
        deepcopy: bool = True
    ) -> Callable[[Callable[[], tuple[T, ...]]], LazyDescriptor[T, tuple[T, ...]]]: ...

    @classmethod
    def volatile(
        cls: type[Self],
        *,
        plural: bool = False,
        deepcopy: bool = True
    ) -> Callable[[Callable], LazyDescriptor]:

        def result(
            method: Callable
        ) -> LazyDescriptor:
            assert isinstance(method, staticmethod)
            return LazyDescriptor(
                method=method.__func__,
                is_property=False,
                plural=plural,
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
        cache_capacity: int = 128
    ) -> Callable[[Callable[..., T]], LazyDescriptor[T, T]]: ...

    @overload
    @classmethod
    def property[T](
        cls: type[Self],
        *,
        plural: Literal[True],
        cache_capacity: int = 128
    ) -> Callable[[Callable[..., tuple[T, ...]]], LazyDescriptor[T, tuple[T, ...]]]: ...

    @classmethod
    def property(
        cls: type[Self],
        *,
        plural: bool = False,
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
                freeze=True,
                deepcopy=False,
                cache_capacity=cache_capacity
            )

        return result
