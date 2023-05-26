import itertools as it
from typing import (
    Hashable,
    Iterable,
    Iterator,
    TypeVar
)


_T = TypeVar("_T")
_T0 = TypeVar("_T0")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


class IterUtils:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @classmethod
    def unzip_pairs(
        cls,
        pair_iters: Iterable[tuple[_T0, _T1]]
    ) -> tuple[Iterator[_T0], Iterator[_T1]]:
        iterator_0, iterator_1 = it.tee(pair_iters)
        return (
            (item_0 for item_0, _ in iterator_0),
            (item_1 for _, item_1 in iterator_1)
        )

    @classmethod
    def unzip_triplets(
        cls,
        triplet_iters: Iterable[tuple[_T0, _T1, _T2]]
    ) -> tuple[Iterator[_T0], Iterator[_T1], Iterator[_T2]]:
        iterator_0, iterator_1, iterator_2 = it.tee(triplet_iters, 3)
        return (
            (item_0 for item_0, _, _ in iterator_0),
            (item_1 for _, item_1, _ in iterator_1),
            (item_2 for _, _, item_2 in iterator_2)
        )
