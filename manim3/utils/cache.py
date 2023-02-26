__all__ = ["CacheUtils"]


from typing import Hashable

from xxhash import xxh3_64_digest


class CacheUtils:
    @classmethod
    def hash_items(
        cls,
        *items: Hashable
    ) -> bytes:
        return xxh3_64_digest(
            b"".join(
                bytes(hex(hash(item)), encoding="ascii")
                for item in items
            )
        )

    @classmethod
    def dict_as_hashable(
        cls,
        d: dict[Hashable, Hashable]
    ) -> Hashable:
        return tuple(sorted(d.items()))
