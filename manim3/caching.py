#__all__ = ["hash_items"]


#from typing import Hashable

#from xxhash import xxh3_64_hexdigest


#def hash_items(*items: Hashable) -> str:
#    return xxh3_64_hexdigest(
#        b"".join(
#            bytes(hex(hash(item)), encoding="ascii")
#            for item in items
#        )
#    )
