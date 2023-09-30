#from typing import (
#    Callable,
#    TypeVar
#)

#from ...mobjects.mobject.mobject import Mobject
#from .transform_to import TransformTo


#_MobjectT = TypeVar("_MobjectT", bound=Mobject)


#class TransformToCopy(TransformTo):
#    __slots__ = ()

#    def __init__(
#        self,
#        mobject: _MobjectT,
#        func: Callable[[_MobjectT], _MobjectT]
#    ) -> None:
#        super().__init__(
#            start_mobject=mobject,
#            stop_mobject=func(mobject.copy())
#        )
