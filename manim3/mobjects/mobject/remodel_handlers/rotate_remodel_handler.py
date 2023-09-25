#import numpy as np
#from scipy.spatial.transform import Rotation

#from ....constants.custom_typing import (
#    NP_3f8,
#    NP_44f8
#)
#from .remodel_handler import RemodelHandler


#class RotateRemodelHandler(RemodelHandler):
#    __slots__ = ("_rotvec",)

#    def __init__(
#        self,
#        rotvec: NP_3f8
#    ) -> None:
#        super().__init__()
#        self._rotvec: NP_3f8 = rotvec

#    def _remodel(
#        self,
#        alpha: float | NP_3f8 = 1.0
#    ) -> NP_44f8:
#        m = np.identity(4)
#        m[:3, :3] = Rotation.from_rotvec(self._rotvec * alpha).as_matrix()
#        return m
