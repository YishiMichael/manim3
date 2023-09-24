#from ....constants.custom_typing import NP_3f8
#from ..mobject import Mobject
#from .about import About


#class AboutEdge(About):
#    __slots__ = ("_edge",)

#    def __init__(
#        self,
#        edge: NP_3f8
#    ) -> None:
#        super().__init__()
#        self._edge: NP_3f8 = edge

#    def _get_about_position(
#        self,
#        mobject: Mobject
#    ) -> NP_3f8:
#        return mobject.get_bounding_box_position(self._edge)
