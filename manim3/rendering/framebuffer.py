#__all__ = ["Framebuffer"]


#import atexit

#import moderngl

#from ..rendering.context import ContextSingleton
#from ..rendering.temporary_resource import TemporaryResource


#_AttachmentT = moderngl.Texture | moderngl.Renderbuffer


#class Framebuffer(TemporaryResource[tuple[tuple[_AttachmentT, ...], _AttachmentT | None], moderngl.Framebuffer]):
#    def __init__(
#        self,
#        *,
#        color_attachments: tuple[_AttachmentT, ...],
#        depth_attachment: _AttachmentT | None
#    ):
#        super().__init__((color_attachments, depth_attachment))

#    @classmethod
#    def _new_instance(cls, parameters: tuple[tuple[_AttachmentT, ...], _AttachmentT | None]) -> moderngl.Framebuffer:
#        color_attachments, depth_attachment = parameters
#        framebuffer = ContextSingleton().framebuffer(
#            color_attachments=tuple(color_attachments),
#            depth_attachment=depth_attachment
#        )
#        atexit.register(lambda: framebuffer.release())
#        return framebuffer
