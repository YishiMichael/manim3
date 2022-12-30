__all__ = ["ContextSingleton"]


import moderngl


class ContextSingleton:
    _INSTANCE: moderngl.Context | None = None

    def __new__(cls):
        #raise NotImplementedError
        assert cls._INSTANCE is not None
        return cls._INSTANCE

    @classmethod
    def set(cls, ctx: moderngl.Context) -> None:
        assert cls._INSTANCE is None
        cls._INSTANCE = ctx
