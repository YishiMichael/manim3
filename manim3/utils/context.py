__all__ = ["ContextSingleton"]


import moderngl


class ContextSingleton:
    _INSTANCE: moderngl.Context | None = None

    def __new__(cls):
        assert cls._INSTANCE is not None
        return cls._INSTANCE

    @classmethod
    def set(cls, ctx: moderngl.Context) -> None:
        cls._INSTANCE = ctx
