#__all__ = ["TemporaryResource"]


#from abc import (
#    ABC,
#    abstractmethod
#)
#from typing import (
#    Generic,
#    TypeVar
#)


#_T = TypeVar("_T")
#_ParamsT = TypeVar("_ParamsT", bound=tuple)


#class TemporaryResource(Generic[_ParamsT, _T], ABC):
#    _VACANT_INSTANCES: dict[_ParamsT, list[_T]]

#    def __init_subclass__(cls) -> None:
#        cls._VACANT_INSTANCES = {}

#    def __init__(self, parameters: _ParamsT):
#        if (vacant_instances := self._VACANT_INSTANCES.get(parameters)) is not None and vacant_instances:
#            instance = vacant_instances.pop()
#        else:
#            instance = self._new_instance(parameters)
#        self._parameters: _ParamsT = parameters
#        self._instance: _T = instance

#    def __enter__(self) -> _T:
#        return self._instance

#    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
#        self._VACANT_INSTANCES.setdefault(self._parameters, []).append(self._instance)

#    @classmethod
#    @abstractmethod
#    def _new_instance(cls, parameters: _ParamsT) -> _T:
#        pass
