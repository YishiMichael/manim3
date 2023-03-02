__all__ = [
    "LazyWrapper",
    "lazy_collection",
    "lazy_object",
    "lazy_object_shared",
    "lazy_object_unwrapped",
    "lazy_property",
    "lazy_property_shared",
    "lazy_property_unwrapped"
]


import inspect
import re
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    TypeVar
)

from bidict import bidict

from ..lazy.core import (
    LazyCollection,
    LazyCollectionDescriptor,
    LazyEntity,
    LazyObject,
    LazyObjectDescriptor,
    LazyPropertyDescriptor
)


_T = TypeVar("_T")
_HashableT = TypeVar("_HashableT", bound=Hashable)
_LazyEntityT = TypeVar("_LazyEntityT", bound="LazyEntity")
_LazyObjectT = TypeVar("_LazyObjectT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")


class LazyWrapper(Generic[_T], LazyObject):
    __slots__ = ("__value",)

    def __init__(
        self,
        value: _T
    ):
        super().__init__()
        self.__value: _T = value

    @property
    def value(self) -> _T:
        return self.__value

    #@classmethod
    #def wrap_method(
    #    cls,
    #    method: Callable[..., _T]
    #) -> "Callable[..., LazyWrapper[_T]]":
    #    def new_method(
    #        kls: type[_InstanceT],
    #        *args: Any,
    #        **kwargs: Any
    #    ) -> LazyWrapper[_T]:
    #        return cls(method(kls, *args, **kwargs))
    #    return new_method


#class LazyObjectUnwrappedDescriptor(Generic[_InstanceT, _T], LazyObjectDescriptor[_InstanceT, LazyWrapper[_T]]):
#    __slots__ = ()

#    def __init__(
#        self,
#        method: Callable[[type[_InstanceT]], _T]
#    ) -> None:
#        @wraps(method)
#        def new_method(
#            cls: type[_InstanceT]
#        ) -> LazyWrapper[_T]:
#            return LazyWrapper(method(cls))

#        super().__init__(new_method)

#    def __set__(
#        self,
#        instance: _InstanceT,
#        obj: _T | LazyWrapper[_T]
#    ) -> None:
#        if not isinstance(obj, LazyWrapper):
#            obj = LazyWrapper(obj)
#        super().__set__(instance, obj)


#class LazyObjectSharedDescriptor(LazyObjectUnwrappedDescriptor[_InstanceT, _HashableT]):
#    __slots__ = ("content_to_object_bidict",)

#    def __init__(
#        self,
#        method: Callable[[type[_InstanceT]], _HashableT]
#        #key: Callable[[_T], _KeyT]
#    ) -> None:
#        super().__init__(method)
#        #self.key: Callable[[_T], _KeyT] = key
#        self.content_to_object_bidict: bidict[_HashableT, LazyWrapper[_HashableT]] = bidict()

#    def __set__(
#        self,
#        instance: _InstanceT,
#        obj: _HashableT
#    ) -> None:
#        def restock_method(
#            cached_object: LazyWrapper[_HashableT]
#        ) -> None:
#            self.content_to_object_bidict.inverse.pop(cached_object)

#        #key = self.key(obj)
#        if (cached_object := self.content_to_object_bidict.get(obj)) is None:
#            cached_object = LazyWrapper(obj)
#            self.content_to_object_bidict[obj] = cached_object
#            cached_object._at_restock(restock_method)
#        super().__set__(instance, cached_object)


#class LazyPropertyUnwrappedDescriptor(
#    Generic[_InstanceT, _ParameterSpec, _T], LazyPropertyDescriptor[_InstanceT, _ParameterSpec, LazyWrapper[_T]]
#):
#    __slots__ = ("restock_methods",)

#    def __init__(
#        self,
#        method: Callable[Concatenate[type[_InstanceT], _ParameterSpec], _T]
#    ) -> None:
#        @wraps(method)
#        def new_method(
#            cls: type[_InstanceT],
#            *args: _ParameterSpec.args,
#            **kwargs: _ParameterSpec.kwargs
#        ) -> LazyWrapper[_T]:
#            return LazyWrapper(method(cls, *args, **kwargs))

#        super().__init__(new_method)
#        self.restock_methods: list[Callable[[_T], None]] = []

#    def restocker(
#        self,
#        restock_method: Callable[[_T], None]
#    ) -> Callable[[_T], None]:
#        self.restock_methods.append(restock_method)
#        return restock_method

#    def handle_new_property(
#        self,
#        entity: LazyWrapper[_T]
#    ) -> LazyWrapper[_T]:
#        for restock_method in self.restock_methods:
#            entity._at_restock(lambda obj: restock_method(obj.value))
#        return entity


#class LazyPropertySharedDescriptor(LazyPropertyUnwrappedDescriptor[_InstanceT, _ParameterSpec, _HashableT]):
#    __slots__ = ("content_to_object_bidict",)

#    def __init__(
#        self,
#        method: Callable[Concatenate[type[_InstanceT], _ParameterSpec], _HashableT]
#        #key: Callable[[_T], _KeyT]
#    ) -> None:
#        super().__init__(method)
#        #self.key: Callable[[_T], _KeyT] = key
#        self.content_to_object_bidict: bidict[_HashableT, LazyWrapper[_HashableT]] = bidict()

#    def handle_new_property(
#        self,
#        entity: LazyWrapper[_HashableT]
#    ) -> LazyWrapper[_HashableT]:
#        def restock_method(
#            cached_object: LazyWrapper[_HashableT]
#        ) -> None:
#            self.content_to_object_bidict.inverse.pop(cached_object)

#        super().handle_new_property(entity)
#        key = entity.value
#        if (cached_object := self.content_to_object_bidict.get(key)) is None:
#            cached_object = entity
#            self.content_to_object_bidict[key] = cached_object
#            cached_object._at_restock(restock_method)
#        else:
#            entity._restock()
#        return cached_object


#class DAGNode(ABC):
#    __slots__ = (
#        "_node_children",
#        #"_node_descendants",
#        "_node_parents",
#        #"_node_ancestors",
#        #"_expired"
#        "_restock_callbacks"
#    )

#    _VACANT_INSTANCES: "ClassVar[list[DAGNode]]"
#    _VARIABLE_DESCRS: "ClassVar[list[LazyObjectDescriptor]]"
#    #_VARIABLE_DESCR_TO_PROPERTY_DESCRS: "ClassVar[dict[LazyObjectDescriptor, tuple[LazyPropertyDescriptor, ...]]]"
#    #_PROPERTY_DESCR_TO_VARIABLE_DESCRS: "ClassVar[dict[LazyPropertyDescriptor, tuple[LazyObjectDescriptor, ...]]]"
#    #_PROPERTY_DESCR_TO_PARAMETER_DESCRS: "ClassVar[dict[LazyPropertyDescriptor, tuple[LazyObjectDescriptor | LazyPropertyDescriptor, ...]]]"
#    #_SLOT_DESCRS: ClassVar[tuple[LazySlot, ...]] = ()

#    #def __init_subclass__(cls) -> None:
#    #    descrs: dict[str, LazyObjectDescriptor | LazyPropertyDescriptor] = {
#    #        name: descr
#    #        for parent_cls in cls.__mro__[::-1]
#    #        for name, descr in parent_cls.__dict__.items()
#    #        if isinstance(descr, LazyObjectDescriptor | LazyPropertyDescriptor)
#    #    }
#    #    #slots: dict[str, LazySlot] = {}
#    #    #for parent_cls in cls.__mro__[::-1]:
#    #    #    for name, method in parent_cls.__dict__.items():
#    #    #        if name in descrs:
#    #    #            assert isinstance(method, LazyObjectDescriptor | LazyPropertyDescriptor)
#    #    #            #cls._check_annotation_matching(method.return_annotation, covered_descr.return_annotation)
#    #    #        if isinstance(method, LazyObjectDescriptor | LazyPropertyDescriptor):
#    #    #            descrs[name] = method
#    #    #        #if (covered_slot := slots.get(name)) is not None:
#    #    #        #    assert isinstance(covered_slot, LazySlot)
#    #    #        #if isinstance(method, LazySlot):
#    #    #        #    slots[name] = method

#    #    property_descr_to_parameter_descrs = {
#    #        descr: tuple(descrs[name] for name in descr.parameters)
#    #        for descr in descrs.values()
#    #        if isinstance(descr, LazyPropertyDescriptor)
#    #    }
#    #    #for descr in descrs.values():
#    #    #    if not isinstance(descr, LazyPropertyDescriptor):
#    #    #        continue
#    #    #    param_descrs: list[LazyObjectDescriptor | LazyPropertyDescriptor] = []
#    #    #    for name in descr.parameters:
#    #    #        param_descr = descrs[name]
#    #    #        #cls._check_annotation_matching(param_descr.return_annotation, param_annotation)
#    #    #        param_descrs.append(param_descr)
#    #    #    property_descr_to_parameter_descrs[descr] = tuple(param_descrs)

#    #    def traverse(property_descr: LazyPropertyDescriptor, occurred: set[LazyObjectDescriptor]) -> Generator[LazyObjectDescriptor, None, None]:
#    #        for name in property_descr.parameters:
#    #            param_descr = descrs[name]
#    #            if isinstance(param_descr, LazyObjectDescriptor):
#    #                yield param_descr
#    #                occurred.add(param_descr)
#    #            else:
#    #                yield from traverse(param_descr, occurred)

#    #    property_descr_to_variable_descrs = {
#    #        property_descr: tuple(traverse(property_descr, set()))
#    #        for property_descr in descrs.values()
#    #        if isinstance(property_descr, LazyPropertyDescriptor)
#    #    }
#    #    variable_descr_to_property_descrs = {
#    #        variable_descr: tuple(
#    #            property_descr
#    #            for property_descr, variable_descrs in property_descr_to_variable_descrs.items()
#    #            if variable_descr in variable_descrs
#    #        )
#    #        for variable_descr in descrs.values()
#    #        if isinstance(variable_descr, LazyObjectDescriptor)
#    #    }

#    #    cls._VACANT_INSTANCES = []
#    #    cls._VARIABLE_DESCR_TO_PROPERTY_DESCRS = variable_descr_to_property_descrs
#    #    cls._PROPERTY_DESCR_TO_VARIABLE_DESCRS = property_descr_to_variable_descrs
#    #    cls._PROPERTY_DESCR_TO_PARAMETER_DESCRS = property_descr_to_parameter_descrs
#    #    #cls._SLOT_DESCRS = tuple(slots.values())
#    #    return super().__init_subclass__()

#    def __new__(cls: type[Self], *args, **kwargs):
#        if (instances := cls._VACANT_INSTANCES):
#            instance = instances.pop()
#            assert isinstance(instance, cls)
#        else:
#            instance = super().__new__(cls)
#        return instance

#    def __init__(self) -> None:
#        super().__init__()
#        #self._nodes: list[LazyObjectNode] = []
#        self._node_children: list[DAGNode] = []
#        #self._node_descendants: list[LazyObject] = [self]
#        self._node_parents: list[DAGNode] = []
#        #self._node_ancestors: list[LazyObject] = [self]
#        self._restock_callbacks: list[Callable[[Self], None]] | None = []

#        #self._expired: bool = False

#    #def __delete__(self) -> None:
#    #    self._restock()

#    def _iter_descendants(self) -> "Generator[DAGNode, None, None]":
#        occurred: set[DAGNode] = set()

#        def iter_descendants(node: DAGNode) -> Generator[DAGNode, None, None]:
#            if node in occurred:
#                return
#            occurred.add(node)
#            yield node
#            for child in node._node_children:
#                yield from iter_descendants(child)

#        yield from iter_descendants(self)
#        #stack: list[LazyObject] = [self]
#        #while stack:
#        #    node = stack.pop()
#        #    if node in occurred:
#        #        continue
#        #    yield node
#        #    occurred.add(node)
#        #    stack.extend(reversed(node._node_children))

#    def _iter_ancestors(self) -> "Generator[DAGNode, None, None]":
#        occurred: set[DAGNode] = set()

#        def iter_ancestors(node: DAGNode) -> Generator[DAGNode, None, None]:
#            if node in occurred:
#                return
#            occurred.add(node)
#            yield node
#            for child in node._node_parents:
#                yield from iter_ancestors(child)

#        yield from iter_ancestors(self)
#        #stack: list[LazyObject] = [self]
#        #occurred: set[LazyObject] = set()
#        #while stack:
#        #    node = stack.pop()
#        #    if node in occurred:
#        #        continue
#        #    yield node
#        #    occurred.add(node)
#        #    stack.extend(reversed(node._node_parents))


#    #@classmethod
#    #def _check_annotation_matching(cls, child_annotation: _Annotation, parent_annotation: _Annotation) -> None:
#    #    error_message = f"Type annotation mismatched: `{child_annotation}` is not compatible with `{parent_annotation}`"
#    #    if isinstance(child_annotation, TypeVar) or isinstance(parent_annotation, TypeVar):
#    #        if isinstance(child_annotation, TypeVar) and isinstance(parent_annotation, TypeVar):
#    #            assert child_annotation == parent_annotation, error_message
#    #        return

#    #    def to_classes(annotation: _Annotation) -> tuple[type, ...]:
#    #        return tuple(
#    #            child.__origin__ if isinstance(child, GenericAlias) else
#    #            Callable if isinstance(child, Callable) else child
#    #            for child in (
#    #                annotation.__args__ if isinstance(annotation, UnionType) else (annotation,)
#    #            )
#    #        )

#    #    assert all(
#    #        any(
#    #            issubclass(child_cls, parent_cls)
#    #            for parent_cls in to_classes(parent_annotation)
#    #        )
#    #        for child_cls in to_classes(child_annotation)
#    #    ), error_message

#    def _bind_children(self, *nodes: "DAGNode"):
#        if (invalid_nodes := [
#            node for node in self._iter_ancestors()
#            if node in nodes
#        ]):
#            raise ValueError(f"Nodes `{invalid_nodes}` have already included `{self}`")
#        self._node_children.extend(nodes)
#        #for ancestor in self._node_ancestors:
#        #    ancestor._node_descendants.update(nodes)
#        for node in nodes:
#            node._node_parents.append(self)
#            #for descendant in self._node_descendants:
#            #    descendant._node_ancestors.append(self)
#        return self

#    def _unbind_children(self, *nodes: "DAGNode"):
#        if (invalid_nodes := [
#            node for node in nodes
#            if node not in self._node_children
#        ]):
#            raise ValueError(f"Nodes `{invalid_nodes}` are not children of `{self}`")
#        #self._node_children.difference_update(nodes)
#        #for ancestor in self._node_ancestors:
#        #    ancestor._node_descendants.difference_update(nodes)
#        for node in nodes:
#            self._node_children.remove(node)
#            node._node_parents.remove(self)
#            if not node._node_parents:
#                node._restock()
#            #for descendant in self._node_descendants:
#            #    descendant._node_ancestors.remove(self)
#        return self

#    #def _copy(self):
#    #    cls = self.__class__
#    #    result = cls.__new__(cls)
#    #    for variable_descr in self._VARIABLE_DESCR_TO_PROPERTY_DESCRS:
#    #        variable = variable_descr.instance_to_object_dict.get(self)
#    #        variable_descr.__set__(result, variable)
#    #    for slot in cls.__slots__:
#    #        result.__setattr__(slot, self.__getattribute__(slot))
#    #    #for slot_descr in self._SLOT_DESCRS:
#    #    #    slot_descr._copy_value(self, result)
#    #    return result

#    def _restock(self) -> None:
#        #for variable_descr in self._VARIABLE_DESCR_TO_PROPERTY_DESCRS:
#        #    variable_descr.__set__(self, None)
#        #for slot_descr in self._SLOT_DESCRS:
#        #    slot_descr.instance_to_value_dict.pop(self, None)
#        for node in self._iter_descendants():
#            if (callbacks := node._restock_callbacks) is None:
#                continue
#            for callback in callbacks:
#                callback(node)
#            callbacks.clear()
#            node.__class__._VACANT_INSTANCES.append(node)

#    def _at_restock(self, callback: Callable[[Self], None]) -> None:
#        if (callbacks := self._restock_callbacks) is not None:
#            callbacks.append(callback)


#class LazyObject(DAGNode):
#    _LAZY_DESCRIPTORS: "ClassVar[list[LazyDescriptor[DAGNode, LazyObject]]]"
#    #_LAZY_COLLECTION_DESCRIPTORS: "ClassVar[list[LazyCollectionDescriptor]]"
#    #_LAZY_PROPERTY_DESCRIPTORS: "ClassVar[list[LazyPropertyDescriptor]]"
#    #_LAZY_DATACLASS: ClassVar[type]

#    def __init_subclass__(cls) -> None:
#        attrs: dict[str, Any] = {
#            name: attr
#            for parent_cls in reversed(cls.__mro__)
#            for name, attr in parent_cls.__dict__.items()
#        }
#        descrs: dict[str, LazyDescriptor] = {
#            name: attr
#            for name, attr in attrs.items()
#            if isinstance(attr, LazyDescriptor)
#        }
#        cls._LAZY_DESCRIPTORS = list(descrs.values())

#        for descr in descrs.values():
#            if not isinstance(descr, LazyPropertyDescriptor):
#                continue
#            descr._setup_callables(descrs)

#        #cls._LAZY_OBJECT_DESCRIPTORS = [
#        #    descr for descr in descrs.values()
#        #    if isinstance(descr, LazyObjectDescriptor)
#        #]
#        #cls._LAZY_COLLECTION_DESCRIPTORS = [
#        #    descr for descr in descrs.values()
#        #    if isinstance(descr, LazyCollectionDescriptor)
#        #]
#        #cls._LAZY_PROPERTY_DESCRIPTORS = [
#        #    descr for descr in descrs.values()
#        #    if isinstance(descr, LazyPropertyDescriptor)
#        #]
#        #for name, descr in descrs.items():
#        #    descr.name = name
#        #cls._LAZY_DATACLASS = make_dataclass(
#        #    f"_{cls.__name__}__dataclass",
#        #    [
#        #        (name, Any, descr.get_field())
#        #        for name, descr in descrs.items()
#        #    ],
#        #    order=True,
#        #    kw_only=True,
#        #    slots=True
#        #)

#    def __init__(self) -> None:
#        super().__init__()
#        #self._lazy_data: Any = self._LAZY_DATACLASS()

#        children: list[DAGNode] = []
#        for descr in self.__class__._LAZY_DESCRIPTORS:
#            if isinstance(descr, LazyObjectDescriptor):
#                if (default_object := descr._default_object) is None:
#                    default_object = descr.method()
#                    default_object._restock_callbacks = None  # Never restock
#                    descr._default_object = default_object
#                descr.initialize(self, default_object)
#                children.append(default_object)
#            elif isinstance(descr, LazyCollectionDescriptor):
#                default_collection = descr.method()
#                descr.initialize(self, default_collection)
#                children.append(default_collection)

#        self._bind_children(*children)

    #@overload
    #def _descr_get(
    #    self,
    #    descr: "LazyObjectDescriptor[_LazyObjectT, _InstanceT]"
    #) -> _LazyObjectT: ...

    #@overload
    #def _descr_get(
    #    self,
    #    descr: "LazyCollectionDescriptor[_DAGNodeT, _InstanceT]"
    #) -> "LazyCollection[_DAGNodeT]": ...

    #@overload
    #def _descr_get(
    #    self,
    #    descr: "LazyPropertyDescriptor[_DAGNodeT, _InstanceT]"
    #) -> "LazyPropertyRecord[_DAGNodeT]": ...

    #def _descr_get(
    #    self,
    #    descr: """Union[
    #        LazyObjectDescriptor[_LazyObjectT, _InstanceT],
    #        LazyCollectionDescriptor[_DAGNodeT, _InstanceT],
    #        LazyPropertyDescriptor[_DAGNodeT, _InstanceT]
    #    ]"""
    #) -> """Union[
    #    _LazyObjectT,
    #    LazyCollection[_DAGNodeT],
    #    LazyPropertyRecord[_DAGNodeT]
    #]""":
    #    return self._lazy_data.__getattribute__(descr.name)

    ##@overload
    #def set_object(
    #    self,
    #    descr: "LazyObjectDescriptor[_LazyObjectT, _InstanceT]",
    #    value: _LazyObjectT
    #) -> None:
    #    self._lazy_data.__setattr__(descr.name, value)

    #@overload
    #def set_object(
    #    self,
    #    descr: "LazyCollectionDescriptor[_DAGNodeT, _InstanceT]",
    #    value: "LazyCollection[_DAGNodeT]"
    #) -> None: ...

    #@overload
    #def set_object(
    #    self,
    #    descr: "LazyPropertyDescriptor[_DAGNodeT, _InstanceT]",
    #    value: "LazyPropertyRecord[_DAGNodeT]"
    #) -> None: ...

    #def set_object(
    #    self,
    #    descr: """Union[
    #        LazyObjectDescriptor[_LazyObjectT, _InstanceT],
    #        LazyPropertyDescriptor[_DAGNodeT, _InstanceT]
    #    ]""",
    #    value: """Union[
    #        _LazyObjectT,
    #        LazyPropertyRecord[_DAGNodeT]
    #    ]"""
    #) -> None:
    #    return self._lazy_data.__setattr__(descr.name, value)


#class LazyCollection(Generic[_DAGNodeT], DAGNode):
#    __slots__ = ()
#
#    def __len__(self) -> int:
#        return self._node_children.__len__()
#
#    @overload
#    def __getitem__(self, index: slice) -> list[_DAGNodeT]:
#        ...
#
#    @overload
#    def __getitem__(self, index: int) -> _DAGNodeT:
#        ...
#
#    def __getitem__(self, index: slice | int) -> list[_DAGNodeT] | _DAGNodeT:
#        return self._node_children.__getitem__(index)
#
#    def add(self, *nodes: _DAGNodeT):
#        self._bind_children(*nodes)
#        return self
#
#    def remove(self, *nodes: _DAGNodeT):
#        self._unbind_children(*nodes)
#        return self


#class LazyPropertyRecord(Generic[_DAGNodeT]):
#    #__slots__ = ()
#    __slots__ = ("_slot",)
#
#    def __init__(self) -> None:
#        self._slot: _DAGNodeT | None = None

    #def get(self) -> _DAGNodeT | None:
    #    if not self._node_children:
    #        return None
    #    return self._node_children[0]

    #def set(self, node: _DAGNodeT) -> None:
    #    if self.get() is node:
    #        return
    #    if self._node_children:
    #        self._unbind_children(self._node_children[0])
    #    self._bind_children(node)

    #def bind(self, *nodes: DAGNode) -> None:
    #    self._bind_children(*nodes)

    #def expire(self) -> None:
    #    self._unbind_children(*self._node_children)

    #def __init__(self) -> None:
    #    self._node: _DAGNodeT | None = None
    #    #self._expired: bool = True

#    __slots__ = ("__storage",)

#    def __init__(self, storage: list[_LazyObjectT] | None = None):
#        if storage is None:
#            storage = []
#        self.__storage: list[_LazyObjectT] = storage

#    def __iter__(self) -> "Iterator[_LazyObjectT]":
#        return iter(self.__storage)

#    @overload
#    def __getitem__(self, index: int) -> "_LazyObjectT": ...

#    @overload
#    def __getitem__(self, index: slice) -> "list[_LazyObjectT]": ...

#    def __getitem__(self, index: int | slice) -> "_LazyObjectT | list[_LazyObjectT]":
#        return self.__storage.__getitem__(index)

#    def add(self, *lazy_objects: _LazyObjectT) -> "LazyCollection[_LazyObjectT]":
#        storage = self.__storage[:]
#        storage.extend(lazy_objects)
#        return LazyCollection(storage)

#    def remove(self, *lazy_objects: _LazyObjectT) -> "LazyCollection[_LazyObjectT]":
#        storage = self.__storage[:]
#        for lazy_object in lazy_objects:
#            storage.remove(lazy_object)
#        return LazyCollection(storage)


#class LazyDescriptor(Generic[_LazyObjectT]):
#    def __init__(self, method: Callable[..., _LazyObjectT]):
#        #self.name: str = method.__name__
#        self.method: Callable[..., _LazyObjectT] = method
#        #self.signature: inspect.Signature = inspect.signature(method)
#        #self.restock_method: Callable[[_T], None] | None = None

    #@property
    #def parameters(self) -> dict[str, _Annotation]:
    #    return {
    #        f"_{parameter.name}_": parameter.annotation
    #        for parameter in list(self.signature.parameters.values())
    #    }

    #@property
    #def return_annotation(self) -> _Annotation:
    #    return self.signature.return_annotation

    #def _restock(self, data: _T) -> None:
    #    if self.restock_method is not None:
    #        self.restock_method(data)
    #    elif isinstance(data, LazyBase):
    #        data._restock()

    #def restocker(self, restock_method: Callable[[_T], None]) -> Callable[[_T], None]:
    #    self.restock_method = restock_method
    #    return restock_method


#class LazyDescriptor(Generic[_DAGNodeT, _InstanceT]):
#    def __init__(self, name: str) -> None:
#        self.name: str = name
#        #self.default_factory: Callable[..., _DAGNodeT] = default_factory
#        #self.name: str = init_method.__name__
#        self.instance_to_node_dict: dict[_InstanceT, _DAGNodeT] = {}

#    @overload
#    def __get__(
#        self,
#        instance: None,
#        owner: type[_InstanceT] | None = None
#    ): ...  # TODO: typing

#    @overload
#    def __get__(
#        self,
#        instance: _InstanceT,
#        owner: type[_InstanceT] | None = None
#    ) -> _DAGNodeT: ...

#    def __get__(
#        self,
#        instance: _InstanceT | None,
#        owner: type[_InstanceT] | None = None
#    )  | _DAGNodeT:
#        if instance is None:
#            return self
#        if (node := self.get(instance)) is None:
#            node = self.missing(instance)
#            self.instance_to_node_dict[instance] = node
#        return node

#    def __set__(
#        self,
#        instance: _InstanceT,
#        value: _DAGNodeT
#    ) -> None:
#        self.instance_to_node_dict[instance] = value

#    def initialize(
#        self,
#        instance: _InstanceT,
#        value: _DAGNodeT
#    ) -> None:
#        assert instance not in self.instance_to_node_dict
#        self.instance_to_node_dict[instance] = value

#    def pop(
#        self,
#        instance: _InstanceT
#    ) -> _DAGNodeT:
#        return self.instance_to_node_dict.pop(instance)

#    def get(
#        self,
#        instance: _InstanceT
#    ) -> _DAGNodeT | None:
#        return self.instance_to_node_dict.get(instance)

#    def missing(
#        self,
#        instance: _InstanceT
#    ) -> _DAGNodeT:
#        raise KeyError

#    #def get_field(self) -> Any:
#    #    pass


#class LazyObjectDescriptor(LazyDescriptor[_LazyObjectT, _InstanceT]):
#    def __init__(
#        self,
#        method: Callable[[], _LazyObjectT]
#    ) -> None:
#        super().__init__(method.__name__)
#        self.method: Callable[[], _LazyObjectT] = method
#        self._default_object: _LazyObjectT | None = None

        #def default_factory() -> _LazyObjectT:
        #    if (default_object := self._default_object) is None:
        #        default_object = method()
        #        default_object._restock_callbacks = None  # Never restock
        #    self._default_object = default_object
        #    return default_object

        #super().__init__(method)
        #assert not self.parameters
        #self.method: Callable[[], _LazyObjectT] = method
        #self.object_type: type[_LazyObjectT] = _get_type_from_annotation(
        #    inspect.signature(method).return_annotation
        #)
        #self.name: str = method.__name__
        #self.instance_to_object_dict: dict[_InstanceT, _LazyObjectT] = {}
        #self.variable_to_instances_dict: dict[_LazyObjectT, list[_InstanceT]] = {}

    #@overload
    #def __get__(
    #    self,
    #    instance: None,
    #    owner: type[_InstanceT] | None = None
    #): ...  # TODO: typing

    #@overload
    #def __get__(
    #    self,
    #    instance: _InstanceT,
    #    owner: type[_InstanceT] | None = None
    #) -> _LazyObjectT: ...

    #def __get__(
    #    self,
    #    instance: _InstanceT | None,
    #    owner: type[_InstanceT] | None = None
    #)  | _LazyObjectT:
    #    if instance is None:
    #        return self
    #    return instance._descr_get(self)
    #    #if (variable := self.instance_to_object_dict.get(instance)) is None:
    #    #    variable = self._get_initial_value()
    #    #    self.instance_to_object_dict[instance] = variable
    #    #return variable
    #    #return self.instance_to_object_dict.get(instance, self.default_variable)

    #def __set__(
    #    self,
    #    instance: _InstanceT,
    #    lazy_object: _LazyObjectT
    #) -> None:
    #    #assert isinstance(variable, LazyWrapper)
    #    #self._set_data(instance, variable)
    #    old_object = self.__get__(instance)
    #    instance._unbind_children(old_object)
    #    for descr in instance.__class__._LAZY_DESCRIPTORS:
    #        if not isinstance(descr, LazyPropertyDescriptor):
    #            continue
    #        if (expired_node := descr.get(instance)) is None:
    #            continue
    #        #record: LazyPropertyRecord = descr.__get__(instance)
    #        expired_node._unbind_children(*expired_node._node_children)
    #    #instance.set_object(self, lazy_object)
    #    super().__set__(instance, lazy_object)
    #    #self.instance_to_node_dict[instance] = lazy_object
    #    instance._bind_children(lazy_object)

        #if self.instance_to_object_dict.get(instance) is variable:
        #    return
        #self._clear_instance_variable(instance)
        #for property_descr in instance.__class__._VARIABLE_DESCR_TO_PROPERTY_DESCRS[self]:
        #    property_descr._clear_instance_variable_tuple(instance)
        #if variable is None:
        #    return
        #self.instance_to_object_dict[instance] = variable
        #self.variable_to_instances_dict.setdefault(variable, []).append(instance)

    #def get_field(self) -> Any:
    #    def factory() -> _LazyObjectT:
    #        if (default_object := self._default_object) is None:
    #            default_object = self.method()
    #            default_object._restock_callbacks = None  # Never restock
    #        self._default_object = default_object
    #        return default_object
    #    return field(default_factory=factory)

    #def _get_default_object(self) -> _LazyObjectT:
    #    if (default_object := self._default_object) is None:
    #        default_object = self.method()
    #        self._default_object = default_object
    #    return default_object

    #@property
    #def default_variable(self) -> _LazyObjectT:
    #    if self._default_object is None:
    #        self._default_object = self.method()
    #    return self._default_object

    #def _get_data(self, instance: _InstanceT) -> LazyWrapper[_T]:
    #    return self.instance_to_object_dict.get(instance, self.default_variable)

    #def _set_data(self, instance: _InstanceT, variable: LazyWrapper[_T] | None) -> None:
    #    if self.instance_to_object_dict.get(instance) is variable:
    #        return
    #    self._clear_instance_variable(instance)
    #    for property_descr in instance.__class__._VARIABLE_DESCR_TO_PROPERTY_DESCRS[self]:
    #        property_descr._clear_instance_variable_tuple(instance)
    #    if variable is None:
    #        return
    #    self.instance_to_object_dict[instance] = variable
    #    self.variable_to_instances_dict.setdefault(variable, []).append(instance)

    #def _clear_instance_variable(self, instance: _InstanceT) -> None:
    #    if (variable := self.instance_to_object_dict.pop(instance, None)) is None:
    #        return
    #    self.variable_to_instances_dict[variable].remove(instance)
    #    if self.variable_to_instances_dict[variable]:
    #        return
    #    self.variable_to_instances_dict.pop(variable)
    #    variable._restock()  # TODO
    #    #self._restock(variable.data)  # TODO


#class LazyCollectionDescriptor(Generic[_DAGNodeT, _InstanceT], LazyDescriptor[LazyCollection[_DAGNodeT], _InstanceT]):
#    def __init__(
#        self,
#        method: Callable[[], LazyCollection[_DAGNodeT]]
#    ) -> None:
#        super().__init__(method.__name__)
#        self.method: Callable[[], LazyCollection[_DAGNodeT]] = method
        #self.object_type: type[_DAGNodeT] = _get_type_from_annotation(
        #    inspect.signature(method).return_annotation.__args__[0]
        #)
        #self.instance_to_collection_dict: dict[_InstanceT, LazyCollection[_DAGNodeT]] = {}

    #@overload
    #def __get__(
    #    self,
    #    instance: None,
    #    owner: type[_InstanceT] | None = None
    #): ...  # TODO: typing

    #@overload
    #def __get__(
    #    self,
    #    instance: _InstanceT,
    #    owner: type[_InstanceT] | None = None
    #) -> LazyCollection[_DAGNodeT]: ...

    #def __get__(
    #    self,
    #    instance: _InstanceT | None,
    #    owner: type[_InstanceT] | None = None
    #)  | LazyCollection[_DAGNodeT]:
    #    if instance is None:
    #        return self
    #    return instance._descr_get(self)

    #def __set__(
    #    self,
    #    instance: _InstanceT,
    #    value: Never
    #) -> None:
    #    raise RuntimeError("Attempting to set a collection object directly")

    #def get_field(self) -> Any:
    #    def factory() -> LazyCollection[_DAGNodeT]:
    #        return self.method()
    #    return field(default_factory=factory)


#class LazyPropertyDescriptor(LazyDescriptor[_DAGNodeT, _InstanceT]):
    #def __init__(
    #    self,
    #    method: Callable[..., _DAGNodeT]
    #) -> None:
    #    #def default_factory() -> LazyPropertyRecord[_DAGNodeT]:
    #    #    return LazyPropertyRecord()

    #    super().__init__(method.__name__)
    #    #super().__init__(method)
    #    #assert self.parameters

    #    #parameter_items = [
    #    #    (name, False) if re.fullmatch(r"_\w+_", name := parameter.name) else (f"_{name}_", True)
    #    #    for parameter in inspect.signature(method).parameters.values()
    #    #]
    #    #parameter_tuple = tuple(parameter for parameter, _ in parameter_items)
    #    #is_lazy_value_tuple = tuple(is_lazy_value for _, is_lazy_value in parameter_items)

    #    #def new_method(*args) -> _DAGNodeT:
    #    #    return method(*(
    #    #        arg.value if is_lazy_value else arg
    #    #        for arg, is_lazy_value in zip(args, is_lazy_value_tuple, strict=True)
    #    #    ))

    #    #self.method: Callable[..., _DAGNodeT] = method
    #    #self.object_type: type[_DAGNodeT] = _get_type_from_annotation(
    #    #    inspect.signature(method).return_annotation
    #    #)
    #    self.method: Callable[..., _DAGNodeT] = method
    #    self.get_parameters_from_instance: Callable[[_InstanceT], tuple] = NotImplemented
    #    self.get_property_from_parameters: Callable[[tuple], _DAGNodeT] = NotImplemented
    #    #self.parameters: tuple[str, ...] = parameter_tuple
    #    self.parameters_property_bidict: bidict[tuple, _DAGNodeT] = bidict()
    #    #self.property_to_parameters_dict: dict[_DAGNodeT, tuple] = {}
    #    #self.instance_to_property_record_dict: dict[_InstanceT, LazyPropertyRecord[_DAGNodeT]] = {}
    #    #self.instance_to_variable_tuple_dict: dict[_InstanceT, tuple[_LazyObjectT, ...]] = {}
    #    #self.variable_tuple_to_instances_dict: dict[tuple[_LazyObjectT, ...], list[_InstanceT]] = {}
    #    #self.variable_tuple_to_property_dict: dict[tuple[_LazyObjectT, ...], _LazyObjectT] = {}

    #@overload
    #def __get__(
    #    self,
    #    instance: None,
    #    owner: type[_InstanceT] | None = None
    #): ...

    #@overload
    #def __get__(
    #    self,
    #    instance: _InstanceT,
    #    owner: type[_InstanceT] | None = None
    #) -> _DAGNodeT: ...

    #def __get__(
    #    self,
    #    instance: _InstanceT | None,
    #    owner: type[_InstanceT] | None = None
    #)  | _DAGNodeT:
    #    if instance is None:
    #        return self

    #    def flatten_deepest(obj: tuple | DAGNode) -> Generator[DAGNode, None, None]:
    #        if not isinstance(obj, tuple):
    #            yield obj
    #        else:
    #            for child_obj in obj:
    #                yield from flatten_deepest(child_obj)

    #    def restock_method(prop: _DAGNodeT) -> None:
    #        #parameters = self.property_to_parameters_dict.pop(prop)
    #        self.parameters_property_bidict.inverse.pop(prop)

    #    record = super().__get__(instance)
    #    #record = self.instance_to_property_record_dict[instance]
    #    if (prop := record._slot) is None:
    #        parameters = self.get_parameters_from_instance(instance)
    #        if (prop := self.parameters_property_bidict.get(parameters)) is None:
    #            record.bind(*flatten_deepest(parameters))
    #            prop = self.get_property_from_parameters(parameters)
    #            self.parameters_property_bidict[parameters] = prop
    #            #self.property_to_parameters_dict[prop] = parameters
    #            prop._at_restock(restock_method)

    #        record._slot = prop
    #        #record._expired = False
    #    #if (prop := self.instance_to_property_record_dict.get(instance)) is None:
    #    #    prop = self.instance_method(instance)
    #    #    self.instance_to_property_dict[instance] = prop
    #    
    #    return prop
    #    #if (variable_tuple := self.instance_to_variable_tuple_dict.get(instance)) is None:
    #    #    variable_tuple = tuple(
    #    #        variable_descr.__get__(instance)
    #    #        for variable_descr in instance.__class__._PROPERTY_DESCR_TO_VARIABLE_DESCRS[self]
    #    #    )
    #    #    self.instance_to_variable_tuple_dict[instance] = variable_tuple
    #    #self.variable_tuple_to_instances_dict.setdefault(variable_tuple, []).append(instance)
    #    #if (result := self.variable_tuple_to_property_dict.get(variable_tuple)) is None:
    #    #    result = self.method(*(
    #    #        param_descr.__get__(instance)
    #    #        for param_descr in instance.__class__._PROPERTY_DESCR_TO_PARAMETER_DESCRS[self]
    #    #    ))
    #    #    self.variable_tuple_to_property_dict[variable_tuple] = result
    #    #return result

    #def missing(
    #    self,
    #    instance: _InstanceT
    #) -> _DAGNodeT:
    #    def flatten_deepest(obj: tuple | DAGNode) -> Generator[DAGNode, None, None]:
    #        if not isinstance(obj, tuple):
    #            yield obj
    #        else:
    #            for child_obj in obj:
    #                yield from flatten_deepest(child_obj)

    #    def restock_method(prop: _DAGNodeT) -> None:
    #        #parameters = self.property_to_parameters_dict.pop(prop)
    #        self.parameters_property_bidict.inverse.pop(prop)

    #    parameters = self.get_parameters_from_instance(instance)
    #    if (prop := self.parameters_property_bidict.get(parameters)) is None:
    #        prop = self.get_property_from_parameters(parameters)
    #        prop._bind_children(*flatten_deepest(parameters))
    #        self.parameters_property_bidict[parameters] = prop
    #        #self.property_to_parameters_dict[prop] = parameters
    #        prop._at_restock(restock_method)
    #    return prop

    #def __set__(
    #    self,
    #    instance: _InstanceT,
    #    value: Never
    #) -> None:
    #    raise RuntimeError("Attempting to set a readonly lazy property")

    #def _setup_callables(self, descrs: dict[str, LazyDescriptor]) -> None:
    #    #parameter_names = list(inspect.signature(self.method).parameters)
    #    parameter_items: list[tuple[tuple[LazyDescriptor, ...], bool]] = [
    #        (tuple(
    #            descrs[name] for name in re.findall(r"_\w+?_(?=_|$)", name)
    #        ), is_lazy_value)
    #        for name, is_lazy_value in (
    #            (name, False) if re.fullmatch(r"_\w+_", name) else (f"_{name}_", True)
    #            for name in inspect.signature(self.method).parameters
    #        )
    #    ]

    #    def apply_deepest(
    #        callback: Callable[[DAGNode], DAGNode],
    #        obj: tuple | DAGNode
    #    ) -> tuple | DAGNode:
    #        if not isinstance(obj, tuple):
    #            return callback(obj)
    #        return tuple(
    #            apply_deepest(callback, child_obj)
    #            for child_obj in obj
    #        )

    #    def construct_obj_from_descr_chain(
    #        descr_chain: tuple[LazyDescriptor, ...],
    #        obj: tuple | DAGNode
    #    ) -> tuple | DAGNode:
    #        for descr in descr_chain:
    #            if isinstance(descr, LazyObjectDescriptor | LazyPropertyDescriptor):
    #                obj = apply_deepest(lambda instance: descr.__get__(instance), obj)
    #            elif isinstance(descr, LazyCollectionDescriptor):
    #                obj = apply_deepest(lambda instance: tuple(descr.__get__(instance)._node_children), obj)
    #            else:
    #                raise TypeError
    #        return obj
    #        #if not descr_chain:
    #        #    return obj
    #        #descr = descr_chain[0]
    #        #rest_chain = descr_chain[1:]
    #        #if isinstance(descr, LazyObjectDescriptor | LazyPropertyDescriptor):
    #        #    return construct_obj_from_descr_chain(
    #        #        rest_chain,
    #        #        apply_deepest(lambda instance: descr.__get__(instance), obj)
    #        #    )
    #        #if isinstance(descr, LazyCollectionDescriptor):
    #        #    return construct_obj_from_descr_chain(
    #        #        rest_chain,
    #        #        apply_deepest(lambda instance: tuple(descr.__get__(instance)._node_children), obj)
    #        #    )
    #        #raise TypeError

    #    def get_parameters_from_instance(instance: _InstanceT) -> tuple:
    #        return tuple(
    #            construct_obj_from_descr_chain(
    #                descr_chain, instance
    #            )
    #            for descr_chain, _ in parameter_items
    #        )

    #    def get_property_from_parameters(parameters: tuple) -> _DAGNodeT:
    #        return self.method(*(
    #            apply_deepest(lambda lazy_value: lazy_value.value, parameter) if is_lazy_value else parameter
    #            for parameter, (_, is_lazy_value) in zip(parameters, parameter_items, strict=True)
    #        ))

    #    self.get_parameters_from_instance = get_parameters_from_instance
    #    self.get_property_from_parameters = get_property_from_parameters


    #def _clear_instance_variable_tuple(self, instance: _InstanceT) -> None:
    #    if (variable_tuple := self.instance_to_variable_tuple_dict.pop(instance, None)) is None:
    #        return
    #    self.variable_tuple_to_instances_dict[variable_tuple].remove(instance)
    #    if self.variable_tuple_to_instances_dict[variable_tuple]:
    #        return
    #    self.variable_tuple_to_instances_dict.pop(variable_tuple)
    #    if (property_ := self.variable_tuple_to_property_dict.pop(variable_tuple, None)) is None:
    #        return
    #    property_._restock()
    #    #self._restock(property_)

    #def get_field(self) -> Any:
    #    def factory() -> LazyPropertyRecord[_DAGNodeT]:
    #        return LazyPropertyRecord()
    #    return field(default_factory=factory)


#def _get_type_from_annotation(
#    annotation: Any
#) -> type:
#    return annotation.__origin__ if isinstance(annotation, GenericAlias) else \
#        type(None) if not isinstance(annotation, type) else annotation


#class _SelfPlaceholder(LazyObject):
#    __slots__ = ()


class lazy_object(LazyObjectDescriptor[_InstanceT, _LazyObjectT]):
    __slots__ = ()

    def __init__(
        self,
        cls_method: Callable[[type[_InstanceT]], _LazyObjectT]
    ):
        #method = cls_method.__func__
        #return_annotation = inspect.signature(method).return_annotation

        #object_type: type[_LazyObjectT] | None = None
        #if isinstance(return_annotation, str):
        #    object_type = _SelfPlaceholder
        #elif isinstance(return_annotation, type) and not isinstance(return_annotation, GenericAlias):
        #    if issubclass(return_annotation, LazyObject):
        #        object_type = return_annotation
        #if object_type is None:
        #    raise TypeError
        super().__init__(
            #object_type=object_type,
            method=cls_method.__func__
        )


class lazy_object_unwrapped(Generic[_InstanceT, _T], LazyObjectDescriptor[_InstanceT, LazyWrapper[_T]]):
    __slots__ = ()

    def __init__(
        self,
        cls_method: Callable[[type[_InstanceT]], _T]
    ):
        method = cls_method.__func__

        def new_method(
            cls: type[_InstanceT]
        ) -> LazyWrapper[_T]:
            return LazyWrapper(method(cls))

        super().__init__(
            #object_type=LazyWrapper,
            method=new_method
        )

    def __set__(
        self,
        instance: _InstanceT,
        obj: _T | LazyWrapper[_T]
    ) -> None:
        if not isinstance(obj, LazyWrapper):
            obj = LazyWrapper(obj)
        super().__set__(instance, obj)

    #def __new__(
    #    cls,
    #    method: Union[
    #        Callable[[type[_InstanceT]], _LazyObjectT],
    #        Callable[[type[_InstanceT]], LazyCollection[_LazyEntityT]],
    #        Callable[[type[_InstanceT]], _T]
    #    ]
    #) -> Union[
    #    LazyObjectDescriptor[_InstanceT, _LazyObjectT],
    #    LazyCollectionDescriptor[_InstanceT, _LazyEntityT],
    #    LazyObjectUnwrappedDescriptor[_InstanceT, _T]
    #]:
    #    method = method.__func__
    #    return_type = _get_type_from_annotation(
    #        inspect.signature(method).return_annotation
    #    )
    #    if issubclass(return_type, LazyObject):
    #        return LazyObjectDescriptor(method)
    #    if issubclass(return_type, LazyCollection):
    #        return LazyCollectionDescriptor(method)
    #    return LazyObjectUnwrappedDescriptor(method)


class lazy_object_shared(Generic[_InstanceT, _HashableT], LazyObjectDescriptor[_InstanceT, LazyWrapper[_HashableT]]):
    __slots__ = ("content_to_object_bidict",)

    def __init__(
        self,
        cls_method: Callable[[type[_InstanceT]], _HashableT]
        #key: Callable[[_T], _KeyT]
    ) -> None:
        self.content_to_object_bidict: bidict[_HashableT, LazyWrapper[_HashableT]] = bidict()

        method = cls_method.__func__

        def new_method(
            cls: type[_InstanceT]
        ) -> LazyWrapper[_HashableT]:
            return LazyWrapper(method(cls))

        super().__init__(
            #object_type=LazyWrapper,
            method=new_method
        )
        #self.key: Callable[[_T], _KeyT] = key

    def __set__(
        self,
        instance: _InstanceT,
        obj: _HashableT
    ) -> None:
        def cleanup_method(
            cached_object: LazyWrapper[_HashableT]
        ) -> None:
            self.content_to_object_bidict.inverse.pop(cached_object)

        #key = self.key(obj)
        if (cached_object := self.content_to_object_bidict.get(obj)) is None:
            cached_object = LazyWrapper(obj)
            self.content_to_object_bidict[obj] = cached_object
            cached_object._at_restock(cleanup_method)
        super().__set__(instance, cached_object)


class lazy_collection(LazyCollectionDescriptor[_InstanceT, _LazyObjectT]):
    __slots__ = ()

    def __init__(
        self,
        cls_method: Callable[[type[_InstanceT]], LazyCollection[_LazyObjectT]]
    ):
        #method = cls_method.__func__
        #return_annotation = inspect.signature(method).return_annotation

        #object_type: type[_LazyObjectT] | None = None
        #if isinstance(return_annotation, str):
        #    object_type = _SelfPlaceholder
        #elif return_annotation.__origin__ is LazyCollection:
        #    object_type = return_annotation.__args__[0]
        #if object_type is None:
        #    raise TypeError
        super().__init__(
            #object_type=object_type,
            method=cls_method.__func__
        )


class lazy_property(LazyPropertyDescriptor[_InstanceT, _LazyEntityT]):
    __slots__ = ()

    def __init__(
        self,
        cls_method: Callable[..., _LazyEntityT]
    ):
        #method = cls_method.__func__
        #return_annotation = inspect.signature(method).return_annotation

        #object_type: type[_LazyEntityT] | None = None
        #is_collection: bool = False
        #if isinstance(return_annotation, type) and not isinstance(return_annotation, GenericAlias):
        #    if issubclass(return_annotation, LazyObject):
        #        object_type = return_annotation
        #elif hasattr(return_annotation, "__origin__"):
        #    is_collection = True
        #    if return_annotation.__origin__ is LazyCollection:
        #        object_type = return_annotation.__args__[0]
        #if object_type is None:
        #    raise TypeError

        new_method, parameter_chains = self.wrap_parameters(cls_method.__func__)
        super().__init__(
            #object_type=object_type,
            #is_collection=is_collection,
            method=new_method,
            parameter_chains=parameter_chains
        )

    @classmethod
    def wrap_parameters(
        cls,
        method: Callable[..., _LazyEntityT]
    ) -> tuple[Callable[..., _LazyEntityT], tuple[tuple[str, ...], ...]]:
        parameter_items = tuple(
            (name, False) if re.fullmatch(r"_\w+_", name) else (f"_{name}_", True)
            for name in tuple(inspect.signature(method).parameters)[1:]  # remove `cls`
        )
        parameter_chains = tuple(
            tuple(re.findall(r"_\w+?_(?=_|$)", parameter_name))
            for parameter_name, _ in parameter_items
        )
        assert all(
            "".join(parameter_chain) == parameter_name
            for parameter_chain, (parameter_name, _) in zip(parameter_chains, parameter_items, strict=True)
        )
        requires_unwrapping_tuple = tuple(requires_unwrapping for _, requires_unwrapping in parameter_items)

        def parameters_wrapped_method(
            kls: type[_InstanceT],
            *args: Any,
            **kwargs: Any
        ) -> _LazyEntityT:
            return method(kls, *(
                arg if not requires_unwrapping else cls.apply_deepest(lambda obj: obj.value, arg)
                for arg, requires_unwrapping in zip(args, requires_unwrapping_tuple, strict=True)
            ), **kwargs)
        return parameters_wrapped_method, parameter_chains


class lazy_property_unwrapped(LazyPropertyDescriptor[_InstanceT, LazyWrapper[_T]]):
    __slots__ = ("restock_methods",)

    def __init__(
        self,
        cls_method: Callable[..., _T]
    ) -> None:
        self.restock_methods: list[Callable[[_T], None]] = []
        parameters_wrapped_method, parameter_chains = lazy_property.wrap_parameters(cls_method.__func__)

        def new_method(
            cls: type[_InstanceT],
            *args: Any,
            **kwargs: Any
        ) -> LazyWrapper[_T]:
            entity = LazyWrapper(parameters_wrapped_method(cls, *args, **kwargs))
            for restock_method in self.restock_methods:
                entity._at_restock(lambda obj: restock_method(obj.value))
            return entity

        super().__init__(
            #object_type=LazyWrapper,
            #is_collection=False,
            method=new_method,
            parameter_chains=parameter_chains
        )

    def restocker(
        self,
        restock_method: Callable[[_T], None]
    ) -> Callable[[_T], None]:
        self.restock_methods.append(restock_method)
        return restock_method

    #def handle_new_property(
    #    self,
    #    entity: LazyWrapper[_T]
    #) -> LazyWrapper[_T]:
    #    for restock_method in self.restock_methods:
    #        entity._at_restock(lambda obj: restock_method(obj.value))
    #    return entity

    #def __new__(
    #    cls,
    #    method: Union[
    #        Callable[Concatenate[type[_InstanceT], _ParameterSpec], _LazyObjectT],
    #        Callable[Concatenate[type[_InstanceT], _ParameterSpec], _T]
    #    ]
    #) -> Union[
    #    LazyPropertyDescriptor[_InstanceT, _ParameterSpec, _LazyObjectT],
    #    LazyPropertyUnwrappedDescriptor[_InstanceT, _ParameterSpec, _T]
    #]:
    #    method = method.__func__
    #    return_type = _get_type_from_annotation(
    #        inspect.signature(method).return_annotation
    #    )
    #    if issubclass(return_type, LazyObject):
    #        return LazyPropertyDescriptor(method)
    #    return LazyPropertyUnwrappedDescriptor(method)


class lazy_property_shared(LazyPropertyDescriptor[_InstanceT, LazyWrapper[_HashableT]]):
    __slots__ = (
        "restock_methods",
        "content_to_object_bidict"
    )

    def __init__(
        self,
        cls_method: Callable[..., _HashableT]
    ) -> None:
        self.content_to_object_bidict: bidict[_HashableT, LazyWrapper[_HashableT]] = bidict()

        def cleanup_method(
            cached_object: LazyWrapper[_HashableT]
        ) -> None:
            self.content_to_object_bidict.inverse.pop(cached_object)

        parameters_wrapped_method, parameter_chains = lazy_property.wrap_parameters(cls_method.__func__)

        def new_method(
            cls: type[_InstanceT],
            *args: Any,
            **kwargs: Any
        ) -> LazyWrapper[_HashableT]:
            content = parameters_wrapped_method(cls, *args, **kwargs)
            if (cached_object := self.content_to_object_bidict.get(content)) is None:
                cached_object = LazyWrapper(content)
                self.content_to_object_bidict[content] = cached_object
                cached_object._at_restock(cleanup_method)
            for restock_method in self.restock_methods:
                cached_object._at_restock(lambda obj: restock_method(obj.value))
            return cached_object

        super().__init__(
            #object_type=LazyWrapper,
            #is_collection=False,
            method=new_method,
            parameter_chains=parameter_chains
        )
        self.restock_methods: list[Callable[[_HashableT], None]] = []

    def restocker(
        self,
        restock_method: Callable[[_HashableT], None]
    ) -> Callable[[_HashableT], None]:
        self.restock_methods.append(restock_method)
        return restock_method


#@overload
#def lazy_property(
#    method: Callable[[_InstanceT, ...], _LazyObjectT]
#) -> LazyObjectDescriptor[_LazyObjectT, _InstanceT]: ...


#@overload
#def lazy_property(
#    method: Callable[[_InstanceT], LazyCollection[_LazyEntityT]]
#) -> LazyCollectionDescriptor[LazyCollection[_LazyEntityT], _InstanceT]: ...


#@overload
#def lazy_property(
#    method: Callable[[_InstanceT], _T]
#) -> LazyObjectRawDescriptor[_T, _InstanceT]: ...


#def lazy_property(
#    method: Union[
#        Callable[[_InstanceT], _LazyObjectT],
#        Callable[[_InstanceT], LazyCollection[_LazyEntityT]],
#        Callable[[_InstanceT], _T]
#    ]
#) -> Union[
#    LazyObjectDescriptor[_LazyObjectT, _InstanceT],
#    LazyCollectionDescriptor[LazyCollection[_LazyEntityT], _InstanceT],
#    LazyObjectRawDescriptor[_T, _InstanceT]
#]:
#    return_type = _get_type_from_annotation(
#        inspect.signature(method).return_annotation
#    )
#    if issubclass(return_type, LazyObject):
#        return LazyObjectDescriptor(method)
#    if issubclass(return_type, LazyCollection):
#        return LazyCollectionDescriptor(method)
#    return LazyObjectRawDescriptor(method)


#def lazy_property_shared(
#    method: Callable[..., _HashableT]
#) -> LazyObjectSharedDescriptor[_HashableT, _InstanceT]:
#    return LazyObjectSharedDescriptor(method)


##class lazy_object(Generic[_LazyObjectT, _InstanceT]):
##    __slots__ = ()
##
##    def __new__(
##        cls,
##        method: Callable[[], _LazyObjectT]
##    ) -> LazyObjectDescriptor[_LazyObjectT, _InstanceT]:
##        return LazyObjectDescriptor(method)


#class lazy_collection(Generic[_LazyEntityT, _InstanceT]):
#    __slots__ = ()

#    def __new__(
#        cls,
#        method: Callable[[], LazyCollection[_LazyEntityT]]
#    ) -> LazyCollectionDescriptor[_LazyEntityT, _InstanceT]:
#        return LazyCollectionDescriptor(method)


#class lazy_property(Generic[_LazyEntityT, _InstanceT]):
#    __slots__ = ()

#    def __new__(
#        cls,
#        method: Callable[..., _LazyEntityT]
#    ) -> LazyPropertyDescriptor[_LazyEntityT, _InstanceT]:
#        return LazyPropertyDescriptor(method)


#class lazy_object_raw(Generic[_T, _InstanceT]):
#    __slots__ = ()

#    def __new__(
#        cls,
#        method: Callable[[], _T]
#    ) -> LazyObjectRawDescriptor[_T, _InstanceT]:
#        return LazyObjectRawDescriptor(method)


#class lazy_object_shared(Generic[_T, _InstanceT, _KeyT]):
#    __slots__ = ("key",)

#    def __init__(
#        self,
#        key: Callable[[_T], _KeyT]
#    ):
#        self.key: Callable[[_T], _KeyT] = key

#    def __call__(
#        self,
#        method: Callable[[], _T]
#    ) -> LazyObjectSharedDescriptor[_T, _InstanceT, _KeyT]:
#        return LazyObjectSharedDescriptor(method, self.key)


#class lazy_property_raw(Generic[_T, _InstanceT]):
#    __slots__ = ()

#    def __new__(
#        cls,
#        method: Callable[..., _T]
#    ) -> LazyPropertyRawDescriptor[_T, _InstanceT]:
#        return LazyPropertyRawDescriptor(method)


#class lazy_property_shared(Generic[_T, _InstanceT, _KeyT]):
#    __slots__ = ("key",)

#    def __init__(
#        self,
#        key: Callable[[_T], _KeyT]
#    ):
#        self.key: Callable[[_T], _KeyT] = key

#    def __call__(
#        self,
#        method: Callable[..., _T]
#    ) -> LazyPropertySharedDescriptor[_T, _InstanceT, _KeyT]:
#        return LazyPropertySharedDescriptor(method, self.key)
