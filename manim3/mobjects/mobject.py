import itertools as it
from typing import (
    Iterable,
    Iterator,
    overload
)
import weakref

from ..lazy.lazy import Lazy
from ..models.model import (
    Model,
    StyleMeta
)
from ..models.cameras.camera import Camera
from ..models.cameras.perspective_camera import PerspectiveCamera
from ..rendering.framebuffer import OITFramebuffer


class Mobject(Model):
    __slots__ = (
        "_parents",
        "_real_ancestors"
    )

    def __init__(self) -> None:
        super().__init__()
        # If `_parents` and `_real_ancestors` are implemented with `LazyDynamicContainer` also,
        # loops will pop up in the DAG of the lazy system.
        self._parents: weakref.WeakSet[Mobject] = weakref.WeakSet()
        self._real_ancestors: weakref.WeakSet[Mobject] = weakref.WeakSet()

    def __iter__(self) -> "Iterator[Mobject]":
        return iter(self._children_)

    @overload
    def __getitem__(
        self,
        index: int
    ) -> "Mobject": ...

    @overload
    def __getitem__(
        self,
        index: slice
    ) -> "list[Mobject]": ...

    def __getitem__(
        self,
        index: int | slice
    ) -> "Mobject | list[Mobject]":
        return self._children_.__getitem__(index)

    # family matters
    # These methods implement a DAG (directed acyclic graph).

    @Lazy.variable_collection
    @classmethod
    def _children_(cls) -> "list[Mobject]":
        return []

    @Lazy.variable_collection
    @classmethod
    def _real_descendants_(cls) -> "list[Mobject]":
        return []

    @classmethod
    def _refresh_families(
        cls,
        *mobjects: "Mobject"
    ) -> None:

        def iter_descendants_by_children(
            mobject: Mobject
        ) -> Iterator[Mobject]:
            yield mobject
            for child in mobject._children_:
                yield from iter_descendants_by_children(child)

        def iter_ancestors_by_parents(
            mobject: Mobject
        ) -> Iterator[Mobject]:
            yield mobject
            for parent in mobject._parents:
                yield from iter_ancestors_by_parents(parent)

        for ancestor in dict.fromkeys(it.chain.from_iterable(
            iter_ancestors_by_parents(mobject)
            for mobject in mobjects
        )):
            ancestor._real_descendants_.reset(dict.fromkeys(it.chain.from_iterable(
                iter_descendants_by_children(child)
                for child in ancestor._children_
            )))
        for descendant in dict.fromkeys(it.chain.from_iterable(
            iter_descendants_by_children(mobject)
            for mobject in mobjects
        )):
            descendant._real_ancestors.clear()
            descendant._real_ancestors.update(dict.fromkeys(it.chain.from_iterable(
                iter_ancestors_by_parents(parent)
                for parent in descendant._parents
            )))

    def iter_children(self) -> "Iterator[Mobject]":
        yield from self._children_

    def iter_parents(self) -> "Iterator[Mobject]":
        yield from self._parents

    def iter_descendants(
        self,
        *,
        broadcast: bool = True
    ) -> "Iterator[Mobject]":
        yield self
        if broadcast:
            yield from self._real_descendants_

    def iter_ancestors(
        self,
        *,
        broadcast: bool = True
    ) -> "Iterator[Mobject]":
        yield self
        if broadcast:
            yield from self._real_ancestors

    def add(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject not in self._children_
        ]
        if (invalid_mobjects := [
            mobject for mobject in filtered_mobjects
            if mobject in self.iter_ancestors()
        ]):
            raise ValueError(f"Circular relationship occurred when adding {invalid_mobjects} to {self}")
        self._children_.extend(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._parents.add(self)
        self._refresh_families(self)
        return self

    def discard(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject in self._children_
        ]
        self._children_.eliminate(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._parents.remove(self)
        self._refresh_families(self, *filtered_mobjects)
        return self

    def added_by(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject not in self._parents
        ]
        if (invalid_mobjects := [
            mobject for mobject in filtered_mobjects
            if mobject in self.iter_descendants()
        ]):
            raise ValueError(f"Circular relationship occurred when adding {self} to {invalid_mobjects}")
        self._parents.update(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._children_.append(self)
        self._refresh_families(self)
        return self

    def discarded_by(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject in self._parents
        ]
        self._parents.difference_update(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._children_.remove(self)
        self._refresh_families(self, *filtered_mobjects)
        return self

    def clear(self):
        self.discard(*self.iter_children())
        return self

    def copy(self):
        # Copy all descendants. The result is not bound to any mobject.
        result = self._copy()
        real_descendants = list(self._real_descendants_)
        real_descendants_copy = [
            descendant._copy()
            for descendant in real_descendants
        ]
        descendants = [self, *real_descendants]
        descendants_copy = [result, *real_descendants_copy]

        def match_copies(
            mobjects: Iterable[Mobject]
        ) -> Iterator[Mobject]:
            return (
                descendants_copy[descendants.index(mobject)] if mobject in descendants else mobject
                for mobject in mobjects
            )

        result._children_.reset(match_copies(self._children_))
        result._parents.clear()
        for real_descendant, real_descendant_copy in zip(real_descendants, real_descendants_copy, strict=True):
            real_descendant_copy._children_.reset(match_copies(real_descendant._children_))
            real_descendant_copy._parents.clear()
            real_descendant_copy._parents.update(match_copies(real_descendant._parents))

        self._refresh_families(*descendants_copy)
        return result

    # render

    @StyleMeta.register()
    @Lazy.variable
    @classmethod
    def _camera_(cls) -> Camera:
        return PerspectiveCamera()

    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        pass

    # meta methods

    def concatenate(self):
        StyleMeta._concatenate(*self.iter_children())(self)()
        self.clear()
        return self
