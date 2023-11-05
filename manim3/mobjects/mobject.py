from __future__ import annotations


import itertools
import weakref
from typing import (
    Callable,
    ClassVar,
    Iterator,
    Self,
    overload,
    override
)

from ..animatables.animatable.animatable import AnimatableMeta
from ..animatables.camera import Camera
from ..animatables.model import Model
from ..lazy.lazy import Lazy
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..toplevel.toplevel import Toplevel


class Mobject(Model):
    __slots__ = (
        "__weakref__",
        "_children",
        "_descendants",
        "_parents",
        "_ancestors"
    )

    _special_slot_copiers: ClassVar[dict[str, Callable]] = {
        "_parents": lambda o: weakref.WeakSet(),
        "_ancestors": lambda o: weakref.WeakSet()
    }

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._children: tuple[Mobject, ...] = ()
        self._descendants: tuple[Mobject, ...] = ()
        self._parents: weakref.WeakSet[Mobject] = weakref.WeakSet()
        self._ancestors: weakref.WeakSet[Mobject] = weakref.WeakSet()

    def __iter__(
        self: Self
    ) -> Iterator[Mobject]:
        return iter(self._children)

    @overload
    def __getitem__(
        self: Self,
        index: int
    ) -> Mobject: ...

    @overload
    def __getitem__(
        self: Self,
        index: slice
    ) -> tuple[Mobject, ...]: ...

    def __getitem__(
        self: Self,
        index: int | slice
    ) -> Mobject | tuple[Mobject, ...]:
        return self._children.__getitem__(index)

    # render

    @AnimatableMeta.register_converter()
    @Lazy.volatile(deepcopy=False)
    @staticmethod
    def _camera_() -> Camera:
        return Toplevel.scene._camera

    def _render(
        self: Self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        pass

    # family matters
    # These methods implement a DAG (directed acyclic graph).

    @classmethod
    def _refresh_families(
        cls: type[Self],
        *mobjects: Mobject
    ) -> None:

        def iter_descendants_by_children(
            mobject: Mobject
        ) -> Iterator[Mobject]:
            yield mobject
            for child in mobject._children:
                yield from iter_descendants_by_children(child)

        def iter_ancestors_by_parents(
            mobject: Mobject
        ) -> Iterator[Mobject]:
            yield mobject
            for parent in mobject._parents:
                yield from iter_ancestors_by_parents(parent)

        for ancestor in dict.fromkeys(itertools.chain.from_iterable(
            iter_ancestors_by_parents(mobject)
            for mobject in mobjects
        )):
            descendants = tuple(dict.fromkeys(itertools.chain.from_iterable(
                iter_descendants_by_children(child)
                for child in ancestor._children
            )))
            ancestor._descendants = descendants
            ancestor._proper_siblings_ = descendants

        for descendant in dict.fromkeys(itertools.chain.from_iterable(
            iter_descendants_by_children(mobject)
            for mobject in mobjects
        )):
            ancestors = tuple(dict.fromkeys(itertools.chain.from_iterable(
                iter_descendants_by_children(parent)
                for parent in descendant._parents
            )))
            descendant._ancestors.clear()
            descendant._ancestors.update(ancestors)

    def iter_children(
        self: Self
    ) -> Iterator[Mobject]:
        yield from self._children

    def iter_parents(
        self: Self
    ) -> Iterator[Mobject]:
        yield from self._parents

    def iter_descendants(
        self: Self,
        *,
        broadcast: bool = True
    ) -> Iterator[Mobject]:
        yield self
        if broadcast:
            yield from self._descendants

    def iter_ancestors(
        self: Self,
        *,
        broadcast: bool = True
    ) -> Iterator[Mobject]:
        yield self
        if broadcast:
            yield from self._ancestors

    def add(
        self: Self,
        *mobjects: Mobject
    ) -> Self:
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject not in self._children
        ]
        if (invalid_mobjects := [
            mobject for mobject in filtered_mobjects
            if mobject in self.iter_ancestors()
        ]):
            raise ValueError(f"Circular relationship occurred when adding {invalid_mobjects} to {self}")
        children = list(self._children)
        for mobject in filtered_mobjects:
            children.append(mobject)
            mobject._parents.add(self)
        self._children = tuple(children)
        type(self)._refresh_families(self)
        return self

    def discard(
        self: Self,
        *mobjects: Mobject
    ) -> Self:
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject in self._children
        ]
        children = list(self._children)
        for mobject in filtered_mobjects:
            children.remove(mobject)
            mobject._parents.remove(self)
        self._children = tuple(children)
        type(self)._refresh_families(self, *filtered_mobjects)
        return self

    def clear(
        self: Self
    ) -> Self:
        self.discard(*self.iter_children())
        return self

    @override
    def copy(
        self: Self
    ) -> Self:
        # Copy all descendants. The result is not bound to any mobject.
        result = super().copy()
        descendants: tuple[Mobject, ...] = (self, *(
            descendant for descendant in self._proper_siblings_
            if isinstance(descendant, Mobject)
        ))
        descendants_copy: tuple[Mobject, ...] = (result, *(
            descendant_copy for descendant_copy in result._proper_siblings_
            if isinstance(descendant_copy, Mobject)
        ))
        for descendant, descendant_copy in zip(descendants, descendants_copy, strict=True):
            descendant_copy._children = tuple(
                descendants_copy[descendants.index(child)]
                for child in descendant._children
            )
            descendant_copy._parents.update(
                descendants_copy[descendants.index(parent)]
                for parent in descendant._parents
                if parent in descendants
            )
        type(self)._refresh_families(*descendants_copy)
        return result
