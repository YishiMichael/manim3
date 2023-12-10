from __future__ import annotations


import itertools
import weakref
from typing import (
    Iterator,
    Self
)

from ..animatables.camera import Camera
from ..animatables.lighting import Lighting
from ..animatables.model import Model
from ..lazy.lazy import Lazy
from ..rendering.vertex_array import VertexArray
from ..toplevel.toplevel import Toplevel


class Mobject(Model):
    __slots__ = (
        "__weakref__",
        "_children",
        "_proper_descendants",
        "_parents",
        "_proper_ancestors"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._children: list[Mobject] = []
        self._proper_descendants: list[Mobject] = []
        self._parents: weakref.WeakSet[Mobject] = weakref.WeakSet()
        self._proper_ancestors: weakref.WeakSet[Mobject] = weakref.WeakSet()

    def __iter__(
        self: Self
    ) -> Iterator[Mobject]:
        yield from self._children

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

        for proper_ancestor in dict.fromkeys(itertools.chain.from_iterable(
            iter_ancestors_by_parents(mobject)
            for mobject in mobjects
        )):
            proper_descendants = dict.fromkeys(itertools.chain.from_iterable(
                iter_descendants_by_children(child)
                for child in proper_ancestor._children
            ))
            proper_ancestor._proper_descendants.clear()
            proper_ancestor._proper_descendants.extend(proper_descendants)
            proper_ancestor._proper_siblings_ = tuple(proper_descendants)

        for proper_descendant in dict.fromkeys(itertools.chain.from_iterable(
            iter_descendants_by_children(mobject)
            for mobject in mobjects
        )):
            proper_ancestors = dict.fromkeys(itertools.chain.from_iterable(
                iter_descendants_by_children(parent)
                for parent in proper_descendant._parents
            ))
            proper_descendant._proper_ancestors.clear()
            proper_descendant._proper_ancestors.update(proper_ancestors)

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
            yield from self._proper_descendants

    def iter_ancestors(
        self: Self,
        *,
        broadcast: bool = True
    ) -> Iterator[Mobject]:
        yield self
        if broadcast:
            yield from self._proper_ancestors

    def add(
        self: Self,
        *mobjects: Mobject
    ) -> Self:
        if (invalid_mobjects := tuple(
            mobject for mobject in mobjects
            if mobject in self.iter_ancestors()
        )):
            raise ValueError(f"Circular relationship occurred when adding {invalid_mobjects} to {self}")
        for mobject in dict.fromkeys(mobjects):
            if mobject in self._children:
                continue
            self._children.append(mobject)
            mobject._parents.add(self)
        type(self)._refresh_families(self, *mobjects)
        return self

    def discard(
        self: Self,
        *mobjects: Mobject
    ) -> Self:
        for mobject in dict.fromkeys(mobjects):
            if mobject not in self._children:
                continue
            self._children.remove(mobject)
            mobject._parents.remove(self)
        type(self)._refresh_families(self, *mobjects)
        return self

    def clear(
        self: Self
    ) -> Self:
        self.discard(*self.iter_children())
        return self

    def copy(
        self: Self
    ) -> Self:
        # Copy all descendants. The result is not bound to any mobject.
        result = super().copy()
        descendants: list[Mobject] = [self, *(
            descendant for descendant in self._proper_siblings_
            if isinstance(descendant, Mobject)
        )]
        descendants_copy: list[Mobject] = [result, *(
            descendant_copy for descendant_copy in result._proper_siblings_
            if isinstance(descendant_copy, Mobject)
        )]

        for descendant, descendant_copy in zip(descendants, descendants_copy, strict=True):
            descendant_copy._children = [
                descendants_copy[descendants.index(child)]
                for child in descendant._children
            ]
            descendant_copy._proper_descendants = []
            descendant_copy._parents = weakref.WeakSet(
                descendants_copy[descendants.index(parent)]
                for parent in descendant._parents
                if parent in descendants
            )
            descendant_copy._proper_ancestors = weakref.WeakSet()

        type(self)._refresh_families(*descendants_copy)
        return result

    # render

    @Lazy.volatile(deepcopy=False)
    @staticmethod
    def _camera_() -> Camera:
        return Toplevel._get_scene()._camera

    @Lazy.volatile(deepcopy=False)
    @staticmethod
    def _lighting_() -> Lighting:
        return Toplevel._get_scene()._lighting

    def _iter_vertex_arrays(
        self: Self
    ) -> Iterator[VertexArray]:
        yield from ()

    def bind_camera(
        self: Self,
        camera: Camera,
        *,
        broadcast: bool = True,
    ) -> Self:
        for mobject in self.iter_descendants(broadcast=broadcast):
            mobject._camera_ = camera
        return self

    def bind_lighting(
        self: Self,
        lighting: Lighting,
        *,
        broadcast: bool = True,
    ) -> Self:
        for mobject in self.iter_descendants(broadcast=broadcast):
            mobject._lighting_ = lighting
        return self
