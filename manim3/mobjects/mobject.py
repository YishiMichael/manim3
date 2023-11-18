from __future__ import annotations


import itertools
import weakref
from typing import (
    Iterator,
    Self,
    overload
)

from ..animatables.camera import Camera
from ..animatables.lighting import Lighting
from ..animatables.model import Model
from ..lazy.lazy import Lazy
from ..rendering.framebuffers.framebuffer import Framebuffer
from ..rendering.vertex_array import (
    ModernglBuffers,
    VertexArray
)
from ..toplevel.toplevel import Toplevel


class Mobject(Model):
    __slots__ = (
        "__weakref__",
        "_children",
        "_proper_descendants",
        "_parents",
        "_proper_ancestors",
        "_moderngl_buffers"
    )

    #_special_slot_copiers: ClassVar[dict[str, Callable]] = {
    #    "_parents": lambda o: weakref.WeakSet(),
    #    "_proper_ancestors": lambda o: weakref.WeakSet()
    #}

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._children: list[Mobject] = []
        self._proper_descendants: list[Mobject] = []
        self._parents: weakref.WeakSet[Mobject] = weakref.WeakSet()
        self._proper_ancestors: weakref.WeakSet[Mobject] = weakref.WeakSet()
        self._moderngl_buffers: ModernglBuffers | None = None

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
    ) -> list[Mobject]: ...

    def __getitem__(
        self: Self,
        index: int | slice
    ) -> Mobject | list[Mobject]:
        return self._children.__getitem__(index)

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
            descendant_copy._moderngl_buffers = None

        type(self)._refresh_families(*descendants_copy)
        return result

    # render

    @Lazy.volatile(deepcopy=False)
    @staticmethod
    def _camera_() -> Camera:
        return Toplevel.scene._camera

    @Lazy.volatile(deepcopy=False)
    @staticmethod
    def _lighting_() -> Lighting:
        return Toplevel.scene._lighting

    def _get_vertex_array(
        self: Self
    ) -> VertexArray | None:
        return None

    def _render(
        self: Self,
        target_framebuffer: Framebuffer
    ) -> None:
        if (vertex_array := self._get_vertex_array()) is None:
            return
        if (moderngl_buffers := self._moderngl_buffers) is None:
            moderngl_buffers = vertex_array.fetch_moderngl_buffers()
            self._moderngl_buffers = moderngl_buffers
        #print(self, self._moderngl_buffers)
        vertex_array.render(moderngl_buffers, target_framebuffer)

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
