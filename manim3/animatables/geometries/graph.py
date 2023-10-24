from __future__ import annotations


import itertools
from typing import (
    Iterator,
    Literal,
    Self,
    Unpack,
    override
)

import numpy as np

from ...constants.custom_typing import (
    NP_i4,
    NP_x2i4,
    NP_x3f8,
    NP_xi4,
    NP_xf8
)
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..animatable.actions import ActionMeta
from ..animatable.animatable import (
    Animatable,
    AnimatableActions,
    AnimatableInterpolateAnimation,
    AnimatablePiecewiseAnimation,
    DynamicAnimatable
)
from ..animatable.animation import (
    AnimateKwargs,
    Animation
)
from ..animatable.piecewiser import Piecewiser
#from .animatable.leaf_animatable import (
#    LeafAnimatable,
#    LeafAnimatableInterpolateInfo
#)
#from ..mobject.mobject_attributes.mobject_attribute import (
#    InterpolateHandler,
#    MobjectAttribute
#)


class GraphActions(AnimatableActions):
    __slots__ = ()

    @ActionMeta.register
    @classmethod
    @override
    def interpolate(
        cls: type[Self],
        dst: Graph,
        src_0: Graph,
        src_1: Graph
    ) -> Iterator[Animation]:
        yield GraphInterpolateAnimation(dst, src_0, src_1)

    @ActionMeta.register
    @classmethod
    @override
    def piecewise(
        cls: type[Self],
        dst: Graph,
        src: Graph,
        piecewiser: Piecewiser
    ) -> Iterator[Animation]:
        yield GraphPiecewiseAnimation(dst, src, piecewiser)


class Graph(GraphActions, Animatable):
    __slots__ = ()

    def __init__(
        self: Self,
        positions: NP_x3f8 | None = None,
        edges: NP_x2i4 | None = None
    ) -> None:
        super().__init__()
        if positions is not None:
            self._positions_ = positions
        if edges is not None:
            self._edges_ = edges

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _positions_() -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _edges_() -> NP_x2i4:
        return np.zeros((0, 2), dtype=np.int32)

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _cumlengths_(
        positions: NP_x3f8,
        edges: NP_x2i4
    ) -> NP_xf8:
        lengths = SpaceUtils.norm(positions[edges[:, 1]] - positions[edges[:, 0]])
        return np.insert(np.cumsum(lengths), 0, 0.0)

    #@classmethod
    #def _interpolate(
    #    cls,
    #    graph_0: "Graph",
    #    graph_1: "Graph"
    #) -> "GraphInterpolateHandler":
    #    positions_0, positions_1, edges = cls._general_interpolate(
    #        graph_0=graph_0,
    #        graph_1=graph_1,
    #        disjoints_0=np.zeros((0,), dtype=np.int32),
    #        disjoints_1=np.zeros((0,), dtype=np.int32)
    #    )
    #    return GraphInterpolateHandler(
    #        positions_0=positions_0,
    #        positions_1=positions_1,
    #        edges=edges
    #    )

    #@classmethod
    #def _interpolate(
    #    cls: type[Self],
    #    src_0: Self,
    #    src_1: Self
    #) -> GraphInterpolateInfo:
    #    return GraphInterpolateInfo(src_0, src_1)

    #def _interpolate(
    #    self: Self,
    #    src_0: Self,
    #    src_1: Self
    #) -> Updater:
    #    return GraphInterpolateUpdater(self, src_0, src_1)

    #@classmethod
    #def _get_cumlengths(
    #    cls,
    #    positions: NP_x3f8,
    #    edges: NP_x2i4
    #) -> NP_xf8:
    #    lengths = SpaceUtils.norm(positions[edges[:, 1]] - positions[edges[:, 0]])
    #    return np.insert(np.cumsum(lengths), 0, 0.0)

    #@classmethod
    #def _get_disjoints(
    #    cls,
    #    edges: NP_x2i4
    #) -> NP_xi4:
    #    return np.flatnonzero(edges[:-1, 1] - edges[1:, 0])

    @classmethod
    def _general_interpolate(
        cls: type[Self],
        graph_0: Graph,
        graph_1: Graph,
        disjoints_0: NP_xi4,
        disjoints_1: NP_xi4
    ) -> tuple[NP_x3f8, NP_x3f8, NP_x2i4]:
        positions_0 = graph_0._positions_
        positions_1 = graph_1._positions_
        edges_0 = graph_0._edges_
        edges_1 = graph_1._edges_

        centroid_0: NP_x3f8 | None = None
        centroid_1: NP_x3f8 | None = None
        if len(edges_0):
            samples_0 = positions_0[edges_0.flatten()]
            centroid_0 = (np.max(samples_0, axis=0, keepdims=True) + np.min(samples_0, axis=0, keepdims=True)) / 2.0
        if len(edges_1):
            samples_1 = positions_1[edges_1.flatten()]
            centroid_1 = (np.max(samples_1, axis=0, keepdims=True) + np.min(samples_1, axis=0, keepdims=True)) / 2.0

        if centroid_0 is None or centroid_1 is None:
            if centroid_0 is not None:
                return positions_0, np.repeat(centroid_0, len(positions_0), axis=0), edges_0
            if centroid_1 is not None:
                return positions_1, np.repeat(centroid_1, len(positions_1), axis=0), edges_1
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,), dtype=np.int32)

        cumlengths_0 = graph_0._cumlengths_
        cumlengths_1 = graph_1._cumlengths_
        full_knots_0 = cumlengths_0 * cumlengths_1[-1]
        full_knots_1 = cumlengths_1 * cumlengths_0[-1]
        knots_0 = full_knots_0[1:-1]
        knots_1 = full_knots_1[1:-1]

        outline_edges_0, outline_positions_0, interpolated_indices_0 = cls._get_decomposed_edges(
            positions=positions_0,
            edges=edges_0,
            insertions=np.arange(len(edges_1) - 1) + len(positions_0),
            full_knots=full_knots_0,
            values=knots_1,
            side="right"
        )
        outline_edges_1, outline_positions_1, interpolated_indices_1 = cls._get_decomposed_edges(
            positions=positions_1,
            edges=edges_1,
            insertions=np.arange(len(edges_0) - 1) + len(positions_1),
            full_knots=full_knots_1,
            values=knots_0,
            side="left"
        )

        #disjoints_0 = Graph._get_disjoints(edges=edges_0)
        #disjoints_1 = Graph._get_disjoints(edges=edges_1)
        #print(np.flatnonzero(edges_0[:-1, 1] - edges_0[1:, 0]))
        #print(disjoints_0)
        #print(len(edges_0))
        if len(disjoints_0) >= 2 and len(disjoints_1) >= 2:
            inlay_edges_0 = Graph._reassemble_edges(
                edges=edges_0,
                transition_indices=disjoints_0[1:-1] - 1,
                prepend=edges_0[disjoints_0[0], 0],
                append=edges_0[disjoints_0[-1] - 1, 1],
                insertion_indices=np.searchsorted(
                    disjoints_0[1:-1] - 1,
                    interpolated_indices_0[disjoints_1[1:-1] - 1],
                    side="right"
                ).astype(np.int32),
                insertions=disjoints_1[1:-1] - 1 + len(positions_0)
            )
            inlay_edges_1 = Graph._reassemble_edges(
                edges=edges_1,
                transition_indices=disjoints_1[1:-1] - 1,
                prepend=edges_1[disjoints_1[0], 0],
                append=edges_1[disjoints_1[-1] - 1, 1],
                insertion_indices=np.searchsorted(
                    disjoints_1[1:-1] - 1,
                    interpolated_indices_1[disjoints_0[1:-1] - 1],
                    side="left"
                ).astype(np.int32),
                insertions=disjoints_0[1:-1] - 1 + len(positions_1)
            )
        else:
            inlay_edges_0 = np.zeros((0, 2), dtype=np.int32)
            inlay_edges_1 = np.zeros((0, 2), dtype=np.int32)
        interpolated_positions_0, interpolated_positions_1, edges = Graph._get_unique_positions(
            positions_0=np.concatenate((
                positions_0,
                outline_positions_0,
                centroid_0
            )),
            positions_1=np.concatenate((
                positions_1,
                outline_positions_1,
                centroid_1
            )),
            edges_0=np.concatenate((
                outline_edges_0,
                inlay_edges_0,
                np.ones_like(inlay_edges_1) * (len(positions_0) + len(edges_1) - 1)
            )),
            edges_1=np.concatenate((
                outline_edges_1,
                np.ones_like(inlay_edges_1) * (len(positions_1) + len(edges_0) - 1),
                inlay_edges_1
            ))
        )
        return interpolated_positions_0, interpolated_positions_1, edges

    @classmethod
    def _interpolate_positions(
        cls: type[Self],
        positions: NP_x3f8,
        edges: NP_x2i4,
        full_knots: NP_xf8,
        values: NP_xf8,
        indices: NP_xi4
    ) -> NP_x3f8:
        residues = (values - full_knots[indices]) / np.maximum(full_knots[indices + 1] - full_knots[indices], 1e-6)
        return SpaceUtils.lerp(
            positions[edges[indices, 0]],
            positions[edges[indices, 1]],
            residues[:, None]
        )

    @classmethod
    def _reassemble_edges(
        cls: type[Self],
        edges: NP_x2i4,
        transition_indices: NP_xi4,
        prepend: NP_i4,
        append: NP_i4,
        insertion_indices: NP_xi4,
        insertions: NP_xi4
    ) -> NP_x2i4:
        return np.column_stack((
            np.insert(np.insert(
                edges[transition_indices + 1, 0],
                insertion_indices,
                insertions
            ), 0, prepend),
            np.append(np.insert(
                edges[transition_indices, 1],
                insertion_indices,
                insertions
            ), append)
        ))

    @classmethod
    def _get_decomposed_edges(
        cls: type[Self],
        positions: NP_x3f8,
        edges: NP_x2i4,
        insertions: NP_xi4,
        full_knots: NP_xf8,
        values: NP_xf8,
        side: Literal["left", "right"]
    ) -> tuple[NP_x2i4, NP_x3f8, NP_xi4]:
        interpolated_indices = np.searchsorted(full_knots[1:-1], values, side=side).astype(np.int32)
        decomposed_edges = cls._reassemble_edges(
            edges=edges,
            transition_indices=np.arange(len(edges) - 1),
            prepend=edges[0, 0],
            append=edges[-1, 1],
            insertion_indices=interpolated_indices,
            insertions=insertions
        )
        interpolated_positions = cls._interpolate_positions(
            positions=positions,
            edges=edges,
            full_knots=full_knots,
            values=values,
            indices=interpolated_indices
        )
        return decomposed_edges, interpolated_positions, interpolated_indices

    @classmethod
    def _get_unique_positions(
        cls: type[Self],
        positions_0: NP_x3f8,
        positions_1: NP_x3f8,
        edges_0: NP_x2i4,
        edges_1: NP_x2i4
    ) -> tuple[NP_x3f8, NP_x3f8, NP_x2i4]:
        unique_edges, edges_inverse = np.unique(
            np.array((edges_0.flatten(), edges_1.flatten())),
            axis=1,
            return_inverse=True
        )
        return (
            positions_0[unique_edges[0]],
            positions_1[unique_edges[1]],
            edges_inverse.reshape((-1, 2))
        )

    #def _get_interpolate_updater(
    #    self: "Graph",
    #    graph_0: "Graph",
    #    graph_1: "Graph"
    #) -> "GraphInterpolateUpdater":
    #    return GraphInterpolateUpdater(
    #        graph=self,
    #        graph_0=graph_0,
    #        graph_1=graph_1
    #    )

    #def partial(
    #    self,
    #    alpha_to_segments: Callable[[float], tuple[NP_xf8, list[int]]]
    #) -> "GraphPartialUpdater":
    #    return GraphPartialUpdater(
    #        graph=self,
    #        original_graph=self._copy(),
    #        alpha_to_segments=alpha_to_segments
    #    )

    #def set_from_parameters(
    #    self,
    #    positions: NP_x3f8,
    #    edges: NP_x2i4
    #):
    #    self._positions_ = positions
    #    self._edges_ = edges
    #    return self

    #def set_from_empty(self):
    #    self.set_from_parameters(
    #        positions=np.zeros((0, 3)),
    #        edges=np.zeros((0, 2), dtype=np.int32)
    #    )

    #def set_from_graph(
    #    self,
    #    graph: "Graph"
    #):
    #    self.set_from_parameters(
    #        positions=graph._positions_,
    #        edges=graph._edges_
    #    )

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
        #rate: Rate = Rates.linear(),
        #rewind: bool = False,
        #run_alpha: float = 1.0,
        #infinite: bool = False
    ) -> DynamicGraph[Self]:
        return DynamicGraph(self, **kwargs)

    def as_parameters(
        self: Self,
        positions: NP_x3f8,
        edges: NP_x2i4
    ) -> Self:
        self._positions_ = positions
        self._edges_ = edges
        return self

    def as_empty(
        self: Self
    ) -> Self:
        return self.as_parameters(
            positions=np.zeros((0, 3)),
            edges=np.zeros((0, 2), dtype=np.int32)
        )

    def split(
        self: Self,
        #dst_tuple: tuple[Self, ...],
        dsts: tuple[Self, ...],
        alphas: NP_xf8
    ) -> Self:
        edges = self._edges_
        if not len(edges):
            for dst in dsts:
                dst.as_empty()
            return self

        cls = type(self)
        positions = self._positions_
        cumlengths = self._cumlengths_
        values = alphas * cumlengths[-1]
        interpolated_indices = np.searchsorted(cumlengths[1:-1], values)
        all_positions = np.concatenate((
            positions,
            cls._interpolate_positions(
                positions=positions,
                edges=edges,
                full_knots=cumlengths,
                values=values,
                indices=interpolated_indices
            )
        ))
        for dst, (prepend, append), (interpolated_index_0, interpolate_index_1) in zip(
            dsts,
            itertools.pairwise(np.array((edges[0, 0], *(np.arange(len(alphas)) + len(positions)), edges[-1, 1]))),
            itertools.pairwise(np.array((0, *interpolated_indices, len(edges) - 1))),
            strict=True
        ):
            dst.as_parameters(
                positions=all_positions,  # TODO: simplify: remove unused positions
                edges=cls._reassemble_edges(
                    edges=edges,
                    transition_indices=np.arange(interpolated_index_0, interpolate_index_1),
                    prepend=prepend,
                    append=append,
                    insertion_indices=np.zeros((0,), dtype=np.int32),
                    insertions=np.zeros((0,), dtype=np.int32)
                )
            )
        return self
        #return tuple(
        #    cls(
        #        positions=all_positions,
        #        edges=cls._reassemble_edges(
        #            edges=edges,
        #            transition_indices=np.arange(interpolated_index_0, interpolate_index_1),
        #            prepend=prepend,
        #            append=append,
        #            insertion_indices=np.zeros((0,), dtype=np.int32),
        #            insertions=np.zeros((0,), dtype=np.int32)
        #        )
        #    )
        #    for (prepend, append), (interpolated_index_0, interpolate_index_1) in zip(
        #        itertools.pairwise(np.array((edges[0, 0], *(np.arange(len(alphas)) + len(positions)), edges[-1, 1]))),
        #        itertools.pairwise(np.array((0, *interpolated_indices, len(edges) - 1))),
        #        strict=True
        #    )
        #)  # TODO: simplify: remove unused positions

    def concatenate(
        self: Self,
        #dst: Self,
        srcs: tuple[Self, ...]
    ) -> Self:
        if not srcs:
            return self.as_empty()

        offsets = np.insert(np.cumsum([
            len(graph._positions_)
            for graph in srcs[:-1]
        ], dtype=np.int32), 0, 0)
        return self.as_parameters(
            positions=np.concatenate([
                graph._positions_
                for graph in srcs
            ]),
            edges=np.concatenate([
                graph._edges_ + offset
                for graph, offset in zip(srcs, offsets, strict=True)
            ])
        )


class DynamicGraph[GraphT: Graph](GraphActions, DynamicAnimatable[GraphT]):
    __slots__ = ()


class GraphInterpolateAnimation[GraphT: Graph](AnimatableInterpolateAnimation[GraphT]):
    __slots__ = ()

    def __init__(
        self: Self,
        dst: GraphT,
        src_0: GraphT,
        src_1: GraphT
    ) -> None:
        super().__init__(dst, src_0, src_1)
        self._graph_0_ = src_0.copy()
        self._graph_1_ = src_1.copy()

    @Lazy.variable()
    @staticmethod
    def _graph_0_() -> GraphT:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _graph_1_() -> GraphT:
        return NotImplemented

    @Lazy.property()
    @staticmethod
    def _interpolate_info_(
        graph_0: GraphT,
        graph_1: GraphT
    ) -> tuple[NP_x3f8, NP_x3f8, NP_x2i4]:
        return Graph._general_interpolate(
            graph_0=graph_0,
            graph_1=graph_1,
            disjoints_0=np.zeros((0,), dtype=np.int32),
            disjoints_1=np.zeros((0,), dtype=np.int32)
        )

    def interpolate(
        self: Self,
        dst: GraphT,
        alpha: float
    ) -> None:
        positions_0, positions_1, edges = self._interpolate_info_
        dst.as_parameters(
            positions=SpaceUtils.lerp(positions_0, positions_1, alpha),
            edges=edges
        )

    def becomes(
        self: Self,
        dst: GraphT,
        src: GraphT
    ) -> None:
        dst.as_parameters(
            positions=src._positions_,
            edges=src._edges_
        )


class GraphPiecewiseAnimation[GraphT: Graph](AnimatablePiecewiseAnimation[GraphT]):
    __slots__ = ()

    @classmethod
    def split(
        cls: type[Self],
        dsts: tuple[GraphT, ...],
        src: GraphT,
        alphas: NP_xf8
    ) -> None:
        src.split(dsts, alphas)

    @classmethod
    def concatenate(
        cls: type[Self],
        dst: GraphT,
        srcs: tuple[GraphT, ...]
    ) -> None:
        dst.concatenate(srcs)


#class GraphInterpolateInfo(AnimatableInterpolateInfo[Graph]):
#    __slots__ = (
#        "_positions_0",
#        "_positions_1",
#        "_edges"
#    )
#
#    def __init__(
#        self: Self,
#        src_0: Graph,
#        src_1: Graph
#    ) -> None:
#        super().__init__()
#        positions_0, positions_1, edges = Graph._general_interpolate(
#            graph_0=src_0,
#            graph_1=src_1,
#            disjoints_0=np.zeros((0,), dtype=np.int32),
#            disjoints_1=np.zeros((0,), dtype=np.int32)
#        )
#        self._positions_0: NP_x3f8 = positions_0
#        self._positions_1: NP_x3f8 = positions_1
#        self._edges: NP_x2i4 = edges
#
#    def interpolate(
#        self: Self,
#        dst: Graph,
#        alpha: float
#    ) -> None:
#        dst._positions_ = SpaceUtils.lerp(self._positions_0, self._positions_1, alpha)
#        dst._edges_ = self._edges


#class GraphPartialUpdater(Updater[Graph]):
#    __slots__ = (
#        "_graph",
#        "_original_graph",
#        "_alpha_to_segments"
#    )

#    def __init__(
#        self,
#        graph: Graph,
#        original_graph: Graph,
#        alpha_to_segments: Callable[[float], tuple[NP_xf8, list[int]]]
#    ) -> None:
#        super().__init__(graph)
#        self._original_graph: Graph = original_graph
#        self._alpha_to_segments: Callable[[float], tuple[NP_xf8, list[int]]] = alpha_to_segments

#    def update(
#        self,
#        alpha: float
#    ) -> None:
#        split_alphas, concatenate_indices = self._alpha_to_segments(alpha)
#        graphs = Graph._split(self._original_graph, split_alphas)
#        graph = Graph._concatenate([graphs[index] for index in concatenate_indices])
#        Graph._copy_lazy_content(self._instance, graph)
#        #mobjects = [equivalent_cls() for _ in range(len(split_alphas) + 1)]
#        #equivalent_cls._split_into(
#        #    dst_mobject_list=mobjects,
#        #    src_mobject=original_mobject,
#        #    alphas=split_alphas
#        #)
#        #equivalent_cls._concatenate_into(
#        #    dst_mobject=mobject,
#        #    src_mobject_list=[mobjects[index] for index in concatenate_indices]
#        #)


##class GraphInterpolateHandler(InterpolateHandler[Graph]):
##    __slots__ = (
##        "_positions_0",
##        "_positions_1",
##        "_edges"
##    )

##    def __init__(
##        self,
##        positions_0: NP_x3f8,
##        positions_1: NP_x3f8,
##        edges: NP_x2i4
##    ) -> None:
##        super().__init__()
##        self._positions_0: NP_x3f8 = positions_0
##        self._positions_1: NP_x3f8 = positions_1
##        self._edges: NP_x2i4 = edges

##    def _interpolate(
##        self,
##        alpha: float
##    ) -> Graph:
##        return Graph(
##            positions=SpaceUtils.lerp(self._positions_0, self._positions_1, alpha),
##            edges=self._edges
##        )
