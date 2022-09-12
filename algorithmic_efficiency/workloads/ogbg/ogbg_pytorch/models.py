# Ported to PyTorch from
# https://github.com/google/init2winit/blob/master/init2winit/model_lib/gnn.py.
from typing import Callable, Optional, Tuple

import jax.tree_util as tree
from jraph import GraphsTuple
import torch
from torch import nn


def _make_mlp(in_dim, hidden_dims, dropout_rate):
    """Creates a MLP with specified dimensions."""
    layers = []
    for dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(in_features=in_dim, out_features=dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
        )
    return nn.Sequential(*layers)


class GNN(nn.Module):
    """Defines a graph network.
    The model assumes the input data is a jraph.GraphsTuple without global
    variables. The final prediction will be encoded in the globals.
    """

    latent_dim: int = 256
    hidden_dims: Tuple[int] = (256,)
    dropout_rate: float = 0.1
    num_message_passing_steps: int = 5

    def __init__(self, num_outputs: int = 128) -> None:
        super().__init__()
        self.num_outputs = num_outputs
        # in_features are specifically chosen for the ogbg workload.
        self.node_embedder = nn.Linear(in_features=9, out_features=self.latent_dim)
        self.edge_embedder = nn.Linear(in_features=3, out_features=self.latent_dim)

        graph_network_layers = []
        for st in range(self.num_message_passing_steps):
            # Constants in in_dims are based on the requirements of the GraphNetwork.
            if st == 0:
                in_dim = self.latent_dim * 3 + self.num_outputs
                last_in_dim = self.latent_dim * 2 + self.num_outputs
            else:
                in_dim = self.hidden_dims[-1] * 4
                last_in_dim = self.hidden_dims[-1] * 3

            graph_network_layers.append(
                GraphNetwork(
                    update_edge_fn=_make_mlp(
                        in_dim, self.hidden_dims, self.dropout_rate
                    ),
                    update_node_fn=_make_mlp(
                        in_dim, self.hidden_dims, self.dropout_rate
                    ),
                    update_global_fn=_make_mlp(
                        last_in_dim, self.hidden_dims, self.dropout_rate
                    ),
                )
            )
        self.graph_network = nn.Sequential(*graph_network_layers)

        self.decoder = nn.Linear(
            in_features=self.hidden_dims[-1], out_features=self.num_outputs
        )

    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        graph = graph._replace(
            globals=torch.zeros(
                [graph.n_node.shape[0], self.num_outputs], device=graph.n_node.device
            )
        )
        graph = graph._replace(nodes=self.node_embedder(graph.nodes))
        graph = graph._replace(edges=self.edge_embedder(graph.edges))

        graph = self.graph_network(graph)

        # Map globals to represent the final result
        graph = graph._replace(globals=self.decoder(graph.globals))

        return graph.globals


class GraphNetwork(nn.Module):
    """Returns a method that applies a configured GraphNetwork.
    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261
    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    than the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.
    In addition this implementation supports softmax attention over incoming
    edge features.
    Example usage::
      gn = GraphNetwork(update_edge_function,
      update_node_function, **kwargs)
      # Conduct multiple rounds of message passing with the same parameters:
      for _ in range(num_message_passing_steps):
        graph = gn(graph)
    Args:
      update_edge_fn: function used to update the edges or None to deactivate edge
        updates.
      update_node_fn: function used to update the nodes or None to deactivate node
        updates.
      update_global_fn: function used to update the globals or None to deactivate
        globals updates.
    Returns:
      A method that applies the configured GraphNetwork.
    """

    def __init__(
        self,
        update_edge_fn: Optional[Callable] = None,
        update_node_fn: Optional[Callable] = None,
        update_global_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.update_edge_fn = update_edge_fn
        self.update_node_fn = update_node_fn
        self.update_global_fn = update_global_fn

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """Applies a configured GraphNetwork to a graph.
        This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261
        There is one difference. For the nodes update the class aggregates over the
        sender edges and receiver edges separately. This is a bit more general
        the algorithm described in the paper. The original behaviour can be
        recovered by using only the receiver edge aggregations for the update.
        In addition this implementation supports softmax attention over incoming
        edge features.
        Many popular Graph Neural Networks can be implemented as special cases of
        GraphNets, for more information please see the paper.
        Args:
          graph: a `GraphsTuple` containing the graph.
        Returns:
          Updated `GraphsTuple`.
        """
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
        if not tree.tree_all(tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
            raise ValueError(
                "All node arrays in nest must contain the same number of nodes."
            )

        sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
        received_attributes = tree.tree_map(lambda n: n[receivers], nodes)
        # Here we scatter the global features to the corresponding edges,
        # giving us tensors of shape [num_edges, global_feat].
        global_edge_attributes = tree.tree_map(
            lambda g: torch.repeat_interleave(g, n_edge, dim=0), globals_
        )

        if self.update_edge_fn:
            edge_fn_inputs = torch.cat(
                [edges, sent_attributes, received_attributes, global_edge_attributes],
                dim=-1,
            )
            edges = self.update_edge_fn(edge_fn_inputs)

        if self.update_node_fn:
            sent_attributes = tree.tree_map(
                lambda e: scatter_sum(e, senders, dim=0, dim_size=sum_n_node), edges
            )
            received_attributes = tree.tree_map(
                lambda e: scatter_sum(e, receivers, dim=0, dim_size=sum_n_node), edges
            )
            # Here we scatter the global features to the corresponding nodes,
            # giving us tensors of shape [num_nodes, global_feat].
            global_attributes = tree.tree_map(
                lambda g: torch.repeat_interleave(g, n_node, dim=0), globals_
            )
            node_fn_inputs = torch.cat(
                [nodes, sent_attributes, received_attributes, global_attributes], dim=-1
            )
            nodes = self.update_node_fn(node_fn_inputs)

        if self.update_global_fn:
            n_graph = n_node.shape[0]
            graph_idx = torch.arange(n_graph, device=graph.n_node.device)
            # To aggregate nodes and edges from each graph to global features,
            # we first construct tensors that map the node to the corresponding graph.
            # For example, if you have `n_node=[1,2]`, we construct the tensor
            # [0, 1, 1]. We then do the same for edges.
            node_gr_idx = torch.repeat_interleave(graph_idx, n_node, dim=0)
            edge_gr_idx = torch.repeat_interleave(graph_idx, n_edge, dim=0)
            # We use the aggregation function to pool the nodes/edges per graph.
            node_attributes = tree.tree_map(
                lambda n: scatter_sum(n, node_gr_idx, dim=0, dim_size=n_graph), nodes
            )
            edge_attributes = tree.tree_map(
                lambda e: scatter_sum(e, edge_gr_idx, dim=0, dim_size=n_graph), edges
            )
            # These pooled nodes are the inputs to the global update fn.
            global_fn_inputs = torch.cat(
                [node_attributes, edge_attributes, globals_], dim=-1
            )
            globals_ = self.update_global_fn(global_fn_inputs)

        return GraphsTuple(
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge,
        )


# Forked from
# github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/scatter.py.
def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    r"""
    |
    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px
    |
    Reduces all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`.
    The applied reduction is here defined as a sum.
    Formally, if :attr:`src` and :attr:`index` are :math:`n`-dimensional
    tensors with size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
    and :attr:`dim` = `i`, then :attr:`out` must be an :math:`n`-dimensional
    tensor with size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`.
    Moreover, the values of :attr:`index` must be between :math:`0` and
    :math:`y - 1`, although no specific ordering of indices is required.
    The :attr:`index` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.
    For one-dimensional tensors, the operation computes
    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j
    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.
    .. note::
        This operation is implemented via atomic operations on the GPU and is
        therefore **non-deterministic** since the order of parallel operations
        to the same value is undetermined.
        For floating-point variables, this results in a source of variance in
        the result.
    :param src: The source tensor.
    :param index: The indices of elements to scatter.
    :param dim: The axis along which to index. (default: :obj:`-1`)
    :param out: The destination tensor.
    :param dim_size: If :attr:`out` is not given, automatically create output
        with size :attr:`dim_size` at dimension :attr:`dim`.
        If :attr:`dim_size` is not given, a minimal sized output tensor
        according to :obj:`index.max() + 1` is returned.
    :rtype: :class:`Tensor`
    .. code-block:: python
        src = torch.randn(10, 6, 64)
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        # Broadcasting in the first and last dim.
        out = scatter_sum(src, index, dim=1)
        print(out.size())
    .. code-block::
        torch.Size([10, 3, 64])
    """
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


# Forked from
# github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/utils.py.
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src
