"""
Runs fwd pass with random graphs for OGBG GNN models and compares outputs.
Run with:
  python3 tests/dropout_fix/ogbg_pytorch/test_model_equivalence.py
"""

import os
import random

from absl.testing import absltest
from absl.testing import parameterized
from jraph import GraphsTuple
import numpy as np
import torch
from torch.testing import assert_close

from algoperf.workloads.ogbg.ogbg_pytorch.models import GNN as OriginalModel
from algoperf.workloads.ogbg.ogbg_pytorch.models_dropout import \
    GNN as CustomModel

B, N, E = 8, 20, 40          # graphs, nodes/graph, edges/graph
NODE_FDIM, EDGE_FDIM = 9, 3  # expected feature dims
DEVICE = 'cuda'

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
SEED = 1996


def _rand_graph():
    total_nodes, total_edges = B * N, B * E
    nodes = torch.randn(total_nodes, NODE_FDIM, device=DEVICE)
    edges = torch.randn(total_edges, EDGE_FDIM, device=DEVICE)
    senders, receivers = [], []
    for i in range(B):
        offset = i * N
        s = torch.randint(N, (E,), device=DEVICE) + offset
        r = torch.randint(N, (E,), device=DEVICE) + offset
        senders.append(s), receivers.append(r)
    senders = torch.cat(senders); receivers = torch.cat(receivers)
    n_node = torch.full((B,), N, device=DEVICE, dtype=torch.int32)
    n_edge = torch.full((B,), E, device=DEVICE, dtype=torch.int32)
    return GraphsTuple(nodes, edges, receivers, senders, None, n_node, n_edge)


class GNNEquivalenceTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='0.0', dropout_rate=0.0),
        dict(testcase_name='0.2', dropout_rate=0.2),
        dict(testcase_name='0.7', dropout_rate=0.7),
        dict(testcase_name='1.0', dropout_rate=1.0),
    )
    def test_forward(self, dropout_rate):
        """Test different dropout_rates."""

        orig = OriginalModel(dropout_rate=dropout_rate).to(DEVICE)
        cust = CustomModel().to(DEVICE)
        orig.load_state_dict(cust.state_dict())  # sync weights

        graph = _rand_graph()

        for mode in ('train', 'eval'):
            getattr(orig, mode)()
            getattr(cust, mode)()

            torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
            y1 = orig(graph)

            torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
            y2 = cust(graph, dropout_rate=dropout_rate)

            assert_close(y1, y2, atol=0, rtol=0)

            if mode == 'eval':  # one extra test: omit dropout at eval
                torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
                y2 = cust(graph)
                assert_close(y1, y2, atol=0, rtol=0)


    @parameterized.named_parameters(
      dict(testcase_name=''),
    )
    def test_default_dropout(self):
        """Test default dropout_rate."""
        
        orig = OriginalModel().to(DEVICE)
        cust = CustomModel().to(DEVICE)
        orig.load_state_dict(cust.state_dict())  # sync weights

        graph = _rand_graph()

        for mode in ('train', 'eval'):
            getattr(orig, mode)()
            getattr(cust, mode)()

            torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
            y1 = orig(graph)

            torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
            y2 = cust(graph)

            assert_close(y1, y2, atol=0, rtol=0)


if __name__ == '__main__':
    absltest.main()
