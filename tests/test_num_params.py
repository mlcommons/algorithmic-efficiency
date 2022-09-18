import jax
import torch
import jax.numpy as jnp
import jax.random as jax_rng
import jraph
import pytest

from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.models import \
    ResNet18 as JaxResNet_c10
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.models import \
    ResNet50 as JaxResNet
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.models import \
    resnet18 as PyTorchResNet_c10
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.models import \
    resnet50 as PyTorchResNet
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_jax.models import \
    ViT as JaxViT
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_pytorch.models import \
    ViT as PyTorchViT
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.models import \
    Conformer as JaxConformer
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.models import \
    ConformerConfig as JaxConformerConfig
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.model import \
    ConformerConfig as PytorchConformerConfig
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.model import \
    ConformerEncoderDecoder as PytorchConformer
from algorithmic_efficiency.workloads.mnist.mnist_jax.workload import \
    _Model as JaxMLP
from algorithmic_efficiency.workloads.mnist.mnist_pytorch.workload import \
    _Model as PyTorchMLP
from algorithmic_efficiency.workloads.ogbg.ogbg_jax.models import GNN as JaxGNN
from algorithmic_efficiency.workloads.ogbg.ogbg_pytorch.models import \
    GNN as PyTorchGNN
from algorithmic_efficiency.workloads.wmt.wmt_jax.models import \
    Transformer as JaxTransformer
from algorithmic_efficiency.workloads.wmt.wmt_jax.models import \
    TransformerConfig
from algorithmic_efficiency.workloads.wmt.wmt_pytorch.models import \
    Transformer as PyTorchTransformer

WORKLOADS = ['mnist', 'cifar', 'imagenet_resnet', 'imagenet_vit', 'wmt', 'ogbg', 'librispeech_conformer']


@pytest.mark.parametrize('workload', WORKLOADS)
def test_matching_num_params(workload):
  jax_model, pytorch_model = get_models(workload)
  # Count parameters of both models.
  num_jax_params = sum(x.size for x in jax.tree_leaves(jax_model))
  num_pytorch_params = sum(
      p.numel() for p in pytorch_model.parameters() if p.requires_grad)
  assert num_jax_params == num_pytorch_params


def get_models(workload):
  init_rngs = {'params': jax_rng.PRNGKey(0), 'dropout': jax_rng.PRNGKey(1)}
  if workload == 'mnist':
    # Init Jax model.
    init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
    jax_model = JaxMLP().init(init_rngs, init_val, train=True)['params']
    # Init PyTorch model.
    pytorch_model = PyTorchMLP()
  elif workload == 'cifar':
    # Init Jax model.
    input_shape = (1, 32, 32, 3)
    model_init = jax.jit(JaxResNet_c10(num_classes=10, dtype=jnp.float32).init)
    jax_model = model_init(init_rngs, jnp.ones(input_shape,
                                               jnp.float32))["params"]
    # Init PyTorch model.
    pytorch_model = PyTorchResNet_c10(num_classes=10)
  elif workload == 'imagenet_resnet':
    # Init Jax model.
    input_shape = (1, 224, 224, 3)
    jax_model = JaxResNet(
        num_classes=1000,
        dtype=jnp.float32).init(init_rngs, jnp.ones(input_shape,
                                                    jnp.float32))['params']
    # Init PyTorch model.
    pytorch_model = PyTorchResNet()
  elif workload == 'imagenet_vit':
    # Init Jax model.
    input_shape = (1, 224, 224, 3)
    jax_model = JaxViT(num_classes=1000).init(
        init_rngs, jnp.ones(input_shape, jnp.float32))['params']
    # Init PyTorch model.
    pytorch_model = PyTorchViT()
  elif workload == 'librispeech_conformer':
    jax_model = JaxConformer(JaxConformerConfig(num_attention_heads=1))
    pytorch_model = PytorchConformer(PytorchConformerConfig(num_attention_heads))
    
    # Init Jax model
    input_shape = [(320000,), (320000,)]
    fake_input_batch = [jnp.zeros((2, *x), jnp.float32) for x in input_shape]
    jax_model = jax_model.init(init_rngs,train=False,
                              *fake_input_batch)["params"]

    # Run model once to initialize lazy layers
    wave = torch.randn(2, 320000)
    pad = torch.zeros_like(wave)
    pytorch_model(wave, pad)
    
  elif workload == 'wmt':
    # Init Jax model.
    input_shape = (16, 256)
    target_shape = (16, 256)
    jax_model = JaxTransformer(TransformerConfig).init(
        init_rngs,
        jnp.ones(input_shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))['params']
    # Init PyTorch model.
    pytorch_model = PyTorchTransformer()
  elif workload == 'ogbg':
    # Init Jax model.
    fake_batch = jraph.GraphsTuple(
        n_node=jnp.asarray([1]),
        n_edge=jnp.asarray([1]),
        nodes=jnp.ones((1, 9)),
        edges=jnp.ones((1, 3)),
        globals=jnp.zeros((1, 128)),
        senders=jnp.asarray([0]),
        receivers=jnp.asarray([0]))
    jax_model = JaxGNN(num_outputs=128).init(
        init_rngs, fake_batch, train=False)['params']
    # Init PyTorch model.
    pytorch_model = PyTorchGNN(num_outputs=128)
  else:
    raise ValueError(f'Models for workload {workload} are not available.')
  return jax_model, pytorch_model
