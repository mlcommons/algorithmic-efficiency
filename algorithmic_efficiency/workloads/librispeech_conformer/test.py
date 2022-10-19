import torch
import jax 
import jax.numpy as jnp 
import jax.random as jax_rng
import flax 
import utils 

from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.models import \
    Conformer as JaxConformer
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.models import \
    ConformerConfig as JaxConformerConfig
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.model import \
    ConformerConfig as PytorchConformerConfig
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.model import \
    ConformerEncoderDecoder as PytorchConformer

jax_model = JaxConformer(JaxConformerConfig())
pytorch_model = PytorchConformer(PytorchConformerConfig())
init_rngs = {'params': jax_rng.PRNGKey(0), 'dropout': jax_rng.PRNGKey(1)}
# Init Jax model
input_shape = [(320000,), (320000,)]
fake_input_batch = [jnp.zeros((2, *x), jnp.float32) for x in input_shape]
jax_model = jax_model.init(
    init_rngs, train=False, *fake_input_batch)["params"]

# Run model once to initialize lazy layers
wave = torch.randn(2, 320000)
pad = torch.zeros_like(wave)
pytorch_model(wave, pad)

def key_transform(k):
    new_key = []
    for i in k:
        if 'ModuleList' in i:
            continue 
        elif 'Linear' in i:
            if 'NonDynamicallyQuantizableLinear' in i:
                i = 'out'
            else:
                i=i.replace('Linear','Dense')
        elif 'Conv1d' in i:
            i=i.replace('Conv1d','Conv')
        elif 'MHSAwithQS' in i:
            i=i.replace('MHSAwithQS','SelfAttention')
        elif 'weight' in i:
            i=i.replace('weight','kernel')
        new_key.append(i)
    return tuple(new_key)

def sd_transform(sd):
    out = {}
    for k in sd:
        if 'Attention' in ''.join(k):
            if 'in_proj' in k[-1]:
                new_key = k[:-1]
                chunks = sd[k].chunk(3)
                for t,c in zip(['query','key','value'],chunks):
                    out[new_key+(t,k[-1].split('_')[-1])] = c
            else:
                out[k] = sd[k]
        else:
            out[k] = sd[k]
    return out



t2j = utils.Torch2Jax(torch_model = pytorch_model, jax_model=jax_model)
t2j.key_transform(key_transform)
t2j.sd_transform(sd_transform)
t2j.diff()
