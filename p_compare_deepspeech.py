from algoperf.workloads.librispeech_deepspeech.librispeech_jax import workload as deepspeech_workload_jit
from algoperf.workloads.librispeech_deepspeech.librispeech_jax import old_workload as deepspeech_workload_pmap

import jax
import jax.numpy as jnp
import numpy as np
from algoperf import jax_sharding_utils
import pprint

data_dir = "/data/librispeech"
tokenizer_path = "/data/librispeech/spm_model.vocab"

batch_size = 8


def print_shape(pytree):
    shape = jax.tree.map(lambda x: x.shape, pytree)
    pprint.pprint(shape)


def allclose(a, b):
    return jax.tree.map(lambda x, y: jnp.allclose(x, y, atol=1e-2, rtol=1e-2), a, b)


def shard_batch_for_jit(x):
    return jax.tree.map(jax.device_put(x, jax_sharding_utils.get_batch_dim_sharding()))


def shard_batch_for_pmap(x):
    return x.reshape((jax.local_device_count(), -1, *x.shape[1:]))


print("=" * 20, " Starting ", "=" * 20)
# rngs
# initialize
seed = 0
rng = jax.random.key(seed)
rng, data_rng, model_rng = jax.random.split(rng, 3)

jit_workload = deepspeech_workload_jit.LibriSpeechDeepSpeechWorkload(
    tokenizer_vocab_path=tokenizer_path
) # <- this one uses shard_map for the .model_fn
pmap_workload = deepspeech_workload_pmap.LibriSpeechDeepSpeechWorkload(
    tokenizer_vocab_path=tokenizer_path
)


# input pipeline
input_pipeline = jit_workload._build_input_queue(
    data_rng, split="train", data_dir=data_dir, global_batch_size=batch_size
)

x = next(input_pipeline)

# initialize
params, model_state = jit_workload.init_model_fn(model_rng)
_, _ = pmap_workload.init_model_fn(rng)

# forward pass
rng, step_rng = jax.random.split(rng)


def run_workload(workload):


    def _loss_fn(params):
        logits, new_model_state = workload.model_fn(
            params, x, model_state, "train", rng, update_batch_norm=True
        )
        print(f"logits={logits.flatten()[:10]}")
        loss_dict = workload.loss_fn(
            label_batch=x["targets"], logits_batch=logits, mask_batch=x.get("weights")
        )
        summed_loss = loss_dict["summed"]
        n_valid_examples = loss_dict["n_valid_examples"]
        return summed_loss, (n_valid_examples, new_model_state)

    # y = workload.model_fn(params, x, model_state, "train", rng, update_batch_norm=True)
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(params)

    loss = summed_loss / n_valid_examples
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))
    print(jax.tree.map(lambda g: jnp.linalg.norm(g), grad))

    print(f"loss {loss:.5g}, grad_norm {grad_norm:.5e}")

    return loss, grad_norm

print("Run pmap_workload")
run_workload(pmap_workload)

print("Run jit workload")
run_workload(jit_workload)
