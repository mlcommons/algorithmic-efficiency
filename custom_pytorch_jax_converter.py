import torch
import numpy as np
import jax
import jax.numpy as jnp
import logging
import copy
import copy
from jax.tree_util import tree_map


def use_pytorch_weights(file_name: str):
    """
    Jax default parameter structure:
    dict_keys(['Dense_0', 'Dense_1', 'Dense_2', 'Dense_3', 'Dense_4', 'Dense_5', 'Dense_6', 'Dense_7', 'embedding_table'])

    Pytorch stateduct structure:
    dict_keys(['embedding_chunk_0', 'embedding_chunk_1', 'embedding_chunk_2', 'embedding_chunk_3', 'bot_mlp.0.weight', 'bot_mlp.0.bias', 'bot_mlp.2.weight', 'bot_mlp.2.bias', 'bot_mlp.4.weight', 'bot_mlp.4.bias', 'top_mlp.0.weight', 'top_mlp.0.bias', 'top_mlp.2.weight', 'top_mlp.2.bias', 'top_mlp.4.weight', 'top_mlp.4.bias', 'top_mlp.6.weight', 'top_mlp.6.bias', 'top_mlp.8.weight', 'top_mlp.8.bias'])


    The following function converts the PyTorch weights to the Jax format
    """
    
    jax_copy = {}

    # Load PyTorch state_dict lazily to CPU
    state_dict = torch.load(file_name, map_location='cpu')
    print(state_dict.keys())

    # Convert PyTorch tensors to NumPy arrays
    numpy_weights = {k: v.cpu().numpy() for k, v in state_dict.items()}

    # --- Embedding Table ---
    embedding_table = np.concatenate([
        numpy_weights[f'embedding_chunk_{i}'] for i in range(4)
    ], axis=0)  # adjust axis if chunking is not vertical

    jax_copy['embedding_table'] = jnp.array(embedding_table)

    # --- Bot MLP: Dense_0 to Dense_2 ---
    for i, j in zip([0, 2, 4], range(3)):
        jax_copy[f'Dense_{j}'] = {}
        jax_copy[f'Dense_{j}']['kernel'] = jnp.array(numpy_weights[f'bot_mlp.{i}.weight'].T)
        jax_copy[f'Dense_{j}']['bias'] = jnp.array(numpy_weights[f'bot_mlp.{i}.bias'])

    # --- Top MLP: Dense_3 to Dense_7 ---
    for i, j in zip([0, 2, 4, 6, 8], range(3, 8)):
        jax_copy[f'Dense_{j}'] = {}
        jax_copy[f'Dense_{j}']['kernel'] = jnp.array(numpy_weights[f'top_mlp.{i}.weight'].T)
        jax_copy[f'Dense_{j}']['bias'] = jnp.array(numpy_weights[f'top_mlp.{i}.bias'])

    del state_dict
    return jax_copy


def maybe_unreplicate(pytree):
    """If leading axis matches device count, strip it assuming it's pmap replication."""
    num_devices = jax.device_count()
    return jax.tree_util.tree_map(
        lambda x: x[0] if isinstance(x, jax.Array) and x.shape[0] == num_devices else x,
        pytree
    )


def move_to_cpu(tree):
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, device=jax.devices("cpu")[0]), tree)


def are_weights_equal(params1, params2, atol=1e-6, rtol=1e-6):
    """Compares two JAX PyTrees of weights and logs where they differ, safely handling PMAP replication."""
    # Attempt to unreplicate if needed

    params1 = maybe_unreplicate(params1)
    params2 = maybe_unreplicate(params2)

    params1 = move_to_cpu(params1)
    params2 = move_to_cpu(params2)

    all_equal = True

    def compare_fn(p1, p2):
        nonlocal all_equal
        if not jnp.allclose(p1, p2, atol=atol, rtol=rtol):
            logging.info("❌ Mismatch found:")
            logging.info(f"Shape : {p1.shape}, Shape 2: {p2.shape}")
            logging.info(f"Max diff: {jnp.max(jnp.abs(p1 - p2))}")
            all_equal = False
        return jnp.allclose(p1, p2, atol=atol, rtol=rtol)

    try:
        jax.tree_util.tree_map(compare_fn, params1, params2)
    except Exception as e:
        logging.info("❌ Structure mismatch or error during comparison:", exc_info=True)
        return False

    if all_equal:
        logging.info("✅ All weights are equal (within tolerance)")
    del params1
    del params2
    return all_equal