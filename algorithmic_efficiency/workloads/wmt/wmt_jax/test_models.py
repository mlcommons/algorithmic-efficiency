"""Tests for WMT model sharding functionality."""

import jax
import jax.numpy as jnp
import numpy as np

from algorithmic_efficiency.workloads.wmt.wmt_jax import models
from algorithmic_efficiency.workloads.wmt.wmt_jax.workload import WmtWorkload
from algorithmic_efficiency.workloads.wmt import tokenizer

from algorithmic_efficiency.sharding_utils import get_mesh, get_naive_sharding, get_replicated_sharding
from algorithmic_efficiency import sharding_utils


def test_eval_step_sharding():
    # Initialize model and workload
    batch_size = 8
    seq_len = 32
    vocab_size = 1000
    
    # Create random input data
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    inputs = jax.random.randint(
        rng, (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32)
    targets = jax.random.randint(
        rng, (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32)
    
    # Initialize workload with sentencepiece tokenizer
    workload = WmtWorkload()
    workload._tokenizer = tokenizer.load_tokenizer('/home/ak4605/data/wmt/wmt_sentencepiece_model')
    params, _ = workload.init_model_fn(init_rng)

    # Initialize cache
    cache = workload.initialize_cache(inputs, 256)

    # Create input batch
    batch = {
        'inputs': inputs,
        'weights': jnp.ones_like(targets, dtype=jnp.float32),
        'targets': targets,
    }

    # Test eval step sharding
    mesh = get_mesh()
    with mesh:
        # Shard the batch
        sharded_batch = jax.tree_map(
            lambda x: jax.device_put(x, get_naive_sharding(x)), batch)
        
        # Replicate params
        sharded_params = jax.tree_map(
            lambda x: jax.device_put(x, get_replicated_sharding()), params)
        
        # Run eval step
        metrics = workload.eval_step(sharded_params, sharded_batch)

        print(metrics)
        
        # Verify metrics are replicated
        assert metrics['loss'].sharding.is_fully_replicated
        assert metrics['accuracy'].sharding.is_fully_replicated
        assert metrics['denominator'].sharding.is_fully_replicated


def test_predict_step_sharding():
    # Initialize model and workload  
    batch_size = 8
    workload = WmtWorkload()
    seq_len = 16
    vocab_size = 1000
    beam_size = 4
    max_decode_len = 32
    
    # Create random input data
    rng = jax.random.PRNGKey(1)
    rng, init_rng = jax.random.split(rng)
    
    inputs = jax.random.randint(
        rng, (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32)
    
    # Initialize workload
    workload = WmtWorkload()
    params, _ = workload.init_model_fn(init_rng)
    
    # Initialize cache
    cache = workload.initialize_cache(inputs, max_decode_len)

    mesh = get_mesh()
    with mesh:
        # Shard inputs
        sharded_inputs = jax.device_put(inputs, get_naive_sharding(inputs))
        
        # Replicate params and cache
        sharded_params = jax.tree_map(
            lambda x: jax.device_put(x, get_replicated_sharding()), params)
        sharded_cache = jax.tree_map(
            lambda x: jax.device_put(x, get_naive_sharding(x)), cache)
        
        # Create jitted predict step
        jitted_predict_step = jax.jit(
            workload.predict_step,
            in_shardings=(
                sharding_utils.get_naive_sharding_spec(), # inputs
                sharding_utils.get_replicated_sharding(), # params
                sharding_utils.get_naive_sharding_tree(cache), # cache
            ),
            static_argnums=(3, 4, 5) # eos_id, max_decode_len, beam_size
        )
        
        # Run predict step
        predictions = jitted_predict_step(
            sharded_inputs,
            sharded_params, 
            sharded_cache,
            2,
            max_decode_len,
            beam_size)
        
        # Verify predictions are sharded on batch dimension
        assert not predictions.sharding.is_fully_replicated
        assert predictions.shape[0] == batch_size


def test_translate_and_calculate_bleu():
    # Initialize model and workload
    batch_size = 8
    seq_len = 16
    vocab_size = 1000
    max_decode_len = 32
    num_batches = 2
    
    # Create random input data
    rng = jax.random.PRNGKey(2)
    rng, init_rng = jax.random.split(rng)
    
    # Create fake dataset iterator
    def fake_ds_iter():
        for _ in range(num_batches):
            yield {
                'inputs': jax.random.randint(
                    rng, (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32),
                'targets': jax.random.randint(
                    rng, (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32),
                'weights': jnp.ones((batch_size, seq_len), dtype=jnp.float32)
            }
    
    # Initialize workload
    workload = WmtWorkload()
    params, _ = workload.init_model_fn(init_rng)
    workload._tokenizer = tokenizer.load_tokenizer('/home/ak4605/data/wmt/wmt_sentencepiece_model')
    # Test translate_and_calculate_bleu sharding
    mesh = get_mesh()
    with mesh:
        # Replicate params
        sharded_params = jax.tree_map(
            lambda x: jax.device_put(x, get_replicated_sharding()), params)
        
        # Run translation
        bleu_score = workload.translate_and_calculate_bleu(
            sharded_params,
            fake_ds_iter(),
            num_batches,
            max_decode_len)
        
        # Verify we get a valid BLEU score
        assert isinstance(bleu_score, float)
        assert 0 <= bleu_score <= 100

def run_tests():
    test_predict_step_sharding()
    print("Predict step sharding test passed!")
    test_eval_step_sharding() 
    print("Eval step sharding test passed!")
    test_translate_and_calculate_bleu()
    print("Translate and calculate BLEU test passed!")
    print("All tests passed!")


if __name__ == '__main__':
    run_tests()
