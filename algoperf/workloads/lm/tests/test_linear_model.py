import jax
import jax.numpy as jnp
import torch

TEST_SEQ_LEN = 512

def test_pytorch_linear():
    from algoperf.workloads.lm.lm_pytorch.models import LinearLayer
    vocab_size = 32000
    model = LinearLayer(vocab_size)
    
    batch_size = 8
    seq_len = TEST_SEQ_LEN
    inputs = torch.randn(batch_size, seq_len, vocab_size)
    outputs = model(inputs)
    
    assert outputs.shape == (batch_size, seq_len, vocab_size)
    assert not torch.isnan(outputs).any()

def test_jax_linear():
    from algoperf.workloads.lm.lm_jax.models import LinearModel

    vocab_size = 32000
    seq_len = TEST_SEQ_LEN
    batch_size = 8
    model = LinearModel(vocab_size)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, seq_len, vocab_size)))
    
    inputs = jax.random.normal(rng, (batch_size, seq_len, vocab_size))
    outputs = model.apply(params, inputs)
    
    assert outputs.shape == (batch_size, seq_len, vocab_size)
    assert not jnp.isnan(outputs).any()

if __name__ == '__main__':
    test_pytorch_linear()
    test_jax_linear()
    print("All tests passed!")
