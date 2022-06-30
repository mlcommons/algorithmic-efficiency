"""Pytorch implementation of DLRM -Small. """

import functools
from typing import Sequence
import math

import numpy as np
import torch
import torch.nn as nn
from absl import logging



def dot_interact(concat_features, device):
    """ Performs feature interaction operation between desnse and sparse features.
    Inpute tensors represent dense or sparse features.
    """
    batch_size = concat_features.shape[0]
    #print(f"BZ inside dotinteraction: {batch_size}")
    #print(f"shape inside dotinteraction: {concat_features.shape}")

    xactions = torch.bmm(concat_features, torch.permute(concat_features, (0, 2, 1)))
    #xactions = torch.matmul(concat_features, torch.permute(concat_features, (0, 2, 1)))

    feature_dim = xactions.shape[-1]

    indices = torch.triu_indices(feature_dim, feature_dim, device=device)

    num_elems = indices.shape[1]

    indices= torch.tile(indices, [1, batch_size])

    indices0 = torch.reshape(torch.tile(torch.reshape(torch.arange(batch_size, device=device), [-1, 1]), [1, num_elems]), 
        [1, -1])

    #print(f"tensors located in: {indices0.device}")
    indices = tuple(torch.cat((indices0, indices), 0))

    #print(f"tensors located in: {indices[0].device}")


    activations = xactions[indices]

    activations = torch.reshape(activations, [batch_size, -1])

    return activations




class DlrmSmall(nn.Module):
    """
    Define a DLRM-Small model.

    vocab_size: list of vocab sizes of embedding tables.
    total_vocab_sizes: sum of embedding table sizes
    mlp_bottom_dims: dims of dense layers of the bottom mlp.
    mlp_top_dims: dims of dense laters of the top mlp. 
    num_dense_features: number of dense features as the bottom mlp input.
    embded_dim: embedding dimension.
    keep_diags: whether to keep the diagonal terms in x @ x.T
    """


    def __init__(self, vocab_sizes, total_vocab_sizes, num_dense_features=13, num_sparse_features=26, mlp_bottom_dims=(512, 256, 128), mlp_top_dims=(1024, 1024, 512, 256, 1), embed_dim=128, device="cuda"):
        super(DlrmSmall, self).__init__()
        self.vocab_sizes = vocab_sizes
        self.total_vocab_sizes = total_vocab_sizes
        self.num_dense_features = num_dense_features
        self.num_sparse_features = num_sparse_features
        self.mlp_bottom_dims = mlp_bottom_dims
        self.mlp_top_dims = mlp_top_dims
        self.embed_dim = embed_dim
        self.device = device

        bottom_mlp_layers = []
        input_dim = self.num_dense_features
        for dense_dim in self.mlp_bottom_dims:
            bottom_mlp_layers.append(nn.Linear(input_dim, dense_dim))
            bottom_mlp_layers.append(nn.ReLU(inplace = True))
            input_dim = dense_dim
        
        self.bot_mlp = nn.Sequential(*bottom_mlp_layers).to(device)
        #self.bot_mlp = nn.Sequential(*bottom_mlp_layers)

        for module in self.bot_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0., math.sqrt(2. / (module.in_features + module.out_features)))
                nn.init.normal_(module.bias.data, 0., math.sqrt(1. /  module.out_features))
        
        


    def _make_top_mlp(self, input_dim):

        input_dims = input_dim
        top_mlp_layers = []
        for output_dims in self.mlp_top_dims[:-1]:
            top_mlp_layers.append(nn.Linear(input_dims, output_dims))
            top_mlp_layers.append(nn.ReLU(inplace=True))
            input_dims = output_dims

        top_mlp_layers.append(nn.Linear(input_dims, self.mlp_top_dims[-1]))
        #self.top_mlp = nn.Sequential(*top_mlp_layers)
        self.top_mlp = nn.Sequential(*top_mlp_layers).to(self.device)

        for module in self.top_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0., math.sqrt(2. / (module.in_features + module.out_features)))
                nn.init.normal_(module.bias.data, 0., math.sqrt(1. /  module.out_features))





    def forward(self, x, train):
        del train

        #bot_mlp_input, cat_features = torch.split(x, [self.num_dense_features, self.num_sparse_features], 1)
        bot_mlp_input, cat_features = x

        cat_features =  cat_features.to(dtype=torch.int32)

        bot_mlp_output = self.bot_mlp(bot_mlp_input)

        batch_size = bot_mlp_output.shape[0]

        feature_stack = torch.reshape(bot_mlp_output, [batch_size, -1, self.embed_dim])



        # Embedding table lookup
        #vocab_sizes = torch.asarray(self.vocab_sizes, dtype=torch.int32)
        vocab_sizes = torch.asarray(self.vocab_sizes, dtype=torch.int32, device=self.device)


        idx_offsets = torch.asarray(
                [0] + list(torch.cumsum(vocab_sizes[:-1], dim=0)), dtype=torch.int32, device=self.device)

        idx_offsets = torch.tile(torch.reshape(idx_offsets, [1, -1]), [batch_size, 1])

        idx_lookup = cat_features + idx_offsets


        # Scale the initialization to fan_in for each slice.
        scale = 1.0 / torch.sqrt(vocab_sizes)
        
        #TODO(rakshithvasuev): Verify correctness with jax version
        # scale = jnp.expand_dims(
        #jnp.repeat(
        #    scale, vocab_sizes, total_repeat_length=self.total_vocab_sizes),
        #-1)
        scale = torch.unsqueeze(
            torch.repeat_interleave(
                scale, vocab_sizes, output_size=self.total_vocab_sizes), dim=-1)
        
        


        self.embedding_table = nn.Embedding(self.total_vocab_sizes, self.embed_dim, device=self.device)
        self.embedding_table.weight.data.uniform_(0, 1)
        self.embedding_table.weight.data = scale * self.embedding_table.weight.data
        
        idx_lookup = torch.reshape(idx_lookup, [-1])

        embed_features = self.embedding_table(idx_lookup)
        embed_features = torch.reshape(embed_features, [batch_size, -1, self.embed_dim])

        feature_stack = torch.cat([feature_stack, embed_features], axis = 1)

        dot_interact_output = dot_interact(concat_features=feature_stack, device=self.device)

        top_mlp_input = torch.cat([bot_mlp_output, dot_interact_output], axis=-1)
        
        mlp_input_dim = top_mlp_input.shape[1]
        #print(mlp_input_dim)
        #mlp_top_dims = self.top_mlp

        self._make_top_mlp(mlp_input_dim)

        #num_layers_top = len(mlp_top_dims)


        logits = self.top_mlp(top_mlp_input)
        return logits





            



        




        
        





if __name__=="__main__":
    

    #device= "cpu"
    device= "cuda"
    vocab_sizes = [1024 * 128] * 26
    total_vocab_sizes = sum(vocab_sizes)
    mlp_bottom_dims = (512, 256, 128)
    mlp_top_dims= (1024, 1024, 512, 256, 1) 

    np.random.seed(12)
    torch.manual_seed(12)


    dlrm = DlrmSmall(vocab_sizes, total_vocab_sizes, 13, 26,  mlp_bottom_dims, mlp_top_dims, 128, device)

    batch_shape = (1000, )
    targets = np.ones(batch_shape)
    targets[0] = 0
    cat_inputs = np.random.rand(batch_shape[0], 26)
    dense_inputs = np.random.rand(batch_shape[0], 13)
    fake_batch = {
        #'inputs': torch.rand(size=(*batch_shape, 13 + 26)),
        #'inputs': torch.FloatTensor(size=(*batch_shape, 13 + 26)).uniform_(200, 500),
        #'cat_inputs': torch.FloatTensor(size=(*batch_shape,  26)).uniform_(200, 10284 * 128).to(device),
        'cat_inputs': torch.FloatTensor(size=(*batch_shape,  26)).uniform_(200, 10284 * 128).to(device),
        'dense_inputs': torch.FloatTensor(size=(*batch_shape, 13)).uniform_(200, 1024).to(device),
        'targets': torch.from_numpy(targets).to(device),
        'weights': torch.randn(batch_shape)
    }


    fake_batch['cat_inputs'] = torch.clamp(fake_batch['cat_inputs'], min=0, max=1024 * 128)


    fake_batch['inputs'] = torch.cat([fake_batch['dense_inputs'], fake_batch['cat_inputs']], dim = 1)
   


    print(dlrm(fake_batch['inputs'],fake_batch['targets']))






