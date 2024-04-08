import json
import os
import struct

from absl import app
from absl import flags
from absl import logging
import numpy as np

flags.DEFINE_integer(
    'held_out_workloads_seed',
    None,
    'Random seed for scoring.'
    'AlgoPerf v0.5 seed: 3438810845')
flags.DEFINE_string('output_filename',
                    'held_out_workloads.json',
                    'Path to file to record sampled held_out workloads.')
FLAGS = flags.FLAGS

HELD_OUT_WORKLOADS = {
    'librispeech': [
        'librispeech_conformer_attention_temperature',
        'librispeech_conformer_layernorm',
        # 'librispeech_conformer_gelu', # Removed due to bug in target setting procedure
        'librispeech_deepspeech_no_resnet',
        'librispeech_deepspeech_norm_and_spec_aug',
        'librispeech_deepspeech_tanh'
    ],
    'imagenet': [
        'imagenet_resnet_silu',
        'imagenet_resnet_gelu',
        'imagenet_resnet_large_bn_init',
        'imagenet_vit_glu',
        'imagenet_vit_post_ln',
        'imagenet_vit_map'
    ],
    'ogbg': ['ogbg_gelu', 'ogbg_silu', 'ogbg_model_size'],
    'wmt': ['wmt_post_ln', 'wmt_attention_temp', 'wmt_glu_tanh'],
    'fastmri': ['fastmri_model_size', 'fastmri_tanh', 'fastmri_layernorm'],
    'criteo1tb': [
        'criteo1tb_layernorm', 'criteo1tb_embed_init', 'criteo1tb_resnet'
    ]
}


def save_held_out_workloads(held_out_workloads, filename):
  with open(filename, "w") as f:
    json.dump(held_out_workloads, f)


def main(_):
  rng_seed = FLAGS.held_out_workloads_seed
  output_filename = FLAGS.output_filename

  if not rng_seed:
    rng_seed = struct.unpack('I', os.urandom(4))[0]

  logging.info('Using RNG seed %d', rng_seed)
  rng = np.random.default_rng(rng_seed)

  sampled_held_out_workloads = []
  for _, v in HELD_OUT_WORKLOADS.items():
    sampled_index = rng.integers(len(v))
    sampled_held_out_workloads.append(v[sampled_index])

  logging.info(f"Sampled held-out workloads: {sampled_held_out_workloads}")
  save_held_out_workloads(sampled_held_out_workloads, output_filename)


if __name__ == '__main__':
  app.run(main)
