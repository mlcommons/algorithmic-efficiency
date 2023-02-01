"""
Runs 10 steps of SGD for each workload and compares results.
Run it as:
  python3 test_traindiffs.py
"""
import pickle
from subprocess import DEVNULL
from subprocess import run
from subprocess import STDOUT

from absl import flags
from absl.testing import absltest
import jax

jax.config.update('jax_platforms', 'cpu')

FLAGS = flags.FLAGS

WORKLOADS = [
    'imagenet_resnet',
    'imagenet_vit',
    'wmt',
    'librispeech_conformer',
    'librispeech_deepspeech',
    'fastmri',
    'ogbg',
]
GLOBAL_BATCH_SIZE = 16


class ModelDiffTest(absltest.TestCase):

  def test_workload(self):
    """
    Compares the multi-gpu jax and ddp-pytorch models for each workload and compares the train and eval metrics collected 
    in the corresponding log files. We launch the multi-gpu jax model and the corresponding ddp-pytorch model separately
    using subprocess because ddp-pytorch models are run using torchrun. Secondly, keeping these separate helps avoid
    CUDA OOM errors resulting from the two frameworks competing with each other for GPU memory.
    """
    # pylint: disable=line-too-long, unnecessary-lambda-assignment
    for workload in WORKLOADS:
      name = f'Testing {workload}'
      jax_logs = '/tmp/jax_log.pkl'
      pyt_logs = '/tmp/pyt_log.pkl'
      run(f'XLA_PYTHON_CLIENT_ALLOCATOR=platform python3 tests/reference_algorithm_tests.py --workload={workload} --framework=jax --global_batch_size={GLOBAL_BATCH_SIZE} --log_file={jax_logs}'
      ' --submission_path=tests/modeldiffs/vanilla_sgd_jax.py --identical=True --tuning_search_space=None --num_train_steps=10',
          shell=True,
          stdout=DEVNULL,
          stderr=STDOUT,
          check=True)
      run(f'XLA_PYTHON_CLIENT_ALLOCATOR=platform torchrun --standalone --nnodes 1 --nproc_per_node 8  tests/reference_algorithm_tests.py --workload={workload} --framework=pytorch --global_batch_size={GLOBAL_BATCH_SIZE} --log_file={pyt_logs}'
      ' --submission_path=tests/modeldiffs/vanilla_sgd_pytorch.py --identical=True --tuning_search_space=None --num_train_steps=10',
          shell=True,
          stdout=DEVNULL,
          stderr=STDOUT,
          check=True)
      with open(jax_logs, 'rb') as f:
        jax_results = pickle.load(f)
      with open(pyt_logs, 'rb') as f:
        pyt_results = pickle.load(f)

      # PRINT RESULTS
      k = next(
          iter(
              filter(lambda k: 'train' in k and 'loss' in k,
                     jax_results['eval_results'][0])))
      header = [
          'Iter',
          'Eval (jax)',
          'Eval (torch)',
          'Grad Norm (jax)',
          'Grad Norm (torch)',
          'Train Loss (jax)',
          'Train Loss (torch)'
      ]
      fmt = lambda l: '|' + '|'.join(map(lambda x: f'{x:^20s}', l)) + '|'
      header = fmt(header)
      pad = (len(header) - len((name))) // 2
      print("=" * pad, name, '=' * (len(header) - len(name) - pad), sep='')
      print(header)
      print('=' * len(header))
      for i in range(10):
        row = map(lambda x: str(round(x, 5)),
                  [
                      jax_results['eval_results'][i][k],
                      pyt_results['eval_results'][i][k],
                      jax_results['scalars'][i]['grad_norm'],
                      pyt_results['scalars'][i]['grad_norm'],
                      jax_results['scalars'][i]['loss'],
                      pyt_results['scalars'][i]['loss'],
                  ])

        print(fmt([f'{i}', *row]))

      print('=' * len(header))


if __name__ == '__main__':
  absltest.main()
