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
    'librispeech_conformer',
    'fastmri'
]
GLOBAL_BATCH_SIZE = 16


class ReferenceSubmissionTest(absltest.TestCase):

  def test_workload(self):
    # pylint: disable=line-too-long, unnecessary-lambda-assignment
    for workload in WORKLOADS:
      name = f'Testing {workload}'
      jax_logs = '/tmp/jax_log.pkl'
      pyt_logs = '/tmp/pyt_log.pkl'
      run(
          f'python3 tests/modeldiffs/train_diff.py --workload={workload} --framework=jax --global_batch_size={GLOBAL_BATCH_SIZE} --log_file={jax_logs}',
          shell=True,
          stdout=DEVNULL,
          stderr=STDOUT,
          check=True)
      run(
          f'torchrun --standalone --nnodes 1 --nproc_per_node 8  tests/modeldiffs/train_diff.py --workload={workload} --framework=pytorch --global_batch_size={GLOBAL_BATCH_SIZE} --log_file={pyt_logs}',
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
          'Grad Norm (torch)'
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
                      pyt_results['scalars'][i]['grad_norm']
                  ])

        print(fmt([f'{i}', *row]))

      print('=' * len(header))


if __name__ == '__main__':
  absltest.main()
