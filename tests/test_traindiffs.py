"""
Runs 10 steps of SGD for each workload and compares results.

See ``docs/CONTRIBUTING.md`` for runtime requirements and commands.
"""

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from subprocess import DEVNULL, STDOUT

from absl import flags
from absl.testing import absltest, parameterized
from numpy import allclose

FLAGS = flags.FLAGS

WORKLOADS = [
  'imagenet_resnet',
  'imagenet_vit',
  'wmt',
  'librispeech_conformer',
  'librispeech_deepspeech',
  'fastmri',
  'ogbg',
  'criteo1tb',
]
GLOBAL_BATCH_SIZE = 16
NUM_TRAIN_STEPS = 10

named_parameters = []
for w in WORKLOADS:
  named_parameters.append(dict(testcase_name=f'{w}', workload=w))


def _run_reference_algorithm(
  workload: str,
  framework: str,
  log_file: Path,
) -> None:
  if framework == 'jax':
    launcher = [sys.executable]
  elif framework == 'pytorch':
    launcher = [
      'torchrun',
      '--standalone',
      '--nnodes',
      '1',
      '--nproc_per_node',
      '8',
    ]
  else:
    raise ValueError(f'Unsupported framework: {framework}')

  command = [
    *launcher,
    '-m',
    'tests.reference_algorithm_tests',
    f'--workload={workload}',
    f'--framework={framework}',
    f'--global_batch_size={GLOBAL_BATCH_SIZE}',
    f'--log_file={log_file}',
    f'--submission_path=tests/modeldiffs/vanilla_sgd_{framework}.py',
    '--identical=True',
    '--tuning_search_space=None',
    f'--num_train_steps={NUM_TRAIN_STEPS}',
  ]
  env = os.environ.copy()
  env['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
  subprocess.run(
    command,
    env=env,
    stdout=DEVNULL,
    stderr=STDOUT,
    check=True,
  )


def _collect_results(workload: str) -> tuple[dict, dict]:
  with tempfile.TemporaryDirectory(prefix=f'traindiff-{workload}-') as temp_dir:
    results_dir = Path(temp_dir)
    jax_logs_path = results_dir / 'jax_log.pkl'
    pytorch_logs_path = results_dir / 'pytorch_log.pkl'

    _run_reference_algorithm(workload, 'jax', jax_logs_path)
    _run_reference_algorithm(workload, 'pytorch', pytorch_logs_path)

    with jax_logs_path.open('rb') as f:
      jax_results = pickle.load(f)
    with pytorch_logs_path.open('rb') as f:
      pytorch_results = pickle.load(f)
  return jax_results, pytorch_results


class ModelDiffTest(parameterized.TestCase):
  @parameterized.named_parameters(*named_parameters)
  def test_workload(self, workload):
    # pylint: disable=line-too-long, unnecessary-lambda-assignment
    """
    Compares the multi-gpu jax and ddp-pytorch models for each workload and compares the train and eval metrics collected
    in the corresponding log files. We launch the multi-gpu jax model and the corresponding ddp-pytorch model separately
    using subprocess because ddp-pytorch models are run using torchrun. Secondly, keeping these separate helps avoid
    CUDA OOM errors resulting from the two frameworks competing with each other for GPU memory.
    """
    name = f'Testing {workload}'
    jax_results, pytorch_results = _collect_results(workload)

    # PRINT RESULTS
    eval_metric_key = next(
      iter(
        filter(
          lambda k: 'train' in k and 'loss' in k, jax_results['eval_results'][0]
        )
      )
    )
    header = [
      'Iter',
      'Eval (jax)',
      'Eval (torch)',
      'Grad Norm (jax)',
      'Grad Norm (torch)',
      'Train Loss (jax)',
      'Train Loss (torch)',
    ]

    def fmt(line):
      return '|' + '|'.join(map(lambda x: f'{x:^20s}', line)) + '|'

    header = fmt(header)
    pad = (len(header) - len((name))) // 2
    print('=' * pad, name, '=' * (len(header) - len(name) - pad), sep='')
    print(header)
    print('=' * len(header))

    for i in range(NUM_TRAIN_STEPS):
      rtol = 1

      row = map(
        lambda x: str(round(x, 5)),
        [
          jax_results['eval_results'][i][eval_metric_key],
          pytorch_results['eval_results'][i][eval_metric_key],
          jax_results['scalars'][i]['grad_norm'],
          pytorch_results['scalars'][i]['grad_norm'],
          jax_results['scalars'][i]['loss'],
          pytorch_results['scalars'][i]['loss'],
        ],
      )
      print(fmt([f'{i}', *row]))
    print('=' * len(header))

    self.assertTrue(  # eval_results
      allclose(
        jax_results['eval_results'][i][eval_metric_key],
        pytorch_results['eval_results'][i][eval_metric_key],
        rtol=rtol,
      )
    )
    self.assertTrue(  # grad_norms
      allclose(
        jax_results['scalars'][i]['grad_norm'],
        pytorch_results['scalars'][i]['grad_norm'],
        rtol=rtol,
      )
    )
    self.assertTrue(  # loss
      allclose(
        jax_results['scalars'][i]['loss'],
        pytorch_results['scalars'][i]['loss'],
        rtol=rtol,
      )
    )


if __name__ == '__main__':
  absltest.main()
