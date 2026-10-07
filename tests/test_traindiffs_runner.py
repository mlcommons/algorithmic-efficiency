"""Unit tests for the training-difference test runner."""

import pickle
import subprocess
from pathlib import Path
from unittest import mock

from absl.testing import absltest

from tests import test_traindiffs


class TrainDiffRunnerTest(absltest.TestCase):
  def test_collect_results_uses_isolated_temporary_files(self):
    result_paths: list[Path] = []

    def write_result(workload: str, framework: str, log_file: Path) -> None:
      del workload
      result_paths.append(log_file)
      with log_file.open('wb') as f:
        pickle.dump({'framework': framework}, f)

    with mock.patch.object(
      test_traindiffs,
      '_run_reference_algorithm',
      side_effect=write_result,
    ):
      jax_results, pytorch_results = test_traindiffs._collect_results('wmt')

    self.assertEqual(jax_results, {'framework': 'jax'})
    self.assertEqual(pytorch_results, {'framework': 'pytorch'})
    self.assertLen(result_paths, 2)
    self.assertEqual(result_paths[0].parent, result_paths[1].parent)
    self.assertNotEqual(result_paths[0], result_paths[1])
    self.assertFalse(result_paths[0].parent.exists())

  def test_collect_results_propagates_framework_failure(self):
    failure = subprocess.CalledProcessError(1, ['python', '-m', 'test'])

    with (
      mock.patch.object(
        test_traindiffs,
        '_run_reference_algorithm',
        side_effect=failure,
      ),
      self.assertRaises(subprocess.CalledProcessError) as raised,
    ):
      test_traindiffs._collect_results('wmt')

    self.assertIs(raised.exception, failure)

  def test_run_reference_algorithm_does_not_use_a_shell(self):
    log_file = Path('/tmp/results with spaces/jax.pkl')

    with mock.patch.object(test_traindiffs.subprocess, 'run') as run:
      test_traindiffs._run_reference_algorithm('wmt', 'jax', log_file)

    command = run.call_args.args[0]
    self.assertEqual(command[0], test_traindiffs.sys.executable)
    self.assertIn(f'--log_file={log_file}', command)
    self.assertNotIn('shell', run.call_args.kwargs)
    self.assertEqual(
      run.call_args.kwargs['env']['XLA_PYTHON_CLIENT_ALLOCATOR'],
      'platform',
    )
    self.assertTrue(run.call_args.kwargs['check'])


if __name__ == '__main__':
  absltest.main()
