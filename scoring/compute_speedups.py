"""File to compute speedups (i.e. geometric means between runtimes)."""

import pickle

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from performance_profile import BASE_WORKLOADS
from performance_profile import get_workloads_time_to_target
from scipy import stats

flags.DEFINE_string('results_txt', None, 'Path to full scoring results file.')
flags.DEFINE_string(
    'base',
    'prize_qualification_baseline',
    'Base submission to compare to. Defaults to the `prize_qualification_baseline`.'
)
flags.DEFINE_string('comparison', None, 'Submission to compute the speedup of.')
flags.DEFINE_boolean('self_tuning_ruleset',
                     False,
                     'Whether the self-tuning ruleset is being scored.')
flags.DEFINE_boolean('save_results',
                     False,
                     'Whether to save the results to disk.')
FLAGS = flags.FLAGS

# These are the old budgets, used in the first iteration of the competition.
MAX_BUDGETS = {
    'criteo1tb': 7703,
    'fastmri': 8859,
    'imagenet_resnet': 63_008,
    'imagenet_vit': 77_520,
    'librispeech_conformer': 61_068,
    'librispeech_deepspeech': 55_506,
    'ogbg': 18_477,
    'wmt': 48_151,
}


def replace_inf(row):
  """Replace ifs with maximum runtime budget (+1 second).

  Args:
      row (pd.Series): The original row.

  Returns:
      pd.Series: The row with infs replaced.
  """
  workload_name = row.name
  # Factor of 3 for self-tuning ruleset
  factor = 3 if FLAGS.self_tuning_ruleset else 1
  max_runtime_workload = factor * MAX_BUDGETS[workload_name]
  row.replace(np.inf, max_runtime_workload + 1, inplace=True)
  return row


def compute_speedup():
  """Compute speedup between two algorithms."""
  # Load results from disk
  with open(FLAGS.results_txt, 'rb') as f:
    results = pickle.load(f)

  # Compute median over runtimes for both training algorithms
  base_results = get_workloads_time_to_target(
      results[FLAGS.base],
      FLAGS.base,
      time_col="score",
      self_tuning_ruleset=FLAGS.self_tuning_ruleset,
  )
  comparison_results = get_workloads_time_to_target(
      results[FLAGS.comparison],
      FLAGS.comparison,
      time_col="score",
      self_tuning_ruleset=FLAGS.self_tuning_ruleset,
  )

  # Merge results
  merged_results = pd.concat([base_results, comparison_results]).transpose()

  # Ignore workload variants (only consider base workloads) for speedup
  merged_results = merged_results.loc[merged_results.index.isin(BASE_WORKLOADS)]

  # Replace infs with maximum runtime budget (+1 second)
  merged_results = merged_results.apply(replace_inf, axis=1)

  # Compute speedup
  merged_results['speedup'] = merged_results[
      f'{FLAGS.comparison}'] / merged_results[f'{FLAGS.base}']
  speedups = merged_results['speedup'].to_numpy()
  mean_speedup = stats.gmean(speedups)  # Geometric mean over workload speedups

  print(merged_results, end='\n\n')
  print(
      f"Average speedup of {FLAGS.comparison} compared to {FLAGS.base}: {mean_speedup} or roughly {(1-mean_speedup):.1%}"
  )

  if FLAGS.save_results:
    # Optionally save results to disk
    print("Saving results to disk...")
    filename = f'{FLAGS.comparison}_vs_{FLAGS.base}_speedup_{(1-mean_speedup):.1%}.csv'
    merged_results.to_csv(filename)


def main(_):
  """Main function to compute speedup between two algorithms."""
  compute_speedup()


if __name__ == '__main__':
  flags.mark_flag_as_required('results_txt')
  flags.mark_flag_as_required('comparison')
  app.run(main)
