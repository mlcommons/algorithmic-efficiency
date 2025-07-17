from absl import flags
from absl import app
from absl import logging

import re
import pandas as pd
import os
import wandb

flags.DEFINE_string(
    'experiment_dir',
    # '/home/kasimbeg/algoperf-runs-internal/experiments/jit_switch_debug_conformer_old_step_hint',
    '/home/kasimbeg/algoperf-runs-internal/experiments/jit_switch_debug_deepspeech_nadamw_jit_branch',
    'Path to experiment dir.')
flags.DEFINE_string(
    'workloads',
    'librispeech_deepspeech_jax',
    'Filter only for workload e.g. fastmri_jax. If None include all workloads in experiment.'
)
flags.DEFINE_string('project_name',
                    'visualize-training-curves-legacy-stephint-deepspeech',
                    'Wandb project name.')
flags.DEFINE_string('run_postfix', 'jit_legacy_lstm', 'Postfix for wandb runs.')

FLAGS = flags.FLAGS

MEASUREMENTS_FILENAME = 'eval_measurements.csv'
TRIAL_DIR_REGEX = 'trial_(\d+)'


def get_filename(trial_dir):
  filename = os.path.join(trial_dir, MEASUREMENTS_FILENAME)
  return filename


def main(_):
  experiment_dir = FLAGS.experiment_dir
  study_dirs = os.listdir(experiment_dir)
  for study_dir in study_dirs:
    if not FLAGS.workloads:
      workload_dirs = os.listdir(os.path.join(experiment_dir, study_dir))
      workload_dirs = [
          w for w in workload_dirs
          if os.path.isdir(os.path.join(experiment_dir, study_dir, w))
      ]
      print(workload_dirs)
    else:
      workload_dirs = FLAGS.workloads.split(',')
    for workload in workload_dirs:
      logging.info(os.path.join(experiment_dir, study_dir, workload))
      trial_dirs = [
          t for t in os.listdir(
              os.path.join(experiment_dir, study_dir, workload))
          if re.match(TRIAL_DIR_REGEX, t)
      ]
      for trial in trial_dirs:
        trial_dir = os.path.join(FLAGS.experiment_dir,
                                 study_dir,
                                 workload,
                                 trial)
        print(trial_dir)
        filename = get_filename(trial_dir)
        if not os.path.exists(filename):
          continue

        # Start a new W&B run
        run = wandb.init(
            project=FLAGS.project_name,
            name=(f'{workload}_{study_dir}_{trial}' + FLAGS.run_postfix))

        # Log the CSV as a versioned Artifact
        artifact = wandb.Artifact(name="training-data", type="dataset")
        artifact.add_file(filename)  # Directly add the file
        run.log_artifact(artifact)

        # Log the metrics for direct visualization
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
          metrics = {col: row[col] for col in df.columns}
          wandb.log(metrics, step=int(row["global_step"]))

        # Finish the W&B run ---
        run.finish()

  return


if __name__ == '__main__':

  app.run(main)
