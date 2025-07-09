from absl import flags
from absl import app
from absl import logging

import matplotlib.pyplot as plt
import re
import pandas as pd
import os
import wandb

flags.DEFINE_string('experiment_dir', None, 'Path to experiment dir.')
flags.DEFINE_string('workload', None, 'Filter only for workload. If None include all workloads in experiment.')
flags.DEFINE_string('project_name', 'visulaize-training-curves', 'Wandb project name.')
flags.DEFINE_string('run_postfix', '', 'Postfix for wandb runs.')

FLAGS = flags.FLAGS

MEASUREMENTS_FILENAME = 'eval_measurements.csv'
TRIAL_DIR_REGEX = 'trial_(\d+)'


def get_filename(trial_dir):
    filename = os.path.join(trial_dir, MEASUREMENTS_FILENAME )
    return filename

def main(_):
    experiment_dir = FLAGS.experiment_dir
    study_dirs = os.listdir(experiment_dir)
    for study_dir in study_dirs:
      workload_dirs = os.listdir(os.path.join(experiment_dir, study_dir))
      workload_dirs = [
          w for w in workload_dirs
          if os.path.isdir(os.path.join(experiment_dir, study_dir, w))
      ]
      print(workload_dirs)
      for workload in workload_dirs:
        data = {
            'workload': workload,
        }
        logging.info(os.path.join(experiment_dir, study_dir, workload))
        trial_dirs = [
            t for t in os.listdir(
                os.path.join(experiment_dir, study_dir, workload))
            if re.match(TRIAL_DIR_REGEX, t)
        ]
        for trial in trial_dirs:
            filename = get_filename(FLAGS.trial_dir)
            # Start a new W&B run
            run = wandb.init(project="visualize-training-curve", name=(f'{workload}_{study_dir}_{trial}' + FLAGS.run_postfix))

            # Log the CSV as a versioned Artifact
            artifact = wandb.Artifact(name="training-data", type="dataset")
            artifact.add_file(filename) # Directly add the file
            run.log_artifact(artifact)

            # Log the metrics for direct visualization
            df = pd.read_csv(filename)
            for index, row in df.iterrows():
                metrics = {col : row[col] for col in df.columns}
                wandb.log(metrics, step=int(row["global_step"]))

            # Finish the W&B run ---
            run.finish()

    return


if __name__ == '__main__':

  app.run(main)