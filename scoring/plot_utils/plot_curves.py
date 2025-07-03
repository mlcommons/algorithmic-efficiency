from absl import flags
from absl import app

import matplotlib.pyplot as plt
import pandas as pd
import os
import wandb

flags.DEFINE_string('trial_dir', None, 'Path to trial dir')
flags.DEFINE_string()
FLAGS = flags.FLAGS

MEASUREMENTS_FILENAME = 'eval_measurements.csv'

def get_filename(trial_dir):
    filename = os.path.join(trial_dir, MEASUREMENTS_FILENAME )
    return filename

def main(_):
    filename = get_filename(FLAGS.trial_dir)

    # Start a new W&B run ---
    run = wandb.init(project="visualize-training-curve", name="conformer_jit")

    # Log the CSV as a versioned Artifact ---
    artifact = wandb.Artifact(name="training-data", type="dataset")
    artifact.add_file(filename) # Directly add the file
    run.log_artifact(artifact)

    # Log the metrics for direct visualization ---
    df = pd.read_csv(filename)
    print(df.columns)
    for index, row in df.iterrows():
        metrics = {col : row[col] for col in df.columns}
        wandb.log(metrics, step=int(row["global_step"]))

    # Finish the W&B run ---
    run.finish()


    return


if __name__ == '__main__':

  app.run(main)