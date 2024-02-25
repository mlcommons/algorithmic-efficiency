import os
import re
import pandas as pd
import scoring_utils
# import scoring.scoring_utils
import pdb

'''
GOAL: controllare gli score per runnare altri stuidies

Input: exp, study, workload
Outputs:
  - for each trial: time to test target, time to val target, MORE
  - fastest trial to valid, its score and hyperparams

  from get_experiment_df
'''


TRIAL_LINE_REGEX = '(.*) --- Tuning run (\d+)/(\d+) ---'
METRICS_LINE_REGEX = '(.*) Metrics: ({.*})'
TRIAL_DIR_REGEX = 'trial_(\d+)'
MEASUREMENTS_FILENAME = 'eval_measurements.csv'
TIMESTAMP = r"-\d{4}(-\d{2}){5}"

# WORKLOADS = workloads_registry.WORKLOADS
WORKLOAD_NAME_PATTERN = '(.*)(_jax|_pytorch)'
BASE_WORKLOADS_DIR = 'algorithmic_efficiency/workloads/'


submission_directory = "/ptmp/najroldi/exp/algoperf"
submission = "nadamw_1"
study_dir = "study_1"
workload = "criteo1tb_pytorch"
experiment_dir = os.path.join(submission_directory, submission)

df = pd.DataFrame()
data = {
  'workload': workload,
}
trial_dirs = [
  t for t in os.listdir(
      os.path.join(experiment_dir, study_dir, workload))
            if re.match(TRIAL_DIR_REGEX, t)
]
for trial in trial_dirs:
  eval_measurements_filepath = os.path.join(
      experiment_dir,
      study_dir,
      workload,
      trial,
      MEASUREMENTS_FILENAME,
  )
  try:
    trial_df = pd.read_csv(eval_measurements_filepath)
  except FileNotFoundError as e:
    # logging.info(f'Could not read {eval_measurements_filepath}')
    continue
  data['trial'] = (trial, experiment_dir)
  data['study'] = study_dir
  for column in trial_df.columns:
    values = trial_df[column].to_numpy()
    data[column] = values
  trial_df = pd.DataFrame([data])
  df = pd.concat([df, trial_df], ignore_index=True)


print(df)

file_path = "results/prova.csv"
# os.makedirs(os.path.dirname(file_path), exist_ok=True)
df.to_csv(file_path, index=False)



# def main(_):
#   dfs = []
#   for workload, group in df.groupby('workload'):
#     validation_metric, validation_target = scoring_utils.get_workload_validation_target(workload)

#     pdb.set_trace()
#     break

#   df = pd.concat(dfs)



# if __name__ == '__main__':
#   # flags.mark_flag_as_required('submission_directory')
#   app.run(main)
