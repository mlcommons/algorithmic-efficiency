import json
from absl import flags
from absl import app
import jax
import os

SUBMISSION_PATH = '/submissions_algorithms/external_tuning/shampoo_submission/submission.py'
TUNING_SEARCH_SPACE = '/submissions_algorithms/external_tuning/shampoo_submission/tuning_search_space.json'
EXPERIMENT_DIR = 'submissions/rolling_leaderboard/external_tuning/shampoo'
FRAMEWORK = 'pytorch'

flags.DEFINE_string('submission_path', 
                    SUBMISSION_PATH,
                    'Path to submission module.')
flags.DEFINE_string('tuning_search_space',
                    TUNING_SEARCH_SPACE,
                    'Path to tuning search space for submission module.')
flags.DEFINE_string('experiment_dir',
                    EXPERIMENT_DIR,
                    'Path to experiment dir where logs will be saved.')
flags.DEFINE_enum('framework',
                  FRAMEWORK,
                  enum_values=['jax', 'pytorch'],
                  help='Can be either pytorch or jax.')  
flags.DEFINE_integer('seed',
                     0, 
                     'RNG seed to to generate study seeds from.')
flags.DEFINE_enum('tuning_ruleset',
                  'external',
                  enum_values=['external', 'self'],
                  help='Which tuning ruleset to score this submission on. Can be external or self.'
                 )

FLAGS = flags.FLAGS


MIN_INT = -2**(31)
MAX_INT = 2**(31) - 1
NUM_TUNING_TRIALS = 5  # For external tuning ruleset
NUM_STUDIES = 3

WORKLOADS = {
  "imagenet_resnet": {
    "dataset": "imagenet"
  },
  "imagenet_vit": {
    "dataset": "imagenet"
  },
  "fastmri": {
    "dataset": "fastmri"
  },
  "ogbg": {
    "dataset": "ogbg"
  },
  "wmt": {
    "dataset": "wmt"
  },
  "librispeech_deepspeech": {
    "dataset": "librispeech"
  },
  "criteo1tb": {
    "dataset": "criteo1tb"
  },
  "librispeech_conformer": {
    "dataset": "librispeech"
  }
}

def main(_):
  workloads=WORKLOADS.keys()
  key = jax.random.key(FLAGS.seed)

  jobs = []

  for workload in workloads:
    # Fold in hash(workload) mod(max(uint32))
    workload_key = jax.random.fold_in(key, hash(workload) % (2**32-1))
    for study_index in range(NUM_STUDIES):
      study_key = jax.random.fold_in(workload_key, study_index)
      for hparam_index in range(NUM_TUNING_TRIALS):
        run_key = jax.random.fold_in(study_key, hparam_index)
        seed = jax.random.randint(run_key, (1,), MIN_INT, MAX_INT)[0].item()
        print(seed)

        # Add workload
        job = {}
        study_dir = os.path.join(FLAGS.experiment_dir, f"study_{study_index}")
        job['framework'] = FLAGS.framework
        job['workload'] = workload
        job['dataset'] = WORKLOADS[workload]['dataset']
        job['submission_path'] = FLAGS.submission_path
        job['tuning_search_space'] = FLAGS.tuning_search_space
        job['experiment_dir'] = study_dir
        job['hparam_start_index'] = hparam_index
        job['hparam_end_index'] = hparam_index + 1
        job['rng_seed'] = seed
        job['num_tuning_trials'] = NUM_TUNING_TRIALS if FLAGS.tuning_ruleset == 'external' else 1

        jobs.append(job)
        print(job)


  # Convert job array to dict with job indices
  job_dict = {}
  for i, job in enumerate(jobs):
    job_dict[f"{i}"] = job

  with open('config.json', 'w') as f:
    json.dump(job_dict, f, indent=4)


if __name__ == '__main__':
  app.run(main)