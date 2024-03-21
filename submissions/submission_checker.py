"""
Expected submission directory structure:

submission_folder/
├── external_tuning
│   ├── algorithm_name
│   │   ├── helper_module.py
│   │   ├── requirements.txt
│   │   ├── submission.py
│   │   └── tuning_search_space.json
│   └── other_algorithm_name
│       ├── requirements.txt
│       ├── submission.py
│       └── tuning_search_space.json
└── self_tuning
    └── algorithm_name
        ├── requirements.txt
        └── submission.py


It is also expected that submission.py has the following APIs:
- init_optimizer_state
- update_params
- get_batch_size
- data_selection

"""

import argparse
import logging
import os
import subprocess

SELF_TUNING = 'self_tuning'
EXTERNAL_TUNING = 'external_tuning'
SUBMISSION_MODULE = 'submission.py'
TUNING_SEARCH_SPACE_FILENAME = 'tuning_search_space.json'


def _check_ruleset_subdirs(submission_dir):
  contents = os.listdir(submission_dir)
  if not ((EXTERNAL_TUNING in contents) or (SELF_TUNING in contents)):
    logging.info(
        f'CHECK FAILED: {submission_dir} does not contain ruleset subdir.')
    return False
  return True


def _check_submission_module(submission_dir):
  for root, dirs, files in os.walk(submission_dir):
    parent_dir = os.path.basename(root)
    if parent_dir == SELF_TUNING or parent_dir == EXTERNAL_TUNING:
      for submission_dir in dirs:
        contents = os.listdir(os.path.join(root, submission_dir))
        if SUBMISSION_MODULE not in contents:
          logging.info(
              f'CHECK FAILED: {parent_dir} does not contain {SUBMISSION_MODULE}'
          )
          return False
  return True


def _check_tuning_search_space_file(submission_dir):
  for root, dirs, files in os.walk(submission_dir):
    parent_dir = os.path.basename(root)
    if parent_dir == EXTERNAL_TUNING:
      for submission_dir in dirs:
        contents = os.listdir(os.path.join(root, submission_dir))
        if TUNING_SEARCH_SPACE_FILENAME not in contents:
          logging.info(
              f'CHECK FAILED: {parent_dir} does not contain {TUNING_SEARCH_SPACE_FILENAME}'
          )
          return False
  return True


def run_checks(submission_dir):
  """Top-level checker function.
    Call individual checkers from this function.
    """
  logging.info('Running repository checks.')

  # Get files and directories
  files = list_files_recursively(submission_dir)
  dirs = list_dirs_recursively(submission_dir)

  # Execute checks
  contains_ruleset_subdirs = _check_ruleset_subdirs(submission_dir)
  contains_submission_module = _check_submission_module(submission_dir)
  contains_tuning_search_space_file = _check_tuning_search_space_file(
      submission_dir)

  if not (contains_ruleset_subdirs or contains_submission_module or
          contains_tuning_search_space_file):
    logging.info('TESTS FAILED.')
    return False

  logging.info('ALL CHECKS PASSED.')
  return True


def get_parser():
  """Parse commandline."""
  parser = argparse.ArgumentParser(
      description='Checks for submission folder for AlgoPerf',)
  parser.add_argument(
      'folder',
      type=str,
      help='the folder for a submission package.',
  )
  parser.add_argument(
      '--log_output',
      type=str,
      default='submission_checker.log',
  )
  return parser


def main():
  parser = get_parser()
  args = parser.parse_args()

  logging.basicConfig(filename=args.log_output, level=logging.INFO)
  logging.getLogger().addHandler(logging.StreamHandler())
  formatter = logging.Formatter("%(levelname)s - %(message)s")
  logging.getLogger().handlers[0].setFormatter(formatter)
  logging.getLogger().handlers[1].setFormatter(formatter)

  valid = run_checks(args.folder)
  return valid


if __name__ == '__main__':
  main()
