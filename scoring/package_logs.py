"""Script to package logs from experiment directory.
Example usage:

python3 package_logs.py --experiment_dir <experiment_dir> --destination_dir <destination_dir>
"""
import os
import shutil

from absl import app
from absl import flags

flags.DEFINE_string('experiment_dir', None, 'Path to experiment.')
flags.DEFINE_string('destination_dir', None, 'Path to save submission logs')

FLAGS = flags.FLAGS


def move_logs(experiment_dir, destination_dir):
  """Copy files from experiment path to destination directory.
    Args:
        experiment_dir: Path to experiment dir.
        destination_dir: Path to destination dir.
    """
  print('in move logs')
  print(destination_dir)
  if not os.path.exists(experiment_dir):
    raise IOError(f'Directory does not exist {destination_dir}')

  for root, dirnames, filenames in os.walk(experiment_dir):
    for filename in filenames:
      if 'checkpoint' not in filename:
        source_path = os.path.join(root, filename)
        relative_path = os.path.relpath(source_path, experiment_dir)
        destination_path = os.path.join(destination_dir, relative_path)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        print(f'Moving {source_path} to {destination_path}')
        shutil.copy(source_path, destination_path)


def main(_):
  flags.mark_flag_as_required('destination_dir')
  flags.mark_flag_as_required('experiment_dir')
  move_logs(FLAGS.experiment_dir, FLAGS.destination_dir)


if __name__ == '__main__':
  app.run(main)
