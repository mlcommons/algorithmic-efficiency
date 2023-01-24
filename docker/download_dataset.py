import os 

from absl import app
from absl import flags
from absl import logging
import tensorflow_datasets as tfds 

# import tensorflow as tf 
# tf.config.set_visible_devices([], 'GPU')

flags.DEFINE_string(
    'data_dir',
    None,
    'The path to the folder where datasets should be downloaded.')
FLAGS = flags.FLAGS

def main(_):
  data_dir = FLAGS.data_dir
  data_dir = os.path.abspath(os.path.expanduser(data_dir))
  
  logging.info('Downloading data to %s...', data_dir)
  tfds.builder('ogbg_molpcba:0.1.2', data_dir=data_dir).download_and_prepare()

if __name__ == '__main__':
  app.run(main)
