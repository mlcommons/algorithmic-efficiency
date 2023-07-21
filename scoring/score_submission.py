import os

from absl import app
from absl import flags
from absl import logging
import scoring_utils

from algorithmic_efficiency import workloads
import scoring


from algorithmic_efficiency import spec


WORKLOADS = {
    'cifar': {
        'workload_path': 'cifar/cifar', 'workload_class_name': 'CifarWorkload'
    },
    'criteo1tb': {
        'workload_path': 'criteo1tb/criteo1tb',
        'workload_class_name': 'Criteo1TbDlrmSmallWorkload',
    },
    'criteo1tb_test': {
        'workload_path': 'criteo1tb/criteo1tb',
        'workload_class_name': 'Criteo1TbDlrmSmallTestWorkload',
    },
    'fastmri': {
        'workload_path': 'fastmri/fastmri',
        'workload_class_name': 'FastMRIWorkload',
    },
    'imagenet_resnet': {
        'workload_path': 'imagenet_resnet/imagenet',
        'workload_class_name': 'ImagenetResNetWorkload',
    },
    'imagenet_vit': {
        'workload_path': 'imagenet_vit/imagenet',
        'workload_class_name': 'ImagenetVitWorkload',
    },
    'librispeech_conformer': {
        'workload_path': 'librispeech_conformer/librispeech',
        'workload_class_name': 'LibriSpeechConformerWorkload',
    },
    'librispeech_deepspeech': {
        'workload_path': 'librispeech_deepspeech/librispeech',
        'workload_class_name': 'LibriSpeechDeepSpeechWorkload',
    },
    'mnist': {
        'workload_path': 'mnist/mnist', 'workload_class_name': 'MnistWorkload'
    },
    'ogbg': {
        'workload_path': 'ogbg/ogbg', 'workload_class_name': 'OgbgWorkload'
    },
    'wmt': {'workload_path': 'wmt/wmt', 'workload_class_name': 'WmtWorkload'},
}



flags.DEFINE_string('experiment_path',
                    None,
                    'Path to directory containing experiment_results')
flags.DEFINE_string('submission_tag', 'my.submission', 'Submission tag.')
flags.DEFINE_string('output_dir',
                    'scoring_results',
                    'Path to save performance profile table and plot.')
FLAGS = flags.FLAGS


def main(_):
  df = scoring_utils.get_experiment_df(FLAGS.experiment_path)
  results = {
      FLAGS.submission_tag: df,
  }
  performance_profile_df = scoring.compute_performance_profiles(
      results,
    #   workload_metadata,
      time_col='score',
      min_tau=1.0,
      max_tau=None,
      reference_submission_tag=None,
      num_points=100,
      scale='linear',
      verbosity=0)
  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
  scoring.plot_performance_profiles(
      performance_profile_df, 'score', save_dir=FLAGS.output_dir)

  logging.info(performance_profile_df)


if __name__ == '__main__':
  flags.mark_flag_as_required('experiment_path')
  app.run(main)
