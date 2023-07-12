from absl import app
from absl import flags
from absl import logging

import scoring_utils 
import scoring
from algorithmic_efficiency import workloads

flags.DEFINE_string('experiment_path',
                    None,
                    'Path to directory containing experiment_results')
flags.DEFINE_string('submission_tag',
                    'my.submission',
                    'Submission tag.')
flags.DEFINE_string('output_dir',
                    'scoring_results',
                    'Path to save performance profile table and plot.')
FLAGS = flags.FLAGS

workload_metadata = {
    'mnist_jax': {
        'target': workloads.mnist.workload.BaseMnistWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'mnist_pytorch': {
        'target': workloads.mnist.workload.BaseMnistWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'criteo1tb_jax': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'criteo1tb_pytorch': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'fastmri_jax': {
        'target': workloads.fastmri.workload.BaseFastMRIWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'fastmri_pytorch': {
        'target': workloads.fastmri.workload.BaseFastMRIWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'imagenet_resnet': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'imagenet_resnet': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'imagenet_vit': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'imagenet_vit': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'librispeech_conformer': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'librispeech_conformer': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'librispeech_deepspeech': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'librispeech_deepspeech': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'ogbg': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'ogbg': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'wmt': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }
    'wmt': {
        'target': workloads.criteo1tb.workload.BaseCriteo1TbDlrmSmallWorkload.validation_target_value,
        'metric': 'validation/accuracy'
    }



}
        

def main(_):
    df = scoring_utils.get_experiment_df(FLAGS.experiment_path)
    results = {
        FLAGS.submission_tag : df,
        }
    # print(results.head)
    performance_profile_df = scoring.compute_performance_profiles(results,
                                                        workload_metadata,
                                                        time_col='score',
                                                        min_tau=1.0,
                                                        max_tau=None,
                                                        reference_submission_tag=None,
                                                        num_points=100,
                                                        scale='linear',
                                                        verbosity=0)
    plot_performance_profiles(performance_profile_df,
                              'score',
                              save_dir=FLAGS.ouptut_dir)
    
    logging.info(performance_profile_df)


if __name__ == '__main__':
    flags.mark_flag_as_required('experiment_path')
    app.run(main)


