from absl import app
from absl import flags
from absl import logging
import os

import scoring_utils 
import scoring
from algorithmic_efficiency import workloads
from algorithmic_efficiency.workloads.mnist.mnist_jax.workload import MnistWorkload
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_jax.workload import Criteo1TbDlrmSmallWorkload
from algorithmic_efficiency.workloads.fastmri.fastmri_jax.workload import FastMRIWorkload
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.workload import ImagenetResNetWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_jax.workload import ImagenetVitWorkload
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.workload import LibriSpeechConformerWorkload
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_jax.workload import LibriSpeechDeepSpeechWorkload
from algorithmic_efficiency.workloads.ogbg.ogbg_jax.workload import OgbgWorkload
from algorithmic_efficiency.workloads.wmt.wmt_jax.workload import WmtWorkload

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
        'target': MnistWorkload().validation_target_value,
        'metric': 'validation/accuracy'
    },
    'mnist_pytorch': {
        'target': MnistWorkload().validation_target_value,
        'metric': 'validation/accuracy'
    },
    'criteo1tb_jax': {
        'target': Criteo1TbDlrmSmallWorkload().validation_target_value,
        'metric': 'validation/loss'
    },
    'criteo1tb_pytorch': {
        'target': Criteo1TbDlrmSmallWorkload().validation_target_value,
        'metric': 'validation/loss'
    },
    'fastmri_jax': {
        'target': FastMRIWorkload().validation_target_value,
        'metric': 'validation/ssim'
    },
    'fastmri_pytorch': {
        'target': FastMRIWorkload().validation_target_value,
        'metric': 'validation/ssim'
    },
    'imagenet_resnet': {
        'target': ImagenetResnetWorkload().validation_target_value,
        'metric': 'validation/accuracy'
    },
    'imagenet_resnet': {
        'target': ImagenetResnetWorkload().validation_target_value,
        'metric': 'validation/accuracy'
    },
    'imagenet_vit': {
        'target': ImagenetVitWorkload().validation_target_value,
        'metric': 'validation/accuracy'
    },
    'imagenet_vit': {
        'target': ImagenetVitWorkload().validation_target_value,
        'metric': 'validation/accuracy'
    },
    'librispeech_conformer': {
        'target': LibriSpeechConformerWorkload().validation_target_value,
        'metric': 'validation/wer'
    },
    'librispeech_conformer': {
        'target': LibriSpeechConformerWorkload().validation_target_value,
        'metric': 'validation/wer'
    },
    'librispeech_deepspeech': {
        'target': LibriSpeechDeepSpeechWorkload().validation_target_value,
        'metric': 'validation/wer'
    },
    'librispeech_deepspeech': {
        'target': LibriSpeechDeepSpeechWorkload().validation_target_value,
        'metric': 'validation/wer'
    },
    'ogbg': {
        'target': OgbgWorkload().validation_target_value,
        'metric': 'validation/mean_average_precision'
    },
    'ogbg': {
        'target': OgbgWorkload().validation_target_value,
        'metric': 'validation/mean_average_precision'
    },
    'wmt': {
        'target': WmtWorkload().validation_target_value,
        'metric': 'validation/bleu'
    },
    'wmt': {
        'target': WmtWorkload().validation_target_value,
        'metric': 'validation/bleu'
    },
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
    print(performance_profile_df)
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    scoring.plot_performance_profiles(performance_profile_df,
                              'score',
                              save_dir=FLAGS.output_dir)
    
    logging.info(performance_profile_df)


if __name__ == '__main__':
    flags.mark_flag_as_required('experiment_path')
    app.run(main)


