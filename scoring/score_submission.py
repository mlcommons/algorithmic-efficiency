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
        'validation_target': MnistWorkload().validation_target_value,
        'test_target': MnistWorkload().test_target_value,
        'validation_metric': 'validation/accuracy',
        'test_metric': 'test/accuracy'
    },
    'mnist_pytorch': {
        'validation_target': MnistWorkload().validation_target_value,
        'test_target': MnistWorkload().test_target_value,
        'validation_metric': 'validation/accuracy',
        'test_metric': 'test/accuracy'
    },
    'criteo1tb_jax': {
        'validation_target': Criteo1TbDlrmSmallWorkload().validation_target_value,
        'test_target': Criteo1TbDlrmSmallWorkload().test_target_value,
        'validation_metric': 'validation/loss',
        'test_metric': 'test/loss',
    },
    'criteo1tb_pytorch': {
        'validation_target': Criteo1TbDlrmSmallWorkload().validation_target_value,
        'test_target': Criteo1TbDlrmSmallWorkload().test_target_value,
        'validaiton_metric': 'validation/loss',
        'test_metric': 'test/loss'
    },
    'fastmri_jax': {
        'validation_target': FastMRIWorkload().validation_target_value,
        'test_target': FastMRIWorkload().test_target_value,
        'validation_metric': 'validation/ssim',
        'test_metric': 'test/ssim',
    },
    'fastmri_pytorch': {
        'validation_target': FastMRIWorkload().validation_target_value,
        'test_target': FastMRIWorkload().test_target_value,
        'validation_metric': 'validation/ssim',
        'test_metric': 'test/ssim',
    },
    'imagenet_resnet_jax': {
        'validation_target': ImagenetResNetWorkload().validation_target_value,
        'test_target': ImagenetResNetWorkload().test_target_value,
        'validation_metric': 'validation/accuracy',
        'test_metric': 'test/accuracy'
    },
    'imagenet_resnet_pytorch': {
        'validation_target': ImagenetResNetWorkload().validation_target_value,
        'test_target': ImagenetResNetWorkload().test_target_value,
        'validation_metric': 'validation/accuracy',
        'test_metric': 'test/accuracy'
    },
    'imagenet_vit_jax': {
        'validation_target': ImagenetVitWorkload().validation_target_value,
        'test_target': ImagenetVitWorkload().test_target_value,
        'validation_metric': 'validation/accuracy',
        'test_metric': 'test/accuracy',
    },
    'imagenet_vit_pytorch': {
        'validation_target': ImagenetVitWorkload().validation_target_value,
        'test_target': ImagenetVitWorkload().test_target_value,
        'validation_metric': 'validation/accuracy',
        'test_metric': 'test/accuracy',
    },
    'librispeech_conformer_jax': {
        'validation_target': LibriSpeechConformerWorkload().validation_target_value,
        'test_target': LibriSpeechConformerWorkload().test_target_value,
        'validation_metric': 'validation/wer',
        'test_metric': 'test/wer',
    },
    'librispeech_conformer_pytorch': {
        'validation_target': LibriSpeechConformerWorkload().validation_target_value,
        'test_target': LibriSpeechConformerWorkload().test_target_value,
        'validation_metric': 'validation/wer',
        'test_metric': 'test/wer',
    },
    'librispeech_deepspeech_jax': {
        'validation_target': LibriSpeechDeepSpeechWorkload().validation_target_value,
        'test_target': LibriSpeechDeepSpeechWorkload().test_target_value,
        'validation_metric': 'validation/wer',
        'test_metric': 'test/wer'
    },
    'librispeech_deepspeech_pytorch': {
        'target': LibriSpeechDeepSpeechWorkload().validation_target_value,
        'validation_target': LibriSpeechDeepSpeechWorkload().validation_target_value,
        'test_target': LibriSpeechDeepSpeechWorkload().test_target_value,
        'validation_metric': 'validation/wer',
    },
    'ogbg_jax': {
        'validation_target': OgbgWorkload().validation_target_value,
        'test_target': OgbgWorkload().test_target_value,
        'validation_metric': 'validation/mean_average_precision',
        'test_metric': 'test/mean_average_precision',
    },
    'ogbg_pytorch': {
        'validation_target': OgbgWorkload().validation_target_value,
        'test_target': OgbgWorkload().test_target_value,
        'validation_metric': 'validation/mean_average_precision',
        'test_metric': 'test/mean_average_precision',
    },
    'wmt_jax': {
        'validation_target': WmtWorkload().validation_target_value,
        'test_target': WmtWorkload().test_target_value,
        'validation_metric': 'validation/bleu',
        'test_metric': 'test/bleu',
    },
    'wmt_pytorch': {
        'validation_target': WmtWorkload().validation_target_value,
        'test_target': WmtWorkload().test_target_value,
        'validation_metric': 'validation/bleu',
        'test_metric': 'test/bleu',
    },
}
        

def main(_):
    df = scoring_utils.get_experiment_df(FLAGS.experiment_path)
    results = {
        FLAGS.submission_tag : df,
        }
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


