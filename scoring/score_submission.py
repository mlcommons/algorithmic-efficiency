from absl import app
from absl import flags
from absl import logging

import scoring_utils 
import scoring

flags.DEFINE_string('experiment_path',
                    None,
                    'Path to directory containing experiment_results')
flags.DEFINE_string('submission_tag',
                    'my.submission',
                    'Submission tag.')
FLAGS = flags.FLAGS

workload_metadata = {
    'mnist_jax': {
        'target': 0.8,
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
    logging.info(performance_profile_df)


if __name__ == '__main__':
    flags.mark_flag_as_required('experiment_path')
    app.run(main)


