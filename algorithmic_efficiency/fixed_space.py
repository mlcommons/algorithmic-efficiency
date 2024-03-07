"""
Nico: generate a fixed space, modeled after halton.generate_search
"""

import collections
from itertools import product
from typing import Any, Callable, Dict, List, Sequence, Text, Tuple, Union

from absl import logging

_DictSearchSpace = Dict[str, Dict[str, Union[str, float, Sequence]]]
_ListSearchSpace = List[Dict[str, Union[str, float, Sequence]]]

def generate_search(search_space: Union[_DictSearchSpace, _ListSearchSpace],
                    num_trials: int) -> List[collections.namedtuple]:
  """Generate a fixed search with the given bounds and scaling.

  Args:linear
    search_space: A dict where the keys are the hyperparameter names, and the
      values are a dict of:
        - {"feasible_points": [...]} for discrete hyperparameters.
    num_trials: the number of hyperparameter points to generate.

  Returns:
    A list of length `num_trials` of namedtuples, each of which has attributes
    corresponding to the given hyperparameters, and values randomly sampled.
  """
  if isinstance(search_space, dict):
    all_hyperparameter_names = list(search_space.keys())
  elif isinstance(search_space, list):
    assert len(search_space) > 0
    all_hyperparameter_names = list(search_space[0].keys())
  else:
    raise AttributeError('tuning_search_space should either be a dict or list.')

  named_tuple_class = collections.namedtuple('Hyperparameters',
                                             all_hyperparameter_names)

  if isinstance(search_space, dict):
    
    # Extracting keys and corresponding feasible points
    keys = search_space.keys()
    values = (search_space[key]['feasible_points'] for key in keys)

    # Generating all possible combinations
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]

    result = [named_tuple_class(**p) for p in combinations]
    
    if len(result) != num_trials:
      raise ValueError(
          "Search space dimension {} differs from num_trials {}".format(
              len(result),
              num_trials
          ))
    
    return result
    
  else:
    hyperparameters = []
    updated_num_trials = min(num_trials, len(search_space))
    if num_trials != len(search_space):
      logging.info(f'--num_tuning_trials was set to {num_trials}, but '
                   f'{len(search_space)} trial(s) found in the JSON file. '
                   f'Updating --num_tuning_trials to {updated_num_trials}.')
    for trial in search_space:
      hyperparameters.append(named_tuple_class(**trial))
    return hyperparameters[:updated_num_trials]
