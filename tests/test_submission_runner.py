"""Sample test for a function from the submission_runner"""

import os

from algorithmic_efficiency.submission_runner import \
    _convert_filepath_to_module


def test_convert_filepath_to_module():
  """Sample test for the `_convert_filepath_to_module` function."""
  test_path = os.path.abspath(__file__)
  module_path = _convert_filepath_to_module(test_path)
  assert ".py" not in module_path
  assert "/" not in module_path
  assert isinstance(module_path, str)
