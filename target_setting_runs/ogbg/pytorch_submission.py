"""PyTorch submission for the target-setting run on OGBG with AdamW."""

from ...target_setting_runs.pytorch_adamw import \
    data_selection  # pylint: disable=unused-import
from ...target_setting_runs.pytorch_adamw import \
    init_optimizer_state  # pylint: disable=unused-import
from ...target_setting_runs.pytorch_adamw import \
    update_params  # pylint: disable=unused-import


def get_batch_size(workload_name):
  # Return the global batch size.
  del workload_name
  return 512
