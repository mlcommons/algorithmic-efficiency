from algorithmic_efficiency.workloads.librispeech_deepspeech import workload


class BaseDeepspeechLibrispeechWorkload(workload.BaseLibrispeechWorkload):

  @property
  def validation_target_value(self) -> float:
    return 0.118232

  @property
  def test_target_value(self) -> float:
    return 0.073397

  @property
  def step_hint(self) -> int:
    """Max num steps the baseline algo was given to reach the target."""
    return 48_000

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 55_506  # ~15.4 hours
