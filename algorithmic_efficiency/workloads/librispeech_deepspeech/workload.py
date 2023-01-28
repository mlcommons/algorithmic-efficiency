from algorithmic_efficiency.workloads.librispeech_conformer import workload


class BaseDeepspeechLibrispeechWorkload(workload.BaseLibrispeechWorkload):

  @property
  def validation_target_value(self):
    return 0.1162

  @property
  def test_target_value(self):
    return 0.067976

  @property
  def step_hint(self) -> int:
    """Max num steps the baseline algo was given to reach the target."""
    return 80_000
