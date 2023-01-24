from algorithmic_efficiency.workloads.librispeech_conformer import workload


class BaseDeepspeechLibrispeechWorkload(workload.BaseLibrispeechWorkload):

  @property
  def target_value(self):
    return 0.1162

  @property
  def step_hint(self) -> int:
    """Max num steps the target setting algo was given to reach the target."""
    return 60_000
