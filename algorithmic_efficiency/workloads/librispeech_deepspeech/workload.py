from algorithmic_efficiency.workloads.librispeech_conformer import \
    workload

class BaseDeepspeechLibrispeechWorkload(workload.BaseLibrispeechWorkload):
  @property
  def target_value(self):
    return 0.12749866