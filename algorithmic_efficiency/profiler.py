"""Profiling code for Jax and PyTorch

Modified from:
https://github.com/Lightning-AI/lightning/tree/master/src/pytorch_lightning/profilers
"""

from collections import defaultdict
from contextlib import contextmanager
import os
import time
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np


class Profiler:

  def __init__(self, local_rank: Optional[int] = None) -> None:
    self._local_rank = local_rank

    self.current_actions: Dict[str, float] = {}
    self.recorded_durations = defaultdict(list)
    self.start_time = time.monotonic()

  def set_local_rank(self, local_rank: int) -> None:
    self._local_rank = local_rank

  @property
  def local_rank(self) -> int:
    return 0 if self._local_rank is None else self._local_rank

  def start(self, action_name: str) -> None:
    if self.local_rank != 0:
      pass
    if action_name in self.current_actions:
      raise ValueError(
          f'Attempted to start {action_name} which has already started.')
    self.current_actions[action_name] = time.monotonic()

  def stop(self, action_name: str) -> None:
    if self.local_rank != 0:
      pass
    end_time = time.monotonic()
    if action_name not in self.current_actions:
      raise ValueError(f'Attempting to stop recording an action '
                       f'({action_name}) which was never started.')
    start_time = self.current_actions.pop(action_name)
    duration = end_time - start_time
    self.recorded_durations[action_name].append(duration)

  @contextmanager
  def profile(self, action_name: str) -> Generator:
    try:
      self.start(action_name)
      yield action_name
    finally:
      self.stop(action_name)

  def _make_report(
      self
  ) -> Tuple[List[Tuple[str, float, float, int, float, float]], int, float]:
    total_duration = time.monotonic() - self.start_time
    report = [(str(a),
               float(np.mean(d)),
               float(np.std(d)),
               len(d),
               float(np.sum(d)),
               100.0 * float(np.sum(d)) / total_duration) for a,
              d in self.recorded_durations.items()]
    report.sort(key=lambda x: x[5], reverse=True)
    total_calls = sum(x[3] for x in report)
    return report, total_calls, total_duration

  def summary(self) -> str:
    sep = os.linesep
    output_string = ''
    output_string += f'Profiler Report{sep}:'

    if len(self.recorded_durations) > 0:
      max_key = max(len(k) for k in self.recorded_durations.keys())

      def log_row(action, mean, std, num_calls, total, per):
        row = f'{sep}|  {action:<{max_key}s}\t|  '
        row += f'{mean:<15}\t|  {std:<15}\t|'
        row += f'  {num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|'
        return row

      header_string = log_row('Action',
                              'Mean Duration (s)',
                              'Std Duration (s)',
                              'Num Calls',
                              'Total Time (s)',
                              'Percentage %')
      output_string_len = len(header_string.expandtabs())
      sep_lines = f'{sep}{"-" * output_string_len}'
      output_string += sep_lines + header_string + sep_lines
      report, total_calls, total_duration = self._make_report()
      output_string += log_row('Total',
                               '-----',
                               '-----',
                               f'{total_calls:}',
                               f'{total_duration:.5}',
                               '100 %')
      output_string += sep_lines
      for action, mean_duration, std_duration, num_calls, \
          total_duration, duration_per in report:
        output_string += log_row(
            action,
            f'{mean_duration:.5}',
            f'{std_duration:.5}',
            f'{num_calls}',
            f'{total_duration:.5}',
            f'{duration_per:.5}',
        )
      output_string += sep_lines
    output_string += sep
    return output_string


class PassThroughProfiler(Profiler):

  def start(self, action_name: str) -> None:
    pass

  def stop(self, action_name: str) -> None:
    pass
