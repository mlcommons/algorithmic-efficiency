
import torch
from collections import deque

class LAWAQueue:
  def __init__(self, maxlen) -> None:
    self._maxlen = int(maxlen)
    self._queue = deque(maxlen=self._maxlen)
  
  def state_dict(self):
    return {key: value for key, value in self.__dict__.items()}
  
  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)
    
  def push(self, params):
    self._queue.append([p.detach().clone(memory_format=torch.preserve_format) for p in params])
  
  def get_last(self):
    return self._queue[-1]
  
  def full(self):
    return (len(self._queue)==self._maxlen)

  def get_avg(self):
    if not self.full():
      raise ValueError("q should be full to compute avg")
    
    q = self._queue
    k = float(self._maxlen)
    q_avg = [torch.zeros_like(p, device=p.device) for p in q[0]]
    for chkpts in q:
      for p_avg,p in zip(q_avg, chkpts):
        p_avg.add_(p/k)
    
    return q_avg