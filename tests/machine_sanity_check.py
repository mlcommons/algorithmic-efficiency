import jaxlib
import torch

try:
  from jaxlib import version as jaxlib_version
except:
  # jaxlib is too old to have version number.
  msg = 'This version of jax requires jaxlib version >= {}.'
  raise ImportError(msg.format('.'.join(map(str, _minimum_jaxlib_version))))

version = tuple(int(x) for x in jaxlib_version.__version__.split('.'))

print('jaxlib version = ', version)
print('NCCL version = ', torch.cuda.nccl.version())