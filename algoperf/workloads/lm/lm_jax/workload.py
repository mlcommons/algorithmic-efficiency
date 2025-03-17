"""LM workload implemented in Jax."""

import functools
from typing import Dict, Optional, Tuple

from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np

from algoperf import param_utils
from algoperf import spec
from algoperf.workloads.lm.workload import BaseLmWorkload


class LmWorkload(BaseLmWorkload):
  pass
