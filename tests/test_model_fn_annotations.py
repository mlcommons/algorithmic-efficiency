import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize(
  'workload_path',
  [
    Path(
      'algoperf/workloads/librispeech_conformer/librispeech_jax/workload.py'
    ),
    Path(
      'algoperf/workloads/librispeech_conformer/librispeech_pytorch/workload.py'
    ),
  ],
)
def test_librispeech_conformer_model_fn_return_annotation(
  workload_path: Path,
):
  module = ast.parse(workload_path.read_text(encoding='utf-8'))
  model_fn = next(
    node
    for node in ast.walk(module)
    if isinstance(node, ast.FunctionDef) and node.name == 'model_fn'
  )

  assert ast.unparse(model_fn.returns) == (
    'Tuple[Tuple[spec.Tensor, spec.Tensor], spec.ModelAuxiliaryState]'
  )
