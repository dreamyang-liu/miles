"""The Megatron train step fetches the per-sample columns a custom loss reads, without importing Megatron."""

import ast

from tests.fast.source_scan import REPO_ROOT

_MODEL = REPO_ROOT / "miles/backends/megatron_utils/model.py"


def _train_step_batch_keys() -> list[str]:
    tree = ast.parse(_MODEL.read_text())
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_forward_backward_pass")
    forward_step = next(n for n in ast.walk(outer) if isinstance(n, ast.FunctionDef) and n.name == "forward_step")
    call = next(
        n
        for n in ast.walk(forward_step)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "get_batch"
    )
    keys = call.args[1]
    assert isinstance(keys, ast.List), "the train step's get_batch keys are expected to be a list literal"
    return [element.value for element in keys.elts if isinstance(element, ast.Constant)]


def test_train_step_fetches_sample_train_metadata() -> None:
    """Sample.train_metadata travels as the "metadata" column; a custom loss sees it only if the step fetches it."""
    assert "metadata" in _train_step_batch_keys()
