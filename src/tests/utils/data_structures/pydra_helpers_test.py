"""Unit tests for Pydra helper functions.

This test defines a trivial Pydra task and a workflow that splits over a list
of tensors, then verifies the outputs after execution in debug (serial) mode.
"""

from typing import Any, Sequence

import torch
from pydra.compose import python, workflow

import senselab  # noqa: F401  # ignore unused import (kept to ensure package init side-effects)


def test_pydra() -> None:
    """Test Pydra workflow submission and execution."""
    @python.define
    def pydra_task(test_input: torch.Tensor) -> torch.Tensor:
        return test_input + 2

    @workflow.define
    def wf_test(x: Sequence[torch.Tensor]) -> Any:  # noqa: ANN401
        node = workflow.add(pydra_task().split(test_input=x))
        return node.out

    tensors: list[torch.Tensor] = [
        torch.tensor([[3, 4], [5, 6]]),
        torch.tensor([[0, 1], [1, 2]]),
    ]

    wf = wf_test(x=tensors)
    res: Any = wf(worker="debug")

    assert res.out[0].equal(torch.tensor([[5, 6], [7, 8]]))
    assert res.out[1].equal(torch.tensor([[2, 3], [3, 4]]))
