"""Tests Pydra Helping functions."""

import pydra
import torch


@pydra.mark.task
def pydra_task(test_input: torch.Tensor) -> torch.Tensor:
    """Task function for Pydra workflow to run."""
    return test_input + 2


def test_pydra() -> None:
    """Test simple tensor pydra workflow."""
    wf = pydra.Workflow(name="wf_test", input_spec=["x"])
    wf.split("x", x=[torch.tensor([[3, 4], [5, 6]]), torch.tensor([[0, 1], [1, 2]])])

    wf.add(pydra_task(name="test_task_task", test_input=wf.lzin.x))
    wf.set_output([("wf_out", wf.test_task_task.lzout.out)])

    with pydra.Submitter(plugin="serial") as sub:
        sub(wf)

    results = wf.result()

    assert results[0].output.wf_out.equal(torch.tensor([[5, 6], [7, 8]]))
    assert results[1].output.wf_out.equal(torch.tensor([[2, 3], [3, 4]]))
