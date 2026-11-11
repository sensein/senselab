"""Total teardown: every step runs, and the first error is raised once all of them did."""

from collections.abc import Callable


def run_all(*steps: Callable[[], object]) -> None:
    """Run each of `steps` in order, even when an earlier one raised; then raise the first error.

    Later errors are attached to the first as notes, so none is lost.

    Raises:
        Exception: the first error a step raised.
    """
    errors: list[Exception] = []
    for step in steps:
        try:
            step()
        except Exception as err:  # noqa: BLE001 - every later step must still run
            errors.append(err)
    if errors:
        first, *later = errors
        for other in later:
            first.add_note(f"a later teardown step also failed: {other!r}")
        raise first
