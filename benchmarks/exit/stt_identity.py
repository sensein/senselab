"""Step 3: one plan hash and equal output identities for `SpeechToText` over LibriSpeech utterances, on any host.

`stt_identity.py plan --work DIR` plans under the packaged profiles of `exit.yaml` and each
`--profile FILE` (a site profile, at least one) and prints the field-level diff, which
must lie in `placement`. `run --work DIR --profile P --executor E --out FILE` runs under profile `P` on `E`
(`in-process`, `pool-local`, `pool-slurm`) and writes record id to output identity; `compare A B` compares two files.
"""

import sys
from collections.abc import Sequence
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import exitlib  # noqa: E402 - beside this script

STEP = "stt"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the step's command line."""
    return exitlib.identity_cli(STEP, argv)


if __name__ == "__main__":
    sys.exit(main())
