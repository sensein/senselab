"""``python -m senselab.utils.reproduce``: the command line in `senselab.utils.reproduce.cli`.

The guard matters: the adapter runner imports every module of this package, ``__main__`` included.
"""

import sys

from senselab.utils.reproduce.cli import main

if __name__ == "__main__":
    sys.exit(main())
