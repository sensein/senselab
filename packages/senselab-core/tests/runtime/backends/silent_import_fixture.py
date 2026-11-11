"""A worker module that blocks at import, so its worker never says hello."""

import threading

threading.Event().wait()
