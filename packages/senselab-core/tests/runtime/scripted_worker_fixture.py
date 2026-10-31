"""A worker that answers each request with the next reply listed in `init["replies"]`.

Each reply is a list of lines. `$ID` in a line is replaced by the request id and `$DIR` by the array
directory. A line `$EXIT:<code>` makes the worker exit at once with that code, and a line starting
`$RAW:` is written without its prefix and without a newline. It lets tests send replies `serve`
never would.
"""

import json
import os
import sys

from senselab_worker import WIRE_VERSION, runtime_identity


def main() -> None:
    """Speak the handshake, then reply from the script until stop or EOF."""
    out = sys.stdout.buffer
    hello = {"hello": {"wire": WIRE_VERSION, "runtime": runtime_identity(), "python": "", "numpy": "", "pid": 0}}
    out.write(json.dumps(hello).encode() + b"\n")
    out.flush()
    init = json.loads(sys.stdin.buffer.readline())["init"]
    replies = iter(init["replies"])
    out.write(b'{"ready": {"load_s": 0}}\n')
    out.flush()
    for line in sys.stdin.buffer:
        msg = json.loads(line)
        if "stop" in msg:
            return
        for reply in next(replies):
            text = reply.replace("$DIR", init["array_dir"]).replace("$ID", msg["id"])
            if text.startswith("$EXIT:"):
                os._exit(int(text[len("$EXIT:") :]))
            out.write(text[len("$RAW:") :].encode() if text.startswith("$RAW:") else text.encode() + b"\n")
            out.flush()


if __name__ == "__main__":
    main()
