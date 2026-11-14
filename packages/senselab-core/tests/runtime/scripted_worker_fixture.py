"""A worker that answers each request with the next reply listed in `init["replies"]`.

Each reply is a list of lines. `$ID` in a line is replaced by the request id, `$DIR` by the array
directory and `$PAD` by `init["pad"]` copies of `y`. A line `$EXIT:<code>` makes the worker exit at
once with that code, and a line starting `$RAW:` is written without its prefix and without a
newline. A request after the last reply raises `RepliesExhausted`. It lets tests send replies
`serve` never would.
"""

import json
import os
import sys

from senselab_worker import WIRE_VERSION, runtime_identity


class RepliesExhausted(RuntimeError):
    """A request arrived after the script's last reply."""


def main() -> None:
    """Speak the handshake, then reply from the script until stop or EOF."""
    out = sys.stdout.buffer
    hello = {"hello": {"wire": WIRE_VERSION, "runtime": runtime_identity(), "python": "", "numpy": "", "pid": 0}}
    out.write(json.dumps(hello).encode() + b"\n")
    out.flush()
    init = json.loads(sys.stdin.buffer.readline())["init"]
    replies = iter(init["replies"])
    pad = "y" * init.get("pad", 0)
    out.write(b'{"ready": {"load_s": 0}}\n')
    out.flush()
    for line in sys.stdin.buffer:
        msg = json.loads(line)
        if "stop" in msg:
            return
        reply_lines = next(replies, None)
        if reply_lines is None:
            raise RepliesExhausted(f"request {msg['id']} arrived after the last scripted reply")
        for reply in reply_lines:
            text = reply.replace("$DIR", init["array_dir"]).replace("$ID", msg["id"]).replace("$PAD", pad)
            if text.startswith("$EXIT:"):
                os._exit(int(text[len("$EXIT:") :]))
            out.write(text[len("$RAW:") :].encode() if text.startswith("$RAW:") else text.encode() + b"\n")
            out.flush()


if __name__ == "__main__":
    main()
