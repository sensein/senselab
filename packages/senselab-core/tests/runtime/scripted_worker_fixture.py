"""A worker that answers each request with the next reply line listed in `init["replies"]`.

`$ID` in a reply is replaced by the request id and `$DIR` by the array directory. It lets tests send
replies `serve` never would.
"""

import json
import sys

from senselab_worker import WIRE_VERSION, runtime_identity


def main() -> None:
    """Speak the handshake, then reply from the script until stop or EOF."""
    out = sys.stdout.buffer
    hello = {"hello": {"wire": WIRE_VERSION, "runtime": runtime_identity(), "python": "", "numpy": "", "pid": 0}}
    out.write(json.dumps(hello).encode() + b"\n")
    out.flush()
    init = json.loads(sys.stdin.buffer.readline())["init"]
    replies = iter(r.replace("$DIR", init["array_dir"]) for r in init["replies"])
    out.write(b'{"ready": {"load_s": 0}}\n')
    out.flush()
    for line in sys.stdin.buffer:
        msg = json.loads(line)
        if "stop" in msg:
            return
        out.write(next(replies).replace("$ID", msg["id"]).encode() + b"\n")
        out.flush()


if __name__ == "__main__":
    main()
