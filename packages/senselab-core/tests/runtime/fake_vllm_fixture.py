"""A stand-in `vllm` executable: run as `python fake_vllm_fixture.py serve <model> --flag=value ...`; loads nothing.

It serves `/health`, `/version` (`FAKE_VLLM_VERSION`, default `0.31.0`), `/v1/models` (id: `--served-model-name`,
else the model argument, or `FAKE_VLLM_ROOT`; root: the model argument) and `POST /v1/chat/completions` (the reply
names `FAKE_VLLM_REPLY_MODEL`, default the served name, and its text is `FAKE_VLLM_CONTENT`, default `ok`) on
`127.0.0.1:<--port>`. It forks one child into its process group, as vLLM's engine process is, and writes
`<pid>.json` (`argv`, `pgid`, `pids`, and `sampler`, its `VLLM_USE_FLASHINFER_SAMPLER`) into `FAKE_VLLM_DIR`.
`FAKE_VLLM_EXIT` makes it exit 1 before listening, after writing `FAKE_VLLM_STDERR` to stderr. A server's launch
passes no such variable, so each is also read from the JSON object in `fake_vllm.json` beside the `bin/vllm` that runs
this fake, which names that file as `FAKE_VLLM_KNOBS`.
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

if os.path.exists(knobs := os.environ.get("FAKE_VLLM_KNOBS", "")):
    with open(knobs) as file:
        os.environ.update(json.load(file))
argv = sys.argv[1:]
assert argv[0] == "serve"
model = argv[1]
flags = dict(arg[2:].partition("=")[::2] for arg in argv[2:] if arg.startswith("--"))
name = flags.get("served-model-name", model)
out = os.environ["FAKE_VLLM_DIR"]


class Handler(BaseHTTPRequestHandler):
    """The fake's endpoints; it logs nothing."""

    def log_message(self, *args: object) -> None:
        """Log nothing."""

    def reply(self, body: object) -> None:
        """Send `body` as a 200 JSON reply."""
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - the http.server hook name
        """Serve the health, version and model probes."""
        if self.path == "/health":
            return self.reply({})
        if self.path == "/version":
            return self.reply({"version": os.environ.get("FAKE_VLLM_VERSION", "0.31.0")})
        return self.reply({"object": "list", "data": [{"id": os.environ.get("FAKE_VLLM_ROOT", name), "root": model}]})

    def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
        """Answer a chat request."""
        self.rfile.read(int(self.headers["Content-Length"]))
        served = os.environ.get("FAKE_VLLM_REPLY_MODEL", name)
        self.reply(
            {
                "model": served,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": os.environ.get("FAKE_VLLM_CONTENT", "ok")}}
                ],
            }
        )


if os.environ.get("FAKE_VLLM_EXIT"):
    print(os.environ.get("FAKE_VLLM_STDERR", ""), file=sys.stderr)
    sys.exit(1)
child = os.fork()
if child == 0:
    while True:
        time.sleep(1)
with open(os.path.join(out, f"{os.getpid()}.json"), "w") as handle:
    record = {"argv": argv, "pgid": os.getpgid(0), "pids": [os.getpid(), child]}
    json.dump({**record, "sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")}, handle)
HTTPServer(("127.0.0.1", int(flags["port"])), Handler).serve_forever()
