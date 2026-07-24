"""WSGI entrypoint for the diarization backend.

Run directly with the venv Python (systemd / launcher do this)::

    /opt/lsml-venv/bin/python senselab_ls/backends/diarization/_wsgi.py -p 9090 --host 0.0.0.0

``app`` is also importable for a WSGI server (e.g. gunicorn
``senselab_ls.backends.diarization._wsgi:app``). Requires ``senselab_ls`` on ``PYTHONPATH``
(the systemd unit sets ``PYTHONPATH=/opt/senselab`` via ``backend.env``).
"""

from __future__ import annotations

import argparse
import os

from label_studio_ml.api import init_app

from senselab_ls.backends.diarization.model import DiarizationBackend

app = init_app(
    model_class=DiarizationBackend,
    basic_auth_user=os.getenv("BASIC_AUTH_USER"),
    basic_auth_pass=os.getenv("BASIC_AUTH_PASS"),
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="senselab diarization ML backend")
    parser.add_argument("-p", "--port", dest="port", type=int, default=9090, help="server port")
    parser.add_argument("--host", dest="host", type=str, default="0.0.0.0", help="server host")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, threaded=True)
