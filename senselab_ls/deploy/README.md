# Deploying the diarization backend

Turnkey artifacts for running `senselab_ls/backends/diarization` on the EC2 box provisioned per
`../AWS_EC2_SETUP.md`.

## Files
- `backend.env.example` — copy to `backend.env` and fill in secrets/paths (never commit the copy).
- `diarization.service` — systemd unit (preferred; expects the repo at `/opt/senselab`, venv at
  `/opt/lsml-venv`, env at `senselab_ls/deploy/backend.env`).
- `run_diarization_backend.sh` — manual launcher (loads `backend.env`, starts on `PORT`, 9090).
- `labeling_config_diarization.xml` — the Label Studio project's labeling config.

## Bring-up (on the instance)
```bash
cd /opt/senselab
cp senselab_ls/deploy/backend.env.example senselab_ls/deploy/backend.env
# edit backend.env: HF_TOKEN, LABEL_STUDIO_URL, LABEL_STUDIO_API_KEY, B2AI_DATASET_ROOT

# option A — systemd
sudo cp senselab_ls/deploy/diarization.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now diarization

# option B — foreground
bash senselab_ls/deploy/run_diarization_backend.sh
```

## Register in Label Studio
1. Create/point a project at the audio tasks; set its labeling config to
   `labeling_config_diarization.xml`.
2. Project → Settings → Model → add `http://<ec2-host>:9090` (scope the security group to the LS
   server's IP; prefer TLS + a token header). Enable "Retrieve predictions when loading a task
   automatically".

## Smoke test
```bash
curl http://<ec2-host>:9090/health
```
First real prediction downloads the pyannote model (needs `HF_TOKEN` + the accepted licence).
