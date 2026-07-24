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
cd /opt/senselab/senselab_ls/deploy
cp backend.env.example backend.env
chmod 600 backend.env
nano backend.env    # fill in the 3 "CHANGE THESE": HF_TOKEN, LABEL_STUDIO_API_KEY,
                    # and the real bucket in B2AI_DATASET_ROOT. The rest can stay as-is.
cd /opt/senselab

# option A — systemd
sudo cp senselab_ls/deploy/diarization.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now diarization

# option B — foreground
bash senselab_ls/deploy/run_diarization_backend.sh
```

## Register in Label Studio (HumanSignal cloud)
(`LABEL_STUDIO_URL` / `LABEL_STUDIO_API_KEY` were set in `backend.env` during bring-up.)
1. Create/point a project at the audio tasks; set its labeling config to
   `labeling_config_diarization.xml`.
2. Ensure the security group allows the HumanSignal egress IPs on `9090` (see
   `../AWS_EC2_SETUP.md` Step 2).
3. Project → Settings → Model → **Connect Model** → URL `http://<ec2-public-dns>:9090`
   (or the `https://` URL if fronted with TLS). Enable "Retrieve predictions when loading a task
   automatically".

**Audio source:** if task `data.audio` is an `s3://` key into the b2ai bucket, the backend reads
it directly via the instance role (no LS round-trip). If audio is uploaded into LS/HumanSignal
storage, it is fetched over the LS API using `LABEL_STUDIO_URL` + `LABEL_STUDIO_API_KEY`.

## Smoke test
```bash
curl http://<ec2-host>:9090/health
```
First real prediction downloads the pyannote model (needs `HF_TOKEN` + the accepted licence).
