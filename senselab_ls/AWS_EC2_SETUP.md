# AWS setup — on-demand EC2 for senselab Label Studio ML backends

Runbook for creating a **persistent, start-on-demand** EC2 instance that hosts the senselab
Label Studio ML backend(s). The instance stays *stopped* (you pay only for its EBS volume) and
is started when you want to serve predictions, then stopped again.

**Every step is given two ways** — **Console** (the AWS web UI, the default/recommended path)
and **CLI** (equivalent `aws` commands). Do a step one way or the other, not both. Placeholders
are in `<ANGLE_BRACKETS>`; never commit real account IDs, key material, or tokens. Region used
throughout: **`us-east-2`**.

## Naming & sizing conventions (used by every step)

| Thing | Value |
|---|---|
| Name tag | `senselab-ls-ml` |
| Region | `us-east-2` |
| Instance type | `g4dn.xlarge` (1× T4 GPU) — CPU fallback `c6i.2xlarge`; bigger `g5.xlarge` (A10G) |
| Root EBS | `gp3`, 100 GB (OS + venv + HF model cache) |
| Key pair | `senselab-ls-ml-key` |
| Security group | `senselab-ls-ml-sg` |
| Backend ports | `9090` diarization, `9091` asr, `9092` scene |

**On-demand economics:** stopped ⇒ you pay only EBS (~$0.08/GB-month ⇒ ~$8/mo at 100 GB).
Running `g4dn.xlarge` ⇒ ~$0.53/hr on-demand.

---

## Step 0 — IAM prerequisite

The creds currently on the dev machine are the scoped **`b2ai-upload-temp`** user, which is
**not authorized for EC2** (verified: `DescribeVpcs`/`DescribeKeyPairs` → `UnauthorizedOperation`).
You need an IAM principal with EC2 rights. Minimum: `AmazonEC2FullAccess` for provisioning +
lifecycle, and (for Step 3) either `IAMFullAccess` or a pre-made instance-profile role.

**Via Console**
1. Sign in to the AWS Console with an admin/EC2-capable user (not `b2ai-upload-temp`).
2. Top-right → confirm the account, and set the **Region** selector to **US East (Ohio)
   us-east-2**.
3. IAM → Users → your user → **Permissions** → confirm `AmazonEC2FullAccess` (attach if absent).

**Via CLI**
```bash
export AWS_PROFILE=<ec2-capable-profile>
export AWS_DEFAULT_REGION=us-east-2
aws sts get-caller-identity   # confirm you are NOT b2ai-upload-temp
```

---

## Step 1 — Key pair (SSH access)

**Via Console**
1. EC2 → **Network & Security → Key Pairs** → **Create key pair**.
2. Name `senselab-ls-ml-key`, type **RSA**, format **.pem** → **Create**.
3. The `.pem` downloads once — move it to `~/.ssh/` and `chmod 400` it.

**Via CLI**
```bash
aws ec2 create-key-pair --key-name senselab-ls-ml-key \
  --query 'KeyMaterial' --output text > ~/.ssh/senselab-ls-ml-key.pem
chmod 400 ~/.ssh/senselab-ls-ml-key.pem
```

> You can skip SSH entirely and use **SSM Session Manager** (Step 7) instead.

---

## Step 2 — Security group

The **Label Studio server connects inbound** to the backend ports — restrict the source to it,
never open `0.0.0.0/0`. **app.humansignal.com (cloud)** connects from three published egress IPs
(verify current values in the HumanSignal SaaS docs before relying on them):

```
3.219.3.197/32   34.237.73.3/32   44.216.17.242/32
```

**Via Console**
1. EC2 → **Network & Security → Security Groups** → **Create security group**.
2. Name `senselab-ls-ml-sg`, VPC = the default VPC.
3. **Inbound rules → Add rule**:
   - SSH `22` — Source: *My IP* (only if using SSH; SSM avoids this).
   - Custom TCP `9090` (+ `9091`/`9092` when ASR/scene land) — add one rule **per HumanSignal
     egress IP** above as the Source.
4. **Create security group**.

**Via CLI**
```bash
SG_ID=$(aws ec2 create-security-group --group-name senselab-ls-ml-sg \
  --description "senselab LS ML backend" \
  --vpc-id "$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)" \
  --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 \
  --cidr "$(curl -s https://checkip.amazonaws.com)/32"          # SSH from your IP (optional)
for ip in 3.219.3.197 34.237.73.3 44.216.17.242; do             # HumanSignal cloud egress IPs
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 9090 --cidr "$ip/32"
done
```

> **HTTP vs HTTPS:** the quickest path is plain HTTP on `9090` restricted to those 3 IPs (the
> payload is audio refs + predictions; audio bytes are fetched separately from S3). For anything
> beyond a test, front the backend with TLS (nginx/caddy) and register the `https://` URL.

---

## Step 3 — Instance role (S3 audio + SSM shell) — recommended

Lets the box (a) read the b2ai dataset **directly from S3** and (b) be reached via SSM without
an open SSH port.

The b2ai dataset lives under `s3://<B2AI_DATASET_BUCKET>/data` (BIDS root: `phenotype/` +
`sub-*/`). The backend reads it via `B2AI_DATASET_ROOT=s3://<B2AI_DATASET_BUCKET>/data`, so the
role needs read on exactly that bucket/prefix (fill in the real bucket name — kept out of this
repo intentionally):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "ListDatasetBucket", "Effect": "Allow", "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::<B2AI_DATASET_BUCKET>",
      "Condition": { "StringLike": { "s3:prefix": ["data/*"] } } },
    { "Sid": "ReadDatasetObjects", "Effect": "Allow", "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::<B2AI_DATASET_BUCKET>/data/*" }
  ]
}
```

> If Label Studio serves audio from a *different* bucket, add that bucket's ARNs too.

**Via Console**
1. IAM → **Policies → Create policy** → JSON tab → paste the policy above → name it
   `senselab-b2ai-s3-read`.
2. IAM → **Roles → Create role** → Trusted entity **AWS service → EC2**.
3. Attach **AmazonSSMManagedInstanceCore** and **senselab-b2ai-s3-read**.
4. Name it `senselab-ls-ml-role` → **Create role** (an instance profile of the same name is
   created automatically for Console launches).

**Via CLI**
```bash
cat > /tmp/ec2-trust.json <<'JSON'
{ "Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
# save the JSON policy above to /tmp/b2ai-s3-read.json first
aws iam create-role --role-name senselab-ls-ml-role \
  --assume-role-policy-document file:///tmp/ec2-trust.json
aws iam attach-role-policy --role-name senselab-ls-ml-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam put-role-policy --role-name senselab-ls-ml-role \
  --policy-name senselab-b2ai-s3-read --policy-document file:///tmp/b2ai-s3-read.json
aws iam create-instance-profile --instance-profile-name senselab-ls-ml-role
aws iam add-role-to-instance-profile --instance-profile-name senselab-ls-ml-role \
  --role-name senselab-ls-ml-role
```

---

## Step 4 — First-boot bootstrap (cloud-init "user data")

You **don't run this by hand.** It is EC2 *user data*: a script the instance runs
**automatically, once, as `root`, on first boot**. You supply it at launch time (Step 5), one of
three ways:

- **Console (recommended):** paste the whole script below into **Advanced details → User data**
  in the Launch wizard (Step 5.8).
- **CLI:** save it to `/tmp/user-data.sh` and pass `run-instances --user-data file:///tmp/user-data.sh`
  (Step 5 CLI block already does this).
- **Manual fallback:** if you launch *without* user data, connect after boot (Step 6) and run the
  same commands yourself as root.

It installs `uv`, clones **this branch** (which contains `senselab_ls/`), builds the venv, and
points the HF cache at the persistent volume. **No secrets here** — `HF_TOKEN` / `LABEL_STUDIO_*`
are set post-boot in Step 6. Progress is logged on the instance to
`/var/log/cloud-init-output.log`; first boot takes several minutes (it pulls torch, etc.), so
check that log before starting the backend.

```bash
#!/bin/bash
set -euxo pipefail
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# Clone the branch that contains senselab_ls/ (drop -b once it merges to the default branch):
git clone -b label-studio-ml-backend-docs https://github.com/sensein/senselab.git /opt/senselab || true
cd /opt/senselab
uv venv /opt/lsml-venv --python 3.12
source /opt/lsml-venv/bin/activate
uv pip install -e ".[audio]"                      # senselab (uv-locked)
uv pip install -r senselab_ls/requirements.txt    # backend extras: label-studio-ml (git) + sdk + redis + rq + boto3

mkdir -p /opt/hf-cache
echo 'HF_HOME=/opt/hf-cache' >> /etc/environment
```

---

## Step 5 — Launch the instance

Use an AWS **Deep Learning Base AMI (Ubuntu)** — it ships the NVIDIA driver, which is all we need
(our pip `torch` bundles its own CUDA runtime). AWS's current offering is
**"Deep Learning Base AMI with Single CUDA (Ubuntu 24.04)"** — that is fine (any Ubuntu DL *Base*
AMI works; the CUDA version and 22.04-vs-24.04 don't matter). Login user is `ubuntu`.
(CPU-only: a plain Ubuntu AMI.)

**Via Console**
1. EC2 → **Instances → Launch instances**.
2. **Name**: `senselab-ls-ml`.
3. **AMI**: search *Deep Learning Base AMI* → pick **Single CUDA (Ubuntu 24.04)**, **64-bit
   (x86)** — `g4dn`/`g5` are x86 instances, so use the x86_64 AMI (not arm64).
4. **Instance type**: `g4dn.xlarge`.
5. **Key pair**: `senselab-ls-ml-key` (or *Proceed without* if SSM-only).
6. **Network settings → Select existing security group** → `senselab-ls-ml-sg`.
7. **Configure storage**: 100 GiB, `gp3`.
8. **Advanced details → IAM instance profile** → `senselab-ls-ml-role`; paste Step 4 into
   **User data**.
9. **Launch instance**.

**Via CLI**
```bash
# first: save the Step 4 bootstrap script to /tmp/user-data.sh
# Find the current DL Base AMI id (the SSM parameter names change; list and pick one):
aws ssm get-parameters-by-path --path /aws/service/deeplearning/ami --recursive \
  --query "Parameters[?contains(Name,'base') && contains(Name,'ubuntu')].Name" --output text | tr '\t' '\n' | grep ami-id
AMI_ID=$(aws ssm get-parameters --names "<paste-the-chosen-parameter-name>" \
  --query 'Parameters[0].Value' --output text)
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" --instance-type g4dn.xlarge \
  --key-name senselab-ls-ml-key --security-group-ids "$SG_ID" \
  --iam-instance-profile Name=senselab-ls-ml-role \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}" \
  --user-data file:///tmp/user-data.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=senselab-ls-ml}]" \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text
```

---

## Step 6 — Connect, configure, and start the backend

Give the Step 4 bootstrap a few minutes after launch to finish before connecting.

**Connect** — Console: EC2 → Instances → select → **Connect** → **Session Manager**; or SSH:
```bash
aws ssm start-session --target "$INSTANCE_ID"
# or:  ssh -i ~/.ssh/senselab-ls-ml-key.pem ubuntu@<public-dns>
```

**1. Confirm the bootstrap finished** (it installed uv, cloned the repo, built `/opt/lsml-venv`):
```bash
tail -n 20 /var/log/cloud-init-output.log     # should show the uv pip installs completing
ls /opt/lsml-venv/bin/label-studio-ml         # this file should exist
```

**2. Set secrets via the env file** (do NOT `export` — the file persists across restarts):
```bash
cd /opt/senselab/senselab_ls/deploy
cp backend.env.example backend.env
chmod 600 backend.env
nano backend.env      # set HF_TOKEN, LABEL_STUDIO_API_KEY, and the real bucket in B2AI_DATASET_ROOT
```

**3. Start the backend** (systemd → auto-restart + survives reboot):
```bash
sudo cp /opt/senselab/senselab_ls/deploy/diarization.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now diarization
sudo systemctl status diarization --no-pager   # want: active (running)
journalctl -u diarization -f                   # watch startup / first model download
```

**4. Health check:**
```bash
curl http://localhost:9090/health              # -> {"status":"UP", ...}
```

**5. Register in Label Studio (HumanSignal cloud):** follow `senselab_ls/deploy/README.md` —
set the project's labeling config, confirm the security group allows the HumanSignal egress IPs
on 9090 (Step 2), then **Connect Model** at `http://<ec2-public-dns>:9090`.

---

## Step 7 — On-demand start / stop

**Via Console**: EC2 → Instances → select → **Instance state → Start** (before a session) /
**Stop** (when done). Stopping halts compute billing; EBS persists.

**Via CLI**
```bash
aws ec2 start-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
# ...serve...
aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
```

> **Auto-stop guard** (avoid runaway cost): an OS idle-shutdown timer, or an EventBridge rule
> that stops the instance nightly.

---

## Step 8 — Elastic IP (stable URL) — optional

The public DNS changes on every start. Either re-register the new URL in Label Studio each
session, or attach an Elastic IP for a stable address (small charge while stopped).

**Via Console**: EC2 → **Elastic IPs → Allocate** → then **Actions → Associate** to the instance.

**Via CLI**
```bash
EIP_ALLOC=$(aws ec2 allocate-address --domain vpc --query AllocationId --output text)
aws ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$EIP_ALLOC"
```

---

## Step 9 — Teardown

**Via Console**: terminate the instance (Instance state → Terminate), then delete the security
group, key pair, and release the Elastic IP.

**Via CLI**
```bash
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"   # deletes the box + root EBS
aws ec2 delete-security-group --group-id "$SG_ID"
aws ec2 delete-key-pair --key-name senselab-ls-ml-key
# aws ec2 release-address --allocation-id "$EIP_ALLOC"
```

---

## Checklist

- [ ] EC2-capable IAM principal (not `b2ai-upload-temp`), region us-east-2
- [ ] Key pair created (or SSM-only)
- [ ] Security group scoped to Label Studio's IP
- [ ] Instance role for S3 + SSM
- [ ] Launched: GPU DL AMI, `g4dn.xlarge`, 100 GB gp3, User data = Step 4 bootstrap
- [ ] `HF_TOKEN` / `LABEL_STUDIO_*` set post-boot (never in user-data)
- [ ] Start/stop verified; Elastic IP if a stable URL is needed
