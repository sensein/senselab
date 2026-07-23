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

Restrict inbound to the **Label Studio server's** IP — that host must reach the backend ports,
not your laptop. Never open `0.0.0.0/0`. If LS is HumanSignal-cloud, whitelist its egress range
or front the box with TLS + a token header.

**Via Console**
1. EC2 → **Network & Security → Security Groups** → **Create security group**.
2. Name `senselab-ls-ml-sg`, VPC = the default VPC.
3. **Inbound rules → Add rule** for each:
   - SSH `22` — Source: *My IP* (only if using SSH).
   - Custom TCP `9090`, `9091`, `9092` — Source: Label Studio's IP/CIDR.
4. **Create security group**.

**Via CLI**
```bash
MY_IP=$(curl -s https://checkip.amazonaws.com)/32
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 create-security-group --group-name senselab-ls-ml-sg \
  --description "senselab LS ML backend" --vpc-id "$VPC_ID" --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MY_IP"
for p in 9090 9091 9092; do
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port $p --cidr "$MY_IP"
done
```

---

## Step 3 — Instance role (S3 audio + SSM shell) — recommended

Lets the box (a) read audio **directly from S3** (the same bucket LS syncs tasks from — faster
than the LS API round-trip) and (b) be reached via SSM without an open SSH port. Scope the S3
policy down to the one bucket in production.

**Via Console**
1. IAM → **Roles → Create role** → Trusted entity **AWS service → EC2**.
2. Attach policies: **AmazonSSMManagedInstanceCore** and **AmazonS3ReadOnlyAccess**.
3. Name it `senselab-ls-ml-role` → **Create role** (an instance profile of the same name is
   created automatically for Console launches).

**Via CLI**
```bash
cat > /tmp/ec2-trust.json <<'JSON'
{ "Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam create-role --role-name senselab-ls-ml-role \
  --assume-role-policy-document file:///tmp/ec2-trust.json
aws iam attach-role-policy --role-name senselab-ls-ml-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam attach-role-policy --role-name senselab-ls-ml-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
aws iam create-instance-profile --instance-profile-name senselab-ls-ml-role
aws iam add-role-to-instance-profile --instance-profile-name senselab-ls-ml-role \
  --role-name senselab-ls-ml-role
```

---

## Step 4 — First-boot bootstrap script (shared by both paths)

This installs `uv`, senselab, and the LS SDK, and puts the HF cache on the persistent volume.
**No secrets here** — `HF_TOKEN` / `LABEL_STUDIO_*` are set post-boot (Step 6). You paste it into
the Console launch wizard's *User data* box, or pass it to `run-instances --user-data`.

```bash
#!/bin/bash
set -euxo pipefail
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

git clone https://github.com/sensein/senselab.git /opt/senselab || true
cd /opt/senselab
uv venv /opt/lsml-venv --python 3.12
source /opt/lsml-venv/bin/activate
uv pip install -e ".[audio]" label-studio-ml label-studio-sdk boto3

mkdir -p /opt/hf-cache
echo 'HF_HOME=/opt/hf-cache' >> /etc/environment
```

---

## Step 5 — Launch the instance

Use an AWS **Deep Learning Base OSS (Ubuntu 22.04)** AMI — NVIDIA driver + CUDA preinstalled.
(CPU-only: a plain Ubuntu 22.04 AMI.)

**Via Console**
1. EC2 → **Instances → Launch instances**.
2. **Name**: `senselab-ls-ml`.
3. **AMI**: search *Deep Learning Base OSS Nvidia Driver GPU Ubuntu 22.04* → select it.
4. **Instance type**: `g4dn.xlarge`.
5. **Key pair**: `senselab-ls-ml-key` (or *Proceed without* if SSM-only).
6. **Network settings → Select existing security group** → `senselab-ls-ml-sg`.
7. **Configure storage**: 100 GiB, `gp3`.
8. **Advanced details → IAM instance profile** → `senselab-ls-ml-role`; paste Step 4 into
   **User data**.
9. **Launch instance**.

**Via CLI**
```bash
AMI_ID=$(aws ssm get-parameters \
  --names /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
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

## Step 6 — Connect, set secrets, start backends

**Via Console**
1. EC2 → Instances → select it → **Connect** → **Session Manager** (or **SSH client** for the
   `ssh` command with your `.pem`).
2. Set secrets and start the backends (see `ML_BACKEND_PLAN.md`):
   ```bash
   export HF_TOKEN=... LABEL_STUDIO_URL=https://<ls-host> LABEL_STUDIO_API_KEY=...
   ```

**Via CLI**
```bash
aws ssm start-session --target "$INSTANCE_ID"
# or: ssh -i ~/.ssh/senselab-ls-ml-key.pem ubuntu@<public-dns>
```

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
