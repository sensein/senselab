# AWS setup — on-demand EC2 for senselab Label Studio ML backends

Runbook for creating a **persistent, start-on-demand** EC2 instance that hosts the senselab
Label Studio ML backend(s). The instance stays *stopped* (you pay only for its EBS volume)
and is started when you want to serve predictions, then stopped again.

> All commands use placeholders in `<ANGLE_BRACKETS>`. Never commit real account IDs, key
> material, or tokens. Region default here is `us-east-2`.

---

## 0. Prerequisites — IAM

The creds currently on the dev machine are the scoped **`b2ai-upload-temp`** user, which is
**not authorized for EC2** (verified: `DescribeVpcs`/`DescribeKeyPairs` → `UnauthorizedOperation`).
You need an IAM principal (user or role) with EC2 admin-ish rights to run this runbook. Minimum
managed policy for provisioning + lifecycle:

- `AmazonEC2FullAccess` (or a scoped policy allowing `ec2:RunInstances`, `Describe*`,
  `Start/Stop/TerminateInstances`, `CreateSecurityGroup`, `AuthorizeSecurityGroupIngress`,
  `CreateKeyPair`, `CreateTags`, `CreateVolume`, `AllocateAddress`).
- `IAMFullAccess` **or** a pre-made instance-profile role (see step 4) — to attach a role that
  lets the box read audio from S3 and be managed via SSM.

Set the working profile/region once:

```bash
export AWS_PROFILE=<ec2-capable-profile>
export AWS_DEFAULT_REGION=us-east-2
aws sts get-caller-identity   # confirm you are NOT b2ai-upload-temp
```

---

## 1. Parameters (edit these)

```bash
NAME=senselab-ls-ml
REGION=us-east-2
INSTANCE_TYPE=g4dn.xlarge          # 1x T4 GPU, 4 vCPU, 16 GB — CPU fallback: c6i.2xlarge
VOLUME_GB=100                      # EBS root: OS + venv + HF model cache
KEY_NAME=${NAME}-key
SG_NAME=${NAME}-sg
MY_IP=$(curl -s https://checkip.amazonaws.com)/32   # your current IP for SSH/health
```

- **GPU vs CPU**: `g4dn.xlarge` (T4) is the cheap GPU default; pyannote diarization + Whisper
  run comfortably. `c6i.2xlarge` works CPU-only but is much slower. Pick larger `g5.xlarge`
  (A10G) if you batch many long files.
- **On-demand economics**: stopped instance ⇒ you pay only EBS (~$0.08/GB-month ⇒ ~$8/mo at
  100 GB). Running `g4dn.xlarge` ⇒ ~$0.53/hr on-demand.

---

## 2. AMI — GPU drivers preinstalled

Use an AWS **Deep Learning Base OSS** AMI (NVIDIA driver + CUDA already set up) so you don't
build drivers by hand. Resolve the latest ID via SSM Parameter Store:

```bash
AMI_ID=$(aws ssm get-parameters \
  --names /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query 'Parameters[0].Value' --output text)
echo "$AMI_ID"
```

(CPU-only: use a plain Ubuntu 22.04 AMI and skip CUDA.)

---

## 3. Key pair + security group

```bash
# SSH key — private key saved locally, chmod 400
aws ec2 create-key-pair --key-name "$KEY_NAME" \
  --query 'KeyMaterial' --output text > ~/.ssh/${KEY_NAME}.pem
chmod 400 ~/.ssh/${KEY_NAME}.pem

# Security group in the default VPC
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" \
  --description "senselab LS ML backend" --vpc-id "$VPC_ID" \
  --query 'GroupId' --output text)

# Inbound: SSH + the ML backend port(s), restricted to your IP.
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr "$MY_IP"
# One rule per backend port (9090 diarization, 9091 asr, 9092 scene ...):
for p in 9090 9091 9092; do
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --protocol tcp --port $p --cidr "$MY_IP"
done
```

> **Who connects?** The *Label Studio server* must reach these ports, not your laptop. Set the
> CIDR to Label Studio's egress IP (or its VPC). If LS is HumanSignal-cloud, whitelist their
> published egress range or front the box with TLS + a token header. Do **not** open `0.0.0.0/0`.

---

## 4. Instance role (S3 audio + SSM shell) — optional but recommended

Lets the box (a) pull audio from S3 without embedding keys and (b) be reached via **SSM Session
Manager** (no SSH, no open port 22). (a) is the recommended audio path: since Label Studio syncs
its tasks *from* an S3 bucket, the backend can read the same `s3://` objects directly instead of
downloading through the LS API — scope `AmazonS3ReadOnlyAccess` down to that one bucket in prod.
Create once:

```bash
# Trust policy
cat > /tmp/ec2-trust.json <<'JSON'
{ "Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam create-role --role-name ${NAME}-role \
  --assume-role-policy-document file:///tmp/ec2-trust.json
aws iam attach-role-policy --role-name ${NAME}-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam attach-role-policy --role-name ${NAME}-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess   # scope down in prod
aws iam create-instance-profile --instance-profile-name ${NAME}-profile
aws iam add-role-to-instance-profile --instance-profile-name ${NAME}-profile \
  --role-name ${NAME}-role
```

---

## 5. Bootstrap (user-data)

Runs on first boot: install `uv`, senselab, and the ML SDK, then leave a systemd-managed
launcher. Secrets (`HF_TOKEN`, `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY`) are **not** baked in
here — set them post-boot via SSM/SSH or an env file on EBS.

```bash
cat > /tmp/user-data.sh <<'EOS'
#!/bin/bash
set -euxo pipefail
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# senselab + LS SDK into a persistent venv on the root EBS volume
git clone https://github.com/<ORG>/senselab.git /opt/senselab || true
cd /opt/senselab
uv venv /opt/lsml-venv --python 3.12
source /opt/lsml-venv/bin/activate
uv pip install -e ".[audio]" label-studio-ml label-studio-sdk

# HF model cache on the (persistent) root volume so it survives stop/start
mkdir -p /opt/hf-cache
echo 'HF_HOME=/opt/hf-cache' >> /etc/environment
EOS

USER_DATA_B64=$(base64 -w0 /tmp/user-data.sh)
```

---

## 6. Launch (create the persistent instance)

```bash
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name=${NAME}-profile \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${VOLUME_GB},VolumeType=gp3}" \
  --user-data "$USER_DATA_B64" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "launched $INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text
```

---

## 7. On-demand lifecycle

```bash
# START when you need predictions
aws ec2 start-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
PUBLIC_DNS=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)

# ... start the ML backend(s) via SSH or SSM (see ML_BACKEND_PLAN.md) ...

# STOP when done (billing for compute stops; EBS persists)
aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
```

- **Public DNS changes on every start.** Either (a) re-register the new URL in Label Studio
  each session, or (b) allocate an **Elastic IP** and associate it (small charge while the
  instance is stopped, but a stable address):
  ```bash
  EIP_ALLOC=$(aws ec2 allocate-address --domain vpc --query AllocationId --output text)
  aws ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$EIP_ALLOC"
  ```
- **Auto-stop guard** (avoid runaway cost): an OS-level idle-shutdown timer, or a scheduled
  Lambda / EventBridge rule that stops the instance nightly.

---

## 8. Access without opening SSH — SSM

With the instance profile from step 4 and SSM agent (preinstalled on DL AMIs):

```bash
aws ssm start-session --target "$INSTANCE_ID"
```

Lets you set secrets and start backends without an inbound SSH rule.

---

## 9. Teardown

```bash
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"      # deletes the box + root EBS
aws ec2 delete-security-group --group-id "$SG_ID"
aws ec2 delete-key-pair --key-name "$KEY_NAME"
# release Elastic IP if allocated:  aws ec2 release-address --allocation-id "$EIP_ALLOC"
```

---

## Checklist

- [ ] EC2-capable IAM profile (not `b2ai-upload-temp`)
- [ ] Key pair + security group scoped to Label Studio's IP
- [ ] Instance role for S3/SSM (optional)
- [ ] Launched with GPU AMI + gp3 root volume sized for HF cache
- [ ] `HF_TOKEN` / `LABEL_STUDIO_*` set post-boot (never in user-data)
- [ ] Start/stop verified; Elastic IP if a stable URL is needed
