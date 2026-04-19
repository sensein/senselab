# EC2 GPU Runner Setup

This document describes how to configure the AWS EC2 instance used as a
self-hosted GitHub Actions runner for GPU integration tests.

The workflow (`tests.yaml`) uses
[machulav/ec2-github-runner](https://github.com/machulav/ec2-github-runner) to
start an ephemeral EC2 instance, run GPU tests, and terminate the instance
automatically.

## Automated setup

Use the setup script for one-command provisioning:

```bash
./scripts/setup-gpu-ci.sh \
  --profile senselab \
  --repo sensein/senselab \
  --region us-east-1 \
  --instance-type g4dn.xlarge \
  --ami <ami-id>
```

The script creates IAM users, security groups, discovers subnets for multi-AZ
failover, and configures all GitHub secrets/variables. See `--help` for options.

## AMI preparation

Start from the **AWS Deep Learning Base AMI (Amazon Linux 2023)** or any
Amazon Linux 2023 AMI with NVIDIA drivers and CUDA pre-installed. The AMI must
be in the same region as the `AWS_REGION` variable configured in GitHub.

### 1. Launch an instance to build the AMI

```bash
aws ec2 run-instances \
  --profile senselab \
  --image-id ami-XXXXXXXX \
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-XXXXXXXX \
  --subnet-id subnet-XXXXXXXX \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=senselab-ami-builder}]'
```

### 2. SSH in as ec2-user and configure

```bash
ssh -i your-key.pem ec2-user@<public-ip>
```

All commands below run as `ec2-user`.

#### Install system dependencies

```bash
sudo dnf install -y jq git
```

#### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

#### Verify GPU access

The CI workflow uses `uv sync` to create a fresh venv on each run, so no
pre-built venv is needed. This allows testing with different PyTorch versions
without rebuilding the AMI. The AMI only needs CUDA drivers and uv.

Verify the NVIDIA driver and CUDA are functional:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Create the AMI

Stop the instance (or use `--no-reboot`), then:

```bash
aws ec2 create-image \
  --profile senselab \
  --instance-id i-XXXXXXXXX \
  --name "senselab-pytorch-gpu-$(date +%Y%m%d)" \
  --description "Amazon Linux 2023 + CUDA + PyTorch + uv for senselab GPU CI" \
  --no-reboot
```

Note the resulting AMI ID — pass it to `setup-gpu-ci.sh --ami`.

### 4. Terminate the builder instance

```bash
aws ec2 terminate-instances --profile senselab --instance-id i-XXXXXXXXX
```

## GitHub configuration

### Secrets (Settings → Secrets → Actions)

| Name | Description |
|------|-------------|
| `AWS_KEY_ID` | IAM access key with EC2 permissions |
| `AWS_KEY_SECRET` | Corresponding secret access key |
| `GH_TOKEN` | GitHub PAT with `repo` scope |

### Variables (Settings → Variables → Actions)

| Name | Example | Description |
|------|---------|-------------|
| `AWS_REGION` | `us-east-1` | Region where the AMI lives |
| `AWS_IMAGE_ID` | `ami-0abc123` | The AMI created above |
| `AWS_INSTANCE_TYPE` | `g4dn.xlarge` | Default GPU instance type |
| `AWS_SUBNET` | `subnet-0abc123` | Primary subnet with internet access |
| `AWS_SECURITY_GROUP` | `sg-0abc123` | Outbound HTTPS only |
| `AWS_AZ_CONFIG` | JSON array | Multi-AZ failover config (see below) |
| `WORKING_DIR` | `/tmp/senselab` | Cache directory on instance |

### Multi-AZ failover

The workflow uses `availability-zones-config` to try multiple AZs when spot
capacity is unavailable. Format:

```json
[
  {"imageId": "ami-xxx", "subnetId": "subnet-az-a", "securityGroupId": "sg-xxx"},
  {"imageId": "ami-xxx", "subnetId": "subnet-az-b", "securityGroupId": "sg-xxx"},
  {"imageId": "ami-xxx", "subnetId": "subnet-az-c", "securityGroupId": "sg-xxx"}
]
```

### GPU label overrides

Add labels to PRs for custom instance selection:

| Label | Effect |
|-------|--------|
| `gpu-instance:<type>` | Exact instance override (e.g., `gpu-instance:p3.2xlarge`) |
| `gpu-multi` | Multi-GPU (g5.12xlarge, 4× A10G) |
| `gpu-family:<family>` | Family default (g4dn, g5, g6, p3, p4d, p5) |
| `gpu-ondemand:true` | On-demand instead of spot pricing |

## IAM policy (minimum permissions)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:CreateTags"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "*"
    }
  ]
}
```

## Updating the AMI

The AMI only contains CUDA drivers and uv — Python dependencies are installed
fresh on each run via `uv sync`. Rebuild the AMI only when:

- NVIDIA drivers need updating
- uv needs updating
- System packages (jq, git, ffmpeg) need updating

To rebuild, repeat steps 1-4 above with a newer base AMI.
# EC2 GPU smoke test - Sat Apr 18 20:44:59 EDT 2026
# EC2 verification Sun Apr 19 17:03:14 EDT 2026
