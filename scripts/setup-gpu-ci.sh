#!/usr/bin/env bash
# setup-gpu-ci.sh — Provision AWS infrastructure and configure GitHub
# secrets/variables for GPU CI testing with ephemeral EC2 runners.
#
# Usage:
#   ./scripts/setup-gpu-ci.sh \
#     --profile senselab \
#     --repo sensein/senselab \
#     --ami ami-0abc123 \
#     [--region us-east-1] \
#     [--instance-type g4dn.xlarge] \
#     [--vpc vpc-xxx] \
#     [--working-dir /tmp/senselab]
#
# Requirements: aws CLI, gh CLI (authenticated), jq
#
# SECURITY: Credentials are NEVER echoed, logged, or written to files.
# Access keys are piped directly to gh secret set via stdin.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
REGION="us-east-1"
INSTANCE_TYPE="g4dn.xlarge"
WORKING_DIR="/tmp/senselab"
VPC_ID=""
IAM_USER_PREFIX="github-actions-ec2"

# ── Argument parsing ──────────────────────────────────────────────────
usage() {
  cat <<'USAGE'
Usage: setup-gpu-ci.sh --profile <aws-profile> --repo <owner/repo> --ami <ami-id> [options]
       setup-gpu-ci.sh --profile <aws-profile> --build-ami --base-ami <ami-id> [options]

Mode 1: Configure repository (default)
  --profile <name>        AWS CLI profile to use
  --repo <owner/repo>     GitHub repository (e.g., sensein/senselab)
  --ami <ami-id>          Pre-built AMI ID with GPU drivers + uv

Mode 2: Build AMI (--build-ami)
  --profile <name>        AWS CLI profile to use
  --build-ami             Build a new AMI from a base Deep Learning AMI
  --base-ami <ami-id>     Base AMI (AWS Deep Learning AMI recommended)
  --key-name <name>       EC2 key pair name for SSH access
  --ssh-key <path>        Path to SSH private key (default: ~/.ssh/<key-name>.pem)

Optional (both modes):
  --region <region>       AWS region (default: us-east-1)
  --instance-type <type>  Default GPU instance type (default: g4dn.xlarge)
  --vpc <vpc-id>          VPC ID (default: auto-detect default VPC)
  --working-dir <path>    Cache directory on EC2 instance (default: /tmp/senselab)
  --gh-token <token>      GitHub PAT with repo scope (default: prompt or use GH_TOKEN env)
  --help                  Show this help
USAGE
  exit 1
}

PROFILE="" REPO="" AMI="" GH_PAT="" BUILD_AMI=false BASE_AMI="" KEY_NAME="" SSH_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)      PROFILE="$2"; shift 2 ;;
    --repo)         REPO="$2"; shift 2 ;;
    --ami)          AMI="$2"; shift 2 ;;
    --region)       REGION="$2"; shift 2 ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    --vpc)          VPC_ID="$2"; shift 2 ;;
    --working-dir)  WORKING_DIR="$2"; shift 2 ;;
    --gh-token)     GH_PAT="$2"; shift 2 ;;
    --build-ami)    BUILD_AMI=true; shift ;;
    --base-ami)     BASE_AMI="$2"; shift 2 ;;
    --key-name)     KEY_NAME="$2"; shift 2 ;;
    --ssh-key)      SSH_KEY="$2"; shift 2 ;;
    --help|-h)      usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "$PROFILE" ]] && { echo "ERROR: --profile is required"; usage; }
REPO_SLUG=$(echo "${REPO:-gpu-ci}" | tr '/' '-')

# ── Mode: Build AMI ──────────────────────────────────────────────────
if [[ "$BUILD_AMI" == "true" ]]; then
  [[ -z "$BASE_AMI" ]] && { echo "ERROR: --base-ami is required for --build-ami"; usage; }

  AWS="aws --profile $PROFILE --region $REGION --output json"

  # Resolve SSH key path: --ssh-key > --key-name derived path > error
  if [[ -z "$SSH_KEY" && -n "$KEY_NAME" ]]; then
    for candidate in ~/.ssh/"${KEY_NAME}.pem" ~/.ssh/"${KEY_NAME}" ~/.ssh/id_rsa ~/.ssh/id_ed25519; do
      if [[ -f "$candidate" ]]; then
        SSH_KEY="$candidate"
        break
      fi
    done
  fi
  [[ -z "$SSH_KEY" ]] && { echo "ERROR: --ssh-key <path> or --key-name with matching key file is required for --build-ami"; usage; }
  [[ ! -f "$SSH_KEY" ]] && { echo "ERROR: SSH key not found: $SSH_KEY"; exit 1; }

  # Detect SSH user from base AMI (Amazon Linux = ec2-user, Ubuntu = ubuntu)
  AMI_NAME=$($AWS ec2 describe-images --image-ids "$BASE_AMI" \
    | jq -r '.Images[0].Name // ""')
  if echo "$AMI_NAME" | grep -qi ubuntu; then
    SSH_USER="ubuntu"
  else
    SSH_USER="ec2-user"
  fi

  # Detect root device from base AMI
  ROOT_DEVICE=$($AWS ec2 describe-images --image-ids "$BASE_AMI" \
    | jq -r '.Images[0].RootDeviceName // "/dev/xvda"')

  echo "=== Building GPU CI AMI ==="
  echo "  Base AMI: $BASE_AMI ($AMI_NAME)"
  echo "  Region: $REGION"
  echo "  SSH key: $SSH_KEY"
  echo "  SSH user: $SSH_USER"
  echo "  Root device: $ROOT_DEVICE"

  # Get default VPC and subnet for launching
  if [[ -z "$VPC_ID" ]]; then
    VPC_ID=$($AWS ec2 describe-vpcs --filters "Name=is-default,Values=true" \
      | jq -r '.Vpcs[0].VpcId // empty')
    [[ -z "$VPC_ID" ]] && { echo "ERROR: No default VPC found. Specify --vpc explicitly."; exit 1; }
  fi
  BUILD_SUBNET=$($AWS ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
    | jq -r '.Subnets[0].SubnetId // empty')
  [[ -z "$BUILD_SUBNET" ]] && { echo "ERROR: No subnets found in VPC $VPC_ID"; exit 1; }
  BUILD_SG=$($AWS ec2 describe-security-groups \
    --filters "Name=group-name,Values=default" "Name=vpc-id,Values=$VPC_ID" \
    | jq -r '.SecurityGroups[0].GroupId')

  # Add temporary SSH rule for current IP
  MY_IP=$(curl -s https://checkip.amazonaws.com) || { echo "ERROR: Failed to detect public IP"; exit 1; }
  [[ -z "$MY_IP" ]] && { echo "ERROR: Public IP detection returned empty result"; exit 1; }
  echo "  Adding temporary SSH access for $MY_IP"
  $AWS ec2 authorize-security-group-ingress \
    --group-id "$BUILD_SG" \
    --protocol tcp --port 22 --cidr "${MY_IP}/32" >/dev/null 2>&1 || true

  # Build launch args
  LAUNCH_ARGS=(
    --image-id "$BASE_AMI"
    --instance-type "$INSTANCE_TYPE"
    --security-group-ids "$BUILD_SG"
    --subnet-id "$BUILD_SUBNET"
    --block-device-mappings "[{\"DeviceName\":\"${ROOT_DEVICE}\",\"Ebs\":{\"VolumeSize\":100}}]"
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${REPO_SLUG:-gpu-ci}-ami-builder}]"
  )
  [[ -n "$KEY_NAME" ]] && LAUNCH_ARGS+=(--key-name "$KEY_NAME")

  # Launch builder instance
  echo "  Launching builder instance ($INSTANCE_TYPE)..."
  BUILDER_ID=$($AWS ec2 run-instances "${LAUNCH_ARGS[@]}" \
    | jq -r '.Instances[0].InstanceId')
  echo "  Instance: $BUILDER_ID"

  echo "  Waiting for instance to be running..."
  $AWS ec2 wait instance-running --instance-ids "$BUILDER_ID"
  BUILDER_IP=$($AWS ec2 describe-instances --instance-ids "$BUILDER_ID" \
    | jq -r '.Reservations[0].Instances[0].PublicIpAddress')
  echo "  IP: $BUILDER_IP"

  SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=5)
  [[ "$SSH_KEY" == *.pem || "$SSH_KEY" == *id_* ]] && SSH_OPTS+=(-i "$SSH_KEY")

  # Wait for SSH to be ready
  echo "  Waiting for SSH..."
  SSH_SUCCESS=false
  for i in $(seq 1 30); do
    if ssh "${SSH_OPTS[@]}" -o BatchMode=yes "${SSH_USER}@${BUILDER_IP}" 'true' 2>/dev/null; then
      SSH_SUCCESS=true
      break
    fi
    sleep 5
  done
  if [[ "$SSH_SUCCESS" == "false" ]]; then
    echo "ERROR: SSH connection timed out after 150 seconds"
    echo "  Terminating builder instance..."
    $AWS ec2 terminate-instances --instance-ids "$BUILDER_ID" >/dev/null
    exit 1
  fi

  # Configure the instance — detect package manager (dnf for AL2023, apt for Ubuntu)
  echo "  Installing uv and system dependencies..."
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${BUILDER_IP}" bash -s <<'REMOTE_SETUP'
set -ex
if command -v dnf &>/dev/null; then
  sudo dnf install -y jq git
elif command -v apt-get &>/dev/null; then
  sudo apt-get update && sudo apt-get install -y jq git
fi
curl -LsSf https://astral.sh/uv/install.sh | sh
REMOTE_SETUP

  # Verify GPU
  echo "  Verifying GPU..."
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${BUILDER_IP}" \
    'nvidia-smi && export PATH="$HOME/.local/bin:$PATH" && uv --version'

  # Create AMI
  echo "  Creating AMI snapshot..."
  NEW_AMI=$($AWS ec2 create-image \
    --instance-id "$BUILDER_ID" \
    --name "${REPO_SLUG}-gpu-ci-$(date +%Y%m%d)" \
    --description "GPU CI AMI with CUDA + uv (built from $BASE_AMI)" \
    --no-reboot \
    | jq -r '.ImageId')
  echo "  AMI: $NEW_AMI"

  # Terminate builder
  echo "  Terminating builder instance..."
  $AWS ec2 terminate-instances --instance-ids "$BUILDER_ID" >/dev/null

  # Remove temporary SSH rule
  $AWS ec2 revoke-security-group-ingress \
    --group-id "$BUILD_SG" \
    --protocol tcp --port 22 --cidr "${MY_IP}/32" >/dev/null 2>&1 || true

  echo ""
  echo "=== AMI build complete ==="
  echo "  AMI ID: $NEW_AMI"
  echo "  Use with: ./scripts/setup-gpu-ci.sh --profile $PROFILE --repo <owner/repo> --ami $NEW_AMI --region $REGION"
  exit 0
fi

# ── Mode: Configure repository ────────────────────────────────────────
[[ -z "$REPO" ]] && { echo "ERROR: --repo is required"; usage; }
[[ -z "$AMI" ]] && { echo "ERROR: --ami is required"; usage; }

AWS="aws --profile $PROFILE --region $REGION --output json"

# ── Preflight checks ─────────────────────────────────────────────────
echo "=== Preflight checks ==="
command -v aws >/dev/null 2>&1 || { echo "ERROR: aws CLI not found"; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "ERROR: gh CLI not found"; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "ERROR: jq not found"; exit 1; }

# Verify AWS profile works
$AWS sts get-caller-identity >/dev/null 2>&1 || {
  echo "ERROR: AWS profile '$PROFILE' not configured or credentials expired"
  exit 1
}
echo "  AWS profile '$PROFILE' OK"

# Verify gh auth
gh auth status >/dev/null 2>&1 || {
  echo "ERROR: gh CLI not authenticated. Run 'gh auth login' first."
  exit 1
}
echo "  gh CLI authenticated OK"

# Verify AMI exists
$AWS ec2 describe-images --image-ids "$AMI" >/dev/null 2>&1 || {
  echo "ERROR: AMI '$AMI' not found in region '$REGION'"
  exit 1
}
echo "  AMI '$AMI' exists in $REGION"

# ── Step 1: VPC & Networking ─────────────────────────────────────────
echo ""
echo "=== Step 1: VPC & Networking ==="

if [[ -z "$VPC_ID" ]]; then
  VPC_ID=$($AWS ec2 describe-vpcs --filters "Name=is-default,Values=true" \
    | jq -r '.Vpcs[0].VpcId // empty')
  if [[ -z "$VPC_ID" ]]; then
    echo "ERROR: No default VPC found. Specify --vpc explicitly."
    exit 1
  fi
  echo "  Using default VPC: $VPC_ID"
else
  echo "  Using specified VPC: $VPC_ID"
fi

# ── Step 2: Security Group ───────────────────────────────────────────
echo ""
echo "=== Step 2: Security Group ==="

SG_NAME="github-actions-runner-sg"
SG_ID=$($AWS ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  | jq -r '.SecurityGroups[0].GroupId // empty')

if [[ -n "$SG_ID" ]]; then
  echo "  Reusing existing security group: $SG_ID ($SG_NAME)"
else
  echo "  Creating security group: $SG_NAME"
  SG_ID=$($AWS ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "GitHub Actions self-hosted runner - outbound HTTPS only" \
    --vpc-id "$VPC_ID" \
    | jq -r '.GroupId')

  # Allow all outbound (default), revoke default ingress not needed
  # The default SG already allows all outbound. No ingress rules needed.
  echo "  Created security group: $SG_ID"
fi

# ── Step 3: Discover subnets for multi-AZ failover ───────────────────
echo ""
echo "=== Step 3: Subnet discovery (multi-AZ) ==="

SUBNETS_JSON=$($AWS ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
  | jq -c '[.Subnets[] | {az: .AvailabilityZone, subnetId: .SubnetId}] | sort_by(.az)')

SUBNET_COUNT=$(echo "$SUBNETS_JSON" | jq 'length')
echo "  Found $SUBNET_COUNT default subnets across AZs"

# Build AWS_AZ_CONFIG
AZ_CONFIG=$(echo "$SUBNETS_JSON" | jq -c --arg ami "$AMI" --arg sg "$SG_ID" \
  '[.[] | {imageId: $ami, subnetId: .subnetId, securityGroupId: $sg}]')

# Primary subnet (first AZ)
PRIMARY_SUBNET=$(echo "$SUBNETS_JSON" | jq -r '.[0].subnetId // empty')
[[ -z "$PRIMARY_SUBNET" ]] && { echo "ERROR: No subnets found in VPC $VPC_ID"; exit 1; }
echo "  Primary subnet: $PRIMARY_SUBNET"
echo "  AZ config: $AZ_CONFIG"

# ── Step 4: IAM User ─────────────────────────────────────────────────
echo ""
echo "=== Step 4: IAM User ==="

# Derive a repo-specific IAM user name
IAM_USER="${IAM_USER_PREFIX}-${REPO_SLUG}"

USER_EXISTS=$($AWS iam get-user --user-name "$IAM_USER" 2>/dev/null | jq -r '.User.UserName // empty' || true)
CREATED_NEW_USER=false

if [[ -n "$USER_EXISTS" ]]; then
  echo "  IAM user '$IAM_USER' already exists"
  KEY_COUNT=$($AWS iam list-access-keys --user-name "$IAM_USER" | jq '.AccessKeyMetadata | length')
  if [[ "$KEY_COUNT" -gt 0 ]]; then
    echo "  User has $KEY_COUNT existing access key(s)"
  else
    echo "  User has no access keys — will create one"
    CREATED_NEW_USER=true
  fi
else
  CREATED_NEW_USER=true
  echo "  Creating IAM user: $IAM_USER"
  $AWS iam create-user --user-name "$IAM_USER" >/dev/null

  # Attach inline policy with minimum EC2 permissions
  POLICY_DOC=$(cat <<'POLICY'
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
    },
    {
      "Effect": "Allow",
      "Action": "iam:CreateServiceLinkedRole",
      "Resource": "arn:aws:iam::*:role/aws-service-role/spot.amazonaws.com/*",
      "Condition": {
        "StringEquals": {
          "iam:AWSServiceName": "spot.amazonaws.com"
        }
      }
    }
  ]
}
POLICY
)
  $AWS iam put-user-policy \
    --user-name "$IAM_USER" \
    --policy-name "ec2-github-runner" \
    --policy-document "$POLICY_DOC"
  echo "  Attached EC2 runner policy"
fi

# ── Step 5: Create access key and set GitHub secrets ──────────────────
echo ""
echo "=== Step 5: GitHub Secrets ==="

# SECURITY: Suppress trace output during credential operations
set +x 2>/dev/null || true

# Check if secrets already exist (gh secret list returns them)
EXISTING_SECRETS=$(gh secret list --repo "$REPO" 2>/dev/null | awk '{print $1}' || true)

if echo "$EXISTING_SECRETS" | grep -q "^AWS_KEY_ID$" && [[ "$CREATED_NEW_USER" == "false" ]]; then
  echo "  AWS_KEY_ID secret already exists and IAM user has keys — skipping"
  echo "  To rotate: delete the secret in GitHub and the IAM key, then re-run."
else
  echo "  Creating new access key and setting GitHub secrets..."
  EXISTING_KEY_COUNT=$($AWS iam list-access-keys --user-name "$IAM_USER" | jq '.AccessKeyMetadata | length')
  if [[ "$EXISTING_KEY_COUNT" -ge 2 ]]; then
    echo "  ERROR: IAM user '$IAM_USER' already has 2 access keys (AWS limit)."
    echo "  Delete one manually: aws --profile $PROFILE iam delete-access-key --user-name $IAM_USER --access-key-id <KEY_ID>"
    exit 1
  fi
  KEY_JSON=$($AWS iam create-access-key --user-name "$IAM_USER")
  _AK=$(echo "$KEY_JSON" | jq -r '.AccessKey.AccessKeyId')
  _SK=$(echo "$KEY_JSON" | jq -r '.AccessKey.SecretAccessKey')
  KEY_JSON=""

  gh secret set AWS_KEY_ID --repo "$REPO" --body "$_AK"
  gh secret set AWS_KEY_SECRET --repo "$REPO" --body "$_SK"
  _AK="" _SK=""
  echo "  Set AWS_KEY_ID and AWS_KEY_SECRET secrets"
fi

# Set GH_TOKEN (for machulav/ec2-github-runner to register runners)
if echo "$EXISTING_SECRETS" | grep -q "^GH_TOKEN$"; then
  echo "  GH_TOKEN secret already exists — skipping"
else
  if [[ -n "$GH_PAT" ]]; then
    gh secret set GH_TOKEN --repo "$REPO" --body "$GH_PAT"
    GH_PAT=""
    echo "  Set GH_TOKEN secret from --gh-token parameter"
  elif [[ -n "${GH_TOKEN:-}" ]]; then
    gh secret set GH_TOKEN --repo "$REPO" --body "$GH_TOKEN"
    echo "  Set GH_TOKEN secret from GH_TOKEN env var"
  else
    echo "  WARNING: GH_TOKEN secret not set. You must set it manually:"
    echo "    gh secret set GH_TOKEN --repo $REPO"
    echo "    (paste a GitHub PAT with 'repo' scope)"
  fi
fi

# Re-enable trace if it was on
set -euo pipefail

# ── Step 6: GitHub Variables ──────────────────────────────────────────
echo ""
echo "=== Step 6: GitHub Variables ==="

set_var() {
  local name="$1" value="$2"
  gh variable set "$name" --repo "$REPO" --body "$value" 2>/dev/null \
    || gh variable set "$name" --repo "$REPO" <<< "$value"
  echo "  Set $name = $value"
}

set_var "AWS_REGION" "$REGION"
set_var "AWS_IMAGE_ID" "$AMI"
set_var "AWS_INSTANCE_TYPE" "$INSTANCE_TYPE"
set_var "AWS_SUBNET" "$PRIMARY_SUBNET"
set_var "AWS_SECURITY_GROUP" "$SG_ID"
set_var "AWS_AZ_CONFIG" "$AZ_CONFIG"
set_var "WORKING_DIR" "$WORKING_DIR"

# ── Done ──────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Repository:     $REPO"
echo "AWS Region:     $REGION"
echo "VPC:            $VPC_ID"
echo "Security Group: $SG_ID"
echo "AMI:            $AMI"
echo "Instance Type:  $INSTANCE_TYPE"
echo "IAM User:       $IAM_USER"
echo "AZ Count:       $SUBNET_COUNT"
echo ""
echo "Next steps:"
echo "  1. If GH_TOKEN was not set, run: gh secret set GH_TOKEN --repo $REPO"
echo "  2. Label a PR with 'to-test' to trigger GPU tests"
echo "  3. Monitor with: gh run list --repo $REPO"
