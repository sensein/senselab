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

Required:
  --profile <name>        AWS CLI profile to use
  --repo <owner/repo>     GitHub repository (e.g., sensein/senselab)
  --ami <ami-id>          Pre-built AMI ID with GPU drivers + venv

Optional:
  --region <region>       AWS region (default: us-east-1)
  --instance-type <type>  Default GPU instance type (default: g4dn.xlarge)
  --vpc <vpc-id>          VPC ID (default: auto-detect default VPC)
  --working-dir <path>    Cache directory on EC2 instance (default: /tmp/senselab)
  --gh-token <token>      GitHub PAT with repo scope (default: prompt or use GH_TOKEN env)
  --help                  Show this help
USAGE
  exit 1
}

PROFILE="" REPO="" AMI="" GH_PAT=""

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
    --help|-h)      usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "$PROFILE" ]] && { echo "ERROR: --profile is required"; usage; }
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
PRIMARY_SUBNET=$(echo "$SUBNETS_JSON" | jq -r '.[0].subnetId')
echo "  Primary subnet: $PRIMARY_SUBNET"
echo "  AZ config: $AZ_CONFIG"

# ── Step 4: IAM User ─────────────────────────────────────────────────
echo ""
echo "=== Step 4: IAM User ==="

# Derive a repo-specific IAM user name
REPO_SLUG=$(echo "$REPO" | tr '/' '-')
IAM_USER="${IAM_USER_PREFIX}-${REPO_SLUG}"

USER_EXISTS=$($AWS iam get-user --user-name "$IAM_USER" 2>/dev/null | jq -r '.User.UserName // empty' || true)

if [[ -n "$USER_EXISTS" ]]; then
  echo "  IAM user '$IAM_USER' already exists"
  echo "  Checking for existing access keys..."
  KEY_COUNT=$($AWS iam list-access-keys --user-name "$IAM_USER" | jq '.AccessKeyMetadata | length')
  if [[ "$KEY_COUNT" -gt 0 ]]; then
    echo "  WARNING: User has $KEY_COUNT existing access key(s)."
    echo "  To rotate: delete old keys via AWS console, then re-run this script."
  fi
else
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

if echo "$EXISTING_SECRETS" | grep -q "^AWS_KEY_ID$"; then
  echo "  AWS_KEY_ID secret already exists — skipping key creation"
  echo "  To rotate: delete the secret in GitHub and the IAM key, then re-run."
else
  echo "  Creating new access key and setting GitHub secrets..."
  # Create key, extract fields, pipe directly to gh — NEVER echo
  KEY_JSON=$($AWS iam create-access-key --user-name "$IAM_USER")

  # Pipe key ID directly to gh secret set
  echo "$KEY_JSON" | jq -r '.AccessKey.AccessKeyId' \
    | gh secret set AWS_KEY_ID --repo "$REPO" --body -

  # Pipe secret key directly to gh secret set
  echo "$KEY_JSON" | jq -r '.AccessKey.SecretAccessKey' \
    | gh secret set AWS_KEY_SECRET --repo "$REPO" --body -

  # Clear the variable immediately
  KEY_JSON=""
  echo "  Set AWS_KEY_ID and AWS_KEY_SECRET secrets"
fi

# Set GH_TOKEN (for machulav/ec2-github-runner to register runners)
if echo "$EXISTING_SECRETS" | grep -q "^GH_TOKEN$"; then
  echo "  GH_TOKEN secret already exists — skipping"
else
  if [[ -n "$GH_PAT" ]]; then
    echo "$GH_PAT" | gh secret set GH_TOKEN --repo "$REPO" --body -
    GH_PAT=""
    echo "  Set GH_TOKEN secret from --gh-token parameter"
  elif [[ -n "${GH_TOKEN:-}" ]]; then
    echo "$GH_TOKEN" | gh secret set GH_TOKEN --repo "$REPO" --body -
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
