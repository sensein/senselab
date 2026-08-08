# Quickstart: AWS GPU Test Orchestration

## Initial Setup (one-time)

### 1. Build the AMI

Follow `.github/EC2_GPU_RUNNER.md` to create a GPU-enabled AMI with pre-installed dependencies. Note the AMI ID.

### 2. Run the setup script

```bash
./scripts/setup-gpu-ci.sh \
  --profile senselab \
  --repo sensein/senselab \
  --region us-east-1 \
  --instance-type g4dn.xlarge \
  --ami ami-YOUR_AMI_ID
```

This creates IAM resources, security group, and configures GitHub secrets/variables.

### 3. Verify

Open a PR, add the `to-test` label. GPU tests should run automatically.

## For Maintainers

### Running GPU tests on a PR

1. Review the PR code (GPU runners execute PR code — only approve trusted changes)
2. Add the `to-test` label
3. GPU tests start automatically
4. Results appear on the PR checks tab
5. Instance terminates automatically after tests

### Requesting a specific GPU

Add one of these labels to the PR:
- `gpu-instance:g5.xlarge` — exact instance type
- `gpu-family:p3` — family default (p3.2xlarge = V100)
- `gpu-multi` — multi-GPU (g5.12xlarge = 4× A10G)
- `gpu-ondemand:true` — use on-demand pricing (instead of spot)

### Updating the AMI

When PyTorch or other heavy dependencies need updating:
1. Launch an instance from the current AMI
2. SSH in and update the venv: `uv pip install --upgrade torch torchaudio`
3. Create a new AMI snapshot
4. Update `AWS_IMAGE_ID` in GitHub repo variables

## For Other Repositories

The setup script is reusable:

```bash
./scripts/setup-gpu-ci.sh \
  --profile senselab \
  --repo sensein/OTHER-REPO \
  --ami ami-SHARED_OR_REPO_AMI
```

Shared AWS resources (VPC, security group) are reused. Only GitHub secrets/variables are repo-specific.
