# Implementation Plan: AWS GPU Test Orchestration Setup

**Branch**: `20260418-120722-aws-gpu-test-setup` | **Date**: 2026-04-18 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/20260418-120722-aws-gpu-test-setup/spec.md`

## Summary

Replace the current senselab EC2 GPU runner setup with a modernized, scriptable approach modeled on nobrainer. The new setup uses machulav/ec2-github-runner@v2.5.2 with multi-AZ spot failover, pre-built AMI with cached dependencies, label-based GPU instance selection, and an automation script reusable across repositories. All AWS CLI operations use `--profile senselab`.

## Technical Context

**Language/Version**: Bash (setup script), YAML (GitHub Actions workflows)
**Primary Dependencies**: machulav/ec2-github-runner@v2.5.2, aws-actions/configure-aws-credentials@v6, aws CLI, gh CLI
**Storage**: N/A (ephemeral instances)
**Testing**: Manual end-to-end verification via PR label
**Target Platform**: GitHub Actions + AWS EC2 (GPU instances)
**Project Type**: CI/CD infrastructure
**Performance Goals**: Instance startup to first test < 5 minutes with pre-built AMI
**Constraints**: All AWS operations via `--profile senselab`; no credentials in logs or version control
**Scale/Scope**: 1 workflow file rewritten, 1 setup script created, 1 AMI built, 1 doc file added

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. UV-Managed Python | PASS | EC2 test jobs use `uv run pytest` and `uv sync` |
| II. Encapsulated Testing | PASS | Tests run in uv virtualenv on ephemeral EC2 instances |
| III. Commit Early and Often | PASS | N/A for infrastructure — single logical change |
| IV. CI Must Stay Green | PASS | GPU tests are opt-in (label-gated), not blocking |
| V. Memory-Driven Anti-Patterns | PASS | No known anti-patterns for this feature |
| VI. No Unnecessary API Calls | N/A | Not applicable to CI infrastructure |
| VII. Simplicity First | PASS | Single workflow, single script, proven pattern from nobrainer |

**PASS** — No violations.

## Project Structure

### Documentation (this feature)

```text
specs/20260418-120722-aws-gpu-test-setup/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0: Technical decisions
├── data-model.md        # Phase 1: Entity model
├── quickstart.md        # Phase 1: Usage guide
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code (repository root)

```text
.github/
├── workflows/
│   └── tests.yaml          # REWRITTEN: EC2 runner jobs modernized
├── EC2_GPU_RUNNER.md        # NEW: Setup documentation (from nobrainer)
scripts/
└── setup-gpu-ci.sh          # NEW: Reusable setup automation script
```

**Structure Decision**: Minimal footprint — one workflow rewrite, one new script, one doc file. The setup script lives in `scripts/` at repo root for discoverability and reuse.

## Phase 0: Research

### Decision 1: Workflow architecture

**Decision**: Replace the 9-job pattern (3× start/test/stop) with the nobrainer 3-job pattern (1× start/test/stop) per test matrix entry.
**Rationale**: The current senselab setup has 3 separate start/test/stop triads (311-core, 311, 312). Nobrainer uses a single triad with matrix strategy. However, senselab needs different dependency configs (core vs full extras), so we'll keep separate triads but modernize each one.
**Alternatives considered**: Single job with matrix — rejected because core vs full extras need different `uv sync` commands and the matrix would be complex.

### Decision 2: machulav/ec2-github-runner version

**Decision**: Upgrade from v2 to v2.5.2 (same as nobrainer)
**Rationale**: v2.5.2 supports `availability-zones-config` for multi-AZ failover (FR-006). The current v2 does not.
**Alternatives considered**: Stay on v2 — rejected because multi-AZ failover requires v2.5.2.

### Decision 3: AWS credentials action

**Decision**: Upgrade from aws-actions/configure-aws-credentials@v5 to @v6
**Rationale**: Nobrainer uses v6. v5 has Node.js 20 deprecation warnings in CI. v6 is the latest stable.
**Alternatives considered**: Keep v5 — rejected due to deprecation warnings.

### Decision 4: GPU instance selection

**Decision**: Adopt nobrainer's label-based instance selection (gpu-instance:TYPE, gpu-family:FAMILY, gpu-multi, gpu-ondemand:true)
**Rationale**: Allows maintainers to test on specific GPU types without workflow changes. Default to `vars.AWS_INSTANCE_TYPE` (g4dn.xlarge).
**Alternatives considered**: Hardcoded instance type — rejected because different tests may need different GPU capabilities.

### Decision 5: AMI base

**Decision**: Use AWS Deep Learning Base AMI (Amazon Linux 2023) with pre-installed senselab venv
**Rationale**: Matches nobrainer approach. Pre-installed heavy dependencies (torch, torchaudio, etc.) reduce startup from ~20min to ~3min.
**Alternatives considered**: Ubuntu AMI — rejected because AL2023 comes with NVIDIA drivers pre-configured.

### Decision 6: Stop-runner condition

**Decision**: Use `if: ${{ always() && needs.start-runner.result == 'success' }}` (nobrainer pattern)
**Rationale**: The current senselab pattern uses complex `job-ran` output flags. Nobrainer's `always() && start-runner.result == 'success'` is simpler and equivalent — it runs cleanup whenever an instance was actually started, regardless of test outcome.
**Alternatives considered**: Keep current `job-ran` flag pattern — rejected as unnecessary complexity.

### Decision 7: Setup script design

**Decision**: Single Bash script (`scripts/setup-gpu-ci.sh`) that uses `aws --profile <name>` and `gh` CLI
**Rationale**: User requested Bash + aws CLI + gh CLI, reusable across repos, no credential leakage. The script accepts `--profile` and `--repo` as parameters.
**Alternatives considered**: Terraform/CloudFormation — rejected as overkill for 5 resources (IAM user, security group, and GitHub vars/secrets).

## Phase 1: Implementation Design

### Step 1: Create setup script (`scripts/setup-gpu-ci.sh`)

The script automates:
1. Create IAM user with minimum EC2 permissions (if not exists)
2. Create access key (output only to `gh secret set`, never to stdout)
3. Create/verify security group (outbound HTTPS only)
4. Identify subnets across AZs for failover
5. Set GitHub secrets (`AWS_KEY_ID`, `AWS_KEY_SECRET`, `GH_TOKEN`) via `gh secret set`
6. Set GitHub variables (`AWS_REGION`, `AWS_IMAGE_ID`, `AWS_INSTANCE_TYPE`, `AWS_SUBNET`, `AWS_SECURITY_GROUP`, `AWS_AZ_CONFIG`, `WORKING_DIR`) via `gh variable set`

Parameters:
- `--profile <aws-profile>` (required)
- `--repo <owner/repo>` (required)
- `--region <region>` (default: us-east-1)
- `--instance-type <type>` (default: g4dn.xlarge)
- `--ami <ami-id>` (required — must be pre-built)
- `--vpc <vpc-id>` (optional — auto-detects default VPC)

Credential safety:
- AWS access keys are piped directly to `gh secret set` via stdin
- No `echo`, `printf`, or logging of secret values
- Script uses `set +x` before credential operations
- `.gitignore` excludes any local state files

### Step 2: Build AMI (documented process)

Document in `.github/EC2_GPU_RUNNER.md`:
1. Launch g4dn.xlarge from AWS Deep Learning Base AMI (AL2023)
2. SSH in, install: jq, git, uv
3. Create pre-installed venv at `~/senselab-env` with: torch, torchaudio, torchvision, transformers, speechbrain, pyannote-audio, pytest
4. Verify GPU: `torch.cuda.is_available()`
5. Create AMI snapshot
6. Note AMI ID for `setup-gpu-ci.sh --ami` parameter

### Step 3: Rewrite `.github/workflows/tests.yaml` EC2 jobs

Replace current 9-job pattern with modernized triads:

**For each test variant (311-core, 311, 312):**

```
start-runner-{variant}:
  if: to-test label + not draft
  uses: machulav/ec2-github-runner@v2.5.2
  with:
    mode: start
    availability-zones-config: ${{ vars.AWS_AZ_CONFIG }}
    market-type: spot (or label-override)

ubuntu-tests-{variant}:
  needs: start-runner-{variant}
  runs-on: ${{ needs.start-runner-{variant}.outputs.label }}
  steps:
    - checkout
    - copy pre-built venv from ~/senselab-env
    - uv pip install project extras
    - verify GPU (nvidia-smi + torch.cuda)
    - install docker
    - uv run pytest ...

stop-runner-{variant}:
  needs: [start-runner-{variant}, ubuntu-tests-{variant}]
  if: always() && needs.start-runner-{variant}.result == 'success'
  uses: machulav/ec2-github-runner@v2.5.2
  with:
    mode: stop
```

Key changes from current:
- machulav/ec2-github-runner v2 → v2.5.2
- aws-actions/configure-aws-credentials v5 → v6
- Remove `job-ran` output flag pattern → use `always() && start-runner.result == 'success'`
- Add `availability-zones-config` for multi-AZ failover
- Add GPU label parsing (gpu-instance:, gpu-family:, gpu-multi, gpu-ondemand:)
- Copy pre-built venv from `~/senselab-env` instead of `~/nobrainer-env`
- Add GPU verification step (torch.cuda.is_available)

### Step 4: Create `.github/EC2_GPU_RUNNER.md`

Documentation adapted from nobrainer's guide with senselab-specific dependencies and venv name.

## Risks

- **AMI region lock**: AMIs are region-specific. Multi-region support requires copying AMIs. Mitigated by defaulting to us-east-1.
- **Spot capacity**: GPU spots can be scarce. Mitigated by multi-AZ failover.
- **Pre-built venv staleness**: When senselab upgrades PyTorch, the AMI must be rebuilt. Documented in EC2_GPU_RUNNER.md.
- **IAM permissions**: Script creates IAM user with EC2 permissions. If org has SCPs restricting IAM, manual setup may be needed.
