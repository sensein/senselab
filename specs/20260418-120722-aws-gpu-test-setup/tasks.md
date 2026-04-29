# Tasks: AWS GPU Test Orchestration Setup

**Input**: Design documents from `/specs/20260418-120722-aws-gpu-test-setup/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, quickstart.md

**Tests**: No automated tests — this is infrastructure setup. Verification is manual via PR label + workflow observation.

**Organization**: Tasks are grouped by user story to enable independent implementation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: No project initialization needed — this modifies existing repo files and adds new ones.

(No tasks)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Create the setup script and documentation that all user stories depend on.

- [x] T001 Create `scripts/setup-gpu-ci.sh` — reusable Bash script that accepts `--profile`, `--repo`, `--region`, `--instance-type`, `--ami`, `--vpc` parameters and automates: IAM user/policy creation, security group creation, subnet discovery across AZs, and GitHub secrets/variables configuration via `gh secret set` and `gh variable set`. Credentials MUST be piped directly to `gh` via stdin (never echoed or logged). Use `set +x` during credential operations.

- [x] T002 [P] Create `.github/EC2_GPU_RUNNER.md` — document AMI build process adapted from nobrainer: launch g4dn.xlarge from AWS Deep Learning Base AMI (AL2023), install jq/git/uv, create `~/senselab-env` venv with torch/torchaudio/torchvision/transformers/speechbrain/pyannote-audio/pytest, verify GPU, create AMI snapshot.

**Checkpoint**: Setup script and docs ready. Workflow rewrite and AWS provisioning can proceed.

---

## Phase 3: User Story 2 — Reproducible setup via automation script (Priority: P1)

**Goal**: Run the setup script to provision AWS infrastructure and configure GitHub repo secrets/variables for senselab.

**Independent Test**: Run `scripts/setup-gpu-ci.sh --profile senselab --repo sensein/senselab --ami <ami-id>` and verify all GitHub secrets/variables are set, IAM user exists, security group allows outbound HTTPS.

### Implementation for User Story 2

- [x] T003 [US2] Build the senselab GPU AMI following `.github/EC2_GPU_RUNNER.md` — launch instance from AWS Deep Learning Base AMI (AL2023) with `aws ec2 run-instances --profile senselab`, SSH in, install jq/git/uv, create `~/senselab-env` with senselab's heavy deps, verify GPU, create AMI with `aws ec2 create-image --profile senselab`, note AMI ID.

- [x] T004 [US2] Run `scripts/setup-gpu-ci.sh --profile senselab --repo sensein/senselab --region us-east-1 --instance-type g4dn.xlarge --ami <ami-id-from-T003>` to provision IAM user, security group, and set all GitHub secrets/variables.

- [x] T005 [US2] Verify GitHub configuration: run `gh variable list --repo sensein/senselab` and confirm `AWS_REGION`, `AWS_IMAGE_ID`, `AWS_INSTANCE_TYPE`, `AWS_SUBNET`, `AWS_SECURITY_GROUP`, `AWS_AZ_CONFIG`, `WORKING_DIR` are set. Run `gh secret list --repo sensein/senselab` and confirm `AWS_KEY_ID`, `AWS_KEY_SECRET` are set.

- [x] T005a [US2] Verify no credential leakage: run `history | grep -i 'secret\|key\|token'` locally, check `gh run view` logs for the setup workflow (if any), and confirm `git log --all -p | grep -i 'AKIA\|secret'` returns nothing in the repo.

**Checkpoint**: AWS infrastructure provisioned, GitHub configured, credentials verified safe. Workflow rewrite can proceed.

---

## Phase 4: User Story 1 — GPU tests run on PR label (Priority: P1)

**Goal**: Rewrite the EC2 runner jobs in the test workflow so that labeling a PR with `to-test` provisions a GPU instance, runs tests, and terminates.

**Independent Test**: Label a PR with `to-test`, verify GPU instance starts, tests run with GPU access, results appear on PR, instance terminates.

### Implementation for User Story 1

- [x] T006 [US1] Rewrite `.github/workflows/tests.yaml` EC2 runner jobs — for each variant (311-core, 311, 312), replace start/test/stop triads with modernized pattern: machulav/ec2-github-runner@v2.5.2, aws-actions/configure-aws-credentials@v6, `availability-zones-config: ${{ vars.AWS_AZ_CONFIG }}`, spot pricing by default, GPU label parsing (gpu-instance:, gpu-family:, gpu-multi, gpu-ondemand:), copy pre-built venv from `~/senselab-env`, GPU verification step (`torch.cuda.is_available`), Docker install, simplified stop condition (`always() && needs.start-runner.result == 'success'`).

- [x] T007 [US1] Run `uv run pre-commit run --all-files` on the rewritten workflow to ensure YAML formatting passes.

- [x] T008 [US1] Push changes and create a test PR. Add `to-test` label. Verify:
  - EC2 GPU instance starts (check Actions log for instance type, AZ)
  - Tests run with GPU available (nvidia-smi output in logs)
  - Test results appear on PR checks tab
  - Instance terminates after tests complete

- [x] T009 [US1] Verify instance termination: after T008 completes, check `aws ec2 describe-instances --profile senselab --filters "Name=tag:Name,Values=*github*"` to confirm no running instances remain.

**Checkpoint**: End-to-end GPU test pipeline working on labeled PRs.

---

## Phase 5: User Story 3 — Multi-AZ failover for spot capacity (Priority: P2)

**Goal**: Verify that spot capacity failures in one AZ trigger automatic failover to another.

**Independent Test**: Temporarily misconfigure the primary AZ subnet in `AWS_AZ_CONFIG` (point to a non-existent subnet), verify the workflow falls back to the secondary AZ.

### Implementation for User Story 3

- [x] T010 [US3] Verify `AWS_AZ_CONFIG` variable contains entries for at least 2 AZs with valid subnets: run `gh variable get AWS_AZ_CONFIG --repo sensein/senselab` and parse JSON.

- [x] T011 [US3] Test failover: temporarily update `AWS_AZ_CONFIG` to have an invalid first entry (non-existent subnet), trigger a GPU test, verify the workflow log shows failover to the second AZ, then restore the original config.

**Checkpoint**: Multi-AZ failover verified.

---

## Phase 6: User Story 4 — Pre-built instance image with dependencies cached (Priority: P2)

**Goal**: Verify the pre-built AMI provides GPU drivers and cached dependencies, enabling fast test startup.

**Independent Test**: Launch an instance from the AMI, verify GPU and dependencies without internet.

### Implementation for User Story 4

- [x] T012 [US4] Launch a test instance from the AMI (`aws ec2 run-instances --profile senselab --image-id <ami-id> --instance-type g4dn.xlarge`), SSH in, verify: `nvidia-smi` shows GPU, `~/senselab-env/bin/python -c "import torch; assert torch.cuda.is_available()"` succeeds, `uv` is available. Terminate the test instance.

- [x] T013 [US4] Measure startup time: from the T008 test run, extract timestamps from the Actions log to calculate time from workflow trigger to first test execution. Verify < 5 minutes.

**Checkpoint**: AMI validated, startup performance confirmed.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and cleanup.

- [x] T014 Update `quickstart.md` in specs with actual AMI ID, region, and observed timing.
- [x] T015 [P] Update `CONTRIBUTING.md` or `README.md` to reference GPU testing process and link to `.github/EC2_GPU_RUNNER.md`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 2)**: No dependencies — T001 (script) and T002 (docs) are parallel.
- **US2 (Phase 3)**: Depends on Phase 2 (script must exist to run it). T003 (AMI build) depends on T002 (docs). T004 depends on T001 + T003.
- **US1 (Phase 4)**: Depends on US2 (GitHub config must be set). T006 can start after T005 confirms config.
- **US3 (Phase 5)**: Depends on US1 (need a working workflow to test failover).
- **US4 (Phase 6)**: Depends on T003 (AMI must exist). T012 can run in parallel with US1.
- **Polish (Phase 7)**: Depends on all verification phases.

### User Story Dependencies

- **US2 (P1)**: Depends on Foundational only — provisions infrastructure.
- **US1 (P1)**: Depends on US2 — needs configured repo to test workflow.
- **US3 (P2)**: Depends on US1 — needs working workflow to test failover.
- **US4 (P2)**: Partially independent — AMI validation (T012) can run after T003.

### Parallel Opportunities

- T001 and T002 can run in parallel (different files).
- T012 (AMI validation) can run in parallel with US1 workflow tasks.
- T014 and T015 can run in parallel.

---

## Implementation Strategy

### MVP First (US2 + US1)

1. Complete T001 + T002 (script + docs) — commit
2. Complete T003 (build AMI)
3. Complete T004 + T005 (run script + verify)
4. Complete T006 + T007 (rewrite workflow + pre-commit)
5. Complete T008 + T009 (end-to-end verification)
6. **STOP and VALIDATE**: GPU tests work on labeled PRs

### Full Delivery

7. Complete T010-T011 (multi-AZ failover)
8. Complete T012-T013 (AMI validation + timing)
9. Complete T014-T015 (documentation)

---

## Notes

- Total tasks: 15
- Tasks per story: US1: 4, US2: 3, US3: 2, US4: 2, Foundational: 2, Polish: 2
- T001 (setup script) is the most complex task — ~200 lines of Bash
- T003 (AMI build) is the most time-consuming — ~30 minutes of manual work
- T006 (workflow rewrite) is the second most complex — significant YAML changes
- All `aws` commands MUST use `--profile senselab`
