# Feature Specification: AWS GPU Test Orchestration Setup

**Feature Branch**: `20260418-120722-aws-gpu-test-setup`
**Created**: 2026-04-18
**Status**: Draft
**Input**: User description: "re-setup gpu test orchestration on aws. read ~/software/neuronets/nobrainer to understand what was needed. we are going to replace the current setup. when providing aws commands or running them use --profile senselab. to the extent it can be automated (using aws and gh) for other repos in the future generate a script to do so without leaking any credentials."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - GPU tests run on PR label (Priority: P1)

A maintainer labels a pull request with `to-test`. The system automatically provisions a GPU-enabled cloud instance, runs the full test suite with GPU acceleration, reports results on the PR, and terminates the instance — all without manual SSH or instance management.

**Why this priority**: This is the core functionality — without it, GPU-dependent tests cannot run in CI, blocking validation of audio/video ML models.

**Independent Test**: Label a test PR with `to-test`, verify that a GPU instance starts, tests execute with GPU available, results appear on the PR, and the instance terminates.

**Acceptance Scenarios**:

1. **Given** a PR is labeled `to-test`, **When** the test workflow triggers, **Then** a GPU instance is provisioned, the test suite runs with GPU access, and results are reported on the PR.
2. **Given** a GPU test run completes (pass or fail), **When** the stop job executes, **Then** the cloud instance is terminated within 2 minutes.
3. **Given** a GPU test run crashes or times out, **When** the stop job executes, **Then** the cloud instance is still terminated (no orphaned instances).

---

### User Story 2 - Reproducible setup via automation script (Priority: P1)

A maintainer runs a single setup script that provisions all required cloud infrastructure (IAM credentials, networking, instance image) and configures the repository's CI secrets/variables — without exposing credentials in logs, scripts, or version control. The same script can be reused for other repositories.

**Why this priority**: Manual setup is error-prone and undocumented. A scriptable approach ensures the infrastructure can be rebuilt, audited, and replicated across repositories (e.g., nobrainer, senselab, future projects).

**Independent Test**: Run the setup script against a fresh repository. Verify all cloud resources are created, GitHub secrets/variables are populated, and the test workflow succeeds on first labeled PR.

**Acceptance Scenarios**:

1. **Given** a repository with no GPU CI configured, **When** the setup script runs with a cloud profile name and repository identifier, **Then** all required cloud resources are created and repository secrets/variables are set.
2. **Given** the setup script has completed, **When** a maintainer inspects shell history, script output, and CI logs, **Then** no cloud credentials, access keys, or tokens appear in plaintext.
3. **Given** the setup script is run for a second repository, **When** providing a different repo identifier, **Then** the script reuses shared cloud resources (VPC, security group) and creates repo-specific secrets/variables.

---

### User Story 3 - Multi-AZ failover for spot capacity (Priority: P2)

When GPU spot instances are unavailable in the primary availability zone, the system automatically retries in other availability zones before falling back to on-demand pricing or failing gracefully.

**Why this priority**: GPU spot capacity is unreliable. Failover prevents CI from being blocked by transient capacity shortages, reducing maintainer intervention.

**Independent Test**: Configure multiple AZs, simulate primary AZ failure (by using a non-existent subnet), verify the system falls back to secondary AZ.

**Acceptance Scenarios**:

1. **Given** spot capacity is unavailable in the primary AZ, **When** the start-runner job executes, **Then** it automatically retries in the next configured AZ.
2. **Given** all AZs are exhausted, **When** no capacity is available, **Then** the workflow fails with a clear error message (not a timeout).

---

### User Story 4 - Pre-built instance image with dependencies cached (Priority: P2)

The instance image comes pre-loaded with GPU drivers, the package manager, and heavy Python dependencies (PyTorch, etc.), so test startup time is dominated by project-specific installation — not by downloading multi-gigabyte frameworks.

**Why this priority**: Without pre-cached dependencies, each test run would spend 10-20 minutes downloading PyTorch alone. A pre-built image cuts startup to under 5 minutes.

**Independent Test**: Launch a fresh instance from the image, verify GPU drivers, package manager, and PyTorch are available without internet downloads.

**Acceptance Scenarios**:

1. **Given** a fresh instance launched from the pre-built image, **When** the test job checks GPU availability, **Then** GPU drivers and CUDA are functional.
2. **Given** the pre-built image, **When** project dependencies are installed, **Then** only project-specific packages need downloading (heavy ML frameworks are already cached).

---

### Edge Cases

- What happens when the GitHub runner registration fails? The stop job still runs and terminates the instance to prevent orphaning.
- What happens when disk space runs out during tests? Tests fail, results are reported, instance is terminated.
- What happens when the cloud credentials expire or are rotated? The setup script can be re-run to update credentials without recreating infrastructure.
- What happens when the instance image becomes outdated? A documented process (or script flag) rebuilds the image with updated dependencies.
- What happens when two PRs are labeled concurrently? Each gets its own independent instance — no resource contention.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provision ephemeral GPU instances on-demand when a PR is labeled for testing.
- **FR-002**: The system MUST terminate GPU instances after test completion (pass, fail, or timeout) with no orphaned instances.
- **FR-003**: The system MUST run the full senselab test suite with GPU access on provisioned instances.
- **FR-004**: The system MUST support multiple Python version test matrices (3.11, 3.12) on GPU instances.
- **FR-005**: The system MUST support testing with different dependency configurations (core-only vs. full extras).
- **FR-006**: The system MUST attempt multiple availability zones before failing when spot capacity is unavailable.
- **FR-007**: The system MUST use a pre-built instance image with GPU drivers and heavy dependencies pre-installed.
- **FR-008**: A setup script MUST automate provisioning of all cloud resources and repository configuration.
- **FR-009**: The setup script MUST NOT expose credentials in logs, output, shell history, or version-controlled files.
- **FR-010**: The setup script MUST be reusable across different repositories by accepting repository identifier and cloud profile as parameters.
- **FR-011**: The system MUST use the named cloud profile (`senselab`) for all cloud CLI operations.
- **FR-012**: The system MUST install Docker on GPU instances for tests requiring containerized environments.
- **FR-013**: The system MUST validate GPU availability (driver check) before running tests.

### Key Entities

- **GPU Instance**: An ephemeral cloud compute instance with GPU, launched per test run and terminated after.
- **Instance Image**: A pre-built machine image containing OS, GPU drivers, package manager, and cached ML dependencies.
- **Setup Script**: A reusable automation script that provisions cloud infrastructure and configures repository CI.
- **AZ Configuration**: A list of availability zone entries (image, subnet, security group) for failover.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: GPU test runs complete end-to-end (provision → test → terminate) within 30 minutes for the full test suite.
- **SC-002**: Instance startup (from label trigger to first test execution) takes under 5 minutes with the pre-built image.
- **SC-003**: Zero orphaned instances after 100 consecutive test runs (including failure scenarios).
- **SC-004**: The setup script configures a new repository for GPU CI in under 15 minutes (excluding image build time).
- **SC-005**: No credentials appear in any CI logs, script output, or version-controlled files.
- **SC-006**: Spot capacity failover succeeds at least 90% of the time across configured availability zones.

## Assumptions

- An AWS account with the `senselab` CLI profile is already configured locally.
- The `gh` CLI is authenticated and has write access to the target repository.
- GPU instance types (e.g., g4dn, g5, p3) are available in the target AWS region.
- The `machulav/ec2-github-runner` GitHub Action (or equivalent) will be used for ephemeral runner management — this is proven in production in the nobrainer repository.
- The repository already has a working test suite that can detect and use GPUs when available.
- Spot pricing is the default; on-demand is a fallback, not the primary mode.
- The setup script will be written in Bash and use `aws` CLI and `gh` CLI — no additional tools required.
- The existing senselab `.github/workflows/tests.yaml` EC2 runner jobs will be replaced, not patched.
