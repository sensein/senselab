# Research: AWS GPU Test Orchestration Setup

## Decision 1: Workflow architecture — keep separate triads vs single matrix

**Decision**: Keep 3 separate start/test/stop triads (311-core, 311, 312)
**Rationale**: Each variant has different `uv sync` commands (core-only vs full extras) and the matrix would require conditional logic inside the test job. Separate triads are explicit and debuggable.
**Alternatives considered**: Single matrix with conditional install steps — rejected for readability. Nobrainer uses a single triad because it has one dependency profile.

## Decision 2: machulav/ec2-github-runner version

**Decision**: v2.5.2
**Rationale**: Required for `availability-zones-config` (multi-AZ failover). Proven in nobrainer production.
**Alternatives considered**: v2 (current) — lacks AZ failover.

## Decision 3: aws-actions/configure-aws-credentials version

**Decision**: v6
**Rationale**: v5 emits Node.js 20 deprecation warnings. v6 is current stable.
**Alternatives considered**: v5 — functional but deprecated.

## Decision 4: Stop-runner condition pattern

**Decision**: `always() && needs.start-runner.result == 'success'`
**Rationale**: Simpler than the current `job-ran` output flag pattern. The `always()` ensures cleanup runs even on test failure. The `start-runner.result == 'success'` check prevents cleanup attempts when no instance was started. Proven in nobrainer.
**Alternatives considered**: Current `job-ran` flags — unnecessary indirection.

## Decision 5: AMI base and pre-installed venv

**Decision**: AWS Deep Learning Base AMI (AL2023) + `~/senselab-env`
**Rationale**: AL2023 comes with NVIDIA drivers and CUDA pre-installed. A pre-built venv with torch/torchaudio/transformers saves ~15 minutes per test run. The venv is copied (not symlinked) so the AMI stays clean.
**Alternatives considered**: Ubuntu + manual NVIDIA install — more setup effort, no benefit.

## Decision 6: Setup script credential safety

**Decision**: Pipe access keys directly to `gh secret set` via stdin
**Rationale**: Never writes credentials to files, stdout, or shell history. Uses `aws iam create-access-key --output json` piped to `jq` piped to `gh secret set -f-`. The `set +x` ensures trace output is suppressed during credential operations.
**Alternatives considered**: Write to temp file + delete — risky if script is interrupted. OIDC federation — ideal but requires more AWS setup (role trust policy) beyond script scope.

## Decision 7: GPU label parsing

**Decision**: Adopt nobrainer's full label system (gpu-instance:TYPE, gpu-family:FAMILY, gpu-multi, gpu-ondemand:true)
**Rationale**: Flexible, proven, zero-config for default case. Advanced users can request specific GPUs for debugging.
**Alternatives considered**: Simple instance type variable only — insufficient for multi-GPU testing or GPU family experiments.
