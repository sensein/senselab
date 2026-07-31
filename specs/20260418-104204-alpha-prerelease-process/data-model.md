# Data Model: Alpha Prerelease Process

## Entities

### Configuration: `.autorc`

| Field | Type | Description |
|-------|------|-------------|
| onlyPublishWithReleaseLabel | boolean | Whether Auto requires a `release` label to create releases. Set to `false` (gating handled by workflow condition). |
| baseBranch | string | Primary stable release branch (`"main"`). |
| prereleaseBranches | string[] | Branches that trigger pre-releases (`["alpha"]`). |
| author | string | Git author for release commits. |
| noVersionPrefix | boolean | Omit "v" prefix from version tags (`true`). |
| plugins | string[] | Auto plugins (`["git-tag"]`). |

### Workflow: `.github/workflows/release.yaml`

| Aspect | Value |
|--------|-------|
| Trigger event | `pull_request: types: [closed]` |
| Trigger branches | `[main, alpha]` |
| Run condition | PR merged AND (has `release` label OR targets `alpha`) |
| Tool | Intuit Auto v11.2.1 |
| Command | `auto shipit -vv` |
| Secret | `AUTO_ORG_TOKEN` |

### Version Tags

| Branch | Tag format | Example | PyPI version |
|--------|-----------|---------|-------------|
| main | `MAJOR.MINOR.PATCH` | `1.4.0` | `1.4.0` |
| alpha | `MAJOR.MINOR.PATCH-alpha.N` | `1.4.0-alpha.0` | `1.4.0a0` |

Note: PEP 440 normalizes `1.4.0-alpha.0` to `1.4.0a0` for PyPI. hatch-vcs handles this automatically.
