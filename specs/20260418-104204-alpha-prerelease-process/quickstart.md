# Quickstart: Alpha Prerelease Process

## For Maintainers

### Creating an alpha release

1. Create a PR targeting the `alpha` branch
2. Get CI tests passing and approval
3. Merge the PR
4. Auto automatically creates an alpha tag + GitHub pre-release + PyPI publish

No labels needed. Every merge to `alpha` creates a release.

### Creating a stable release

Same as before:
1. Create a PR targeting `main`
2. Add the `release` label
3. Merge the PR
4. Auto creates a stable tag + GitHub release + PyPI publish

### Keeping alpha in sync with main

Periodically merge main into alpha to pick up stable fixes:
```bash
git checkout alpha
git merge main
git push origin alpha
```

## For Users

### Installing the latest alpha

```bash
pip install senselab --pre
```

### Installing a specific alpha version

```bash
pip install senselab==1.4.0a0
```

### Installing the latest stable (unchanged)

```bash
pip install senselab
```
