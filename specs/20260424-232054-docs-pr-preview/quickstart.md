# Quickstart: Docs PR Preview + Coverage

## Build docs locally
```bash
uv run pdoc src/senselab -t docs_style/pdoc-theme --docformat google
```

## Check doc.md coverage
```bash
for dir in src/senselab/audio/tasks/*/; do
  [ -f "$dir/doc.md" ] && echo "✓ $(basename $dir)" || echo "✗ $(basename $dir)"
done
```

## Test PR preview workflow
Open a PR with a docstring change and verify the preview comment appears.
