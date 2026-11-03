"""One YAML loader: text dates, duplicate refusal, typed errors."""

from pathlib import Path

import pytest
from senselab_core.runtime.config import ConfigError, load_yaml


@pytest.mark.logic
def test_one_yaml_loader_refuses_duplicates_and_keeps_text_dates(tmp_path: Path) -> None:
    """A date stays text, a repeated key at any depth is refused, and the error names the kind and source."""
    assert load_yaml("when: 2026-11-01\n", "doc") == {"when": "2026-11-01"}
    with pytest.raises(ConfigError, match="duplicate keys") as err:
        load_yaml("a: {b: 1, b: 2}\n", "doc")
    assert (err.value.what, err.value.label) == ("doc", "<text>")
    path = tmp_path / "x.yaml"
    path.write_text("a: 1\na: 2\n", "utf-8")
    with pytest.raises(ConfigError, match=str(path)):
        load_yaml(path, "doc")


@pytest.mark.logic
def test_loader_refuses_unreadable_and_non_mapping(tmp_path: Path) -> None:
    """A missing file, invalid YAML and a list are each a `ConfigError`."""
    binary = tmp_path / "binary.yaml"
    binary.write_bytes(b"\xff\xfe\x00")
    for source in (tmp_path / "absent.yaml", binary, "a: [", "- 1\n"):
        with pytest.raises(ConfigError):
            load_yaml(source, "doc")
