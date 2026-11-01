"""Reading the packaged `@context` and `senselab:` vocabulary files (`data/prov/`)."""

import functools
import json
from importlib import resources

from senselab_worker.canonical import unique_object

from senselab_core.kernel.prov import Vocabulary


def read_prov_file(package: str, file_name: str) -> object:
    """Return the JSON of `<package>/data/prov/<file_name>`; a key repeated within one object is refused."""
    text = resources.files(package).joinpath("data", "prov", file_name).read_text("utf-8")
    return json.loads(text, object_pairs_hook=unique_object)


def load_vocabulary(package: str, file_name: str, base: Vocabulary | None = None) -> Vocabulary:
    """Return `base` extended by the vocabulary file a package ships in its `data/prov/`.

    Raises:
        VocabularyError: if the file is not a `senselab-vocab@1` document over `base`.
    """
    return Vocabulary.from_document(read_prov_file(package, file_name), base)


@functools.cache
def core_vocabulary() -> Vocabulary:
    """Return the core vocabulary, `senselab_core.runtime`'s `data/prov/senselab.jsonld`."""
    return load_vocabulary("senselab_core.runtime", "senselab.jsonld")


def context_document() -> object:
    """Return the published `@context` document, `data/prov/context.jsonld`."""
    return read_prov_file("senselab_core.runtime", "context.jsonld")
