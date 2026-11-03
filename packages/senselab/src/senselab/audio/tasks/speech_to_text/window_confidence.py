"""Token confidence of a CrisperWhisper decode: each window's final ids, teacher-forced once, scored from raw logits.

`capture_windows` records the ids each window finally emits; `ct2_teacher_logits` and
`transformers_teacher_logits` give the raw logits an engine assigns those ids; `window_confidence` scores them with
`token_confidence_from_logits`, one block per window. `torch`, `numpy` and `ctranslate2` are imported only where
logits are scored, so importing this module loads no framework.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from senselab.audio.tasks.speech_to_text.token_confidence import token_confidence_from_logits

__all__ = [
    "Window",
    "capture_windows",
    "ct2_teacher_logits",
    "transformers_teacher_logits",
    "window_confidence",
]

DECODES = ("generate_with_repair", "generate_with_repair_and_attention")
"""The engine methods that decode one window, each `(features, prompt_tokens, ...)` returning its ids first."""
REPLACES = "cross_attention_for_tokens"
"""The engine method `(features, prompt_tokens, gen_ids)` the library calls when it replaces a window's ids."""


@dataclass(frozen=True)
class Window:
    """One decoded window: its encoder input, its prompt, and the ids it emits."""

    features: Any
    prompt_tokens: list[int]
    ids: list[int]


@contextlib.contextmanager
def capture_windows(engine: Any) -> Iterator[list[Window]]:  # noqa: ANN401 - a crisperwhisper engine
    """Wrap the `DECODES` and `REPLACES` of `engine` for the block, recording each window's final ids.

    A decode records a window; a `REPLACES` call on the same `features` replaces that window's ids (coverage
    fallback, early-EOT recovery). Yields the windows in decode order. The wrappers return what the caller would
    have received and are removed on exit.
    """
    windows: list[Window] = []
    index: dict[int, int] = {}

    def _record(features: Any, prompt_tokens: Sequence[int], ids: Sequence[int]) -> None:  # noqa: ANN401
        window = Window(features, [int(t) for t in prompt_tokens], [int(t) for t in ids])
        if id(features) in index:
            windows[index[id(features)]] = window
        else:
            index[id(features)] = len(windows)
            windows.append(window)

    def _decode(original: Callable[..., Any]) -> Callable[..., Any]:
        def call(features: Any, prompt_tokens: Sequence[int], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            result = original(features, prompt_tokens, *args, **kwargs)
            _record(features, prompt_tokens, result[0] if isinstance(result, tuple) else result)
            return result

        return call

    def _replace(original: Callable[..., Any]) -> Callable[..., Any]:
        def call(features: Any, prompt_tokens: Sequence[int], gen_ids: Sequence[int]) -> Any:  # noqa: ANN401
            result = original(features, prompt_tokens, gen_ids)
            _record(features, prompt_tokens, gen_ids)
            return result

        return call

    wrappers = {name: _decode for name in DECODES} | {REPLACES: _replace}
    originals = {name: getattr(engine, name) for name in wrappers}
    own = {name for name in wrappers if name in vars(engine)}
    for name, wrap in wrappers.items():
        setattr(engine, name, wrap(originals[name]))
    try:
        yield windows
    finally:
        for name, original in originals.items():
            if name in own:
                setattr(engine, name, original)
            else:
                delattr(engine, name)


def _host(view: Any) -> Any:  # noqa: ANN401 - a ctranslate2 StorageView or a numpy array in, a numpy array out
    import numpy  # noqa: PLC0415 - loads only where ct2 logits are scored

    if isinstance(view, numpy.ndarray):
        return view.astype(numpy.float32, copy=False)
    import ctranslate2  # noqa: PLC0415 - as above

    return numpy.array(view.to(ctranslate2.DataType.float32).to_device(ctranslate2.Device.cpu))


def ct2_teacher_logits(engine: Any, window: Window) -> Any:  # noqa: ANN401 - a CT2Engine in, a numpy array out
    """Return the raw logits `(len(ids), vocab)` with which the ct2 engine predicts each of the window's ids.

    `prefill` of the prompt gives the row predicting `ids[0]`; `forward_batch` over `ids[:-1]` gives the rest.

    Raises:
        ValueError: if the model returns a row count other than `len(ids)`.
    """
    import numpy  # noqa: PLC0415 - loads only where ct2 logits are scored

    state, first = engine.model.prefill(window.features, window.prompt_tokens)
    head = _host(first)
    rows = [head.reshape(-1, head.shape[-1])[-1:]]
    if len(window.ids) > 1:
        rest = _host(engine.model.forward_batch(state, window.ids[:-1]))
        rows.append(rest.reshape(-1, rest.shape[-1]))
    logits = numpy.concatenate(rows)
    if logits.shape[0] != len(window.ids):
        raise ValueError(f"{logits.shape[0]} logit rows for {len(window.ids)} generated tokens")
    return logits


def transformers_teacher_logits(engine: Any, window: Window) -> Any:  # noqa: ANN401 - an engine in, a tensor out
    """Return the raw logits `(len(ids), vocab)` with which the transformers engine predicts each of the window's ids.

    One forward of `prompt + ids[:-1]`; its last `len(ids)` positions predict the ids.
    """
    import torch  # noqa: PLC0415 - torch loads only where logits are scored

    decoder = torch.tensor([window.prompt_tokens + window.ids[:-1]], device=engine.device, dtype=torch.long)
    with torch.no_grad():
        out = engine.model(input_features=window.features, decoder_input_ids=decoder, use_cache=False)
    return out.logits[0, -len(window.ids) :].float().cpu()


def window_confidence(
    engine: Any,  # noqa: ANN401 - a crisperwhisper engine
    windows: Sequence[Window],
    teacher_logits: Callable[[Any, Window], Any],
    special_token_ids: Optional[set[int]] = None,
) -> list[dict[str, Any]]:
    """Return one `token_confidence_from_logits` block per window with ids, scored from `teacher_logits`."""
    import torch  # noqa: PLC0415 - torch loads only where logits are scored

    blocks: list[dict[str, Any]] = []
    for window in windows:
        if window.ids:
            logits = torch.as_tensor(teacher_logits(engine, window))
            blocks.extend(
                token_confidence_from_logits(
                    logits=list(logits), sequences=torch.tensor([window.ids]), special_token_ids=special_token_ids
                )
            )
    return blocks
