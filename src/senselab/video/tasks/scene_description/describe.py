"""Structured frame descriptions from a VLM, with one repair attempt per invalid completion."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from senselab.video.tasks.scene_description.backends import GenerationRequest, VLMBackend
from senselab.video.tasks.scene_description.schema import (
    DESCRIBE_PROMPT,
    SceneDescription,
    parse_scene_json,
    repair_prompt,
)


class FrameDescription(BaseModel):
    """The VLM's description of one frame.

    Attributes:
        description: The validated description, or None when neither attempt parsed.
        parse_ok: Whether ``description`` is set.
        repaired: Whether the description came from the repair attempt.
        raw_text: The first completion.
        repair_text: The repair completion, when one was requested.
        error: The last parse error, when ``parse_ok`` is False; the first one when repaired.
        seconds: Generation time summed over both attempts.
        prompt_tokens: Prompt tokens of the first attempt.
        generation_tokens: Generated tokens summed over both attempts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    description: Optional[SceneDescription] = None
    parse_ok: bool = False
    repaired: bool = False
    raw_text: str = ""
    repair_text: Optional[str] = None
    error: Optional[str] = None
    seconds: float = 0.0
    prompt_tokens: int = 0
    generation_tokens: int = 0


def describe_frames(
    images: Sequence[np.ndarray],
    backend: VLMBackend,
    prompt: str = DESCRIBE_PROMPT,
    batch_size: int = 8,
) -> List[FrameDescription]:
    """Describe each image with the VLM and validate the result against ``SceneDescription``.

    Every image is sent with ``prompt``. A completion that does not parse is sent back once,
    text-only, with the parse error and a request for corrected JSON; ``parse_ok`` records whether
    either attempt validated.

    Args:
        images: RGB ``uint8`` arrays.
        backend: A loaded VLM backend; its model stays resident across calls.
        prompt: The description prompt.
        batch_size: Images sent to the backend per request.

    Returns:
        One ``FrameDescription`` per image, in order.
    """
    out: List[FrameDescription] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        first = backend.generate([GenerationRequest(prompt=prompt, image=img) for img in chunk])
        records: List[FrameDescription] = []
        retry: List[int] = []
        for i, gen in enumerate(first):
            desc, err = parse_scene_json(gen.text)
            records.append(
                FrameDescription(
                    description=desc,
                    parse_ok=desc is not None,
                    raw_text=gen.text,
                    error=err,
                    seconds=gen.seconds,
                    prompt_tokens=gen.prompt_tokens,
                    generation_tokens=gen.generation_tokens,
                )
            )
            if desc is None:
                retry.append(i)
        if retry:
            second = backend.generate(
                [GenerationRequest(prompt=repair_prompt(records[i].raw_text, records[i].error or "")) for i in retry]
            )
            for i, gen in zip(retry, second):
                rec = records[i]
                desc, err = parse_scene_json(gen.text)
                rec.repair_text = gen.text
                rec.seconds += gen.seconds
                rec.generation_tokens += gen.generation_tokens
                if desc is not None:
                    rec.description, rec.parse_ok, rec.repaired = desc, True, True
                else:
                    rec.error = err
        out.extend(records)
    return out
