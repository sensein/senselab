"""An instruction-tuned LLM reading a redacted transcript back, in its own isolated venv.

The fifth PII engine, and the only one that is asked to *explain* rather than to mark. It runs after
the detector cascade has already planned and applied its redactions, over the text those redactions
produced, and it is off unless a run turns it on.

Its product is a chain of thought and a list of concerns, not a set of spans: what it can do that the
detectors cannot is say *why* a residue is identifying — a date plus a street plus an employer that
no single detector flags. It therefore never edits a released artifact and never widens a redaction;
a caller uses it to withhold.

The loop is the caller's, not the model's: one review per call, and a caller that wants the model to
see the effect of its own concerns masks them and calls again. See
``specs/20260817-triage-workflow-dag/llm-check.md``.

Heavy dependencies (``transformers``, ``torch``, ``accelerate``, ``compressed-tensors``) live in an
isolated venv built by :func:`senselab.utils.subprocess_venv.ensure_venv`, the same way the detector
cascade's own venv does, so the host stays off a fixed torch stack.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

logger = logging.getLogger("senselab")

REVIEW_VENV = "pii-redaction-review"
REVIEW_PYTHON = "3.12"

REVIEW_REQUIREMENTS = [
    "transformers>=4.57",
    "torch>=2.8,<2.9",
    "accelerate>=1.0",
    "compressed-tensors>=0.12",
]

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"
DEFAULT_REF = "main"
_REASONING_HEADING = "REASONING:"
_FINDINGS_HEADING = "FINDINGS:"

_PROMPT = (
    "You are auditing a transcript that has already been automatically redacted. Every [CATEGORY] "
    "token marks text that was removed. Your job is to find anything still present that could "
    "identify the speaker, alone or in combination with the rest of the text.\n\n"
    "Answer in exactly two parts.\n"
    "First, a section headed REASONING: your full reasoning, in prose, including what you "
    "considered and rejected.\n"
    "Second, a section headed FINDINGS: a JSON array. Each element is an object with keys "
    '"text" (the exact substring that concerns you), "category" (one of PERSON, LOCATION, '
    'DATE_TIME, ORGANIZATION, ID, CONTACT, OTHER) and "why" (one sentence). Return [] if nothing '
    "remains.\n\nTRANSCRIPT:\n"
)


@dataclass
class ReviewFinding:
    """One residue the model believes is still identifying.

    Attributes:
        text: The substring it named.
        category: Its category, uppercased.
        why: Its one-sentence reason.
    """

    text: str
    category: str
    why: str


@dataclass
class ReviewResult:
    """One pass of the reviewer over one text.

    Attributes:
        available: Whether the model ran at all. ``False`` with a populated ``failure`` is the only
            honest answer when it did not — an empty ``findings`` under ``available=False`` reads
            identically to "the model found nothing", which is the one wrong answer here.
        reasoning: The model's chain of thought, verbatim. The point of the step.
        findings: What it flagged.
        failure: ``None`` on success; otherwise why the reviewer did not run.
        model_id: The repo the review was asked of.
        revision: The resolved 40-hex commit it loaded, or ``None``.
        raw: The model's completion, unparsed, when parsing recovered nothing.
    """

    available: bool
    reasoning: str = ""
    findings: list[ReviewFinding] = field(default_factory=list)
    failure: Optional[str] = None
    model_id: str = ""
    revision: Optional[str] = None
    raw: str = ""


# Worker payload (stdin JSON):
#   {
#     "text": str,
#     "prompt": str,
#     "model_id": str,
#     "model_path": str | None,   # staged snapshot dir, named by the commit it holds
#     "revision": str | None,     # resolved commit SHA, never a mutable ref
#     "max_new_tokens": int,
#   }
# Worker output (stdout JSON):
#   {"completion": str, "revision": str | None}
_REVIEW_WORKER_SCRIPT = r"""
import json, os, re, sys


def main():
    args = json.loads(sys.stdin.read())
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = args["model_id"]
    revision = args.get("revision")
    target = args.get("model_path") or model_id
    loaded_revision = revision
    if args.get("model_path"):
        basename = os.path.basename(os.path.normpath(args["model_path"]))
        loaded_revision = basename if re.fullmatch(r"[0-9a-f]{40}", basename) else revision
        tokenizer = AutoTokenizer.from_pretrained(target)
        model = AutoModelForCausalLM.from_pretrained(target, dtype="auto", device_map="auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(target, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(target, revision=revision, dtype="auto", device_map="auto")
    model.eval()

    messages = [{"role": "user", "content": args["prompt"] + args["text"]}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=int(args["max_new_tokens"]), do_sample=False
        )
    completion = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(json.dumps({"completion": completion, "revision": loaded_revision}))


try:
    main()
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


def _staged_snapshot(repo_id: str, revision: str) -> Optional[str]:
    """The local snapshot directory for a staged commit, or ``None`` when it could not be staged.

    Args:
        repo_id: The HuggingFace repo id.
        revision: The resolved 40-hex commit SHA.

    Returns:
        The ``snapshots/<sha>/`` directory as a string, or ``None`` — in which case the worker keeps
        the repo id as its load target and loads online with the same SHA.
    """
    from senselab.utils.dependencies import resolve_model

    try:
        _sha, snapshot = resolve_model(repo_id, revision)
    except Exception as exc:  # noqa: BLE001 — an unstageable model is a fallback, not a crash
        logger.warning(f"redaction review: {repo_id}@{revision} could not be staged ({exc}); loading online.")
        return None
    return str(snapshot)


def parse_completion(completion: str) -> tuple[str, list[ReviewFinding]]:
    """Split the model's answer into its reasoning and its findings.

    The findings array is looked for after the last ``FINDINGS:`` heading rather than at the first
    ``[`` in the whole completion, because the reasoning routinely quotes the transcript's own
    ``[CATEGORY]`` placeholders and splitting on those would truncate it at the first quotation.

    Args:
        completion: The model's raw text.

    Returns:
        ``(reasoning, findings)``. The reasoning is returned even when the array is missing or
        unparsable, because the reasoning is what the step exists to capture. A malformed array
        yields no findings rather than raising — a caller reads ``available`` to tell that from a
        clean pass.
    """
    marker = completion.rfind(_FINDINGS_HEADING)
    head = completion if marker == -1 else completion[:marker]
    tail = completion if marker == -1 else completion[marker + len(_FINDINGS_HEADING) :]
    start, end = tail.find("["), tail.rfind("]")
    if marker == -1 and start != -1:
        head = completion[:start]
    reasoning = head.replace(_REASONING_HEADING, " ").replace(_FINDINGS_HEADING, " ").strip()
    if start == -1 or end < start:
        return reasoning, []
    try:
        parsed = json.loads(tail[start : end + 1])
    except ValueError:
        return reasoning, []
    if not isinstance(parsed, list):
        return reasoning, []
    findings = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        findings.append(
            ReviewFinding(
                text=text,
                category=str(item.get("category") or "OTHER").upper(),
                why=str(item.get("why") or ""),
            )
        )
    return reasoning, findings


def review_redacted_text(
    text: str,
    *,
    model_id: str = DEFAULT_MODEL,
    ref: str = DEFAULT_REF,
    max_new_tokens: int = 1024,
    timeout_s: int = 1800,
) -> ReviewResult:
    """Ask the reviewer to read one redacted transcript back, reporting failure rather than raising.

    The ref is resolved to a commit SHA here and only the SHA reaches the worker, so a load can
    never go back through a pointer that may have moved.

    Args:
        text: The redacted transcript.
        model_id: The HuggingFace repo. Defaults to the QAT w4a16 Gemma-4 31B checkpoint, which is
            the variant that fits one ordinary GPU; see
            ``specs/20260817-triage-workflow-dag/config-derivations.md``.
        ref: The ref to resolve. Never passed to a load.
        max_new_tokens: Generation ceiling. The reasoning is the product, so this is not small.
        timeout_s: Wall-clock ceiling on the worker, model load included.

    Returns:
        The review. ``available`` is ``True`` only when the worker answered; every other path —
        venv build failure, missing weights, out of memory, timeout, a worker that raised — returns
        ``available=False`` with a populated ``failure`` and no findings.
    """
    try:
        from senselab.utils.model_revision import resolve_revision

        revision = resolve_revision(model_id, ref)
    except Exception as exc:  # noqa: BLE001 — an unresolvable ref is a recorded absence, not a crash
        return ReviewResult(available=False, failure=f"revision: {type(exc).__name__}: {exc}", model_id=model_id)

    try:
        venv_dir = ensure_venv(REVIEW_VENV, REVIEW_REQUIREMENTS, python_version=REVIEW_PYTHON)
        env = hf_subprocess_env(model_id, revision, base_env=_clean_subprocess_env())
        payload = json.dumps(
            {
                "text": text,
                "prompt": _PROMPT,
                "model_id": model_id,
                "model_path": _staged_snapshot(model_id, revision),
                "revision": revision,
                "max_new_tokens": int(max_new_tokens),
            }
        )
        result = subprocess.run(
            [venv_python(venv_dir), "-c", _REVIEW_WORKER_SCRIPT],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        output = parse_subprocess_result(result, "redaction review")
    except Exception as exc:  # noqa: BLE001 — every failure mode becomes a recorded absence
        return ReviewResult(
            available=False,
            failure=f"{type(exc).__name__}: {exc}",
            model_id=model_id,
            revision=revision,
        )

    completion = str(output.get("completion") or "")
    reasoning, findings = parse_completion(completion)
    return ReviewResult(
        available=True,
        reasoning=reasoning,
        findings=findings,
        model_id=model_id,
        revision=output.get("revision") or revision,
        raw="" if reasoning else completion,
    )


def review_payload(result: ReviewResult) -> dict[str, Any]:
    """One review as the mapping a provenance store records.

    Args:
        result: The review.

    Returns:
        The mapping. ``reasoning`` is the chain of thought verbatim, which is what the step is for.
    """
    return {
        "available": result.available,
        "reasoning": result.reasoning,
        "findings": [
            {"text": finding.text, "category": finding.category, "why": finding.why} for finding in result.findings
        ],
        "failure": result.failure,
        "model_id": result.model_id,
        "revision": result.revision,
    }
