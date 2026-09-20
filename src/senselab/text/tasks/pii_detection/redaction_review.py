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

The venv holds **one long-lived worker per process**, started on the first review and reused by every
later one, so the weights are loaded once rather than once per review. :func:`shutdown_review_worker`
ends it; it also ends at interpreter exit and when its stdin reaches EOF, so a killed parent does not
leave weights resident. See ``specs/20260817-triage-workflow-dag/llm-check-amortised-load.md``.
"""

from __future__ import annotations

import atexit
import json
import logging
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, venv_python

logger = logging.getLogger("senselab")

REVIEW_VENV = "pii-redaction-review"
REVIEW_PYTHON = "3.12"

REVIEW_REQUIREMENTS = [
    "transformers>=5.8",
    "torch>=2.8,<2.9",
    "accelerate>=1.0",
    "compressed-tensors>=0.15",
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
        elapsed_s: Wall-clock seconds this call took, a load it paid for included.
        load_s: How many of those seconds went on starting the worker and loading the weights.
            ``0.0`` when an already-running worker served the call.
        output_tokens: How many tokens the model generated, or ``None`` when it did not answer.
        peak_reserved_mib: Device memory the worker's allocator held at the end of the generation,
            before it was emptied. ``0`` on a CPU worker and on a call that did not answer.
        resident_mib: Device memory it holds between reviews — the weights and nothing else, which
            is what a second process on the same card has to live beside.
    """

    available: bool
    reasoning: str = ""
    findings: list[ReviewFinding] = field(default_factory=list)
    failure: Optional[str] = None
    model_id: str = ""
    revision: Optional[str] = None
    raw: str = ""
    elapsed_s: float = 0.0
    load_s: float = 0.0
    output_tokens: Optional[int] = None
    peak_reserved_mib: int = 0
    resident_mib: int = 0


# A line protocol over the worker's stdin/stdout, one JSON object per line.
#
# Load payload, sent once (stdin):
#   {
#     "model_id": str,
#     "model_path": str | None,   # staged snapshot dir, named by the commit it holds
#     "revision": str | None,     # resolved commit SHA, never a mutable ref
#   }
# Load reply (stdout):        {"ready": True, "revision": str | None, "load_s": float}
#
# Review request (stdin):     {"text": str, "prompt": str, "max_new_tokens": int}
# Review reply (stdout):      {"completion": str, "generate_s": float, "output_tokens": int,
#                              "peak_reserved_mib": int, "resident_mib": int}
# Stop request (stdin):       {"stop": True}
#
# The worker empties the CUDA caching allocator after each generation and reports what it held
# before and holds after. A process that keeps the weights must not also keep every transient
# buffer one generation touched: the allocator never returns blocks to the driver on its own, and
# this worker does not share an address space with the graph that needs the rest of the card. See
# ``specs/20260817-triage-workflow-dag/llm-check-amortised-load.md``.
#
# Every reply carries the ``_WORKER_MARKER`` prefix, so a library writing to the real stdout cannot
# be mistaken for one; ``sys.stdout`` is redirected to stderr before any heavy import for the same
# reason. A raised exception is reported as {"error": {"type": str, "message": str}} and ends the
# worker: a failure mid-generation is an out-of-memory or a dead CUDA context far more often than it
# is a bad request, and a process in that state cannot be trusted with the next one.
_WORKER_MARKER = "@@SENSELAB_REVIEW@@"

_REVIEW_WORKER_SCRIPT = (
    r"""
import json, os, re, sys, time

MARKER = "%s"
_replies = sys.stdout
sys.stdout = sys.stderr


def emit(payload):
    _replies.write(MARKER + json.dumps(payload) + "\n")
    _replies.flush()


def load(args):
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
    return torch, tokenizer, model, loaded_revision


def main():
    started = time.monotonic()
    torch, tokenizer, model, loaded_revision = load(json.loads(sys.stdin.readline()))
    emit({"ready": True, "revision": loaded_revision, "load_s": round(time.monotonic() - started, 3)})
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("stop"):
            return
        began = time.monotonic()
        messages = [{"role": "user", "content": request["prompt"] + request["text"]}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=int(request["max_new_tokens"]), do_sample=False
            )
        answer = generated[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(answer, skip_special_tokens=True)
        tokens = int(answer.shape[-1])
        generate_s = round(time.monotonic() - began, 3)
        peak, resident = 0, 0
        if torch.cuda.is_available():
            peak = int(torch.cuda.memory_reserved() / 2**20)
            del answer, generated, inputs
            torch.cuda.empty_cache()
            resident = int(torch.cuda.memory_reserved() / 2**20)
        emit(
            {
                "completion": completion,
                "generate_s": generate_s,
                "output_tokens": tokens,
                "peak_reserved_mib": peak,
                "resident_mib": resident,
            }
        )


try:
    main()
except Exception as exc:
    emit({"error": {"type": type(exc).__name__, "message": str(exc)}})
    sys.exit(1)
"""
    % _WORKER_MARKER
)


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


class ReviewWorkerError(RuntimeError):
    """A worker that did not start, did not answer, or reported an exception of its own.

    Its message is already in the shape a ``ReviewResult.failure`` carries — the worker-side
    exception type and message, or what the host observed instead — so a caller reports ``str(exc)``
    rather than prefixing its own type name and nesting one report inside another.
    """


def _failure(exc: BaseException) -> str:
    """One exception as the string a recorded absence carries.

    Args:
        exc: What went wrong.

    Returns:
        The message alone for a :class:`ReviewWorkerError`, which already names the failing type;
        otherwise the type and the message, because nothing else would say what raised.
    """
    return str(exc) if isinstance(exc, ReviewWorkerError) else f"{type(exc).__name__}: {exc}"


class _ReviewWorker:
    """One venv subprocess with the weights loaded, answering review requests until it is stopped.

    Attributes:
        model_id: The repo whose weights it holds.
        revision: The 40-hex commit those weights are, as the worker read it back.
        load_s: How long starting it and loading the weights took.
    """

    def __init__(self, model_id: str, revision: str) -> None:
        """Record what the worker will be asked to load. Starting it is :meth:`start`.

        Args:
            model_id: The HuggingFace repo.
            revision: The resolved 40-hex commit.
        """
        self.model_id = model_id
        self.revision: Optional[str] = revision
        self.load_s = 0.0
        self._process: Optional[subprocess.Popen[str]] = None
        self._replies: queue.Queue[dict[str, Any]] = queue.Queue()
        self._noise: deque[str] = deque(maxlen=40)

    def start(self, timeout_s: int) -> None:
        """Build the venv if needed, stage the commit, spawn the worker and wait for its weights.

        Args:
            timeout_s: Wall-clock ceiling on the load.

        Raises:
            ReviewWorkerError: If the worker died, raised, or did not report ready in time.
        """
        venv_dir = ensure_venv(REVIEW_VENV, REVIEW_REQUIREMENTS, python_version=REVIEW_PYTHON)
        model_path = _staged_snapshot(self.model_id, str(self.revision))
        env = hf_subprocess_env(self.model_id, str(self.revision), base_env=_clean_subprocess_env())
        self._process = subprocess.Popen(  # noqa: S603 — the interpreter is this repo's own venv
            [venv_python(venv_dir), "-c", _REVIEW_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._pump_replies, daemon=True).start()
        threading.Thread(target=self._pump_noise, daemon=True).start()
        began = time.monotonic()
        self._send({"model_id": self.model_id, "model_path": model_path, "revision": self.revision})
        reply = self._await(timeout_s)
        self.revision = reply.get("revision") or self.revision
        self.load_s = round(time.monotonic() - began, 3)

    @property
    def alive(self) -> bool:
        """Whether the worker process is still running and still holding its weights."""
        return self._process is not None and self._process.poll() is None

    def review(self, text: str, max_new_tokens: int, timeout_s: int) -> dict[str, Any]:
        """Ask the loaded model for one review.

        Args:
            text: The redacted transcript.
            max_new_tokens: Generation ceiling.
            timeout_s: Wall-clock ceiling on this generation alone.

        Returns:
            The worker's reply: ``completion``, ``generate_s`` and ``output_tokens``.

        Raises:
            ReviewWorkerError: If the worker died, raised, or did not answer in time.
        """
        self._send({"text": text, "prompt": _PROMPT, "max_new_tokens": int(max_new_tokens)})
        return self._await(timeout_s)

    def close(self) -> None:
        """End the worker, releasing the weights. Safe to call on a worker that never started."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.write(json.dumps({"stop": True}) + "\n")
                process.stdin.flush()
                process.stdin.close()
            process.wait(timeout=5)
        except Exception:  # noqa: BLE001 — a worker that will not stop politely is killed
            process.kill()
            try:
                process.wait(timeout=30)
            except Exception:  # noqa: BLE001, S110 — nothing further is owed to an unreapable child
                pass

    def _send(self, payload: dict[str, Any]) -> None:
        """Write one request line, turning a closed pipe into the failure the caller reports.

        Args:
            payload: The request.

        Raises:
            ReviewWorkerError: If the worker is gone or its stdin will not take the line.
        """
        if self._process is None or self._process.stdin is None:
            raise ReviewWorkerError("redaction review worker is not running")
        try:
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise ReviewWorkerError(f"redaction review worker closed its input: {exc}\n{self._tail()}") from exc

    def _await(self, timeout_s: int) -> dict[str, Any]:
        """Wait for one reply, killing the worker on anything that is not one.

        Args:
            timeout_s: How long to wait.

        Returns:
            The reply.

        Raises:
            ReviewWorkerError: On a timeout, a dead worker, or a worker-reported exception.
        """
        try:
            reply = self._replies.get(timeout=timeout_s)
        except queue.Empty:
            self.close()
            raise ReviewWorkerError(f"redaction review worker did not answer in {timeout_s}s\n{self._tail()}") from None
        if "error" in reply:
            self.close()
            error = reply["error"] or {}
            raise ReviewWorkerError(f"{error.get('type', 'RuntimeError')}: {error.get('message', 'unknown error')}")
        if reply.get("eof"):
            self.close()
            raise ReviewWorkerError(f"redaction review worker exited without answering\n{self._tail()}")
        return reply

    def _pump_replies(self) -> None:
        """Drain stdout into the reply queue, forwarding anything unmarked to the noise tail."""
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if line.startswith(_WORKER_MARKER):
                try:
                    self._replies.put(json.loads(line[len(_WORKER_MARKER) :]))
                except ValueError:
                    self._noise.append(line.rstrip())
            elif line.strip():
                self._noise.append(line.rstrip())
        self._replies.put({"eof": True})

    def _pump_noise(self) -> None:
        """Drain stderr so a chatty loader cannot fill its pipe and deadlock the worker."""
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            if line.strip():
                self._noise.append(line.rstrip())

    def _tail(self) -> str:
        """The worker's last lines of output, for a failure message that says what it was doing."""
        return "\n".join(self._noise)


_WORKER_LOCK = threading.Lock()
_WORKER: Optional[_ReviewWorker] = None
_WORKER_KEY: Optional[tuple[str, str]] = None
_WORKER_REFUSED: Optional[str] = None


def shutdown_review_worker() -> None:
    """End the process's review worker, releasing its weights, and clear any recorded refusal.

    A later :func:`review_redacted_text` starts a new one. Registered to run at interpreter exit, so
    a caller only needs this to hand the memory back earlier than that, or to retry a load that
    failed.
    """
    global _WORKER, _WORKER_KEY, _WORKER_REFUSED
    with _WORKER_LOCK:
        worker, _WORKER, _WORKER_KEY, _WORKER_REFUSED = _WORKER, None, None, None
    if worker is not None:
        worker.close()


atexit.register(shutdown_review_worker)


def _worker_for(model_id: str, revision: str, timeout_s: int) -> tuple[_ReviewWorker, float]:
    """The running worker holding this commit, started if there is not a live one already.

    Args:
        model_id: The HuggingFace repo.
        revision: The resolved 40-hex commit.
        timeout_s: Wall-clock ceiling on a load, when one is needed.

    Returns:
        ``(worker, load_s)`` — the worker with its weights loaded, and the seconds this call spent
        loading them, which is ``0.0`` when an already-running worker was reused.

    Raises:
        ReviewWorkerError: If a worker could not be started, now or earlier in this process.
    """
    global _WORKER, _WORKER_KEY, _WORKER_REFUSED
    if _WORKER_REFUSED is not None:
        raise ReviewWorkerError(_WORKER_REFUSED)
    if _WORKER is not None and _WORKER_KEY == (model_id, revision) and _WORKER.alive:
        return _WORKER, 0.0
    if _WORKER is not None:
        _WORKER.close()
        _WORKER, _WORKER_KEY = None, None
    worker = _ReviewWorker(model_id, revision)
    try:
        worker.start(timeout_s)
    except Exception as exc:
        worker.close()
        _WORKER_REFUSED = _failure(exc)
        raise
    _WORKER, _WORKER_KEY = worker, (model_id, revision)
    return worker, worker.load_s


def review_redacted_text(
    text: str,
    *,
    model_id: str = DEFAULT_MODEL,
    ref: str = DEFAULT_REF,
    max_new_tokens: int = 1024,
    timeout_s: int = 1800,
) -> ReviewResult:
    """Ask the reviewer to read one redacted transcript back, reporting failure rather than raising.

    The ref is resolved to a commit SHA before the worker starts and only the SHA reaches it, so a
    load can never go back through a pointer that may have moved. The worker outlives the call: the
    first review in a process pays the load and every later one does not, and ``load_s`` on the
    result says which this was.

    Calls are serialised — one worker holds one copy of the weights — so a concurrent caller waits.

    Args:
        text: The redacted transcript.
        model_id: The HuggingFace repo. Defaults to the QAT w4a16 Gemma-4 31B checkpoint, which is
            the variant that fits one ordinary GPU; see
            ``specs/20260817-triage-workflow-dag/config-derivations.md``.
        ref: The ref to resolve. Never passed to a load.
        max_new_tokens: Generation ceiling. The reasoning is the product, so this is not small.
        timeout_s: Wall-clock ceiling, applied to the load and to the generation separately.

    Returns:
        The review. ``available`` is ``True`` only when the worker answered; every other path —
        venv build failure, missing weights, out of memory, timeout, a worker that raised — returns
        ``available=False`` with a populated ``failure`` and no findings. A worker that could not be
        started is recorded once and reported without a retry for the rest of the process, so a host
        with no reachable GPU costs one load attempt rather than one per recording.
    """
    began = time.monotonic()
    try:
        from senselab.utils.model_revision import resolve_revision

        revision = resolve_revision(model_id, ref)
    except Exception as exc:  # noqa: BLE001 — an unresolvable ref is a recorded absence, not a crash
        return ReviewResult(
            available=False,
            failure=f"revision: {type(exc).__name__}: {exc}",
            model_id=model_id,
            elapsed_s=round(time.monotonic() - began, 3),
        )

    with _WORKER_LOCK:
        try:
            worker, load_s = _worker_for(model_id, revision, timeout_s)
            output = worker.review(text, max_new_tokens, timeout_s)
        except Exception as exc:  # noqa: BLE001 — every failure mode becomes a recorded absence
            return ReviewResult(
                available=False,
                failure=_failure(exc),
                model_id=model_id,
                revision=revision,
                elapsed_s=round(time.monotonic() - began, 3),
            )
        loaded_revision = worker.revision

    completion = str(output.get("completion") or "")
    reasoning, findings = parse_completion(completion)
    return ReviewResult(
        available=True,
        reasoning=reasoning,
        findings=findings,
        model_id=model_id,
        revision=loaded_revision or revision,
        raw="" if reasoning else completion,
        elapsed_s=round(time.monotonic() - began, 3),
        load_s=load_s,
        output_tokens=output.get("output_tokens"),
        peak_reserved_mib=int(output.get("peak_reserved_mib") or 0),
        resident_mib=int(output.get("resident_mib") or 0),
    )


def review_payload(result: ReviewResult) -> dict[str, Any]:
    """One review as the mapping a provenance store records.

    Args:
        result: The review.

    Returns:
        The mapping. ``reasoning`` is the chain of thought verbatim, which is what the step is for.
        ``elapsed_s`` and ``load_s`` are what the round cost and how much of that was the weights,
        and the two ``_mib`` fields are what it held on the device at its peak and between reviews,
        so a store answers the time and the memory without a stopwatch outside the graph.
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
        "elapsed_s": result.elapsed_s,
        "load_s": result.load_s,
        "output_tokens": result.output_tokens,
        "peak_reserved_mib": result.peak_reserved_mib,
        "resident_mib": result.resident_mib,
    }
