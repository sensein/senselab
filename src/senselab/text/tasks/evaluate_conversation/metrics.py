"""Metrics to assess performance on tutor response.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.

All other functions are value-independent.
"""

from typing import Dict, List

import sacrebleu as sb
import textstat
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEUScore


def Rouge(*args: List, **kwargs: Dict) -> rouge_scorer.RougeScorer:
    """Wrapper for rouge_scorer's RougeScorer class."""
    return rouge_scorer.RougeScorer(*args, **kwargs)


Rouge.__doc__ = rouge_scorer.RougeScorer.__doc__


def sentence_bleu_sacre(*args: List, **kwargs: Dict) -> BLEUScore:
    """Wrapper for sacrebleu's sentence_bleu function."""
    return sb.sentence_bleu(*args, **kwargs)


sentence_bleu_sacre.__doc__ = sb.sentence_bleu.__doc__


def word_count(*args: List, **kwargs: Dict) -> int:
    """Wrapper for textstat's lexicon_count function."""
    return textstat.lexicon_count(*args, **kwargs)


word_count.__doc__ = textstat.lexicon_count.__doc__


correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)
