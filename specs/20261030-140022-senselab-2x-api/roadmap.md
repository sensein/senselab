# Roadmap: the senselab flywheel (after 2.x)

Status: a future goal (P-36). It informs 2.x decisions; it does not add 2.x scope.

## Why

Triage became brittle by iterating on a fixed set of recordings from one dataset. Each found
error became a rule or threshold for that case. A flywheel turns each found error into
**labelled data**, refits or fine-tunes on it, and measures the change on held-out data. That is
the P-34/P-35 discipline applied to the science, not only to the code.

## The loop

```
datasets ──► pipelines measure ──► decisions + uncertainty ──► select items for review
   ▲                                                                   │
   │                                                     humans and agents annotate and review
   │                                                                   │
promote new profile or model ◄── evaluate on benchmarks + held-out labels ◄── fit profiles / fine-tune models
```

1. **Measure.** Pipelines run on datasets. Decisions carry uncertainty.
2. **Select.** A selection policy picks items worth a human's time: uncertainty, disagreement
   between backends, novelty, coverage gaps.
3. **Review and annotate.** Humans, and agents labelled as agents, review items and annotate
   spans. Disagreements are kept, never overwritten.
4. **Fit.** Labels feed `fit` tasks: threshold profiles, calibration, and fine-tuned models.
5. **Evaluate.** Candidates are scored on public benchmarks and held-out label sets, within
   measured bands.
6. **Promote.** A promotion is a recorded change of profile or model identity. Identity-based
   caching recomputes only what the promotion affects.

## What 2.x must keep possible

These are hooks, not features. Each is cheap now and expensive to retrofit.

| Hook | In 2.x |
|---|---|
| H-1 Annotations carry a producer (human, agent or model), a guideline version and a confidence | `Annotation.producer` (design §4.1); add the guideline version and confidence fields |
| H-2 Human labels are declared, identified inputs, so anything fitted on them is reproducible | I-1, I-9 (design §4.8) |
| H-3 `fit` produces artifacts; fine-tuned weights are models with content identity, training provenance and inherited licenses | `fit` (§4.14), local-weight hashing (§4.3), I-9 |
| H-4 Routing an item to review is an ordinary route branch; "awaiting review" is an environment-caused `Absent` that is never cached | `route`, `Absent(cause)` (§4.5, I-6) |
| H-5 Results can carry uncertainty in a standard optional field | result models (§4.1); generalise triage's `Estimate` |
| H-6 Datasets are versioned by manifest hash and carry annotation layers (BIDS derivatives) | dataset adapters (§4.6) |
| H-7 Human labels are source data: never deleted, pseudonymous, no identifiers (P-28) | provenance store; owner-label handling |

## Not in 2.x

- selection (active-learning) policies;
- annotation and review UIs beyond today's review pages;
- fine-tuning orchestration;
- inter-rater and label-quality models;
- promotion workflows.

These are designed once 2.x is in place. Triage's decision layer is then rebuilt as the first
flywheel application, rather than ported (design §6.2, Phase 4).
