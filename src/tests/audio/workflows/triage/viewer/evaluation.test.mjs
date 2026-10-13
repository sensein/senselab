// The evaluate-triage summary: each count, and that each count's facet term selects exactly the
// rows it counted. Every row here is synthetic.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const viewer = join(here, '..', '..', '..', '..', '..', 'senselab', 'audio', 'workflows', 'triage', 'viewer')
const require = createRequire(import.meta.url)
const A = require(join(viewer, 'axes.js'))
globalThis.SchemaAxes = A
const F = require(join(viewer, 'facets.js'))
const E = require(join(viewer, 'evaluation.js'))

function rows() {
  return [
    {
      trimmable: true, trim_s: 4.5, clip_state: 'kept_inconsistent', clip_s: 0.2,
      q_raw_issues: ['noise_floor', 'clipping'], q_resolved_by_enhanced: ['noise_floor'], q_unresolved: ['clipping'],
      ms_signals: ['diarization', 'reviewer'], ms_agreement: 'diarization+reviewer',
    },
    {
      trimmable: false, trim_s: 0.3, clip_state: 'withdrawn_only', clip_s: 0,
      q_raw_issues: ['low_snr'], q_resolved_by_enhanced: [], q_unresolved: ['low_snr'],
      ms_signals: ['reviewer'], ms_agreement: 'reviewer_only',
    },
    {
      trimmable: true, trim_s: 2.0, clip_state: 'none', clip_s: 0,
      q_raw_issues: [], q_resolved_by_enhanced: [], q_unresolved: [],
      ms_signals: [], ms_agreement: 'none',
    },
    {
      trimmable: null, trim_s: null, clip_state: 'kept_consistent', clip_s: 1.5,
      q_raw_issues: null, q_resolved_by_enhanced: null, q_unresolved: null,
      ms_signals: ['diarization'], ms_agreement: 'diarization_only',
    },
  ]
}

function find(sections, title) {
  return sections.find((s) => s.title === title)
}

function items(sections) {
  const out = []
  for (const s of sections) {
    if (s.items) out.push(...s.items)
    if (s.rows) for (const r of s.rows) out.push(r.raw, r.resolved, r.unresolved)
  }
  return out
}

test('trimmable counts and their trim seconds', () => {
  const trim = find(E.summarise(rows()), 'task extent: trimmable')
  assert.deepEqual(trim.items.map((i) => [i.term, i.count, i.seconds]), [['true', 2, 6.5], ['false', 1, 0.3]])
})

test('raw quality against enhanced, per check', () => {
  const q = find(E.summarise(rows()), 'raw quality → enhanced')
  const by = Object.fromEntries(q.rows.map((r) => [r.check, [r.raw.count, r.resolved.count, r.unresolved.count]]))
  assert.deepEqual(by, { noise_floor: [1, 1, 0], low_snr: [1, 0, 1], clipping: [1, 0, 1], dropout: [0, 0, 0] })
})

test('clipping by re-assessment state, with clipped seconds', () => {
  const c = find(E.summarise(rows()), 'clipping and its re-assessment')
  assert.deepEqual(c.items.map((i) => [i.term, i.count, i.seconds]), [
    ['kept_inconsistent', 1, 0.2], ['kept_consistent', 1, 1.5], ['withdrawn_only', 1, 0], ['none', 1, null],
  ])
})

test('multi-speaker by signal and by agreement', () => {
  const m = find(E.summarise(rows()), 'another speaker inside the task extent')
  const by = Object.fromEntries(m.items.map((i) => [i.facet + ':' + i.term, i.count]))
  assert.equal(by['ms_signals:diarization'], 2)
  assert.equal(by['ms_signals:reviewer'], 2)
  assert.equal(by['ms_signals:separation'], 0)
  assert.equal(by['ms_agreement:diarization+reviewer'], 1)
  assert.equal(by['ms_agreement:reviewer_only'], 1)
  assert.equal(by['ms_agreement:none'], 1)
})

test('every count names a catalogued facet whose term selects exactly that many rows', () => {
  const data = rows()
  for (const it of items(E.summarise(data))) {
    assert.ok(F.BY_NAME[it.facet], it.facet + ' is not a facet')
    const model = new F.FacetModel(data)
    model.toggle(it.facet, it.term)
    const mask = model.mask()
    let n = 0
    for (let i = 0; i < mask.length; i++) if (mask[i]) n++
    assert.equal(n, it.count, it.facet + ' = ' + it.term)
  }
})
