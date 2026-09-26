// The facet model: composition, the two counts, absence as a value, and the mask the view draws.
// Every row here is synthetic.

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
globalThis.SchemaFacets = F
const CorpusView = require(join(viewer, 'corpus.js'))

// The view schedules its paint on a frame; node has no frames, so the paint never runs and what
// these tests read is the selection the paint would have drawn.
globalThis.requestAnimationFrame = () => 0
globalThis.cancelAnimationFrame = () => {}
globalThis.window = { devicePixelRatio: 1 }

const ABSENT = F.ABSENT

/**
 * A small corpus with every shape the panel must handle: a closed vocabulary, an open one, nulls,
 * and a set column.
 */
function corpus () {
  const rows = []
  const spec = [
    // [n, task, verdict, conformance_speech, gate_failed_names]
    [10, 'story-recall', 'pass', 'true', []],
    [6, 'story-recall', 'flag', 'false', ['response_min_s']],
    [4, 'story-recall', 'pass', null, []],
    [7, 'free-speech', 'pass', 'true', []],
    [3, 'free-speech', 'flag', null, ['response_min_s', 'items_min']],
    [2, 'cough', 'discard', 'undetermined', ['items_min']],
    [1, 'cough', 'flag', null, null]
  ]
  for (const [n, task, verdict, conf, failed] of spec) {
    for (let i = 0; i < n; i++) {
      rows.push({
        __i: rows.length,
        task,
        verdict,
        conformance_speech: conf,
        gate_failed_names: failed,
        duration_s: 1 + rows.length
      })
    }
  }
  return rows
}

const ROWS = corpus() // 33 rows

test('the fixture is the size every count below is read against', () => {
  assert.equal(ROWS.length, 33)
})

// ------------------------------------------------------------------ the catalogue

test('every categorical and set column is offered, and the two unusable ones are refused', () => {
  const names = F.CATALOGUE.map(f => f.col.name)
  for (const n of ['verdict', 'task', 'release', 'release_ground', 'route_state',
    'route_airway', 'conformance_speech', 'gate_train_min_s_passed']) {
    assert.ok(names.includes(n), `${n} must be offered as a facet`)
  }
  assert.ok(!names.includes('participant'), 'participant is a list, not a facet')
  assert.ok(!names.includes('session'))
  assert.match(F.REFUSED.participant, /1,527 levels/)
  // and nothing numeric is offered: a number is brushed on an axis, not ticked in a list
  for (const f of F.CATALOGUE) assert.ok(['categorical', 'set'].includes(f.col.kind), f.col.name)
})

test('all 21 gate outcomes are facets, and the set columns face by membership', () => {
  const passed = F.CATALOGUE.filter(f => /^gate_.*_passed$/.test(f.col.name))
  assert.equal(passed.length, 21)
  assert.equal(F.BY_NAME.gate_failed_names.mode, 'set')
  assert.equal(F.BY_NAME.flag_nodes.mode, 'set')
  assert.equal(F.BY_NAME.verdict.mode, 'scalar')
})

// ------------------------------------------------------------------ counts

test('a value reports the whole-corpus total and, with nothing chosen, the same available', () => {
  const m = new F.FacetModel(ROWS)
  const v = m.values('task')
  const story = v.values.find(x => x.term === 'story-recall')
  assert.equal(story.total, 20)
  assert.equal(story.available, 20)
  assert.equal(v.values.find(x => x.term === 'free-speech').total, 10)
  assert.equal(v.values.find(x => x.term === 'cough').total, 3)
  assert.equal(m.before(), 33)
  assert.equal(m.after(), 33)
})

test('absence is a value with its own count, and it is always last', () => {
  const m = new F.FacetModel(ROWS)
  const v = m.values('conformance_speech')
  const last = v.values[v.values.length - 1]
  assert.equal(last.term, ABSENT)
  assert.equal(last.absent, true)
  assert.equal(last.total, 8) // 4 + 3 + 1
  assert.equal(v.values.filter(x => x.absent).length, 1)
})

test('a closed vocabulary keeps its declared order and an open one sorts by count', () => {
  const m = new F.FacetModel(ROWS)
  assert.deepEqual(
    m.values('verdict').values.map(x => x.term),
    ['pass', 'flag', 'discard', ABSENT]
  )
  assert.deepEqual(
    m.values('task').values.map(x => x.term),
    ['story-recall', 'free-speech', 'cough', ABSENT]
  )
})

// ------------------------------------------------------------------ narrowing

test('choosing one value narrows the drawn set to exactly that value count', () => {
  const m = new F.FacetModel(ROWS)
  m.toggle('task', 'story-recall')
  assert.equal(m.before(), 33)
  assert.equal(m.after(), 20)
  const mask = m.mask()
  assert.equal(mask.length, 33)
  assert.equal(mask.reduce((a, b) => a + b, 0), 20)
  ROWS.forEach((r, i) => assert.equal(mask[i], r.task === 'story-recall' ? 1 : 0))
})

test('two values of one facet are a union, not an intersection', () => {
  const m = new F.FacetModel(ROWS)
  m.toggle('task', 'story-recall')
  m.toggle('task', 'cough')
  assert.equal(m.after(), 23)
})

test('two facets are an intersection', () => {
  const m = new F.FacetModel(ROWS)
  m.toggle('task', 'story-recall')
  m.toggle('verdict', 'pass')
  assert.equal(m.after(), 14) // 10 + 4
  m.toggle('verdict', 'flag')
  assert.equal(m.after(), 20) // pass|flag over story-recall
})

test('a sibling value keeps a meaningful count after its neighbour is chosen', () => {
  const m = new F.FacetModel(ROWS)
  m.toggle('verdict', 'pass')
  const v = m.values('verdict')
  // `available` for a facet excludes that facet's own constraint, so `flag` is not zeroed
  assert.equal(v.values.find(x => x.term === 'pass').available, 21)
  assert.equal(v.values.find(x => x.term === 'flag').available, 10)
  assert.equal(v.values.find(x => x.term === 'discard').available, 2)
  // but another facet's counts *do* narrow, which is the whole point
  const t = m.values('task')
  assert.equal(t.values.find(x => x.term === 'story-recall').available, 14)
  assert.equal(t.values.find(x => x.term === 'cough').available, 0)
  assert.equal(t.values.find(x => x.term === 'story-recall').total, 20)
})

test('selecting absent keeps exactly the nulls, and never the empty sets', () => {
  const m = new F.FacetModel(ROWS)
  m.toggle('conformance_speech', ABSENT)
  assert.equal(m.after(), 8)
  const mask = m.mask()
  ROWS.forEach((r, i) => assert.equal(mask[i], r.conformance_speech == null ? 1 : 0))
})

test('a set facet is membership, and its absent bucket is null rather than the empty set', () => {
  const m = new F.FacetModel(ROWS)
  const v = m.values('gate_failed_names')
  assert.equal(v.mode, 'set')
  assert.equal(v.values.find(x => x.term === 'response_min_s').total, 9) // 6 + 3
  assert.equal(v.values.find(x => x.term === 'items_min').total, 5) // 3 + 2
  assert.equal(v.values[v.values.length - 1].total, 1) // the one row carrying null, not []
  m.toggle('gate_failed_names', 'items_min')
  assert.equal(m.after(), 5)
  m.toggle('gate_failed_names', 'response_min_s')
  assert.equal(m.after(), 11) // union: 9 + 5 - 3 rows carrying both
})

test('clearing one facet and clearing all restore the denominator exactly', () => {
  const m = new F.FacetModel(ROWS)
  m.toggle('task', 'cough')
  m.toggle('verdict', 'flag')
  assert.equal(m.after(), 1)
  m.clear('verdict')
  assert.equal(m.after(), 3)
  m.clear()
  assert.equal(m.after(), 33)
  assert.equal(m.mask(), null, 'no facet means no mask at all, not an all-ones mask')
  assert.deepEqual(m.activeNames(), [])
  assert.equal(m.chosenCount(), 0)
})

test('toggling the same value twice is a no-op on the drawn set', () => {
  const m = new F.FacetModel(ROWS)
  m.toggle('task', 'cough')
  m.toggle('task', 'cough')
  assert.equal(m.after(), 33)
  assert.equal(m.mask(), null)
})

test('a term that names an Object.prototype key is coded like any other', () => {
  // A task or a gate name is a value the producer wrote, not a key we chose, so the term index
  // must not resolve `constructor` or `__proto__` against Object.prototype.
  const rows = [
    { __i: 0, task: 'constructor', verdict: 'pass' },
    { __i: 1, task: '__proto__', verdict: 'pass' },
    { __i: 2, task: 'constructor', verdict: 'flag' },
    { __i: 3, task: 'story-recall', verdict: 'pass' }
  ]
  const m = new F.FacetModel(rows)
  const v = m.values('task')
  assert.equal(v.values.find(x => x.term === 'constructor').total, 2)
  assert.equal(v.values.find(x => x.term === '__proto__').total, 1)
  m.toggle('task', 'constructor')
  assert.equal(m.after(), 2)
  assert.deepEqual(Array.from(m.mask()), [1, 0, 1, 0])
})

// ------------------------------------------------------------------ the base mask

test('the base mask is the brushes, and the facet counts are read against it', () => {
  const m = new F.FacetModel(ROWS)
  // a brush that keeps only the first 12 rows: 10 story-recall/pass and 2 story-recall/flag
  const base = new Uint8Array(33)
  for (let i = 0; i < 12; i++) base[i] = 1
  m.setBase(base)
  assert.equal(m.before(), 12)
  assert.equal(m.after(), 12)
  const v = m.values('verdict')
  assert.equal(v.values.find(x => x.term === 'pass').available, 10)
  assert.equal(v.values.find(x => x.term === 'flag').available, 2)
  assert.equal(v.values.find(x => x.term === 'pass').total, 21, 'total is the corpus, not the brush')
  m.toggle('verdict', 'flag')
  assert.equal(m.after(), 2)
})

// ------------------------------------------------------------------ the view

test('the view draws the intersection of the brushes and the facet mask', () => {
  const view = new CorpusView(stubCanvas(), stubCanvas())
  view.setRows(ROWS)
  view.setAxes(['task', 'verdict'])
  assert.equal(view.selectedCount, 33)

  const m = new F.FacetModel(ROWS)
  m.toggle('task', 'story-recall')
  view.setFacetMask(m.mask())
  assert.equal(view.selectedCount, 20)
  assert.equal(view.brushMask.reduce((a, b) => a + b, 0), 33, 'the brush mask is untouched by a facet')

  view.setBrush('verdict', { terms: ['pass'] })
  assert.equal(view.selectedCount, 14)
  assert.equal(view.brushMask.reduce((a, b) => a + b, 0), 21)

  view.clearBrushes()
  assert.equal(view.selectedCount, 20, 'clearing brushes leaves the facets standing')

  view.setFacetMask(null)
  assert.equal(view.selectedCount, 33)
})

test('a facet never changes which values are absent on an axis', () => {
  const view = new CorpusView(stubCanvas(), stubCanvas())
  view.setRows(ROWS)
  view.setAxes(['conformance_speech'])
  const before = view.summaries[0].absent
  const m = new F.FacetModel(ROWS)
  m.toggle('task', 'story-recall')
  view.setFacetMask(m.mask())
  assert.equal(view.summaries[0].absent, before)
})

/** A canvas with just enough of a 2-D context for the view to lay out without painting. */
function stubCanvas () {
  const noop = () => {}
  const ctx = new Proxy({}, {
    get: (_, k) => (k === 'canvas' ? {} : (k === 'measureText' ? (() => ({ width: 10 })) : noop))
  })
  return { clientWidth: 1200, clientHeight: 700, width: 1200, height: 700, getContext: () => ctx }
}
