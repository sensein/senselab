// The axis model and the drawing rule for nulls. Every row here is synthetic.

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
const CorpusView = require(join(viewer, 'corpus.js'))

function rows(...specs) {
  return specs.map((s, i) => Object.assign({ __i: i }, s))
}

// -------------------------------------------------------------- the catalogue

test('the measurement names are the twenty-nine recording_vectors.py declares', () => {
  const all = A.SCALAR_MEASUREMENTS.concat(A.VECTOR_MEASUREMENTS, A.MATRIX_MEASUREMENTS, A.CATEGORICAL_MEASUREMENTS)
  assert.equal(all.length, 29)
  assert.equal(A.SCALAR_MEASUREMENTS.length, 17)
  assert.equal(A.VECTOR_MEASUREMENTS.length, 1)
  assert.equal(A.MATRIX_MEASUREMENTS.length, 1)
  assert.equal(A.CATEGORICAL_MEASUREMENTS.length, 10)
  assert.equal(new Set(all).size, 29)
})

test('a vector and a matrix are offered but refused, with the reason on the option', () => {
  for (const name of ['m_ddk_position_realised_mass', 'm_ddk_cv_instrument_reading']) {
    const col = A.BY_NAME[name]
    assert.ok(col, `${name} must be in the catalogue so it is never silently missing`)
    assert.equal(col.assignable, false)
    assert.match(col.reason, /no single number/)
    assert.throws(() => A.summarise(name, rows({})), /not assignable/)
  }
})

test('a set column is refused but its size and its _n are assignable', () => {
  assert.equal(A.BY_NAME['m_carrier_rejected'].assignable, false)
  assert.equal(A.BY_NAME['m_carrier_rejected.size'].assignable, true)
  assert.equal(A.BY_NAME['m_carrier_rejected_n'].assignable, true)
  assert.equal(A.BY_NAME['flag_nodes'].assignable, false)
  assert.equal(A.BY_NAME['flag_nodes.size'].assignable, true)
})

test('the nine sentinel-only categoricals say they are presence flags', () => {
  assert.equal(A.SENTINEL_ONLY.length, 9)
  assert.ok(!A.SENTINEL_ONLY.includes('carrier_rejected'))
  assert.match(A.BY_NAME['m_sweep_extent'].reason, /presence flag/)
  assert.match(A.BY_NAME['m_carrier_rejected'].reason, /gate names/)
})

test('the first three default axes are participant, task, verdict, and there are ten', () => {
  assert.deepEqual(A.DEFAULT_AXES.slice(0, 3), ['participant', 'task', 'verdict'])
  assert.equal(A.DEFAULT_AXES.length, 10)
  assert.equal(A.MAX_AXES, 10)
  A.DEFAULT_AXES.forEach((n) => assert.ok(A.BY_NAME[n] && A.BY_NAME[n].assignable, `${n} must be assignable`))
})

test('the corpus read asks for no binary block and for no vector or matrix', () => {
  const cols = A.corpusColumns()
  for (const b of ['wave_minmax', 'env_dbfs', 'continuity', 'spans', 'span_labels', 'span_squim',
    'asr_words', 'asr_word_text', 'pii_marks', 'pii_category', 'branch_lanes',
    'm_ddk_position_realised_mass', 'm_ddk_cv_instrument_reading']) {
    assert.ok(!cols.includes(b), `${b} must not be in the first read`)
  }
  for (const s of ['participant', 'task', 'verdict', 'duration_s', 'flags_n', 'pii_findings_n', 'stem']) {
    assert.ok(cols.includes(s), `${s} must be in the first read`)
  }
})

// -------------------------------------------------------------- null is absent

test('a null never lands on the value scale', () => {
  const s = A.summarise('duration_s', rows({ duration_s: 1 }, { duration_s: null }, { duration_s: 9 }))
  assert.equal(s.present, 2)
  assert.equal(s.absent, 1)
  assert.equal(s.min, 1)
  assert.equal(s.max, 9)
  assert.equal(A.position(s, null), null)
  assert.equal(A.position(s, 1), 0)
  assert.equal(A.position(s, 9), 1)
})

test('a null does not pull the domain down to zero', () => {
  const s = A.summarise('m_glide_extent_semitones', rows(
    { m_glide_extent_semitones: 12, m_glide_extent_semitones_n: 1 },
    { m_glide_extent_semitones: null, m_glide_extent_semitones_n: 0 },
    { m_glide_extent_semitones: 20, m_glide_extent_semitones_n: 1 },
  ))
  assert.equal(s.min, 12)
  assert.notEqual(s.min, 0)
})

test('pii_findings_n keeps zero as a value and null as absent', () => {
  const s = A.summarise('pii_findings_n', rows(
    { pii_findings_n: 0 }, { pii_findings_n: null }, { pii_findings_n: 4 },
  ))
  assert.equal(s.present, 2)
  assert.equal(s.absent, 1)
  assert.equal(A.position(s, 0), 0)
  assert.equal(A.position(s, null), null)
  assert.match(A.BY_NAME.pii_findings_n.nullMeans, /0 means scanned and clean/)
})

// -------------------------------------------------------------- categoricals

test('a categorical axis is ordered categories, with a declared order where one exists', () => {
  const s = A.summarise('verdict', rows(
    { verdict: 'flag' }, { verdict: 'pass' }, { verdict: 'pass' }, { verdict: 'discard' },
  ))
  assert.equal(s.kind, 'categorical')
  assert.deepEqual(s.categories, ['pass', 'flag', 'discard'])
  assert.equal(A.position(s, 'pass'), 1)
  assert.equal(A.position(s, 'discard'), 0)
  assert.equal(A.position(s, 'nope'), null)
})

test('an open vocabulary orders by frequency, ties broken by name', () => {
  const s = A.summarise('task', rows(
    { task: 'b' }, { task: 'a' }, { task: 'a' }, { task: 'c' },
  ))
  assert.deepEqual(s.categories, ['a', 'b', 'c'])
})

test('a categorical axis can be ordered alphabetically instead', () => {
  const s = A.summarise('task', rows({ task: 'z' }, { task: 'z' }, { task: 'a' }), 'alphabetical')
  assert.deepEqual(s.categories, ['a', 'z'])
})

test('a single-category axis sits at the middle rather than dividing by zero', () => {
  const s = A.summarise('verdict', rows({ verdict: 'pass' }))
  assert.equal(A.position(s, 'pass'), 0.5)
})

test('a numeric axis with one distinct value widens rather than dividing by zero', () => {
  const s = A.summarise('duration_s', rows({ duration_s: 7 }, { duration_s: 7 }))
  assert.ok(s.flat)
  assert.ok(s.min < 7 && s.max > 7)
  assert.ok(Number.isFinite(A.position(s, 7)))
})

// -------------------------------------------------------------- the reduction

test('an m_ axis says it is a mean and counts the rows that fold more than one reading', () => {
  const s = A.summarise('m_interruptions', rows(
    { m_interruptions: 3, m_interruptions_n: 1 },
    { m_interruptions: 4, m_interruptions_n: 7 },
    { m_interruptions: null, m_interruptions_n: 0 },
  ))
  assert.equal(s.folded, 1)
  assert.equal(s.foldedMax, 7)
  const text = A.caption(s)
  assert.match(text, /mean/)
  assert.match(text, /1 of 2 fold >1 reading \(max 7\)/)
  assert.match(text, /1 absent/)
})

test('an axis where nothing folds says so rather than implying a reduction', () => {
  const s = A.summarise('m_voiced_duration_s', rows(
    { m_voiced_duration_s: 3, m_voiced_duration_s_n: 1 },
    { m_voiced_duration_s: 4, m_voiced_duration_s_n: 1 },
  ))
  assert.equal(s.folded, 0)
  assert.match(A.caption(s), /every reading is a single measurement/)
})

test('a plain column carries no reduction claim', () => {
  const s = A.summarise('duration_s', rows({ duration_s: 1 }))
  assert.equal(s.col.reduction, null)
  assert.ok(!A.caption(s).includes('mean'))
})

test('a .size axis reads the list length and keeps a null list absent', () => {
  const s = A.summarise('flag_nodes.size', rows(
    { flag_nodes: [] }, { flag_nodes: ['SPEECH', 'VOICE'] }, { flag_nodes: null },
  ))
  assert.equal(s.present, 2)
  assert.equal(s.absent, 1)
  assert.equal(s.min, 0)
  assert.equal(s.max, 2)
})

// -------------------------------------------------------------- the drawing rule

function view(axisNames, data) {
  const v = new CorpusView({ getContext: () => ({}), clientWidth: 1000, clientHeight: 600 },
    { getContext: () => ({}) })
  v.setRows(data)
  v.axes = axisNames
  v.summaries = axisNames.map((n) => A.summarise(n, data))
  v.layout()
  return v
}

test('an absent value breaks the polyline; it is never a vertex on the value band', () => {
  const data = rows({ duration_s: 1, flags_n: 0, pii_findings_n: 5 },
    { duration_s: 9, flags_n: 2, pii_findings_n: null })
  const v = view(['duration_s', 'flags_n', 'pii_findings_n'], data)
  const drawn = v.segmentsFor(v.vertices(data[1]))
  assert.equal(drawn.segments.length, 1, 'only the duration→flags segment is drawn')
  assert.equal(drawn.stubs.length, 1, 'and one stub runs to the rail')
  const complete = v.segmentsFor(v.vertices(data[0]))
  assert.equal(complete.segments.length, 2)
  assert.equal(complete.stubs.length, 0)
})

test('the absent vertex carries y = null, so the rail can never be read as a value', () => {
  const data = rows({ duration_s: 1, pii_findings_n: null })
  const v = view(['duration_s', 'pii_findings_n'], data)
  const verts = v.vertices(data[0])
  assert.equal(verts[1].y, null)
  assert.equal(verts[1].absent, true)
  assert.ok(verts[0].y > v.geom.bandTop - 1 && verts[0].y < v.geom.bandBottom + 1)
})

test('a zero is a vertex on the band and an absent is not, on the same axis', () => {
  const data = rows({ pii_findings_n: 0 }, { pii_findings_n: null }, { pii_findings_n: 8 })
  const v = view(['pii_findings_n'], data)
  assert.equal(v.vertices(data[0])[0].absent, false)
  assert.equal(v.vertices(data[1])[0].absent, true)
  assert.notEqual(v.vertices(data[0])[0].y, null)
})

test('a run of absences leaves no segment at all rather than a flat line along the rail', () => {
  const data = rows({ duration_s: null, flags_n: null, pii_findings_n: null })
  const v = view(['duration_s', 'flags_n', 'pii_findings_n'], data)
  const drawn = v.segmentsFor(v.vertices(data[0]))
  assert.equal(drawn.segments.length, 0)
  assert.equal(drawn.stubs.length, 0)
})

// -------------------------------------------------------------- brushing

test('brushing a numeric axis keeps the range and drops the absent', () => {
  const data = rows({ duration_s: 1 }, { duration_s: 5 }, { duration_s: 9 }, { duration_s: null })
  const v = view(['duration_s'], data)
  v.brushes = { duration_s: { lo: 4, hi: 6 } }
  v.applyBrushes()
  assert.deepEqual(Array.from(v.selected), [0, 1, 0, 0])
  assert.equal(v.selectedCount, 1)
})

test('absent can be included or isolated, and is excluded by default', () => {
  const data = rows({ pii_findings_n: 0 }, { pii_findings_n: null }, { pii_findings_n: 3 })
  const v = view(['pii_findings_n'], data)
  v.brushes = { pii_findings_n: { lo: 0, hi: 1 } }
  v.applyBrushes()
  assert.deepEqual(Array.from(v.selected), [1, 0, 0])
  v.brushes = { pii_findings_n: { lo: 0, hi: 1, absent: 'include' } }
  v.applyBrushes()
  assert.deepEqual(Array.from(v.selected), [1, 1, 0])
  v.brushes = { pii_findings_n: { absent: 'only' } }
  v.applyBrushes()
  assert.deepEqual(Array.from(v.selected), [0, 1, 0])
})

test('brushing a categorical axis keeps the chosen terms', () => {
  const data = rows({ verdict: 'pass' }, { verdict: 'flag' }, { verdict: 'discard' })
  const v = view(['verdict'], data)
  v.brushes = { verdict: { terms: ['flag', 'discard'] } }
  v.applyBrushes()
  assert.deepEqual(Array.from(v.selected), [0, 1, 1])
})

test('two brushes intersect', () => {
  const data = rows(
    { verdict: 'pass', duration_s: 1 }, { verdict: 'flag', duration_s: 1 },
    { verdict: 'flag', duration_s: 9 },
  )
  const v = view(['verdict', 'duration_s'], data)
  v.brushes = { verdict: { terms: ['flag'] }, duration_s: { lo: 0, hi: 5 } }
  v.applyBrushes()
  assert.deepEqual(Array.from(v.selected), [0, 1, 0])
})

test('removing an axis drops its brush rather than filtering invisibly', () => {
  const data = rows({ verdict: 'pass', duration_s: 1 }, { verdict: 'flag', duration_s: 9 })
  const v = view(['verdict', 'duration_s'], data)
  v.brushes = { verdict: { terms: ['flag'] } }
  v.setAxes(['duration_s'])
  assert.deepEqual(Object.keys(v.brushes), [])
  assert.equal(v.selectedCount, 2)
})
