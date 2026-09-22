// The decoders, against the byte layouts in specs/20260922-compact-recording-vectors/schema.md.
// Every fixture here is synthetic. No byte in this file came from a recording.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const viewer = join(here, '..', '..', '..', '..', '..', 'senselab', 'audio', 'workflows', 'triage', 'viewer')
const require = createRequire(import.meta.url)
const D = require(join(viewer, 'decode.js'))

const bytes = (...xs) => new Uint8Array(xs)
const u16le = (v) => [v & 0xff, (v >> 8) & 0xff]

// -------------------------------------------------------------- the worked example

test('the worked example in the schema document decodes as written', () => {
  // One span, row A, 1.0 s to 2.0 s of a recording whose time_scale_s is 4.0.
  const block = bytes(0x02, 0x00, 0x40, 0x00, 0x80)
  const spans = D.decodeSpans(block, 4.0)
  assert.equal(spans.length, 1)
  assert.equal(spans[0].row, 'A')
  assert.equal(spans[0].rowIndex, 2)
  assert.ok(Math.abs(spans[0].t0 - 1.0) < 1e-4, `t0 was ${spans[0].t0}`)
  assert.ok(Math.abs(spans[0].t1 - 2.0) < 1e-4, `t1 was ${spans[0].t1}`)
  // and the exact values the document quotes for the decode
  assert.ok(Math.abs(spans[0].t0 - 1.00002) < 1e-5)
  assert.ok(Math.abs(spans[0].t1 - 2.00003) < 1e-5)
})

test('the worked example is little-endian, and big-endian is a different answer', () => {
  const bigEndian = bytes(0x02, 0x40, 0x00, 0x80, 0x00)
  const spans = D.decodeSpans(bigEndian, 4.0)
  assert.ok(Math.abs(spans[0].t0 - 1.0) > 0.9, 'a big-endian read must not land on 1.0 s')
})

test('full scale is exactly 65535, not 65536', () => {
  assert.equal(D.TIME_SCALE, 65535)
  assert.equal(D.decodeTime(65535, 4.0), 4.0)
  assert.notEqual(D.decodeTime(32768, 4.0), 2.0)
})

test('time_scale_s is the denominator, and duration_s would give a different answer', () => {
  const block = bytes(0x02, ...u16le(16384), ...u16le(32768))
  const withScale = D.decodeSpans(block, 4.0)[0]
  const withDuration = D.decodeSpans(block, 8.0)[0]
  assert.ok(Math.abs(withScale.t0 - 1.0) < 1e-4)
  assert.ok(Math.abs(withDuration.t0 - 2.0) < 1e-4)
})

// -------------------------------------------------------------- null is not empty

test('a null block decodes to null and an empty block to the empty list', () => {
  for (const fn of [D.decodeSpans, D.decodeAsrWords, D.decodePiiMarks, D.decodeBranchLanes]) {
    assert.equal(fn(null, [], 1.0), null, `${fn.name} on null`)
  }
  assert.deepEqual(D.decodeSpans(bytes(), 1.0), [])
  assert.deepEqual(D.decodeAsrWords(bytes(), [], 1.0), [])
  assert.deepEqual(D.decodePiiMarks(bytes(), [], 1.0), [])
  assert.deepEqual(D.decodeBranchLanes(bytes(), [], 1.0), [])
})

test('a never-scanned recording decodes to null, never to a clean empty list', () => {
  assert.equal(D.decodePiiMarks(null, null, 30.0), null)
  assert.deepEqual(D.decodePiiMarks(bytes(), [], 30.0), [])
})

test('null traces stay null and do not become a floor of zeros', () => {
  assert.equal(D.decodeEnvelopeDbfs(null), null)
  assert.equal(D.decodeContinuity(null), null)
  assert.equal(D.decodeWaveMinmax(null, 0.5), null)
  assert.equal(D.decodeMatrix(null, 2), null)
})

// -------------------------------------------------------------- the quantisers

test('the value quantiser spans exactly its declared range', () => {
  assert.equal(D.decodeValue(0, -100, 0), -100)
  assert.equal(D.decodeValue(255, -100, 0), 0)
  assert.ok(Math.abs(D.decodeValue(128, -100, 0) + 49.8) < 0.1)
  assert.equal(D.decodeValue(0, 1.0, 4.5), 1.0)
  assert.equal(D.decodeValue(255, 1.0, 4.5), 4.5)
  assert.equal(D.decodeValue(0, -10, 30), -10)
  assert.equal(D.decodeValue(255, -10, 30), 30)
})

test('the declared ranges are the ones the schema names', () => {
  assert.deepEqual(D.ENVELOPE_DBFS_RANGE, [-100.0, 0.0])
  assert.deepEqual(D.CONTINUITY_RANGE, [0.0, 1.05])
  assert.deepEqual(D.SCORE_RANGE, [0.0, 1.0])
  assert.deepEqual(D.SQUIM_RANGES.stoi, [0.0, 1.0])
  assert.deepEqual(D.SQUIM_RANGES.pesq, [1.0, 4.5])
  assert.deepEqual(D.SQUIM_RANGES.si_sdr, [-10.0, 30.0])
})

test('pesq and si_sdr ranges are not interchangeable', () => {
  const block = bytes(...u16le(0), 255, 0, 0)
  const squim = D.decodeSpanSquim(block, 1)[0]
  assert.equal(squim.stoi, 1.0)
  assert.equal(squim.pesq, 1.0)
  assert.equal(squim.si_sdr, -10.0)
})

// -------------------------------------------------------------- fixed widths

test('the fixed traces are exactly 256 and the waveform exactly 512 bytes', () => {
  assert.equal(D.TRACE_POINTS, 256)
  assert.equal(D.decodeEnvelopeDbfs(new Uint8Array(256)).length, 256)
  assert.equal(D.decodeContinuity(new Uint8Array(256)).length, 256)
  assert.equal(D.decodeWaveMinmax(new Uint8Array(512), 1.0).points, 256)
  assert.throws(() => D.decodeEnvelopeDbfs(new Uint8Array(255)), /not the fixed 256/)
  assert.throws(() => D.decodeContinuity(new Uint8Array(257)), /not the fixed 256/)
  assert.throws(() => D.decodeWaveMinmax(new Uint8Array(511), 1.0), /not the fixed 512/)
})

test('the waveform is min then max per bucket, over plus and minus wave_peak', () => {
  const b = new Uint8Array(512)
  b[0] = 0; b[1] = 255
  b[2] = 128; b[3] = 128
  const w = D.decodeWaveMinmax(b, 0.5)
  assert.equal(w.min[0], -0.5)
  assert.equal(w.max[0], 0.5)
  assert.ok(Math.abs(w.min[1]) < 0.002 && Math.abs(w.max[1]) < 0.002)
})

test('a bucket covers i/256 to (i+1)/256 of time_scale_s', () => {
  assert.deepEqual(D.bucketExtent(0, 25.6), [0, 0.1])
  const last = D.bucketExtent(255, 25.6)
  assert.ok(Math.abs(last[1] - 25.6) < 1e-9)
})

test('a record block whose length is not a whole number of records throws', () => {
  assert.throws(() => D.decodeSpans(new Uint8Array(7), 1.0), /whole number of 5-byte spans/)
  assert.throws(() => D.decodeSpanLabels(new Uint8Array(5), [], 1), /whole number of 4-byte span_labels/)
  assert.throws(() => D.decodePiiMarks(new Uint8Array(5), [], 1.0), /whole number of 4-byte pii_marks/)
})

// -------------------------------------------------------------- enumerations

test('the enumerations are the ones recording_vectors.py declares, in order', () => {
  assert.deepEqual(D.SPAN_ROWS, ['E', 'C', 'A', 'S', 'G'])
  assert.deepEqual(D.CLASSIFIERS, ['yamnet', 'hear', 'ast'])
  assert.deepEqual(D.WORD_OUTCOMES, ['agreement', 'variant', 'insertion'])
  assert.deepEqual(D.LANES, ['AIRWAY', 'SPEECH', 'VOICE', 'REDACT'])
  assert.equal(D.UNKNOWN_CODE, 255)
})

test('255 is the documented sentinel and decodes to null, not to the first term', () => {
  const labels = D.decodeSpanLabels(bytes(...u16le(0), 255, 128), ['Speech'], 1)
  assert.equal(labels[0].classifier, null)
  assert.notEqual(labels[0].classifier, 'yamnet')
  const words = D.decodeAsrWords(bytes(...u16le(0), ...u16le(1), 255), ['hi'], 1.0)
  assert.equal(words[0].outcome, null)
})

test('an undocumented enumeration code throws rather than drawing the wrong row', () => {
  assert.throws(() => D.decodeSpans(bytes(5, 0, 0, 0, 0), 1.0), /span row code 5/)
  assert.throws(() => D.decodeBranchLanes(bytes(4, 0, 0, 0, 0), ['x'], 1.0), /branch lane code 4/)
  assert.throws(() => D.decodeSpanLabels(bytes(...u16le(0), 3, 0), ['x'], 1), /classifier code 3/)
})

// -------------------------------------------------------------- parallel columns

test('a parallel string column must have one element per record', () => {
  const block = bytes(...u16le(0), ...u16le(1), 0, ...u16le(1), ...u16le(2), 0)
  assert.equal(D.decodeAsrWords(block, ['a', 'b'], 1.0).length, 2)
  assert.throws(() => D.decodeAsrWords(block, ['a'], 1.0), /1 elements for 2 asr_words records/)
  assert.throws(() => D.decodeAsrWords(block, null, 1.0), /asr_word_text is null while asr_words is present/)
})

test('span_labels and span_squim index the filtered span list, and an overrun throws', () => {
  const spans = D.decodeSpans(bytes(0, 0, 0, ...u16le(100)), 1.0)
  assert.equal(spans.length, 1)
  assert.throws(() => D.decodeSpanLabels(bytes(...u16le(1), 0, 0), ['x'], spans.length), /indexes span 1 of 1/)
  assert.throws(() => D.decodeSpanSquim(bytes(...u16le(3), 0, 0, 0), spans.length), /indexes span 3 of 1/)
})

// -------------------------------------------------------------- the matrix

test('the flattened matrix rebuilds as n / width rows, row-major', () => {
  const m = D.decodeMatrix([1, 2, 3, 4, 5, 6], 3)
  assert.deepEqual(m.rows, [[1, 2, 3], [4, 5, 6]])
  assert.equal(m.width, 3)
  assert.throws(() => D.decodeMatrix([1, 2, 3], 2), /not a whole number of rows of 2/)
  assert.throws(() => D.decodeMatrix([1, 2], 0), /width 0 is not positive/)
})

test('nulls inside a vector survive as nulls, not as zeros', () => {
  const m = D.decodeMatrix([1, null, 3, null], 2)
  assert.deepEqual(m.rows, [[1, null], [3, null]])
})

// -------------------------------------------------------------- the whole row

test('a row with no time scale decodes to all-absent rather than to a zero axis', () => {
  const out = D.decodeRow({ time_scale_s: null, wave_minmax: new Uint8Array(512), wave_peak: 1 })
  assert.equal(out.timeScaleS, null)
  assert.equal(out.wave, null)
  assert.equal(out.spans, null)
})

test('a whole row decodes with its blocks cross-indexed', () => {
  const row = {
    time_scale_s: 10.0,
    wave_peak: 0.25,
    wave_minmax: new Uint8Array(512).fill(128),
    env_dbfs: new Uint8Array(256).fill(200),
    continuity: new Uint8Array(256).fill(255),
    spans: new Uint8Array([2, ...u16le(0), ...u16le(32768), 0, ...u16le(100), ...u16le(200)]),
    span_labels: new Uint8Array([...u16le(0), 1, 200]),
    span_label_name: ['Speech'],
    span_squim: new Uint8Array([...u16le(1), 255, 0, 255]),
    asr_words: new Uint8Array([...u16le(0), ...u16le(6553), 0]),
    asr_word_text: ['caterpillar'],
    pii_marks: new Uint8Array([...u16le(0), ...u16le(6553)]),
    pii_category: ['PERSON'],
    branch_lanes: new Uint8Array([3, ...u16le(0), ...u16le(65535)]),
    branch_lane_role: ['PERSON'],
    m_ddk_cv_instrument_reading: [1, 2, 3, 4],
    m_ddk_cv_instrument_reading_width: 2,
  }
  const out = D.decodeRow(row)
  assert.equal(out.spans.length, 2)
  assert.equal(out.spans[0].row, 'A')
  assert.equal(out.spanLabels[0].classifier, 'hear')
  assert.ok(Math.abs(out.spanLabels[0].score - 200 / 255) < 1e-9)
  assert.equal(out.spanSquim[0].stoi, 1.0)
  assert.equal(out.spanSquim[0].pesq, 1.0)
  assert.equal(out.spanSquim[0].si_sdr, 30.0)
  assert.equal(out.asrWords[0].text, 'caterpillar')
  assert.equal(out.asrWords[0].outcome, 'agreement')
  assert.ok(Math.abs(out.asrWords[0].t1 - 1.0) < 1e-3)
  assert.equal(out.piiMarks[0].category, 'PERSON')
  assert.equal(out.branchLanes[0].lane, 'REDACT')
  assert.ok(Math.abs(out.branchLanes[0].t1 - 10.0) < 1e-9)
  assert.deepEqual(out.matrix.rows, [[1, 2], [3, 4]])
  assert.equal(out.continuity[0], 1.05)
  assert.equal(out.wave.max[0], out.wave.min[0])
})
