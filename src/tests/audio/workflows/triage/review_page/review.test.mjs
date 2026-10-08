// The review page's pure parts: decoding the index, the columns it registers, the axes per evidence
// group, search, the spectrogram unpacking, and the reviewer's export round trip. Every row here is
// synthetic.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const triage = join(here, '..', '..', '..', '..', '..', 'senselab', 'audio', 'workflows', 'triage')
const require = createRequire(import.meta.url)
const A = require(join(triage, 'viewer', 'axes.js'))
globalThis.SchemaAxes = A
const F = require(join(triage, 'viewer', 'facets.js'))
globalThis.SchemaFacets = F
const R = require(join(triage, 'review_page', 'review.js'))

const dict = (values, codes) => ({ values, codes })
const INDEX = {
  build: { id: 'abc123', n: 3, shard_size: 500, shard_dir: 'data', audio_base: '../', spec: { frames: 64, bands: 16, levels: 4 } },
  cols: {
    stem: ['sub-aaaa_ses-1_task-breath', 'sub-bbbb_ses-2_task-harvard', 'sub-cccc_ses-1_task-mpt'],
    participant: dict(['sub-aaaa', 'sub-bbbb', 'sub-cccc'], [0, 1, 2]),
    session: dict(['ses-1', 'ses-2'], [0, 1, 0]),
    task: dict(['breath', 'harvard', 'mpt'], [0, 1, 2]),
    family: dict(['respiration-and-cough-breath', 'harvard-sentences', 'maximum-phonation-time'], [0, 1, 2]),
    branch: dict(['AIRWAY', 'SPEECH', 'VOICE'], [0, 1, 2]),
    verdict: dict(['pass', 'review', 'discard'], [0, 1, 2]),
    release: dict(['as_is', 'withheld'], [0, 1, -1]),
    reason: dict(['weak_events', 'no_task_captured'], [-1, 0, 1]),
    run_status: dict(['complete'], [0, 0, 0]),
    reasons: dict(['weak_events', 'no_task_captured'], [[], [0], [1]]),
    annotations: dict(['task_mismatch'], [[0], [], []]),
    items: dict(['breath_event_db_over_local', 'phonation_found'], [[0], [], [1]]),
    decisive: dict(['breath_event_db_over_local', 'phonation_found'], [[0], [], [1]]),
    extent_duration_s: [3.0, null, null],
    duration_s: [4.0, 9.0, 1.2],
    text: [null, 'the birch canoe slid', null],
  },
  evidence: {
    breath_event_db_over_local: { group: 'breath', unit: 'dB', kind: 'numeric', rows: [0], values: [20.5] },
    phonation_found: { group: 'voice', unit: null, kind: 'categorical', rows: [2], values: ['false'] },
  },
}

test('the index decodes to one row per recording, with evidence under ev: keys', () => {
  const rows = R.decodeRows(INDEX)
  assert.equal(rows.length, 3)
  assert.equal(rows[0]['ev:breath_event_db_over_local'], 20.5)
  assert.equal(rows[1].release, 'withheld')
  assert.equal(rows[2].release, null)
  assert.deepEqual(rows[1].reasons, ['weak_events'])
  assert.equal(rows[0].declared_family, 'respiration-and-cough-breath')
})

test('the page registers its columns, and the facets and axes can read them', () => {
  A.register(R.columnSpecs(INDEX))
  F.refresh()
  assert.equal(A.BY_NAME['ev:breath_event_db_over_local'].kind, 'numeric')
  assert.equal(A.BY_NAME['ev:phonation_found'].kind, 'categorical')
  assert.equal(F.BY_NAME.reason.mode, 'scalar')
  assert.equal(F.BY_NAME.evidence_items.mode, 'set')
  const rows = R.decodeRows(INDEX)
  const s = A.summarise('ev:breath_event_db_over_local', rows)
  assert.equal(s.present, 1)
  assert.equal(s.absent, 2)
})

test('axes are the decision columns, or verdict and reason beside one group of evidence', () => {
  assert.ok(R.axesFor('decision', INDEX).includes('verdict'))
  assert.deepEqual(R.axesFor('breath', INDEX), ['verdict', 'reason', 'ev:breath_event_db_over_local'])
  assert.deepEqual(Object.keys(R.evidenceGroups(INDEX)).sort(), ['breath', 'voice'])
})

test('search reads the stem and a speech transcript; the who box reads participant and session', () => {
  const rows = R.decodeRows(INDEX)
  assert.equal(R.searchMask(rows, '', ''), null)
  assert.deepEqual(Array.from(R.searchMask(rows, 'canoe', '')), [0, 1, 0])
  assert.deepEqual(Array.from(R.searchMask(rows, 'task-', 'ses-1')), [1, 0, 1])
})

test('the spectrogram unpacks two bits a cell, time-major', () => {
  const bytes = new Uint8Array(256)
  bytes[0] = 0b11100100
  const text = Buffer.from(bytes).toString('base64')
  const levels = R.unpackSpec(text, 64, 16)
  assert.deepEqual(Array.from(levels.slice(0, 5)), [0, 1, 2, 3, 0])
})

test('an export carries the owner label columns and imports back to the same decisions', () => {
  const rows = R.decodeRows(INDEX)
  const byStem = Object.fromEntries(rows.map((r) => [r.stem, r]))
  const decisions = { [rows[1].stem]: { verdict: 'discard', note: 'only the room', at: '2026-10-07T00:00:00Z' } }
  const payload = R.exportPayload(byStem, decisions, 'abc123', '2026-10-07T01:00:00Z')
  assert.equal(payload.schema, R.EXPORT_SCHEMA)
  const e = payload.entries[0]
  assert.deepEqual(
    Object.keys(e).slice(0, 8),
    ['listen_set', 'stem', 'family', 'instructed', 'duration_s', 'owner_label', 'owner_note', 'pipeline_at_listen'],
  )
  assert.equal(e.listen_set, 'triage_review_abc123')
  assert.equal(e.owner_label, 'reviewer_discard')
  assert.equal(e.pipeline_at_listen, 'review/withheld (weak_events)')
  assert.deepEqual(R.importDecisions(payload), decisions)
})

test('a transcript names its model by id, with the pipeline source where it differs', () => {
  assert.equal(R.modelLabel({ source: 'whisper', model_id: 'openai/whisper-large-v3-turbo' }), 'openai/whisper-large-v3-turbo (whisper)')
  assert.equal(R.modelLabel({ source: 'canary', model_id: null }), 'canary')
  assert.equal(R.modelLabel(undefined), 'unknown model')
})

test('a consensus word is shaded by its agreement band and lists every model reading', () => {
  assert.equal(R.agreementBand(1), 'ag-all')
  assert.equal(R.agreementBand(0.67), 'ag-most')
  assert.equal(R.agreementBand(0.5), 'ag-some')
  assert.equal(R.agreementBand(0.25), 'ag-few')
  assert.equal(R.agreementBand(null), null)
  const sp = { models: [{ source: 'a', model_id: 'org/a' }, { source: 'b', model_id: 'org/b' }] }
  const text = R.alternatives(sp, [0.5, 'variant', ['birch', null]])
  assert.equal(text, 'variant, agreement 50%\norg/a (a): birch\norg/b (b): (no word)')
})

test('a stream under the corpus root goes through audio_base, one outside it through the source route', () => {
  const build = { audio_base: '../', source_base: '/_source' }
  assert.equal(R.streamUrl('sub-a/ses-1/run/streams/plain.flac', build), '../sub-a/ses-1/run/streams/plain.flac')
  assert.equal(R.streamUrl('/orcd/data/b2ai/sub a.wav', build), '/_source/orcd/data/b2ai/sub%20a.wav')
})

test('the time window zooms about a point and pans, staying inside the recording', () => {
  const whole = { t0: 0, t1: 4 }
  const z = R.zoomWindow(whole, 4, 0.5, 1)
  assert.ok(Math.abs(z.t1 - z.t0 - 2) < 1e-9)
  assert.ok(Math.abs(z.t0 - 0.5) < 1e-9)
  assert.deepEqual(R.zoomWindow(z, 4, 4, 2), { t0: 0, t1: 4 })
  assert.ok(R.zoomWindow(whole, 4, 0.001, 2).t1 - R.zoomWindow(whole, 4, 0.001, 2).t0 >= 0.25 - 1e-9)
  assert.deepEqual(R.panWindow({ t0: 1, t1: 2 }, 4, 5), { t0: 3, t1: 4 })
  assert.deepEqual(R.panWindow({ t0: 1, t1: 2 }, 4, -5), { t0: 0, t1: 1 })
})

test('a span is placed as fractions of the window, and dropped outside it', () => {
  assert.deepEqual(R.placeSpan(1, 2, { t0: 0, t1: 4 }), { left: 0.25, width: 0.25 })
  assert.equal(R.placeSpan(5, 6, { t0: 0, t1: 4 }), null)
  assert.equal(R.placeSpan(null, 1, { t0: 0, t1: 4 }), null)
  assert.deepEqual(R.ticks({ t0: 0, t1: 5 }, 5), [0, 1, 2, 3, 4, 5])
  assert.deepEqual(R.ticks({ t0: 1.05, t1: 1.5 }, 5), [1.1, 1.2, 1.3, 1.4, 1.5])
})

// The shape of the r18 buttercup case: two models, every word of the second losing a tie or absent.
const SP = {
  models: [
    { source: 'asr_crisperwhisper', model_id: 'nyralabs/CrisperWhisper2.0_turbo' },
    { source: 'asr_qwen', model_id: 'Qwen/Qwen3-ASR-1.7B' },
  ],
  own: [
    { source: 'asr_crisperwhisper', model_id: 'nyralabs/CrisperWhisper2.0_turbo', text: 'What are', words: [[0.02, 0.53, 'What'], [0.53, 0.63, 'are']] },
    { source: 'asr_qwen', model_id: 'Qwen/Qwen3-ASR-1.7B', text: 'Barakat', words: [[0.0, 0.96, 'Barakat'], [null, null, 'untimed']] },
  ],
  words: [
    [0.5, 'variant', ['What', 'Barakat'], 0.01, 0.63, 'What'],
    [0.5, 'insertion', ['are', null], 0.53, 0.63, 'are'],
  ],
}

test('each model gets a lane of its own timed words, untimed ones left to the text list', () => {
  const lanes = R.modelLanes(SP)
  assert.deepEqual(lanes.map((l) => l.label), [
    'nyralabs/CrisperWhisper2.0_turbo (asr_crisperwhisper)', 'Qwen/Qwen3-ASR-1.7B (asr_qwen)',
  ])
  assert.deepEqual(lanes[1].tokens, [[0.0, 0.96, 'Barakat']])
})

test('a reading other than the consensus surface is kept, naming the one model that gave it', () => {
  assert.equal(R.tokenKey('Time?'), 'time')
  assert.deepEqual(R.otherReadings(SP, SP.words[0], 'What'), [{ text: 'Barakat', models: ['Qwen/Qwen3-ASR-1.7B (asr_qwen)'] }])
  assert.deepEqual(R.otherReadings(SP, [1, 'agreement', ['what', 'What?'], 0, 1, 'what'], 'what'), [])
  assert.deepEqual(R.readersOf(SP, SP.words[1]), ['nyralabs/CrisperWhisper2.0_turbo (asr_crisperwhisper)'])
})

test('the page plays the recording and enhanced streams, never the task cuts', () => {
  const streams = { recording: '/a.wav', enhanced: 'e.flac', plain: 'p.flac', redacted: 'r.flac', released: 'x.flac',
    task_plain: 'tp.flac', task_enhanced: 'te.flac' }
  assert.deepEqual(R.audioTracks({ streams }).map((t) => t.name), ['recording', 'enhanced', 'released'])
  assert.deepEqual(R.audioTracks({ streams, speech: {} }).map((t) => [t.name, t.timeline]),
    [['recording', true], ['enhanced', true], ['redacted', true], ['released', false]])
})

// A stand-in <audio>: `seekable` says whether setting currentTime lands there (a server answering byte
// ranges) or falls back to 0 (one that does not); `seeked` fires after the assignment, as in a browser.
function fakePlayer({ seekable = true, ready = true } = {}) {
  const t = new EventTarget()
  let at = 0
  Object.assign(t, {
    readyState: ready ? 1 : 0, paused: true, preload: 'none', plays: 0,
    load() { queueMicrotask(() => { t.readyState = 1; t.dispatchEvent(new Event('loadedmetadata')) }) },
    play() { t.paused = false; t.plays += 1; t.dispatchEvent(new Event('play')); return Promise.resolve() },
    pause() { t.paused = true; t.dispatchEvent(new Event('pause')) },
  })
  Object.defineProperty(t, 'currentTime', {
    get: () => at,
    set: (v) => { at = seekable ? v : 0; queueMicrotask(() => t.dispatchEvent(new Event('seeked'))) },
  })
  return t
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 20))

test('play task span waits for metadata, seeks to the extent start, waits for seeked, then plays', async () => {
  const tl = R.timelineOf(40, [6.85, 29.73])
  const a = fakePlayer({ ready: false })
  tl.players.push(a)
  R.playSpan(tl, a, () => assert.fail('a seekable player must not report a failed seek'), 1000)
  assert.equal(a.plays, 0)
  assert.equal(a.preload, 'auto')
  await settle()
  assert.equal(a.currentTime, 6.85)
  assert.equal(a.plays, 1)
  assert.equal(tl.spanEnd, 29.73)
  a.currentTime = 29.8
  R.stopAtSpanEnd(tl)
  assert.equal(a.paused, true)
  assert.equal(tl.spanEnd, null)
})

test('play task span does not play from 0 where the server cannot seek, and says why', async () => {
  const tl = R.timelineOf(40, [6.85, 29.73])
  const a = fakePlayer({ seekable: false })
  tl.players.push(a)
  const failed = []
  R.playSpan(tl, a, (landed, start) => failed.push([landed, start]), 1000)
  await settle()
  assert.deepEqual(failed, [[0, 6.85]])
  assert.equal(a.plays, 0)
  assert.equal(tl.spanEnd, null)
  assert.match(R.CANNOT_SEEK, /triage_review_serve\.py/)
})

test('play task span gives up waiting for seeked and still checks where the player landed', async () => {
  const tl = R.timelineOf(40, [6.85, 29.73])
  const a = fakePlayer({ seekable: false })
  a.addEventListener = function (type, fn, opts) { if (type !== 'seeked') EventTarget.prototype.addEventListener.call(this, type, fn, opts) }
  tl.players.push(a)
  const failed = []
  R.playSpan(tl, a, (landed) => failed.push(landed), 5)
  await settle()
  assert.deepEqual(failed, [0])
  assert.equal(a.plays, 0)
})
