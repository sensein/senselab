// The review page's pure parts: decoding the index, the columns it registers, the axes per evidence
// group, search, the spectrogram unpacking, the transcript alignment and lanes, and the reviewer's
// export round trip. Every row here is synthetic.

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

test('a consensus word lists every model reading', () => {
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

test('the page plays the recording and enhanced streams, never the task cuts', () => {
  const streams = { recording: '/a.wav', enhanced: 'e.flac', plain: 'p.flac', redacted: 'r.flac', released: 'x.flac',
    task_plain: 'tp.flac', task_enhanced: 'te.flac' }
  assert.deepEqual(R.audioTracks({ streams }).map((t) => t.name), ['recording', 'enhanced', 'released'])
  assert.deepEqual(R.audioTracks({ streams, speech: {}, release: 'redacted' }).map((t) => [t.name, t.timeline]),
    [['recording', true], ['enhanced', true], ['redacted', true], ['released', false]])
})

test('the redacted stream is a track only where the release is the redacted copy', () => {
  const streams = { recording: '/a.wav', enhanced: 'e.flac', redacted: 'r.flac' }
  const shown = (release) => R.audioTracks({ streams, speech: {}, release })
    .filter((t) => t.name === 'redacted').map((t) => [t.timeline, t.collapsed])
  assert.deepEqual(shown('redacted'), [[true, false]])
  for (const release of ['as_is', 'withheld', null, undefined]) assert.deepEqual(shown(release), [[false, true]])
  assert.equal(R.NOT_RELEASED_REDACTION, 'redaction considered, not released')
  assert.ok(R.redactionReleased('redacted'))
  assert.ok(!R.redactionReleased('as_is'))
})

test('a PII mark on a recording not released redacted says it was detected and not applied, and why', () => {
  const ground = 'every word REDACT masked is the task\'s own content'
  assert.equal(R.notAppliedTitle('as_is', ground), 'detected, not applied — release as_is: ' + ground)
  assert.equal(R.notAppliedTitle(null, null), 'detected, not applied — release not assessed')
  const marks = [{ dataset: {} }, { dataset: {} }]
  R.markNotApplied({ querySelectorAll: (q) => (q === 'mark.pii' ? marks : []) }, 'as_is', ground)
  for (const m of marks) {
    assert.equal(m.title, R.notAppliedTitle('as_is', ground))
    assert.equal(m.dataset.applied, '0')
  }
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

// The buttercup case's shape: one model repeating a phrase, the other a single word, so most columns
// are one model's insertions and the rest are variants. One word sits inside a PII mark.
const PHRASE = ['What', 'is', 'it', 'like?']
const BUTTERCUP_WORDS = [
  [0.5, 'variant', ['What', 'Barakat'], 0.01, 0.63, 'What'],
  [0.5, 'insertion', ['are', null], 0.53, 0.63, 'are'],
  [0.5, 'insertion', ['the', null], 0.63, 0.8, 'the'],
  [0.5, 'variant', ['time?', 'barakat'], 0.88, 1.2, 'time?'],
]
for (let k = 0; k < 18; k++) {
  const variant = k % 4 === 0 && k < 16
  BUTTERCUP_WORDS.push([0.5, variant ? 'variant' : 'insertion', [PHRASE[k % 4], variant ? 'barakat' : null],
    1.06 + 0.15 * k, 1.2 + 0.15 * k, PHRASE[k % 4]])
}
const word = (i, w) => `<span class="w" data-i="${i}">${w[5]}</span>`
const BUTTERCUP = {
  models: SP.models,
  own: SP.own,
  words: BUTTERCUP_WORDS,
  html: BUTTERCUP_WORDS.map((w, i) => i === 3
    ? `<mark class="pii u-green" data-k="k3" data-c="PERSON"><span class="cat">PERSON</span>${word(i, w)}</mark>`
    : word(i, w)).join(' '),
}

test('each model gets a short initial from its family, distinct across models', () => {
  assert.deepEqual(R.modelInitials(SP.models), ['W', 'Q'])
  assert.deepEqual(R.modelInitials([{ source: 'asr_canary' }, { source: 'asr_crisp', model_id: 'x/canary-1b' }]), ['C', 'C2'])
})

test('a word without a stored outcome takes one from its readings', () => {
  assert.equal(R.outcomeOf([1, null, ['a', 'A'], 0, 1, 'a']), 'agreement')
  assert.equal(R.outcomeOf([0.5, null, ['a', 'b'], 0, 1, 'a']), 'variant')
  assert.equal(R.outcomeOf([0.5, null, ['a', null], 0, 1, 'a']), 'insertion')
  assert.equal(R.outcomeOf([0.5, 'variant', ['a', 'a'], 0, 1, 'a']), 'variant')
})

test('the buttercup alignment has 22 columns: 6 variants and 16 of one model\'s insertions', () => {
  const cols = R.alignmentColumns(BUTTERCUP)
  assert.equal(cols.length, 22)
  assert.deepEqual(R.alignmentSummary(BUTTERCUP),
    { columns: 22, agreement: 0, variant: 6, insertion: 16, insertionsBy: { W: 16, Q: 0 } })
  assert.deepEqual(cols[0].rows.map((r) => [r.initial, r.text, r.chosen]), [['W', 'What', true], ['Q', 'Barakat', false]])
  assert.deepEqual(cols[1].rows.map((r) => [r.initial, r.text]), [['W', 'are'], ['Q', null]])
})

test('the alignment replaces each word with its column and keeps the PII mark around its word', () => {
  const html = R.alignmentHtml(BUTTERCUP)
  assert.equal((html.match(/class="al-col /g) || []).length, 22)
  assert.equal((html.match(/al-col al-variant/g) || []).length, 6)
  assert.equal((html.match(/al-col al-insertion/g) || []).length, 16)
  assert.equal((html.match(/class="al-r al-gap"/g) || []).length, 16)
  assert.ok(!html.includes('class="w"'))
  assert.match(html, /<mark class="pii u-green"[^>]*><span class="cat">PERSON<\/span><span class="al-col al-variant" data-i="3" data-t="0.88" data-e="1.2"/)
  assert.match(html, /<span class="al-t">0\.88<\/span><span class="al-r chosen"><b class="al-m"[^>]*>W<\/b>time\?<\/span><span class="al-r"><b class="al-m"[^>]*>Q<\/b>barakat<\/span>/)
})

test('an agreement column shows its word once, keeping its bracket markup; readings are escaped', () => {
  const sp = {
    models: SP.models,
    words: [[1, 'agreement', ['[laughs]', '[laughs]'], 0, 0.4, '[laughs]'], [0.5, 'variant', ['a<b', 'c'], 0.4, 0.8, 'c']],
    html: '<span class="w" data-i="0"><span class="bracket">[laughs]</span></span> <span class="w" data-i="1">c</span>',
  }
  const html = R.alignmentHtml(sp)
  assert.match(html, /al-agreement[^>]*><span class="al-t">0\.00<\/span><span class="al-r"><span class="bracket">\[laughs\]<\/span><\/span><\/span>/)
  assert.ok(html.includes('a&lt;b'))
  assert.ok(!html.includes('a<b'))
})

test('overlapping lane tokens go to separate rows; a label shows only where it fits its box', () => {
  assert.deepEqual(R.packRows([[0, 1], [0.5, 1.5], [1, 2], [1.2, 1.3]]), [0, 1, 0, 2])
  assert.deepEqual(R.packRows([[0.88, 1.2], [1.06, 1.2], [1.2, 1.33]]), [0, 1, 0])
  assert.equal(R.labelFits(40, 'What', 7), true)
  assert.equal(R.labelFits(20, 'What', 7), false)
  assert.equal(R.labelFits(0, '', 7), true)
})
