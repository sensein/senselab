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
