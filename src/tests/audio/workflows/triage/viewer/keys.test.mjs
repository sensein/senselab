// Stepping through the selection from the keyboard.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const viewer = join(here, '..', '..', '..', '..', '..', 'senselab', 'audio', 'workflows', 'triage', 'viewer')
const require = createRequire(import.meta.url)
const K = require(join(viewer, 'keys.js'))

const selected = Uint8Array.from([0, 1, 0, 1, 1, 0, 1])
// selected rows, in order: 1, 3, 4, 6

test('next and previous move through the selected rows only, in row order', () => {
  assert.equal(K.step(selected, 1, 'next').index, 3)
  assert.equal(K.step(selected, 3, 'next').index, 4)
  assert.equal(K.step(selected, 4, 'prev').index, 3)
  assert.deepEqual(K.step(selected, 4, 'next'), { index: 6, position: 4, total: 4, clamped: false })
})

test('Home and End go to the ends of the selection', () => {
  assert.equal(K.step(selected, 4, 'first').index, 1)
  assert.equal(K.step(selected, 1, 'last').index, 6)
})

test('a step past an end stays there and says so', () => {
  const end = K.step(selected, 6, 'next')
  assert.deepEqual(end, { index: 6, position: 4, total: 4, clamped: true })
  assert.equal(K.describe(end), '4 of 4 · last in the selection')
  const start = K.step(selected, 1, 'prev')
  assert.equal(K.describe(start), '1 of 4 · first in the selection')
})

test('from nothing open, or a row outside the selection, it enters the selection nearby', () => {
  assert.equal(K.step(selected, -1, 'next').index, 1)
  assert.equal(K.step(selected, -1, 'prev').index, 6)
  assert.equal(K.step(selected, 2, 'next').index, 3)
  assert.equal(K.step(selected, 5, 'prev').index, 4)
  assert.equal(K.step(selected, 0, 'prev').index, 1)
})

test('an empty selection moves nowhere', () => {
  const none = K.step(new Uint8Array(3), 0, 'next')
  assert.equal(none.index, -1)
  assert.equal(K.describe(none), 'nothing selected')
})

test('keys typed into a field, or with a modifier, are not steps', () => {
  assert.equal(K.actionFor({ key: 'j', target: { tagName: 'INPUT' } }), null)
  assert.equal(K.actionFor({ key: 'j', target: { tagName: 'SELECT' } }), null)
  assert.equal(K.actionFor({ key: 'j', target: { tagName: 'DIV', isContentEditable: true } }), null)
  assert.equal(K.actionFor({ key: 'j', ctrlKey: true, target: { tagName: 'BODY' } }), null)
  assert.equal(K.actionFor({ key: 'j', target: { tagName: 'BODY' } }), 'next')
  assert.equal(K.actionFor({ key: 'ArrowUp', target: { tagName: 'BODY' } }), 'prev')
  assert.equal(K.actionFor({ key: 'End', target: { tagName: 'CANVAS' } }), 'last')
  assert.equal(K.actionFor({ key: 'x', target: { tagName: 'BODY' } }), null)
})

test('a clicked row reports where it stands', () => {
  assert.equal(K.describe(K.locate(selected, 4)), '3 of 4')
  assert.match(K.describe(K.locate(selected, 2)), /not in the selection/)
})
