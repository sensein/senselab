// The theme: every canvas colour is a token both themes define, and the text stays readable in both.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { readFileSync } from 'node:fs'

const here = dirname(fileURLToPath(import.meta.url))
const viewer = join(here, '..', '..', '..', '..', '..', 'senselab', 'audio', 'workflows', 'triage', 'viewer')
const require = createRequire(import.meta.url)
const Theme = require(join(viewer, 'theme.js'))
const css = readFileSync(join(viewer, 'styles.css'), 'utf8')

/** The custom properties one rule block defines, by its selector as written. */
function block (selector) {
  const at = css.indexOf(selector + ' {')
  assert.ok(at >= 0, `no block ${selector}`)
  const body = css.slice(at + selector.length + 2, css.indexOf('}', at))
  const out = {}
  for (const m of body.matchAll(/--([\w-]+):\s*([^;]+);/g)) out[m[1]] = m[2].trim()
  return out
}

const LIGHT = block(':root')
const LIGHT_ALL = Object.assign({}, block(':root'))
// The first `:root {` holds only color-scheme and --mono; the tokens are in the second.
const second = css.indexOf(':root {', css.indexOf(':root {') + 1)
const lightBody = css.slice(second + 7, css.indexOf('}', second))
for (const m of lightBody.matchAll(/--([\w-]+):\s*([^;]+);/g)) LIGHT_ALL[m[1]] = m[2].trim()
const DARK_SYSTEM = block('  :root:not([data-theme="light"])')
const DARK_CHOSEN = block(':root[data-theme="dark"]')

test('every canvas token is defined by the light theme and by both dark blocks', () => {
  for (const name of Theme.TOKENS) {
    for (const [label, tokens] of [['light', LIGHT_ALL], ['dark (system)', DARK_SYSTEM], ['dark (chosen)', DARK_CHOSEN]]) {
      assert.ok(`c-${name}` in tokens, `--c-${name} missing from the ${label} theme`)
    }
  }
  assert.ok(LIGHT)
})

test('the two dark blocks are the same theme, and the fallback is it', () => {
  assert.deepEqual(DARK_SYSTEM, DARK_CHOSEN)
  for (const name of Theme.TOKENS) assert.equal(Theme.FALLBACK[name], DARK_CHOSEN[`c-${name}`], name)
})

test('no --c- token is defined that the canvases do not read', () => {
  const defined = Object.keys(DARK_CHOSEN).filter(k => k.startsWith('c-')).map(k => k.slice(2))
  assert.deepEqual(defined.sort(), Theme.TOKENS.slice().sort())
})

test('the canvases and the page script carry no colour literal of their own', () => {
  for (const file of ['corpus.js', 'recording.js', 'app.js']) {
    const text = readFileSync(join(viewer, file), 'utf8')
    assert.deepEqual(text.match(/'#[0-9a-fA-F]{3,8}'|rgba?\(/g), null, file)
  }
})

function luminance (hex) {
  const n = hex.replace('#', '')
  const [r, g, b] = [0, 2, 4].map(i => parseInt(n.slice(i, i + 2), 16) / 255)
    .map(c => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4))
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}
function contrast (a, b) {
  const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p)
  return (x + 0.05) / (y + 0.05)
}

test('text reads at 4.5:1 and marks at 3:1 against the ground, in both themes', () => {
  for (const [label, t] of [['light', LIGHT_ALL], ['dark', DARK_CHOSEN]]) {
    for (const ink of ['ink', 'ink-2']) {
      for (const ground of ['ground', 'panel', 'panel-2']) {
        assert.ok(contrast(t[ink], t[ground]) >= 4.5, `${label} ${ink} on ${ground}: ${contrast(t[ink], t[ground]).toFixed(2)}`)
      }
    }
    const marks = ['c-accent', 'c-good', 'c-bad', 'c-warn', 'c-teal', 'c-orange', 'c-violet', 'c-ink-3']
      .concat(Array.from({ length: Theme.SERIES_N }, (_, i) => `c-series-${i}`))
    for (const mark of marks) {
      assert.ok(contrast(t[mark], t['c-ground']) >= 3, `${label} ${mark}: ${contrast(t[mark], t['c-ground']).toFixed(2)}`)
    }
  }
})

test('the toggle cycles system, light, dark', () => {
  assert.deepEqual([Theme.next('system'), Theme.next('light'), Theme.next('dark')], ['light', 'dark', 'system'])
  assert.equal(Theme.next('unknown'), 'light')
})

test('with no document the colour is the dark fallback, and no storage reads as system', () => {
  assert.equal(Theme.color('accent'), Theme.FALLBACK.accent)
  assert.equal(Theme.series().length, Theme.SERIES_N)
  assert.equal(Theme.stored(), 'system')
})
