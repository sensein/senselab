// End-to-end checks for the built viewer page, driven headless against a parquet the shipping
// writer produced. Run with `npx playwright test`; see specs/20260922-compact-recording-vectors.
//
// The page is opened over file:// with no network of any kind, which is how the owner opens it.
// The fixture parquet is synthetic: no corpus data reaches this directory.

import { test, expect } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { existsSync } from 'node:fs'

const here = dirname(fileURLToPath(import.meta.url))
const repo = resolve(here, '../../../../../..')
const PAGE = resolve(repo, 'src/senselab/audio/workflows/triage/viewer/recording_vectors_viewer.html')
const PARQUET = resolve(repo, 'artifacts/viewer_e2e/recording_vectors.parquet')
const SHOTS = resolve(repo, 'artifacts/viewer_e2e')

const GATE_AXIS = 'gate_train_min_s'
const GATE_SLOT = 3

/** Open the page and load the fixture, leaving the corpus painted. */
async function open (page) {
  expect(existsSync(PARQUET), `fixture missing: ${PARQUET} — run make_fixture first`).toBe(true)
  const problems = []
  page.on('pageerror', e => problems.push(String(e)))
  page.on('requestfailed', r => problems.push('requestfailed ' + r.url()))
  await page.goto('file://' + PAGE)
  await page.setInputFiles('#picker', PARQUET)
  await expect(page.locator('#main')).toBeVisible({ timeout: 30000 })
  await expect(page.locator('#selcount')).toContainText('recordings drawn')
  return problems
}

test('the page loads the parquet from disk and reports what it read', async ({ page }) => {
  const problems = await open(page)
  await expect(page.locator('#status')).not.toHaveClass(/bad/)
  const info = await page.locator('#file-info').innerText()
  expect(info).toContain('240 rows')
  expect(info).toContain('schema_version 3')
  const rows = await page.evaluate(() => window.__viewerState.rows.length)
  expect(rows).toBe(240)
  expect(problems, 'the page raised no error and fetched nothing').toEqual([])
  await page.screenshot({ path: resolve(SHOTS, 'loaded.png'), fullPage: false })
})

test('the default axes are the ten the spec derives, and each one renders', async ({ page }) => {
  await open(page)
  const expected = await page.evaluate(() => SchemaAxes.DEFAULT_AXES)
  expect(expected).toHaveLength(10)
  const onRail = await page.$$eval(
    '#axis-rail .axis-cell:not(.add) select.axis-select',
    els => els.map(e => e.value)
  )
  expect(onRail).toEqual(expected)
  // every default actually placed at least one recording, i.e. the axis is not wholly absent
  const present = await page.evaluate(
    n => Array.from({ length: n }, (_, i) => window.__viewerState.view.countPresent(i).present),
    expected.length
  )
  present.forEach((p, i) => expect(p, `${expected[i]} placed nothing`).toBeGreaterThan(0))
})

test('a gate reading takes an axis and its bound is drawn as a reference line', async ({ page }) => {
  await open(page)
  const select = page.locator('#axis-rail .axis-cell:not(.add) select.axis-select').nth(GATE_SLOT)
  await select.selectOption(GATE_AXIS)
  await expect(select).toHaveValue(GATE_AXIS)

  const drawn = await page.evaluate(slot => {
    const view = window.__viewerState.view
    const summary = view.summaries[slot]
    return {
      axis: view.axes[slot],
      kind: summary.kind,
      bounds: view.boundsFor(slot),
      op: summary.col.op,
      boundColumn: summary.col.boundColumn,
      paintsBounds: typeof view.paintBounds === 'function'
    }
  }, GATE_SLOT)

  expect(drawn.axis).toBe(GATE_AXIS)
  expect(drawn.kind).toBe('numeric')
  expect(drawn.boundColumn).toBe('gate_train_min_s_bound')
  expect(drawn.op).toBe('at_least')
  expect(drawn.paintsBounds).toBe(true)
  // one bound governs the whole fixture, and it is the configured 1.0 s
  expect(drawn.bounds).toHaveLength(1)
  expect(drawn.bounds[0].bound).toBe(1)
  expect(drawn.bounds[0].n).toBe(240)

  // the bound sits inside the band, so the reference line is on screen rather than clipped away
  const frac = await page.evaluate(
    slot => SchemaAxes.position(window.__viewerState.view.summaries[slot], 1),
    GATE_SLOT
  )
  expect(frac).not.toBeNull()
  expect(frac).toBeGreaterThan(0)
  expect(frac).toBeLessThan(1)
  await page.screenshot({ path: resolve(SHOTS, 'gate-axis-with-bound.png'), fullPage: false })
})

test('a genuine zero reading is placed on the band and only a null is absent', async ({ page }) => {
  await open(page)
  await page.locator('#axis-rail .axis-cell:not(.add) select.axis-select').nth(GATE_SLOT)
    .selectOption(GATE_AXIS)

  const rail = await page.evaluate(({ slot, gate }) => {
    const state = window.__viewerState
    const view = state.view
    const zeros = state.rows.filter(r => r[gate] === 0).length
    const nulls = state.rows.filter(r => r[gate] == null).length
    const counts = view.countPresent(slot)
    // every exact zero must own a real vertex, never the absent rail
    const zerosPlaced = state.rows
      .filter(r => r[gate] === 0)
      .every(r => {
        const v = view.vertices(r)[slot]
        return v.absent === false && v.y != null && Number.isFinite(v.y)
      })
    // and the null rows must be the only ones on the rail
    const nullsAbsent = state.rows
      .filter(r => r[gate] == null)
      .every(r => view.vertices(r)[slot].absent === true)
    return { zeros, nulls, counts, zerosPlaced, nullsAbsent, atZero: SchemaAxes.position(view.summaries[slot], 0) }
  }, { slot: GATE_SLOT, gate: GATE_AXIS })

  expect(rail.zeros).toBe(80)
  expect(rail.nulls).toBe(80)
  expect(rail.counts.present).toBe(160)
  expect(rail.counts.absent).toBe(80)
  expect(rail.zerosPlaced, 'a 0.0 reading rendered as absent').toBe(true)
  expect(rail.nullsAbsent, 'a null reading rendered as a value').toBe(true)
  expect(rail.atZero).not.toBeNull()

  // and the rail chip reports exactly the nulls, not the zeros
  const chip = page.locator('#axis-rail .axis-cell:not(.add)').nth(GATE_SLOT).locator('button.chip.abs')
  await expect(chip).toHaveText('absent 80')
})

test('selecting a line opens that recording', async ({ page }) => {
  await open(page)
  await expect(page.locator('#rec-panel')).toBeHidden()
  await page.locator('#list .list-item').first().click()
  await expect(page.locator('#rec-panel')).toBeVisible()
  await expect(page.locator('#rec-loading')).toBeHidden({ timeout: 30000 })
  await expect(page.locator('#rec-title')).not.toBeEmpty()
  await expect(page.locator('#rec-canvas')).toBeVisible()
  await expect(page.locator('#rec-decision dt').first()).not.toBeEmpty()
  await expect(page.locator('#rec-transcript-note')).not.toBeEmpty()

  const opened = await page.evaluate(() => ({
    current: window.__viewerState.current,
    focus: window.__viewerState.view.focus,
    lanes: window.__viewerState.recView.hits.length
  }))
  expect(opened.current).toBeGreaterThanOrEqual(0)
  expect(opened.focus).toBe(opened.current)
  expect(opened.lanes).toBeGreaterThan(0)
  await page.screenshot({ path: resolve(SHOTS, 'recording-open.png'), fullPage: false })
})

test('the gate outcome axis separates pass, fail and undetermined', async ({ page }) => {
  await open(page)
  const select = page.locator('#axis-rail .axis-cell:not(.add) select.axis-select').nth(GATE_SLOT)
  await select.selectOption('gate_train_min_s_passed')
  const cats = await page.evaluate(slot => window.__viewerState.view.summaries[slot].categories, GATE_SLOT)
  expect(cats).toEqual(['true', 'undetermined', 'false'])
})
