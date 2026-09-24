// The facet panel, driven headless against the built page over file://.
//
// The fixture is synthetic and reproducible:
//   uv run python scripts/triage_viewer_fixture.py --out artifacts/viewer_e2e
// Point VECTORS_PARQUET at a real parquet to run the same assertions over the real corpus; every
// number below is computed from the loaded rows in the page rather than written as a literal, so
// the suite is honest against either file.

import { test, expect } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { existsSync, readFileSync, mkdirSync } from 'node:fs'

const here = dirname(fileURLToPath(import.meta.url))
const repo = resolve(here, '../../../../../..')
const PAGE = resolve(repo, 'src/senselab/audio/workflows/triage/viewer/recording_vectors_viewer.html')
const PARQUET = process.env.VECTORS_PARQUET || resolve(repo, 'artifacts/viewer_e2e/facets_fixture.parquet')
const COUNTS = resolve(repo, 'artifacts/viewer_e2e/facets_fixture.counts.json')
const SHOTS = resolve(repo, 'artifacts/viewer_e2e')
const REAL = Boolean(process.env.VECTORS_PARQUET)

mkdirSync(SHOTS, { recursive: true })

/** Open the page, load the parquet, and wait until the facet panel has rendered. */
async function open (page) {
  expect(
    existsSync(PARQUET),
    `fixture missing: ${PARQUET} - run "uv run python scripts/triage_viewer_fixture.py --out artifacts/viewer_e2e"`
  ).toBe(true)
  const problems = []
  page.on('pageerror', e => problems.push(String(e)))
  page.on('requestfailed', r => problems.push('requestfailed ' + r.url()))
  await page.goto('file://' + PAGE)
  await page.setInputFiles('#picker', PARQUET)
  await expect(page.locator('#main')).toBeVisible({ timeout: 120000 })
  await expect(page.locator('#facet-list details.facet-group').first()).toBeVisible({ timeout: 60000 })
  return problems
}

/** How many loaded rows satisfy a predicate, counted from the rows and never from the facet model. */
function independently (page, body) {
  return page.evaluate(src => {
    // eslint-disable-next-line no-new-func
    const pred = new Function('r', 'return (' + src + ')')
    return window.__viewerState.rows.filter(pred).length
  }, body)
}

const facet = (page, name) => page.locator(`#facet-list details.facet-group[data-facet="${name}"]`)
const value = (page, name, term) => facet(page, name).locator(`.facet-value[data-term="${term}"]`)

async function choose (page, name, term) {
  const group = facet(page, name)
  if (!(await group.evaluate(e => e.open))) await group.locator('summary').click()
  await value(page, name, term).click()
}

const drawn = page => page.evaluate(() => window.__viewerState.view.selectedCount)

// ------------------------------------------------------------------ the page

test('the page loads the parquet from disk and paints the default axes beside the facets', async ({ page }) => {
  const problems = await open(page)
  await expect(page.locator('#status')).not.toHaveClass(/bad/)

  const shape = await page.evaluate(() => ({
    rows: window.__viewerState.rows.length,
    axes: window.__viewerState.view.axes,
    defaults: SchemaAxes.DEFAULT_AXES,
    drawn: window.__viewerState.view.selectedCount
  }))
  expect(shape.rows).toBeGreaterThan(0)
  expect(shape.axes).toEqual(shape.defaults)
  expect(shape.axes).toHaveLength(10)
  expect(shape.drawn).toBe(shape.rows)

  // every default axis placed at least one recording, so none is a wholly blank column
  const present = await page.evaluate(
    n => Array.from({ length: n }, (_, i) => window.__viewerState.view.countPresent(i).present),
    shape.axes.length
  )
  present.forEach((p, i) => expect(p, `${shape.axes[i]} placed nothing`).toBeGreaterThan(0))

  // the panel sits alongside the plot rather than over it
  const box = await page.locator('#facets').boundingBox()
  const plot = await page.locator('#pc-wrap').boundingBox()
  expect(box.width).toBeGreaterThan(200)
  expect(box.x + box.width).toBeLessThanOrEqual(plot.x + 1)

  await expect(page.locator('#facet-denominator')).toContainText('no facet applied')
  expect(problems, 'the page raised no error and fetched nothing').toEqual([])
  await page.screenshot({ path: resolve(SHOTS, REAL ? 'real-loaded.png' : 'facets-loaded.png') })
})

test('the panel offers every categorical column and refuses participant, with the reason', async ({ page }) => {
  await open(page)
  const offered = await page.$$eval('#facet-list details.facet-group', els => els.map(e => e.dataset.facet))
  for (const n of ['verdict', 'task', 'release', 'release_ground', 'route_state', 'conformance_speech']) {
    expect(offered, `${n} must be offered`).toContain(n)
  }
  expect(offered).not.toContain('participant')
  expect(offered.filter(n => /^gate_.*_passed$/.test(n))).toHaveLength(22)
  expect(await page.evaluate(() => SchemaFacets.REFUSED.participant)).toMatch(/list, not a facet/)
})

// ------------------------------------------------------------------ counts

test('a facet value shows its count before it is chosen, and the count is right', async ({ page }) => {
  await open(page)
  const terms = await page.evaluate(() => window.__viewerState.facets.values('task').values
    .filter(v => !v.absent).slice(0, 6).map(v => v.term))
  expect(terms.length).toBeGreaterThan(0)

  for (const term of terms) {
    const truth = await independently(page, `r.task === ${JSON.stringify(term)}`)
    const cell = value(page, 'task', term)
    await expect(cell).toBeVisible()
    const shown = await cell.evaluate(e => e.innerText.replace(/\s+/g, ' ').trim())
    expect(shown, `the ${term} row must read "<available> / <total>"`)
      .toContain(`${truth.toLocaleString('en-US')} / ${truth.toLocaleString('en-US')}`)
  }

  // and the absent bucket is a value with its own count, not a silent drop
  const nulls = await independently(page, 'r.conformance_speech == null')
  const absent = value(page, 'conformance_speech', '(absent)')
  await expect(absent).toBeVisible()
  await expect(absent).toContainText(String(nulls.toLocaleString('en-US')))
})

test('the fixture the generator wrote and the page read agree value for value', async ({ page }) => {
  test.skip(REAL, 'the counts file describes the synthetic fixture only')
  await open(page)
  const expected = JSON.parse(readFileSync(COUNTS, 'utf8'))
  expect(await page.evaluate(() => window.__viewerState.rows.length)).toBe(expected.rows)
  for (const column of ['task', 'verdict', 'release', 'route_state', 'conformance_speech']) {
    const got = await page.evaluate(
      n => Object.fromEntries(window.__viewerState.facets.values(n).values
        .filter(v => v.total > 0).map(v => [v.term, v.total])),
      column
    )
    expect(got, `${column} counts must match what the producer wrote`).toEqual(expected[column])
  }
})

// ------------------------------------------------------------------ narrowing

test('a facet narrows the drawn set to an independently computed number', async ({ page }) => {
  await open(page)
  const total = await page.evaluate(() => window.__viewerState.rows.length)
  expect(await drawn(page)).toBe(total)

  const term = await page.evaluate(() => window.__viewerState.facets.values('task').values[0].term)
  const truth = await independently(page, `r.task === ${JSON.stringify(term)}`)
  expect(truth).toBeGreaterThan(0)
  expect(truth).toBeLessThan(total)

  await choose(page, 'task', term)
  expect(await drawn(page), `choosing task=${term} must draw exactly its own recordings`).toBe(truth)
  await expect(page.locator('#selcount')).toContainText(`${truth.toLocaleString('en-US')} of`)

  // the header states the denominator on both sides of the narrowing
  const den = await page.locator('#facet-denominator').innerText()
  expect(den).toContain(total.toLocaleString('en-US'))
  expect(den).toContain(truth.toLocaleString('en-US'))
  await expect(page.locator('#facet-chosen .facet-tag')).toHaveCount(1)
  await page.screenshot({ path: resolve(SHOTS, REAL ? 'real-one-facet.png' : 'facets-one.png') })
})

test('values of one facet are a union and two facets are an intersection', async ({ page }) => {
  await open(page)
  const [a, b] = await page.evaluate(
    () => window.__viewerState.facets.values('task').values.filter(v => !v.absent).slice(0, 2).map(v => v.term)
  )
  const union = await independently(page, `r.task === ${JSON.stringify(a)} || r.task === ${JSON.stringify(b)}`)

  await choose(page, 'task', a)
  await choose(page, 'task', b)
  expect(await drawn(page), 'two values of one facet are a union').toBe(union)

  const verdict = await page.evaluate(
    () => window.__viewerState.facets.values('verdict').values.filter(v => v.available > 0)[0].term
  )
  const both = await independently(
    page,
    `(r.task === ${JSON.stringify(a)} || r.task === ${JSON.stringify(b)}) && r.verdict === ${JSON.stringify(verdict)}`
  )
  await choose(page, 'verdict', verdict)
  expect(await drawn(page), 'two facets are an intersection').toBe(both)
  expect(both).toBeLessThanOrEqual(union)
  await expect(page.locator('#facet-chosen .facet-tag')).toHaveCount(3)
  await page.screenshot({ path: resolve(SHOTS, REAL ? 'real-composed.png' : 'facets-composed.png') })
})

test('a sibling value keeps a real count after its neighbour is chosen', async ({ page }) => {
  await open(page)
  const values = await page.evaluate(
    () => window.__viewerState.facets.values('verdict').values.filter(v => v.total > 0 && !v.absent).map(v => v.term)
  )
  test.skip(values.length < 2, 'needs a verdict column with more than one value')
  await choose(page, 'verdict', values[0])
  const sibling = await page.evaluate(
    t => window.__viewerState.facets.values('verdict').values.find(v => v.term === t),
    values[1]
  )
  const truth = await independently(page, `r.verdict === ${JSON.stringify(values[1])}`)
  expect(sibling.available, 'a sibling is counted with this facet left out').toBe(truth)
  expect(sibling.total).toBe(truth)
})

test('absent is selectable, and selects exactly the nulls', async ({ page }) => {
  await open(page)
  const nulls = await independently(page, 'r.conformance_speech == null')
  test.skip(nulls === 0, 'this corpus has no null conformance_speech')
  await choose(page, 'conformance_speech', '(absent)')
  expect(await drawn(page)).toBe(nulls)
})

test('a facet composes with an axis brush rather than replacing it', async ({ page }) => {
  await open(page)
  const term = await page.evaluate(() => window.__viewerState.facets.values('verdict').values[0].term)
  const withFacet = await independently(page, `r.verdict === ${JSON.stringify(term)}`)
  await choose(page, 'verdict', term)
  expect(await drawn(page)).toBe(withFacet)

  // brush duration_s (axis slot 3) to its upper half, from the summary the page itself built
  const cut = await page.evaluate(() => {
    const s = window.__viewerState.view.summaries[3]
    return { name: s.name, lo: (s.min + s.max) / 2, hi: s.max }
  })
  expect(cut.name).toBe('duration_s')
  const both = await independently(
    page,
    `r.verdict === ${JSON.stringify(term)} && r.duration_s != null && r.duration_s >= ${cut.lo}`
  )
  await page.evaluate(c => window.__viewerState.view.setBrush('duration_s', { lo: c.lo, hi: c.hi }), cut)
  expect(await drawn(page), 'the brush and the facet intersect').toBe(both)

  // the facet panel now counts against the brushed denominator, and says so
  const brushed = await independently(page, `r.duration_s != null && r.duration_s >= ${cut.lo}`)
  expect(await page.evaluate(() => window.__viewerState.facets.before())).toBe(brushed)
  expect(await page.evaluate(() => window.__viewerState.facets.after())).toBe(both)

  // clearing the brushes leaves the facet standing
  await page.locator('#clear-brushes').click()
  expect(await drawn(page)).toBe(withFacet)
})

test('clearing the facets restores the full set exactly', async ({ page }) => {
  await open(page)
  const total = await page.evaluate(() => window.__viewerState.rows.length)
  const [t, v] = await page.evaluate(() => [
    window.__viewerState.facets.values('task').values[0].term,
    window.__viewerState.facets.values('verdict').values[0].term
  ])
  await choose(page, 'task', t)
  await choose(page, 'verdict', v)
  expect(await drawn(page)).toBeLessThan(total)

  await expect(page.locator('#facet-clear')).toBeEnabled()
  await page.locator('#facet-clear').click()
  expect(await drawn(page)).toBe(total)
  await expect(page.locator('#facet-chosen .facet-tag')).toHaveCount(0)
  await expect(page.locator('#facet-clear')).toBeDisabled()
  await expect(page.locator('#facet-denominator')).toContainText('no facet applied')
  expect(await page.evaluate(() => window.__viewerState.view.facetMask)).toBeNull()
})

test('the search narrows the panel by facet name and by a value of an opened facet', async ({ page }) => {
  await open(page)
  const visible = () => page.$$eval('#facet-list details.facet-group', els => els.map(e => e.dataset.facet))
  const all = await visible()
  expect(all.length).toBeGreaterThan(10)

  await page.locator('#facet-search').fill('release')
  const byName = await visible()
  expect(byName).toContain('release')
  expect(byName).toContain('release_ground')
  expect(byName).not.toContain('verdict')

  // a value of an already-encoded facet surfaces its group, opened on the match
  const term = await page.evaluate(() => window.__viewerState.facets.values('task').values[0].term)
  await page.locator('#facet-search').fill(term.slice(0, 5))
  const byValue = await visible()
  expect(byValue, `searching "${term}" must surface the task facet`).toContain('task')
  await expect(value(page, 'task', term)).toBeVisible()

  await page.locator('#facet-search').fill('')
  expect(await visible(), 'clearing the search restores every offered facet').toEqual(all)
})

// ------------------------------------------------------------------ zero is not absent

test('a genuine 0.0 reading is a value on the band, and only a null is absent', async ({ page }) => {
  await open(page)
  const gate = await page.evaluate(() => {
    const rows = window.__viewerState.rows
    const names = SchemaAxes.GATES.map(g => 'gate_' + g[0])
      .filter(n => rows.some(r => r[n] === 0) && rows.some(r => r[n] == null))
    return names[0] || null
  })
  expect(gate, 'the corpus must carry a gate with both genuine zeros and nulls').not.toBeNull()

  await page.locator('#axis-rail .axis-cell:not(.add) select.axis-select').nth(3).selectOption(gate)
  const rail = await page.evaluate(g => {
    const state = window.__viewerState
    const view = state.view
    const zeros = state.rows.filter(r => r[g] === 0)
    const nulls = state.rows.filter(r => r[g] == null)
    return {
      zeros: zeros.length,
      nulls: nulls.length,
      counts: view.countPresent(3),
      zerosPlaced: zeros.every(r => {
        const v = view.vertices(r)[3]
        return v.absent === false && v.y != null && Number.isFinite(v.y)
      }),
      nullsAbsent: nulls.every(r => view.vertices(r)[3].absent === true),
      atZero: SchemaAxes.position(view.summaries[3], 0)
    }
  }, gate)

  expect(rail.zeros).toBeGreaterThan(0)
  expect(rail.nulls).toBeGreaterThan(0)
  expect(rail.zerosPlaced, 'a 0.0 reading rendered as absent').toBe(true)
  expect(rail.nullsAbsent, 'a null reading rendered as a value').toBe(true)
  expect(rail.atZero).not.toBeNull()
  expect(rail.counts.absent).toBe(rail.nulls)
  await expect(
    page.locator('#axis-rail .axis-cell:not(.add)').nth(3).locator('button.chip.abs')
  ).toHaveText(`absent ${rail.nulls.toLocaleString('en-US')}`)

  // A facet narrows which lines are drawn, so the rail's counts follow the drawn set by design.
  // What must not change is the rule: among the lines still drawn, a 0.0 is on the band and only a
  // null is on the rail, and the rail's number is exactly the drawn nulls.
  const verdict = await page.evaluate(() => window.__viewerState.facets.values('verdict').values[0].term)
  await choose(page, 'verdict', verdict)
  const after = await page.evaluate(g => {
    const state = window.__viewerState
    const view = state.view
    const shown = state.rows.filter((_, i) => view.selected[i] === 1)
    return {
      counts: view.countPresent(3),
      drawnNulls: shown.filter(r => r[g] == null).length,
      drawnZeros: shown.filter(r => r[g] === 0).length,
      zerosPlaced: shown.filter(r => r[g] === 0).every(r => view.vertices(r)[3].absent === false),
      nullsAbsent: shown.filter(r => r[g] == null).every(r => view.vertices(r)[3].absent === true)
    }
  }, gate)
  expect(after.drawnZeros, 'the facet must leave some genuine zeros drawn').toBeGreaterThan(0)
  expect(after.zerosPlaced, 'a 0.0 rendered as absent once a facet was applied').toBe(true)
  expect(after.nullsAbsent, 'a null rendered as a value once a facet was applied').toBe(true)
  expect(after.counts.absent).toBe(after.drawnNulls)
  expect(after.counts.absent).toBeLessThan(rail.nulls)
  await page.screenshot({ path: resolve(SHOTS, REAL ? 'real-zero-vs-absent.png' : 'facets-zero-vs-absent.png') })
})

// ------------------------------------------------------------------ cost

test('narrowing stays responsive, and the cost is reported', async ({ page }) => {
  await open(page)
  const rows = await page.evaluate(() => window.__viewerState.rows.length)
  const terms = await page.evaluate(
    () => window.__viewerState.facets.values('task').values.filter(v => !v.absent).slice(0, 5).map(v => v.term)
  )
  const samples = []
  for (const term of terms) {
    await choose(page, 'task', term)
    samples.push(await page.evaluate(() => window.__viewerState.facetTiming))
  }
  const worstMask = Math.max(...samples.map(s => s.mask_ms))
  const worstPanel = Math.max(...samples.map(s => s.panel_ms))
  // eslint-disable-next-line no-console
  console.log(`facet click over ${rows.toLocaleString('en-US')} rows: mask+select ` +
    `${worstMask.toFixed(2)} ms worst of ${samples.length}, panel ${worstPanel.toFixed(2)} ms`)
  expect(worstMask, 'building the mask and re-deriving the drawn set must stay inside a frame').toBeLessThan(16)
  expect(worstPanel, 'redrawing the open facet groups must stay inside two frames').toBeLessThan(33)
})
