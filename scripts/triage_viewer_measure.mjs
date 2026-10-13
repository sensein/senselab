// Measure the facet layer of the recording-vectors viewer in a real browser, at corpus scale.
//
//   node scripts/triage_viewer_measure.mjs <recording_vectors.parquet>
//
// Prints one JSON object: how long a facet click takes end to end, how that splits between
// building the mask and redrawing the panel, how long the canvas then takes to repaint, and the
// same repaint for an axis brush as an unchanged control. Encoding costs and the memory the
// encodings hold are measured separately, because they are what the lazy encoding is avoiding.
//
// Needs @playwright/test and its Chromium. The parquet is never read by this file -- the page
// reads it, from disk, exactly as the owner opens it. See
// specs/20260924-recording-vectors-facets/facets.md.
import { chromium } from '@playwright/test'
import { resolve } from 'node:path'

const repo = process.cwd()
const PAGE = resolve(repo, 'src/senselab/audio/workflows/triage/viewer/recording_vectors_viewer.html')
const PARQUET = resolve(process.argv[2])

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } })
await page.goto('file://' + PAGE)
const t0 = Date.now()
await page.setInputFiles('#picker', PARQUET)
await page.waitForSelector('#facet-list details.facet-group', { timeout: 180000 })
const load = Date.now() - t0

const out = await page.evaluate(async () => {
  const s = window.__viewerState
  const F = window.SchemaFacets
  const rows = s.rows.length
  const med = a => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)]

  // 1. encoding one column, cold, for a few columns of different shapes
  const encode = {}
  for (const name of ['task', 'verdict', 'gate_failed_names', 'gate_train_min_s_passed']) {
    const m = new F.FacetModel(s.rows)
    const t = performance.now()
    m.encode(name)
    encode[name] = performance.now() - t
  }

  // 2. encoding every offered column, which is what a lazy panel is avoiding
  const all = new F.FacetModel(s.rows)
  let t = performance.now()
  for (const f of F.CATALOGUE) all.encode(f.col.name)
  const encodeAll = performance.now() - t

  // 3. the mask, with 1..4 facets active
  const model = s.facets
  const terms = model.values('task').values.filter(v => !v.absent).slice(0, 4).map(v => v.term)
  const mask = []
  const panel = []
  const full = []
  const picks = [
    ['task', terms[0]], ['verdict', model.values('verdict').values[0].term],
    ['release', model.values('release').values[0].term],
    ['conformance_speech', model.values('conformance_speech').values[0].term]
  ]
  for (let rep = 0; rep < 5; rep++) {
    model.clear()
    for (const [name, term] of picks) {
      const button = document.querySelector(
        `details[data-facet="${name}"] .facet-value[data-term="${CSS.escape(term)}"]`)
      const a = performance.now()
      if (button) button.click()
      else { model.toggle(name, term) }
      const b = performance.now()
      full.push(b - a)
      mask.push(s.facetTiming.mask_ms)
      panel.push(s.facetTiming.panel_ms)
    }
  }

  // 4. one full click through to the painted frame
  model.clear()
  const painted = []
  for (let rep = 0; rep < 5; rep++) {
    const [name, term] = picks[rep % picks.length]
    const button = document.querySelector(
      `details[data-facet="${name}"] .facet-value[data-term="${CSS.escape(term)}"]`)
    const a = performance.now()
    if (button) button.click()
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))
    painted.push(performance.now() - a)
  }

  // 4b. the same click-to-painted cost for an axis brush, which faceting must not have worsened
  model.clear()
  const brushed = []
  const summary = s.view.summaries[3]
  for (let rep = 0; rep < 5; rep++) {
    const lo = summary.min + (rep / 10) * (summary.max - summary.min)
    const a = performance.now()
    s.view.setBrush('duration_s', { lo, hi: summary.max })
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))
    brushed.push(performance.now() - a)
  }
  s.view.clearBrushes()

  // 5. how much memory the encodings hold
  let bytes = 0
  for (const name of Object.keys(all.encodings)) {
    const e = all.encodings[name]
    bytes += e.mode === 'scalar'
      ? e.codes.byteLength
      : e.postings.reduce((n, p) => n + p.byteLength, 0) + e.absentRows.byteLength
  }

  return {
    rows,
    openGroups: Object.values(s.facetOpen).filter(Boolean).length,
    offered: F.CATALOGUE.length,
    encode,
    encodeAll_ms: encodeAll,
    encodings_bytes: bytes,
    mask_ms_median: med(mask),
    mask_ms_max: Math.max(...mask),
    panel_ms_median: med(panel),
    panel_ms_max: Math.max(...panel),
    click_ms_median: med(full),
    click_ms_max: Math.max(...full),
    click_to_painted_ms_median: med(painted),
    click_to_painted_ms_max: Math.max(...painted),
    brush_to_painted_ms_median: med(brushed),
    brush_to_painted_ms_max: Math.max(...brushed)
  }
})

console.log(JSON.stringify({ load_ms: load, ...out }, null, 1))
await browser.close()
