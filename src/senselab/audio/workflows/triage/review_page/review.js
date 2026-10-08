// The triage review page: three tabs over one selection of recordings.
//
// Explore draws the recording-vectors viewer's parallel coordinates over the decision columns and
// the evidence items; Review lists what the brushes, facets and search admit and shows one
// recording; Decisions holds the reviewer's entries and exports them as JSON whose keys are the
// owner label table's. Per-recording data arrives from side files through `ReviewPage.shard`.

'use strict';

var ReviewPage = (function () {
  var EXPORT_SCHEMA = 'senselab.triage.review';
  var EXPORT_VERSION = 1;
  var VERDICTS = ['pass', 'review', 'discard'];
  var DECISION_AXES = ['declared_family', 'verdict', 'release', 'reason', 'run_status', 'extent_duration_s', 'duration_s'];
  var FACETS = ['verdict', 'release', 'reason', 'branch', 'declared_family', 'run_status', 'annotations',
    'evidence_items', 'decisive_items', 'reasons'];
  var MAX_EVIDENCE_AXES = 8;
  var LIST_PAGE = 200;
  // The LLM reviewer's bookkeeping, left out of the line the page shows.
  var LLM_HIDDEN = ['task_context', 'result_cache', 'prompt_version', 'model_id', 'revision', 'original'];

  // ---------------------------------------------------------------- pure: index to rows

  function decodeScalar(col) {
    return col.codes.map(function (c) { return c < 0 ? null : col.values[c]; });
  }

  function decodeList(col) {
    return col.codes.map(function (codes) { return codes.map(function (c) { return col.values[c]; }); });
  }

  function evidenceKey(name) { return 'ev:' + name; }

  /** One row object per recording, keyed the way the axis catalogue reads them. */
  function decodeRows(index) {
    var cols = index.cols;
    var n = index.build.n;
    var scalars = {};
    ['participant', 'session', 'task', 'family', 'branch', 'verdict', 'release', 'reason', 'run_status']
      .forEach(function (k) { scalars[k] = decodeScalar(cols[k]); });
    var reasons = decodeList(cols.reasons), annotations = decodeList(cols.annotations);
    var items = decodeList(cols.items), decisive = decodeList(cols.decisive);
    var rows = new Array(n);
    for (var i = 0; i < n; i++) {
      rows[i] = {
        index: i,
        stem: cols.stem[i],
        participant: scalars.participant[i],
        session: scalars.session[i],
        task: scalars.task[i],
        declared_family: scalars.family[i],
        branch: scalars.branch[i],
        verdict: scalars.verdict[i],
        release: scalars.release[i],
        reason: scalars.reason[i],
        run_status: scalars.run_status[i],
        reasons: reasons[i],
        annotations: annotations[i],
        evidence_items: items[i],
        decisive_items: decisive[i],
        extent_duration_s: cols.extent_duration_s[i],
        duration_s: cols.duration_s[i],
        text: cols.text[i],
      };
    }
    Object.keys(index.evidence).forEach(function (name) {
      var e = index.evidence[name];
      var key = evidenceKey(name);
      for (var j = 0; j < e.rows.length; j++) rows[e.rows[j]][key] = e.values[j];
    });
    return rows;
  }

  /** The column specs this page adds to the axis catalogue. */
  function columnSpecs(index) {
    var specs = [
      { name: 'branch', kind: 'categorical', group: 'decision', nullMeans: 'no branch owns the declared family' },
      { name: 'reason', kind: 'categorical', group: 'decision', nullMeans: 'a pass with no reason' },
      { name: 'run_status', kind: 'categorical', group: 'decision', nullMeans: 'the fold wrote no run status' },
      { name: 'reasons', kind: 'set', group: 'decision', assignable: false, reason: 'a set of reasons — filter by a term' },
      { name: 'annotations', kind: 'set', group: 'decision', assignable: false, reason: 'a set of annotations — filter by a term' },
      { name: 'evidence_items', kind: 'set', group: 'evidence', label: 'evidence items read', assignable: false,
        reason: 'a set of evidence names — filter by a term' },
      { name: 'decisive_items', kind: 'set', group: 'evidence', label: 'decisive evidence items', assignable: false,
        reason: 'a set of evidence names — filter by a term' },
      { name: 'extent_duration_s', kind: 'numeric', group: 'decision', unit: 's', nullMeans: 'no task extent' },
    ];
    Object.keys(index.evidence).sort().forEach(function (name) {
      var e = index.evidence[name];
      specs.push({
        name: evidenceKey(name), label: name, kind: e.kind === 'numeric' ? 'numeric' : 'categorical',
        group: 'evidence: ' + (e.group || 'other'), unit: e.unit || null,
        nullMeans: 'the fold did not weigh ' + name + ' for this recording',
      });
    });
    return specs;
  }

  /** Evidence groups present, each with its items most-read first. */
  function evidenceGroups(index) {
    var groups = {};
    Object.keys(index.evidence).forEach(function (name) {
      var e = index.evidence[name];
      var g = e.group || 'other';
      (groups[g] = groups[g] || []).push({ name: name, n: e.rows.length });
    });
    Object.keys(groups).forEach(function (g) {
      groups[g].sort(function (a, b) { return b.n - a.n || (a.name < b.name ? -1 : 1); });
    });
    return groups;
  }

  /** The axes for one choice: the decision columns, or verdict and reason beside a group's items. */
  function axesFor(choice, index) {
    if (!choice || choice === 'decision') return DECISION_AXES.slice();
    var items = evidenceGroups(index)[choice] || [];
    return ['verdict', 'reason'].concat(items.slice(0, MAX_EVIDENCE_AXES).map(function (i) { return evidenceKey(i.name); }));
  }

  /** Rows a stem/transcript search and a participant/session filter admit. */
  function searchMask(rows, query, who) {
    var q = (query || '').trim().toLowerCase();
    var w = (who || '').trim().toLowerCase();
    if (!q && !w) return null;
    var out = new Uint8Array(rows.length);
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      var ok = true;
      if (q) ok = r.stem.toLowerCase().indexOf(q) >= 0 || (r.text != null && r.text.indexOf(q) >= 0);
      if (ok && w) ok = String(r.participant || '').toLowerCase().indexOf(w) >= 0 ||
        String(r.session || '').toLowerCase().indexOf(w) >= 0;
      out[i] = ok ? 1 : 0;
    }
    return out;
  }

  function andMasks(a, b) {
    if (!a) return b;
    if (!b) return a;
    var out = new Uint8Array(a.length);
    for (var i = 0; i < a.length; i++) out[i] = a[i] && b[i] ? 1 : 0;
    return out;
  }

  /** Two-bit levels, time-major, out of the base64 text the builder packs. */
  function unpackSpec(text, frames, bands) {
    var bin = typeof atob === 'function' ? atob(text) : Buffer.from(text, 'base64').toString('binary');
    var out = new Uint8Array(frames * bands);
    for (var i = 0; i < out.length; i++) {
      var byte = bin.charCodeAt(i >> 2);
      out[i] = (byte >> ((i & 3) * 2)) & 3;
    }
    return out;
  }

  // ---------------------------------------------------------------- pure: the time window

  var MIN_SPAN_S = 0.25;

  /** The window `[t0, t1]` after zooming by `factor` about time `at`, kept inside `[0, duration]`. */
  function zoomWindow(win, duration, factor, at) {
    var d = duration > 0 ? duration : 1;
    var span = Math.min(d, Math.max(Math.min(MIN_SPAN_S, d), (win.t1 - win.t0) * factor));
    var frac = win.t1 > win.t0 ? (at - win.t0) / (win.t1 - win.t0) : 0.5;
    var t0 = Math.max(0, Math.min(d - span, at - frac * span));
    return { t0: t0, t1: t0 + span };
  }

  /** The window moved by `dt` seconds, kept inside `[0, duration]`. */
  function panWindow(win, duration, dt) {
    var span = win.t1 - win.t0;
    var t0 = Math.max(0, Math.min(Math.max(0, duration - span), win.t0 + dt));
    return { t0: t0, t1: t0 + span };
  }

  /** A span's left edge and width as fractions of the window, or null where it falls outside. */
  function placeSpan(start, end, win) {
    if (start == null || end == null || end < win.t0 || start > win.t1) return null;
    var span = win.t1 - win.t0;
    return { left: (start - win.t0) / span, width: Math.max(0, end - start) / span };
  }

  /** Tick times for a window: a 1-2-5 step giving about `target` ticks. */
  function ticks(win, target) {
    var span = win.t1 - win.t0;
    if (!(span > 0)) return [];
    var raw = span / (target || 5);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 5, 10].map(function (m) { return m * mag; }).filter(function (v) { return v >= raw; })[0];
    var out = [];
    for (var k = Math.ceil(win.t0 / step - 1e-9); k * step <= win.t1 + 1e-9; k++) out.push(Math.round(k * step * 1000) / 1000 + 0);
    return out;
  }

  // ---------------------------------------------------------------- pure: transcripts

  /** A token's comparison key: the consensus normalisation (casefold, alphanumerics and apostrophe). */
  function tokenKey(text) {
    return String(text == null ? '' : text).toLowerCase().replace(/[^\p{L}\p{N}']/gu, '');
  }

  /** The labels of the models that read a consensus word. */
  function readersOf(sp, word) {
    var models = sp.models || [];
    var out = [];
    (word[2] || []).forEach(function (r, k) { if (r !== null && r !== undefined) out.push(modelLabel(models[k])); });
    return out;
  }

  /** The readings of a consensus word other than its surface, each with the models that read it. */
  function otherReadings(sp, word, surface) {
    var models = sp.models || [];
    var shown = tokenKey(surface);
    var byKey = {};
    var order = [];
    (word[2] || []).forEach(function (r, k) {
      if (r === null || r === undefined) return;
      var key = tokenKey(r);
      if (key === shown || !key) return;
      if (!byKey[key]) { byKey[key] = { text: r, models: [] }; order.push(key); }
      byKey[key].models.push(modelLabel(models[k]));
    });
    return order.map(function (k) { return byKey[k]; });
  }

  /** One lane per model, its own words in its own order: `{label, tokens: [[start, end, text]]}`. */
  function modelLanes(sp) {
    return (sp.own || []).map(function (o) {
      return { label: modelLabel(o), tokens: (o.words || []).filter(function (w) { return w[0] != null && w[1] != null; }) };
    });
  }

  // ---------------------------------------------------------------- pure: the reviewer's entries

  function listenSet(buildId) { return 'triage_review_' + buildId; }

  /** One exported entry, in the owner label table's columns plus the reviewer's own verdict. */
  function entryFor(row, decision, buildId) {
    return {
      listen_set: listenSet(buildId),
      stem: row.stem,
      family: row.declared_family,
      instructed: '',
      duration_s: row.duration_s,
      owner_label: 'reviewer_' + decision.verdict,
      owner_note: decision.note || '',
      pipeline_at_listen: row.verdict + (row.release ? '/' + row.release : '') + (row.reason ? ' (' + row.reason + ')' : ''),
      reviewer_verdict: decision.verdict,
      reviewed_at: decision.at,
    };
  }

  function exportPayload(rowsByStem, decisions, buildId, now) {
    var stems = Object.keys(decisions).sort();
    return {
      schema: EXPORT_SCHEMA,
      version: EXPORT_VERSION,
      build: buildId,
      exported_at: now,
      entries: stems.filter(function (s) { return rowsByStem[s]; })
        .map(function (s) { return entryFor(rowsByStem[s], decisions[s], buildId); }),
    };
  }

  /** Decisions out of an exported file: `{stem: {verdict, note, at}}`. */
  function importDecisions(payload) {
    var entries = Array.isArray(payload) ? payload : (payload && payload.entries) || [];
    var out = {};
    entries.forEach(function (e) {
      var verdict = e.reviewer_verdict || String(e.owner_label || '').replace(/^reviewer_/, '');
      if (!e.stem || VERDICTS.indexOf(verdict) < 0) return;
      out[e.stem] = { verdict: verdict, note: e.owner_note || '', at: e.reviewed_at || null };
    });
    return out;
  }

  // ---------------------------------------------------------------- the page

  var state = null;
  var shards = {};
  var waiting = {};

  /** A side file has loaded: its records, in page order, starting at number * shard_size. */
  function shard(number, records) {
    shards[number] = records;
    (waiting[number] || []).forEach(function (cb) { cb(records); });
    delete waiting[number];
  }

  function recordOf(index, cb) {
    var size = state.index.build.shard_size;
    var k = Math.floor(index / size);
    var take = function (records) { cb(records[index - k * size]); };
    if (shards[k]) return take(shards[k]);
    if (waiting[k]) { waiting[k].push(take); return; }
    waiting[k] = [take];
    var s = document.createElement('script');
    s.src = state.index.build.shard_dir + '/shard-' + String(k).padStart(4, '0') + '.js';
    s.onerror = function () { delete waiting[k]; cb(null); };
    document.body.appendChild(s);
  }

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function loadDecisions() {
    try { return JSON.parse(window.localStorage.getItem(state.storeKey) || '{}'); } catch (e) { return {}; }
  }
  function saveDecisions() {
    try { window.localStorage.setItem(state.storeKey, JSON.stringify(state.decisions)); } catch (e) { /* none */ }
    $('rv-dec-n').textContent = Object.keys(state.decisions).length;
  }

  function init() {
    var index = window.REVIEW_INDEX;
    var rows = decodeRows(index);
    SchemaAxes.register(columnSpecs(index));
    SchemaAxes.ORDERINGS.run_status = ['complete', 'incomplete'];
    SchemaFacets.refresh();
    var byStem = {};
    rows.forEach(function (r) { byStem[r.stem] = r; });
    state = {
      index: index, rows: rows, byStem: byStem, storeKey: 'triage-review:' + index.build.id,
      decisions: {}, current: -1, listShown: LIST_PAGE, served: /^https?:$/.test(location.protocol),
    };
    state.decisions = loadDecisions();
    $('rv-dec-n').textContent = Object.keys(state.decisions).length;
    $('rv-mode').textContent = state.served ? 'served: audio plays where the cluster serves it'
      : 'opened from file: quantised spectrograms, no audio';
    Theme.wire($('rv-theme'));
    Theme.onChange(function () { if (state.view) state.view.draw(); if (state.current >= 0) openRecording(state.current); });
    window.addEventListener('resize', function () { if (state.timeline) redraw(state.timeline); });

    var view = new CorpusView($('rv-pc'), $('rv-pc-overlay'));
    state.view = view;
    view.setRows(rows);
    view.onSelectionChange = function (count) {
      $('rv-selected').textContent = count.toLocaleString() + ' of ' + rows.length.toLocaleString() + ' selected';
      state.facets.setBase(view.brushMask);
      renderFacets();
      renderList();
    };
    state.facets = new SchemaFacets.FacetModel(rows);

    var groupSelect = $('rv-axes-group');
    ['decision'].concat(Object.keys(evidenceGroups(index)).sort()).forEach(function (g) {
      var o = el('option', null, g); o.value = g; groupSelect.appendChild(o);
    });
    groupSelect.onchange = function () { setAxes(groupSelect.value); };
    var colour = $('rv-colour');
    ['verdict', 'release', 'reason', 'branch', 'run_status'].forEach(function (c) {
      var o = el('option', null, c); o.value = c; colour.appendChild(o);
    });
    colour.onchange = function () { view.colourBy = colour.value; view.buildColourMap(); view.draw(); };
    $('rv-clear').onclick = function () { view.clearBrushes(); renderAxisLabels(); };
    wirePointer(view);
    setAxes('decision');

    document.querySelectorAll('.rv-tab').forEach(function (b) { b.onclick = function () { showTab(b.dataset.tab); }; });
    $('rv-search').oninput = applyFilters;
    $('rv-who').oninput = applyFilters;
    $('rv-facets-clear').onclick = function () {
      state.facets.activeNames().forEach(function (n) { state.facets.clear(n); });
      applyFilters();
    };
    $('rv-more').onclick = function () { state.listShown += LIST_PAGE; renderList(); };
    $('rv-export').onclick = exportJson;
    $('rv-import').onchange = importJson;
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', function () { view.draw(); renderAxisLabels(); });
    applyFilters();
    renderDecisions();
  }

  function showTab(name) {
    document.querySelectorAll('.rv-tab').forEach(function (b) { b.classList.toggle('on', b.dataset.tab === name); });
    ['explore', 'review', 'decisions'].forEach(function (t) { $('tab-' + t).hidden = t !== name; });
    if (name === 'explore') { state.view.draw(); renderAxisLabels(); }
    if (name === 'decisions') renderDecisions();
  }

  function setAxes(choice) {
    var names = axesFor(choice, state.index).filter(function (n) { return SchemaAxes.BY_NAME[n]; });
    state.view.brushes = {};
    state.view.setAxes(names);
    state.view.buildColourMap();
    state.view.applyBrushes();
    state.view.draw();
    renderAxisLabels();
  }

  function renderAxisLabels() {
    var box = $('rv-axis-labels');
    box.innerHTML = '';
    var view = state.view;
    if (!view.geom) view.layout();
    view.summaries.forEach(function (s, a) {
      var d = el('div', view.brushes[s.name] ? 'brushed' : null, s.col.label);
      d.style.left = view.geom.xs[a] + 'px';
      d.appendChild(el('small', null, SchemaAxes.caption(s)));
      box.appendChild(d);
    });
    var active = Object.keys(view.brushes);
    $('rv-brushes').textContent = active.length ? 'brushed: ' + active.map(function (n) {
      return (SchemaAxes.BY_NAME[n] || { label: n }).label;
    }).join(', ') : '';
  }

  function wirePointer(view) {
    var canvas = $('rv-pc-overlay');
    var drag = null;
    function local(e) { var r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
    function onBand(p) { return view.geom && p.y >= view.geom.bandTop - 6 && p.y <= view.geom.bandBottom + 6; }
    canvas.addEventListener('mousedown', function (e) {
      var p = local(e);
      var a = view.axisAt(p.x);
      if (a >= 0 && onBand(p)) drag = { axis: a, y0: p.y, y1: p.y };
    });
    canvas.addEventListener('mousemove', function (e) {
      var p = local(e);
      if (!drag) {
        var hit = view.pick(p.x, p.y);
        if (hit !== view.hover) { view.hover = hit; view.paintOverlay(); }
        return;
      }
      drag.y1 = p.y;
      view.paintOverlay();
      var ctx = view.octx, x = view.geom.xs[drag.axis];
      var top = Math.min(drag.y0, drag.y1), bot = Math.max(drag.y0, drag.y1);
      ctx.save();
      ctx.fillStyle = Theme.color('brush-fill');
      ctx.fillRect(x - 16, top, 32, bot - top);
      ctx.restore();
    });
    window.addEventListener('mouseup', function (e) {
      if (!drag) return;
      var p = local(e);
      var s = view.summaries[drag.axis];
      var top = Math.min(drag.y0, p.y), bot = Math.max(drag.y0, p.y);
      if (bot - top < 3) view.setBrush(s.name, null);
      else if (s.kind === 'numeric') {
        var hi = view.valueAt(drag.axis, top), lo = view.valueAt(drag.axis, bot);
        view.setBrush(s.name, { lo: Math.min(lo, hi), hi: Math.max(lo, hi) });
      } else {
        var terms = s.categories.filter(function (c) { var y = view.yOf(s, c); return y != null && y >= top - 1 && y <= bot + 1; });
        view.setBrush(s.name, terms.length ? { terms: terms } : null);
      }
      drag = null;
      renderAxisLabels();
    });
    canvas.addEventListener('click', function (e) {
      var p = local(e);
      if (view.axisAt(p.x) >= 0 && onBand(p)) return;
      var hit = view.pick(p.x, p.y);
      if (hit >= 0) { showTab('review'); openRecording(hit); }
    });
  }

  function applyFilters() {
    var mask = andMasks(state.facets.mask(), searchMask(state.rows, $('rv-search').value, $('rv-who').value));
    state.listShown = LIST_PAGE;
    state.view.setFacetMask(mask);
  }

  function renderFacets() {
    var box = $('rv-facet-list');
    var open = {};
    box.querySelectorAll('details[open]').forEach(function (d) { open[d.dataset.name] = true; });
    box.innerHTML = '';
    FACETS.forEach(function (name) {
      if (!SchemaFacets.BY_NAME[name]) return;
      var v = state.facets.values(name);
      var d = el('details', 'rv-facet');
      d.dataset.name = name;
      if (open[name] || v.chosen.length || name === 'verdict' || name === 'reason') d.open = true;
      d.appendChild(el('summary', null, v.label + (v.chosen.length ? ' · ' + v.chosen.length : '')));
      v.values.forEach(function (item) {
        if (!item.total) return;
        var l = el('label', item.available ? null : 'zero');
        var c = el('input'); c.type = 'checkbox'; c.checked = item.chosen;
        c.onchange = function () { state.facets.toggle(name, item.term); applyFilters(); };
        l.appendChild(c);
        l.appendChild(el('span', null, item.label));
        l.appendChild(el('span', 'n', item.available.toLocaleString() + ' / ' + item.total.toLocaleString()));
        d.appendChild(l);
      });
      box.appendChild(d);
    });
  }

  function selectedIndices() {
    var out = [];
    var sel = state.view.selected;
    for (var i = 0; i < sel.length; i++) if (sel[i]) out.push(i);
    return out;
  }

  function verdictChip(v) { return el('span', 'rv-chip v-' + v, v); }

  function renderList() {
    var list = $('rv-list');
    list.innerHTML = '';
    var picked = selectedIndices();
    var reviewed = picked.filter(function (i) { return state.decisions[state.rows[i].stem]; }).length;
    $('rv-list-head').textContent = picked.length.toLocaleString() + ' recordings · ' + reviewed + ' reviewed here';
    picked.slice(0, state.listShown).forEach(function (i) {
      var r = state.rows[i];
      var li = el('li', i === state.current ? 'cur' : null);
      li.dataset.i = i;
      li.appendChild(el('span', 's', r.stem));
      li.appendChild(verdictChip(r.verdict));
      li.appendChild(el('span', 'rv-note', (r.reason || '') + ' · ' + (r.declared_family || 'no family')));
      if (state.decisions[r.stem]) li.appendChild(el('span', 'done', ' ✓ ' + state.decisions[r.stem].verdict));
      li.onclick = function () { openRecording(i); };
      list.appendChild(li);
    });
    $('rv-more').hidden = picked.length <= state.listShown;
  }

  function onKey(e) {
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    if (e.key !== 'j' && e.key !== 'k') return;
    var picked = selectedIndices();
    if (!picked.length) return;
    var at = picked.indexOf(state.current);
    var next = e.key === 'j' ? Math.min(picked.length - 1, at + 1) : Math.max(0, at - 1);
    openRecording(picked[at < 0 ? 0 : next]);
  }

  function fmt(v) {
    if (v == null) return '—';
    if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) >= 100 ? 0 : 2);
    if (typeof v === 'object') return JSON.stringify(v);
    return String(v);
  }

  function openRecording(i) {
    state.current = i;
    var r = state.rows[i];
    document.querySelectorAll('#rv-list li').forEach(function (li) { li.classList.toggle('cur', Number(li.dataset.i) === i); });
    var box = $('rv-detail');
    box.innerHTML = '';
    var head = el('section');
    head.appendChild(el('h2', null, r.stem));
    head.appendChild(verdictChip(r.verdict));
    if (r.release) head.appendChild(el('span', 'rv-chip', r.release));
    head.appendChild(el('span', 'rv-note', [r.declared_family || 'no family', r.branch || 'no branch',
      r.reason ? 'reason ' + r.reason : null, r.run_status === 'incomplete' ? 'incomplete run' : null]
      .filter(Boolean).join(' · ')));
    if (r.reasons.length > 1) head.appendChild(el('div', 'rv-note', 'all reasons: ' + r.reasons.join(', ')));
    if (r.annotations.length) head.appendChild(el('div', 'rv-note', 'annotations: ' + r.annotations.join(', ')));
    box.appendChild(head);
    box.appendChild(entryForm(r));
    var body = el('div', null);
    body.appendChild(el('p', 'rv-note', 'loading…'));
    box.appendChild(body);
    recordOf(i, function (rec) {
      if (state.current !== i) return;
      body.innerHTML = '';
      if (!rec) { body.appendChild(el('p', 'rv-note', 'this recording\'s side file did not load')); return; }
      var time = spectrogramSection(r, rec);
      state.timeline = time.timeline;
      body.appendChild(time);
      body.appendChild(audioSection(rec, time.timeline));
      body.appendChild(evidenceSection(rec));
      if (rec.speech) body.appendChild(speechSection(rec.speech));
      if (rec.missing && rec.missing.length) body.appendChild(el('p', 'rv-note', 'missing: ' + rec.missing.join(', ')));
      body.appendChild(el('p', 'rv-note', 'commit ' + (rec.commit || '—') + ' · config ' + (rec.config_hash || '—')));
    });
  }

  function entryForm(r) {
    var box = el('section', 'rv-entry');
    box.appendChild(el('h3', null, 'your decision'));
    var current = state.decisions[r.stem] || {};
    VERDICTS.forEach(function (v) {
      var l = el('label');
      var c = el('input'); c.type = 'radio'; c.name = 'rv-verdict'; c.value = v; c.checked = current.verdict === v;
      c.onchange = function () { record(r, v, note.value); };
      l.appendChild(c); l.appendChild(document.createTextNode(' ' + v));
      box.appendChild(l);
    });
    var clear = el('button', null, 'clear'); clear.type = 'button';
    clear.onclick = function () { delete state.decisions[r.stem]; saveDecisions(); openRecording(r.index); renderList(); };
    box.appendChild(clear);
    var note = el('textarea'); note.rows = 2; note.placeholder = 'note'; note.value = current.note || '';
    note.oninput = function () { if (state.decisions[r.stem]) record(r, state.decisions[r.stem].verdict, note.value); };
    box.appendChild(note);
    return box;
  }

  function record(r, verdict, note) {
    state.decisions[r.stem] = { verdict: verdict, note: note || '', at: new Date().toISOString() };
    saveDecisions();
    renderList();
  }

  /** One recording's time view: spectrogram, ASR lanes and audio players over one zoomable window. */
  function timelineOf(duration, extent) {
    var d = duration || 1;
    return { d: d, extent: extent, win: { t0: 0, t1: d }, redraw: [], players: [], active: null, spanEnd: null };
  }

  function redraw(tl) { tl.redraw.forEach(function (f) { f(); }); }

  function setWindow(tl, win) { tl.win = win; redraw(tl); }

  function playerAt(tl) { return tl.active || tl.players[0] || null; }

  function seekPlayer(a, t) { whenReady(a, function () { a.currentTime = t; }); }

  var SEEK_TOLERANCE_S = 0.2;
  var SEEK_WAIT_MS = 3000;
  var CANNOT_SEEK = "this server can't seek; serve with triage_review_serve.py";

  /** Run `fn` once the player knows its duration, loading it first where it has not. */
  function whenReady(a, fn) {
    if (a.readyState >= 1) { fn(); return; }
    a.addEventListener('loadedmetadata', fn, { once: true });
    a.preload = 'auto';
    a.load();
  }

  /**
   * Play the task extent on one player: seek to its start, wait for `seeked`, then play until its end.
   * Where the player lands away from the start (a server that cannot answer byte ranges), it does not
   * play, and `onCannotSeek(landedAt, start)` is called instead.
   */
  function playSpan(tl, a, onCannotSeek, waitMs) {
    if (!tl.extent) return;
    var start = tl.extent[0];
    tl.players.forEach(function (b) { if (b !== a && !b.paused) b.pause(); });
    tl.active = a;
    tl.spanEnd = null;
    whenReady(a, function () {
      var settled = false;
      var timer = null;
      var after = function () {
        if (settled) return;
        settled = true;
        if (timer != null) clearTimeout(timer);
        a.removeEventListener('seeked', after);
        if (tl.active !== a) return;
        if (Math.abs(a.currentTime - start) > SEEK_TOLERANCE_S) {
          if (onCannotSeek) onCannotSeek(a.currentTime, start);
          return;
        }
        tl.spanEnd = tl.extent[1];
        var played = a.play();
        if (played && played.catch) played.catch(function () { /* the reviewer can press play */ });
      };
      a.addEventListener('seeked', after);
      timer = setTimeout(after, waitMs == null ? SEEK_WAIT_MS : waitMs);
      a.currentTime = start;
    });
  }

  function stopAtSpanEnd(tl) {
    var a = tl.active;
    if (a && tl.spanEnd != null && a.currentTime >= tl.spanEnd) { a.pause(); tl.spanEnd = null; }
  }

  function watchPlayhead(tl) {
    function step() {
      var a = tl.active;
      stopAtSpanEnd(tl);
      redraw(tl);
      if (a && !a.paused) requestAnimationFrame(step);
    }
    tl.players.forEach(function (a) {
      a.addEventListener('play', function () {
        tl.players.forEach(function (b) { if (b !== a) b.pause(); });
        if (tl.active !== a) tl.spanEnd = null;
        tl.active = a;
        requestAnimationFrame(step);
      });
      a.addEventListener('timeupdate', function () { if (tl.active === a) stopAtSpanEnd(tl); });
      a.addEventListener('seeked', function () { redraw(tl); });
      a.addEventListener('pause', function () { redraw(tl); });
    });
  }

  function now(tl) { var a = tl.active; return a ? a.currentTime : null; }

  function zoomControls(tl) {
    var bar = el('div', 'rv-zoom');
    function button(label, title, fn) {
      var b = el('button', null, label); b.type = 'button'; b.title = title; b.onclick = fn; bar.appendChild(b);
    }
    var mid = function () { return (tl.win.t0 + tl.win.t1) / 2; };
    button('−', 'zoom out', function () { setWindow(tl, zoomWindow(tl.win, tl.d, 2, mid())); });
    button('+', 'zoom in', function () { setWindow(tl, zoomWindow(tl.win, tl.d, 0.5, mid())); });
    button('◀', 'pan left', function () { setWindow(tl, panWindow(tl.win, tl.d, -(tl.win.t1 - tl.win.t0) / 2)); });
    button('▶', 'pan right', function () { setWindow(tl, panWindow(tl.win, tl.d, (tl.win.t1 - tl.win.t0) / 2)); });
    button('all', 'show the whole recording', function () { setWindow(tl, { t0: 0, t1: tl.d }); });
    if (tl.extent) {
      button('task span', 'show the task extent', function () {
        var pad = 0.05 * (tl.extent[1] - tl.extent[0]);
        var t0 = Math.max(0, tl.extent[0] - pad), t1 = Math.min(tl.d, tl.extent[1] + pad);
        setWindow(tl, { t0: t0, t1: Math.max(t1, t0 + Math.min(MIN_SPAN_S, tl.d)) });
      });
    }
    var where = el('span', 'rv-note');
    bar.appendChild(where);
    tl.redraw.push(function () { where.textContent = tl.win.t0.toFixed(2) + '–' + tl.win.t1.toFixed(2) + ' s'; });
    bar.appendChild(el('span', 'rv-note', 'ctrl/⌘-wheel zooms, drag pans, click seeks'));
    return bar;
  }

  /** Wheel-zoom, drag-pan and click-to-seek on one element of the time view. */
  function wireTime(tl, node) {
    function timeAt(e) {
      var box = node.getBoundingClientRect();
      return tl.win.t0 + ((e.clientX - box.left) / Math.max(1, box.width)) * (tl.win.t1 - tl.win.t0);
    }
    node.addEventListener('wheel', function (e) {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        setWindow(tl, zoomWindow(tl.win, tl.d, e.deltaY > 0 ? 1.25 : 0.8, timeAt(e)));
      } else if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        e.preventDefault();
        var box = node.getBoundingClientRect();
        setWindow(tl, panWindow(tl.win, tl.d, (e.deltaX / Math.max(1, box.width)) * (tl.win.t1 - tl.win.t0)));
      }
    }, { passive: false });
    var drag = null;
    node.addEventListener('pointerdown', function (e) {
      if (e.button !== 0) return;
      var tok = e.target.closest ? e.target.closest('.rv-tok') : null;
      drag = { x: e.clientX, win: tl.win, moved: false, at: tok ? Number(tok.dataset.t) : null };
      if (node.setPointerCapture) node.setPointerCapture(e.pointerId);
    });
    node.addEventListener('pointermove', function (e) {
      if (!drag) return;
      var dx = e.clientX - drag.x;
      if (Math.abs(dx) > 3) drag.moved = true;
      if (!drag.moved) return;
      var box = node.getBoundingClientRect();
      setWindow(tl, panWindow(drag.win, tl.d, (-dx / Math.max(1, box.width)) * (drag.win.t1 - drag.win.t0)));
    });
    node.addEventListener('pointerup', function (e) {
      if (!drag) return;
      var moved = drag.moved, at = drag.at;
      drag = null;
      if (moved) return;
      var a = playerAt(tl);
      var t = at != null ? at : timeAt(e);
      if (a) { tl.active = a; tl.spanEnd = null; seekPlayer(a, Math.max(0, Math.min(tl.d, t))); }
    });
    node.addEventListener('pointercancel', function () { drag = null; });
  }

  function spectrogramSection(r, rec) {
    var sec = el('section', 'rv-time');
    sec.appendChild(el('h3', null, 'spectrogram' + (rec.spec_stream ? ' · ' + rec.spec_stream : '') +
      (rec.speech && (rec.speech.own || []).length ? ' and ASR timelines' : '')));
    var tl = timelineOf(r.duration_s, rec.extent);
    sec.timeline = tl;
    sec.appendChild(zoomControls(tl));
    var frame = el('div', 'rv-timeline');
    sec.appendChild(frame);
    if (rec.spec) {
      var spec = state.index.build.spec;
      var levels = unpackSpec(rec.spec, spec.frames, spec.bands);
      var canvas = el('canvas', 'rv-spec');
      frame.appendChild(canvas);
      wireTime(tl, canvas);
      tl.redraw.push(function () { drawSpectrogram(canvas, levels, spec, tl, rec); });
    } else {
      frame.appendChild(el('p', 'rv-note', 'no stream decoded for this recording'));
    }
    if (rec.speech) frame.appendChild(lanesOf(tl, rec.speech));
    var head = el('div', 'rv-playhead');
    frame.appendChild(head);
    tl.redraw.push(function () {
      var t = now(tl);
      var at = t == null ? null : placeSpan(t, t, tl.win);
      head.hidden = !at;
      if (at) head.style.left = (at.left * 100) + '%';
    });
    var legend = el('div', 'rv-legend');
    [['extent', 'accent'], ['task events', 'good'], ['activity', 'teal'], ['issues', 'bad']].forEach(function (p) {
      var s = el('span'); var sw = el('i'); sw.style.background = Theme.color(p[1]);
      s.appendChild(sw); s.appendChild(document.createTextNode(p[0])); legend.appendChild(s);
    });
    sec.appendChild(legend);
    requestAnimationFrame(function () { redraw(tl); });
    return sec;
  }

  /** The consensus lane and one lane per ASR model, on the spectrogram's time axis. */
  function lanesOf(tl, sp) {
    var box = el('div', 'rv-lanes');
    var words = sp.words || [];
    var lanes = [];
    if (words.length) {
      lanes.push({
        label: 'consensus', cls: 'consensus',
        tokens: words.map(function (w, i) { return [w[3], w[4], null, i]; }).filter(function (t) { return t[0] != null; }),
      });
    }
    modelLanes(sp).forEach(function (lane) { lanes.push({ label: lane.label, cls: 'model', tokens: lane.tokens }); });
    lanes.forEach(function (lane) {
      var row = el('div', 'rv-lane ' + lane.cls);
      row.appendChild(el('div', 'rv-lane-name', lane.label + ' · ' + lane.tokens.length + ' token' +
        (lane.tokens.length === 1 ? '' : 's')));
      var track = el('div', 'rv-lane-track');
      row.appendChild(track);
      box.appendChild(row);
      wireTime(tl, track);
      var nodes = lane.tokens.map(function (t) {
        var tok = el('span', 'rv-tok');
        tok.dataset.t = t[0];
        if (t[3] != null) {
          var word = words[t[3]];
          tok.textContent = word[5] || '';
          var band = agreementBand(word[0]);
          if (band) tok.classList.add(band);
          if (word[1] === 'insertion') tok.classList.add('unmatched');
          tok.title = alternatives(sp, word) + '\n' + word[3].toFixed(2) + '–' + word[4].toFixed(2) + ' s';
        } else {
          tok.textContent = t[2];
          tok.title = lane.label + ': ' + t[2] + '\n' + t[0].toFixed(2) + '–' + t[1].toFixed(2) + ' s';
        }
        track.appendChild(tok);
        return { node: tok, start: t[0], end: t[1] };
      });
      tl.redraw.push(function () {
        nodes.forEach(function (n) {
          var at = placeSpan(n.start, n.end, tl.win);
          n.node.hidden = !at;
          if (!at) return;
          n.node.style.left = (at.left * 100) + '%';
          n.node.style.width = 'max(2px, ' + (at.width * 100) + '%)';
        });
      });
    });
    return box;
  }

  function drawSpectrogram(canvas, levels, spec, tl, rec) {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    var top = 22, bottom = h - 16, cellH = (bottom - top) / spec.bands;
    var d = tl.d, win = tl.win, span = win.t1 - win.t0;
    var xOf = function (t) { return Math.max(0, Math.min(w, ((t - win.t0) / span) * w)); };
    var frameS = d / spec.frames;
    var cellW = (frameS / span) * w;
    var shades = ['panel', 'dim', 'muted', 'ink-2'].map(function (t) { return Theme.color(t); });
    var f0 = Math.max(0, Math.floor(win.t0 / frameS)), f1 = Math.min(spec.frames, Math.ceil(win.t1 / frameS));
    for (var f = f0; f < f1; f++) {
      var x = ((f * frameS - win.t0) / span) * w;
      for (var b = 0; b < spec.bands; b++) {
        ctx.fillStyle = shades[levels[f * spec.bands + b]];
        ctx.fillRect(x, bottom - (b + 1) * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
    var ov = rec.overlay || {};
    function bars(spans, y, height, colour) {
      ctx.fillStyle = colour;
      (spans || []).forEach(function (s) {
        if (s[1] < win.t0 || s[0] > win.t1) return;
        ctx.fillRect(xOf(s[0]), y, Math.max(1.5, xOf(s[1]) - xOf(s[0])), height);
      });
    }
    if (rec.extent) {
      ctx.save(); ctx.globalAlpha = 0.18; ctx.fillStyle = Theme.color('accent');
      ctx.fillRect(xOf(rec.extent[0]), top, xOf(rec.extent[1]) - xOf(rec.extent[0]), bottom - top);
      ctx.restore();
      bars([rec.extent], 0, 4, Theme.color('accent'));
    }
    bars(ov.events, 6, 5, Theme.color('good'));
    bars(ov.activity, 13, 3, Theme.color('teal'));
    bars(ov.issues, bottom + 2, 6, Theme.color('bad'));
    ctx.fillStyle = Theme.color('ink-3');
    ctx.font = '10px ui-monospace, Menlo, monospace';
    ctx.textBaseline = 'bottom';
    ticks(win, 6).forEach(function (t) {
      var x = ((t - win.t0) / span) * w;
      ctx.textAlign = x < 20 ? 'left' : x > w - 20 ? 'right' : 'center';
      ctx.fillText((span < 2 ? t.toFixed(2) : t.toFixed(1)) + ' s', x, h);
    });
  }

  /** A stream's URL: under the corpus root through audio_base, outside it through the source route. */
  function streamUrl(path, build) {
    var encoded = path.split('/').map(encodeURIComponent).join('/');
    if (path.charAt(0) === '/') return (build.source_base || '/_source') + encoded;
    return build.audio_base + encoded;
  }

  /** The tracks a recording plays, in page order: each `{name, timeline}`, whether it runs on the recording's clock. */
  function audioTracks(rec) {
    var streams = rec.streams || {};
    var out = [];
    ['recording', 'enhanced'].forEach(function (n) { if (streams[n]) out.push({ name: n, timeline: true }); });
    if (rec.speech && streams.redacted) out.push({ name: 'redacted', timeline: true });
    if (streams.released) out.push({ name: 'released', timeline: false });
    return out;
  }

  function audioSection(rec, tl) {
    var sec = el('section', 'rv-audio');
    sec.appendChild(el('h3', null, 'audio'));
    if (!state.served) {
      sec.appendChild(el('p', 'rv-note', 'Audio plays when this page is served from the cluster over an ssh tunnel; ' +
        'opened from a file it shows the quantised spectrogram only.'));
      return sec;
    }
    var base = state.index.build.audio_base;
    if (rec.extent) {
      sec.appendChild(el('p', 'rv-note', 'task extent ' + rec.extent[0].toFixed(2) + '–' + rec.extent[1].toFixed(2) +
        ' s, marked on each track; “play task span” plays only that'));
    }
    audioTracks(rec).forEach(function (track) {
      var wrap = el('div', 'rv-track');
      wrap.dataset.stream = track.name;
      wrap.appendChild(el('span', 'rv-note', track.name));
      var a = el('audio'); a.controls = true; a.preload = 'none'; a.src = streamUrl(rec.streams[track.name], state.index.build);
      wrap.appendChild(a);
      if (track.timeline && tl) {
        tl.players.push(a);
        wrap.appendChild(spanStrip(tl, a));
        if (rec.extent) {
          var b = el('button', 'rv-play-span', '▶ play task span'); b.type = 'button';
          var note = el('span', 'rv-note rv-seek-note'); note.hidden = true;
          b.onclick = function () {
            note.hidden = true;
            playSpan(tl, a, function (at) {
              note.textContent = CANNOT_SEEK + ' (asked for ' + tl.extent[0].toFixed(2) + ' s, got ' + at.toFixed(2) + ' s)';
              note.hidden = false;
            });
          };
          wrap.appendChild(b);
          wrap.appendChild(note);
        }
      }
      sec.appendChild(wrap);
    });
    if (tl) watchPlayhead(tl);
    if (rec.figure) {
      var link = el('a', null, 'full figure');
      link.href = base + rec.figure.split('/').map(encodeURIComponent).join('/');
      link.target = '_blank'; link.rel = 'noopener';
      sec.appendChild(link);
    }
    return sec;
  }

  /** A track's whole-recording strip: the task extent marked, its own playhead, click to seek. */
  function spanStrip(tl, a) {
    var strip = el('div', 'rv-strip');
    if (tl.extent) {
      var ext = el('i', 'ext');
      ext.style.left = (tl.extent[0] / tl.d * 100) + '%';
      ext.style.width = ((tl.extent[1] - tl.extent[0]) / tl.d * 100) + '%';
      strip.appendChild(ext);
    }
    var head = el('i', 'head');
    strip.appendChild(head);
    strip.onclick = function (e) {
      var box = strip.getBoundingClientRect();
      tl.active = a; tl.spanEnd = null;
      seekPlayer(a, ((e.clientX - box.left) / Math.max(1, box.width)) * tl.d);
    };
    tl.redraw.push(function () {
      head.hidden = tl.active !== a;
      head.style.left = (Math.min(1, a.currentTime / tl.d) * 100) + '%';
    });
    return strip;
  }

  function evidenceSection(rec) {
    var sec = el('section');
    sec.appendChild(el('h3', null, 'evidence (decisive in bold)'));
    var items = rec.evidence || [];
    if (!items.length) { sec.appendChild(el('p', 'rv-note', 'the fold recorded no evidence items')); return sec; }
    var t = el('table', 'rv-ev');
    var hr = el('tr');
    ['item', 'group', 'value', 'unit', 'against', 'effect'].forEach(function (h) { hr.appendChild(el('th', null, h)); });
    t.appendChild(hr);
    items.slice().sort(function (a, b) { return (b.decisive ? 1 : 0) - (a.decisive ? 1 : 0); }).forEach(function (it) {
      var tr = el('tr', it.decisive ? 'decisive' : null);
      tr.appendChild(el('td', null, it.name));
      tr.appendChild(el('td', null, it.group || ''));
      tr.appendChild(el('td', 'v', fmt(it.value)));
      tr.appendChild(el('td', null, it.unit || ''));
      tr.appendChild(el('td', 'v', it.comparison ? it.comparison + ' ' + fmt(it.threshold) : ''));
      tr.appendChild(el('td', null, it.effect || ''));
      t.appendChild(tr);
    });
    sec.appendChild(t);
    return sec;
  }

  /** A recogniser's label: its model id, with the pipeline's source name where that differs. */
  function modelLabel(m) {
    if (!m) return 'unknown model';
    if (m.model_id && m.model_id !== m.source) return m.model_id + (m.source ? ' (' + m.source + ')' : '');
    return m.source || m.model_id || 'unknown model';
  }

  /** The agreement band a consensus word's shading is drawn from. */
  function agreementBand(a) {
    if (a === null || a === undefined) return null;
    if (a >= 0.999) return 'ag-all';
    if (a >= 0.66) return 'ag-most';
    if (a >= 0.34) return 'ag-some';
    return 'ag-few';
  }

  /** What each model read at one consensus word, one line per model. */
  function alternatives(sp, word) {
    var models = sp.models || [];
    var head = word[1] + ', agreement ' + (word[0] === null ? '—' : Math.round(word[0] * 100) + '%');
    return [head].concat((word[2] || []).map(function (r, k) {
      return modelLabel(models[k]) + ': ' + (r === null || r === undefined ? '(no word)' : r);
    })).join('\n');
  }

  /**
   * Shade each consensus word by its agreement and show the models' readings on hover or tap. A word
   * one model alone produced is marked unmatched; a reading a minority gave is shown beside the word.
   */
  function decorateWords(p, sp, detail) {
    var words = sp.words || [];
    p.querySelectorAll('span.w[data-i]').forEach(function (span) {
      var word = words[+span.getAttribute('data-i')];
      if (!word) return;
      var band = agreementBand(word[0]);
      if (band) span.classList.add(band);
      var readers = readersOf(sp, word);
      var text = alternatives(sp, word);
      if (word[1] === 'insertion') {
        span.classList.add('unmatched');
        text = 'only ' + readers.join(', ') + '\n' + text;
      }
      span.title = text;
      span.tabIndex = 0;
      span.onclick = function (e) { e.stopPropagation(); detail.textContent = span.title; detail.hidden = false; };
      var after = span;
      otherReadings(sp, word, word[5] != null ? word[5] : span.textContent).forEach(function (other) {
        var alt = el('span', 'w-alt' + (other.models.length === 1 ? ' unmatched' : ''), other.text);
        alt.title = (other.models.length === 1 ? 'only ' : '') + other.models.join(', ') + ': ' + other.text;
        alt.tabIndex = 0;
        alt.onclick = function (e) { e.stopPropagation(); detail.textContent = alt.title + '\n' + text; detail.hidden = false; };
        after.after(alt);
        after = alt;
      });
    });
  }

  function speechSection(sp) {
    var sec = el('section', 'rv-speech');
    sec.appendChild(el('h3', null, 'transcript, PII and redactions'));
    var shown = sp.shown || {};
    var models = sp.models || [];
    var label;
    if (shown.kind === 'consensus') {
      label = 'consensus of ' + models.length + ' ASR model' + (models.length === 1 ? '' : 's') + ': ' +
        models.map(modelLabel).join(', ') + ' — shaded by agreement; hover or tap a word for each model’s reading';
    } else if (shown.kind === 'model') {
      var m = models.filter(function (x) { return x.source === shown.source; })[0] || { source: shown.source };
      label = 'individual ASR model: ' + modelLabel(m) + ' (the consensus holds no words)';
    } else {
      label = 'no ASR model left a word';
    }
    sec.appendChild(el('div', 'rv-tx-label', label));
    var p = el('p', 'text');
    p.innerHTML = sp.html || '<span class="empty">no words</span>';
    sec.appendChild(p);
    if (shown.kind === 'consensus') {
      var legend = el('div', 'rv-legend rv-ag-legend');
      [['ag-all', 'all agree'], ['ag-most', '≥ 2/3'], ['ag-some', '≥ 1/3'], ['ag-few', 'under 1/3'],
        ['unmatched', 'one model only']].forEach(function (b) {
        var item = el('span'); item.appendChild(el('i', b[0])); item.appendChild(document.createTextNode(b[1]));
        legend.appendChild(item);
      });
      sec.appendChild(legend);
      var detail = el('pre', 'rv-word-detail');
      detail.hidden = true;
      sec.appendChild(detail);
      decorateWords(p, sp, detail);
    }
    var own = sp.own || [];
    if (own.length) {
      var box = el('details', 'rv-own');
      box.appendChild(el('summary', null, 'individual ASR transcripts (' + own.length + ')'));
      own.forEach(function (o) {
        var row = el('div', 'rv-own-row');
        row.appendChild(el('div', 'rv-own-model', modelLabel(o)));
        row.appendChild(el('div', 'rv-own-text', o.text || '(no words)'));
        if ((o.words || []).length) {
          var timed = el('div', 'rv-own-words');
          o.words.forEach(function (w) {
            var item = el('span', 'rv-own-w');
            item.appendChild(el('small', null, w[0] == null ? 'untimed' : w[0].toFixed(2) + '–' + w[1].toFixed(2)));
            item.appendChild(document.createTextNode(' ' + w[2]));
            timed.appendChild(item);
          });
          row.appendChild(timed);
        }
        box.appendChild(row);
      });
      sec.appendChild(box);
    }
    var lines = [];
    if (sp.release_ground) lines.push('release ground: ' + sp.release_ground);
    if (sp.why) lines.push('verdict: ' + sp.why);
    if (sp.redact_why) lines.push('REDACT: ' + sp.redact_why);
    if (sp.condition_kind) lines.push('held condition: ' + sp.condition_kind);
    if (sp.names_proposed && sp.names_proposed.length) lines.push('names proposed for release: ' + sp.names_proposed.join(', '));
    var llm = sp.llm || {};
    var shown = Object.keys(llm).filter(function (k) { return LLM_HIDDEN.indexOf(k) < 0; })
      .map(function (k) { return k + ' ' + JSON.stringify(llm[k]); });
    if (shown.length) lines.push('LLM reviewer: ' + shown.join(' · '));
    lines.forEach(function (l) { sec.appendChild(el('div', 'rv-note', l)); });
    if (sp.pii && sp.pii.length) {
      var t = el('table', 'rv-ev');
      var hr = el('tr');
      ['category', 'detector', 'text', 'in stimulus'].forEach(function (h) { hr.appendChild(el('th', null, h)); });
      t.appendChild(hr);
      sp.pii.forEach(function (f) {
        var tr = el('tr');
        [f.c, f.s, f.h, f.stim === 1 ? 'yes' : f.stim === 0 ? 'no' : '—'].forEach(function (v) { tr.appendChild(el('td', null, v)); });
        t.appendChild(tr);
      });
      sec.appendChild(t);
    }
    return sec;
  }

  function renderDecisions() {
    var body = $('rv-dec-body');
    body.innerHTML = '';
    var stems = Object.keys(state.decisions).sort();
    var counts = {};
    stems.forEach(function (s) {
      var d = state.decisions[s];
      var r = state.byStem[s];
      counts[d.verdict] = (counts[d.verdict] || 0) + 1;
      var tr = el('tr');
      tr.appendChild(el('td', 's', s));
      tr.appendChild(el('td', null, r ? r.declared_family || '' : 'not in this page'));
      tr.appendChild(el('td', null, r ? r.verdict + (r.reason ? ' (' + r.reason + ')' : '') : ''));
      tr.appendChild(el('td', null, d.verdict));
      tr.appendChild(el('td', null, d.note || ''));
      tr.appendChild(el('td', null, d.at || ''));
      var x = el('button', null, 'remove'); x.type = 'button';
      x.onclick = function () { delete state.decisions[s]; saveDecisions(); renderDecisions(); renderList(); };
      var td = el('td'); td.appendChild(x); tr.appendChild(td);
      if (r) tr.querySelector('td.s').onclick = function () { showTab('review'); openRecording(r.index); };
      body.appendChild(tr);
    });
    $('rv-dec-summary').textContent = stems.length + ' decisions' + (stems.length ? ' — ' +
      VERDICTS.map(function (v) { return v + ' ' + (counts[v] || 0); }).join(', ') : '');
  }

  function exportJson() {
    var payload = exportPayload(state.byStem, state.decisions, state.index.build.id, new Date().toISOString());
    var blob = new Blob([JSON.stringify(payload, null, 1)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'triage_review_' + state.index.build.id + '.json';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }

  function importJson(e) {
    var file = e.target.files && e.target.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var got = importDecisions(JSON.parse(reader.result));
        Object.keys(got).forEach(function (s) { state.decisions[s] = got[s]; });
        saveDecisions(); renderDecisions(); renderList();
      } catch (err) { $('rv-dec-summary').textContent = 'could not read that file: ' + err.message; }
    };
    reader.readAsText(file);
    e.target.value = '';
  }

  if (typeof window !== 'undefined' && window.REVIEW_INDEX && typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
  }

  return {
    EXPORT_SCHEMA: EXPORT_SCHEMA,
    EXPORT_VERSION: EXPORT_VERSION,
    decodeRows: decodeRows,
    columnSpecs: columnSpecs,
    evidenceGroups: evidenceGroups,
    axesFor: axesFor,
    searchMask: searchMask,
    unpackSpec: unpackSpec,
    entryFor: entryFor,
    exportPayload: exportPayload,
    importDecisions: importDecisions,
    shard: shard,
    modelLabel: modelLabel,
    agreementBand: agreementBand,
    alternatives: alternatives,
    streamUrl: streamUrl,
    zoomWindow: zoomWindow,
    panWindow: panWindow,
    placeSpan: placeSpan,
    ticks: ticks,
    tokenKey: tokenKey,
    readersOf: readersOf,
    otherReadings: otherReadings,
    modelLanes: modelLanes,
    audioTracks: audioTracks,
    timelineOf: timelineOf,
    playSpan: playSpan,
    stopAtSpanEnd: stopAtSpanEnd,
    CANNOT_SEEK: CANNOT_SEEK,
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = ReviewPage;
