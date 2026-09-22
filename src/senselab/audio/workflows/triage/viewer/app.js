// Wiring: pick the parquet off disk, read what the corpus view needs, draw it, and read a
// recording's binary blocks when its line is selected.
//
// Nothing here is fetched over the network. The page is opened from file:// and the parquet is
// handed to it by the reader.
//
// specs/20260922-compact-recording-vectors/views.md carries what each read costs and why the
// block cache is shaped the way it is.

'use strict';

(function () {
  var BLOCK_COLUMNS = [
    'run_dir',
    'wave_minmax', 'wave_peak', 'env_dbfs', 'floor_dbfs', 'continuity',
    'spans', 'span_labels', 'span_label_name', 'span_squim',
    'asr_words', 'asr_word_text', 'pii_marks', 'pii_category',
    'branch_lanes', 'branch_lane_role',
    'm_ddk_position_realised_mass', 'm_ddk_cv_instrument_reading', 'm_ddk_cv_instrument_reading_width',
  ];

  var state = {
    file: null,
    metadata: null,
    rows: null,
    axes: SchemaAxes.DEFAULT_AXES.slice(),
    view: null,
    recView: null,
    cache: null,
    caching: true,
    warm: null,
    current: -1,
  };

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function status(text, kind) {
    // Two places: the landing carries it before the corpus opens, the topbar after.
    ['status', 'status-2'].forEach(function (id) {
      var n = $(id);
      if (!n) return;
      n.textContent = text;
      n.className = 'status' + (kind ? ' ' + kind : '');
    });
  }

  // ---------------------------------------------------------------- reading the parquet

  /** An AsyncBuffer over a local File: every read is a disk slice, never a network fetch. */
  function bufferFor(file) {
    return {
      byteLength: file.size,
      slice: function (start, end) {
        return file.slice(start, end === undefined ? file.size : end).arrayBuffer();
      },
    };
  }

  async function openFile(file) {
    state.file = file;
    state.cache = null;
    state.current = -1;
    status('reading the footer of ' + file.name + '…');
    var buf = bufferFor(file);
    var t0 = performance.now();
    state.metadata = await Hyparquet.parquetMetadataAsync(buf);
    state.buffer = buf;
    var columns = SchemaAxes.corpusColumns();
    status('reading ' + columns.length + ' corpus columns…');
    var rows = await Hyparquet.parquetReadObjects({
      file: buf, metadata: state.metadata, columns: columns,
      compressors: Hyparquet.compressors, utf8: false,
    });
    var ms = performance.now() - t0;
    var version = rows.length ? rows[0].schema_version : null;
    if (version !== SchemaDecode.SCHEMA_VERSION) {
      status('schema_version is ' + version + ', this page decodes ' + SchemaDecode.SCHEMA_VERSION +
        ' — refusing to draw rather than mis-decode.', 'bad');
      return;
    }
    // An unread column would read as absent on every row rather than as an error.
    var unread = columns.filter(function (c) { return !rows.length || !(c in rows[0]); });
    if (unread.length) {
      status('the parquet has no column ' + unread.join(', ') + ' — refusing to draw an axis that ' +
        'would read as absent everywhere.', 'bad');
      return;
    }
    rows.forEach(function (r, i) { r.__i = i; });
    state.rows = rows;
    $('file-info').textContent = file.name + ' · ' + rows.length.toLocaleString() + ' rows · ' +
      (file.size / 1e6).toFixed(1) + ' MB on disk · schema_version ' + version;
    status('corpus read in ' + ms.toFixed(0) + ' ms (' + columns.length + ' columns; the binary blocks are not read yet)', 'good');
    $('landing').hidden = true;
    $('main').hidden = false;
    buildCorpus();
  }

  /** Every binary block for every row, read one column at a time and kept. */
  async function warmCache(onProgress) {
    if (state.cache) return state.cache;
    if (state.warm) return state.warm;
    state.warm = (async function () {
      var cache = {};
      for (var i = 0; i < BLOCK_COLUMNS.length; i++) {
        var name = BLOCK_COLUMNS[i];
        if (onProgress) onProgress(i, BLOCK_COLUMNS.length, name);
        await new Promise(function (r) { setTimeout(r, 0); });
        var got = await Hyparquet.parquetReadObjects({
          file: state.buffer, metadata: state.metadata, columns: [name],
          compressors: Hyparquet.compressors, utf8: false,
        });
        var column = new Array(got.length);
        for (var r2 = 0; r2 < got.length; r2++) column[r2] = got[r2][name];
        cache[name] = column;
      }
      state.cache = cache;
      state.warm = null;
      if (onProgress) onProgress(BLOCK_COLUMNS.length, BLOCK_COLUMNS.length, null);
      return cache;
    })();
    return state.warm;
  }

  /** One recording's block columns, from the cache when warm and by a narrow read when not. */
  async function readBlocks(index) {
    if (state.cache) {
      var out = {};
      BLOCK_COLUMNS.forEach(function (n) { out[n] = state.cache[n][index]; });
      return out;
    }
    var got = await Hyparquet.parquetReadObjects({
      file: state.buffer, metadata: state.metadata, columns: BLOCK_COLUMNS,
      rowStart: index, rowEnd: index + 1, compressors: Hyparquet.compressors, utf8: false,
    });
    return got[0];
  }

  // ---------------------------------------------------------------- the corpus view

  function buildCorpus() {
    var view = new CorpusView($('pc'), $('pc-overlay'));
    state.view = view;
    view.setRows(state.rows);
    view.setAxes(state.axes);
    view.onSelectionChange = function (n) {
      $('selcount').textContent = n.toLocaleString() + ' of ' + state.rows.length.toLocaleString() + ' recordings drawn';
      renderAxisCaptions();
      renderList();
    };
    view.applyBrushes();
    renderAxisRail();
    renderControls();
    wirePointer(view);
    view.draw();
    window.addEventListener('resize', function () { view.draw(); });
  }

  function columnOptions(select, chosen) {
    var groups = {};
    SchemaAxes.CATALOGUE.forEach(function (c) {
      (groups[c.group] = groups[c.group] || []).push(c);
    });
    Object.keys(groups).forEach(function (g) {
      var og = document.createElement('optgroup');
      og.label = g;
      groups[g].forEach(function (c) {
        var o = document.createElement('option');
        o.value = c.name;
        o.textContent = c.label + (c.assignable ? '' : '  — not assignable');
        if (!c.assignable) { o.disabled = true; o.title = c.reason; }
        if (c.name === chosen) o.selected = true;
        og.appendChild(o);
      });
      select.appendChild(og);
    });
  }

  function renderAxisRail() {
    var rail = $('axis-rail');
    rail.innerHTML = '';
    state.axes.forEach(function (name, i) {
      var cell = el('div', 'axis-cell');
      var head = el('div', 'axis-head');
      var sel = el('select', 'axis-select');
      columnOptions(sel, name);
      sel.onchange = function () { state.axes[i] = sel.value; state.view.setAxes(state.axes); renderAxisRail(); state.view.draw(); };
      head.appendChild(sel);
      var btns = el('div', 'axis-btns');
      btns.appendChild(button('◀', 'move left', function () { swap(i, i - 1); }, i === 0));
      btns.appendChild(button('▶', 'move right', function () { swap(i, i + 1); }, i === state.axes.length - 1));
      btns.appendChild(button('✕', 'remove axis', function () {
        state.axes.splice(i, 1); state.view.setAxes(state.axes); renderAxisRail(); state.view.draw();
      }, state.axes.length <= 1));
      head.appendChild(btns);
      cell.appendChild(head);
      cell.appendChild(el('div', 'axis-caption', ''));
      var brush = el('div', 'axis-brush', '');
      cell.appendChild(brush);
      rail.appendChild(cell);
    });
    if (state.axes.length < SchemaAxes.MAX_AXES) {
      var add = el('div', 'axis-cell add');
      var b = el('button', 'add-btn', '+ add axis (' + state.axes.length + '/' + SchemaAxes.MAX_AXES + ')');
      b.onclick = function () {
        var used = state.axes;
        var next = SchemaAxes.CATALOGUE.filter(function (c) { return c.assignable && used.indexOf(c.name) < 0; })[0];
        if (!next) return;
        state.axes.push(next.name); state.view.setAxes(state.axes); renderAxisRail(); state.view.draw();
      };
      add.appendChild(b);
      rail.appendChild(add);
    }
    renderAxisCaptions();
  }

  function swap(a, b) {
    if (b < 0 || b >= state.axes.length) return;
    var t = state.axes[a]; state.axes[a] = state.axes[b]; state.axes[b] = t;
    state.view.setAxes(state.axes); renderAxisRail(); state.view.draw();
  }

  function button(label, title, fn, disabled) {
    var b = el('button', 'icon-btn', label);
    b.title = title;
    b.disabled = !!disabled;
    b.onclick = fn;
    return b;
  }

  function renderAxisCaptions() {
    var cells = $('axis-rail').querySelectorAll('.axis-cell:not(.add)');
    state.view.summaries.forEach(function (s, i) {
      var cell = cells[i];
      if (!cell) return;
      var live = state.view.countPresent(i);
      var text = SchemaAxes.caption(s);
      cell.querySelector('.axis-caption').textContent = text;
      var brush = cell.querySelector('.axis-brush');
      brush.innerHTML = '';
      var b = state.view.brushes[s.name];
      var chip = el('span', 'chip' + (b ? ' on' : ''),
        b ? describeBrush(s, b) : 'drag the axis to brush');
      brush.appendChild(chip);
      if (b) {
        var clear = el('button', 'chip-x', '✕');
        clear.onclick = function () { state.view.setBrush(s.name, null); renderAxisCaptions(); };
        brush.appendChild(clear);
      }
      if (s.kind === 'numeric') {
        var log = s.scale === 'log';
        var scaleBtn = el('button', 'chip scale' + (log ? ' on' : ''), log ? 'log10' : 'linear');
        scaleBtn.title = s.logCapable
          ? 'switch the value scale; the tick labels stay real values'
          : 'log needs a domain above zero, and this one reaches ' + CorpusView.formatNumber(s.min);
        scaleBtn.disabled = !s.logCapable;
        scaleBtn.onclick = function () {
          state.view.setScale(s.name, log ? 'linear' : 'log');
          state.view.draw();
          renderAxisCaptions();
        };
        brush.appendChild(scaleBtn);
      }
      var abs = el('button', 'chip abs' + (b && b.absent ? ' on' : ''),
        'absent ' + live.absent.toLocaleString());
      abs.title = 'null on this axis: ' + (s.col.nullMeans || 'no reading') +
        '. Click to cycle exclude → include → only.';
      abs.onclick = function () {
        var cur = state.view.brushes[s.name] || {};
        var next = cur.absent === 'include' ? 'only' : cur.absent === 'only' ? undefined : 'include';
        var merged = Object.assign({}, cur, { absent: next });
        var empty = merged.absent === undefined && merged.lo == null && !(merged.terms && merged.terms.length);
        state.view.setBrush(s.name, empty ? null : merged);
        renderAxisCaptions();
      };
      brush.appendChild(abs);
    });
  }

  function describeBrush(s, b) {
    if (b.absent === 'only') return 'absent only';
    if (s.kind === 'numeric' && b.lo != null) {
      return CorpusView.formatNumber(b.lo) + ' … ' + CorpusView.formatNumber(b.hi) +
        (b.absent === 'include' ? ' + absent' : '');
    }
    if (b.terms && b.terms.length) {
      return b.terms.length === 1 ? String(b.terms[0]) : b.terms.length + ' categories';
    }
    return b.absent === 'include' ? 'absent included' : 'brushed';
  }

  function renderControls() {
    var colour = $('colour-by');
    colour.innerHTML = '';
    columnOptions(colour, state.view.colourBy);
    colour.onchange = function () { state.view.colourBy = colour.value; state.view.buildColourMap(); state.view.draw(); renderLegend(); };
    $('opacity').oninput = function () {
      var v = Number($('opacity').value);
      state.view.opacity = v === 0 ? null : v / 1000;
      $('opacity-label').textContent = v === 0 ? 'auto' : (v / 1000).toFixed(3);
      state.view.draw();
    };
    $('stubs').onchange = function () { state.view.showStubs = $('stubs').checked; state.view.draw(); };
    $('clear-brushes').onclick = function () { state.view.clearBrushes(); renderAxisCaptions(); };
    $('reset-axes').onclick = function () {
      state.axes = SchemaAxes.DEFAULT_AXES.slice();
      state.view.setAxes(state.axes); renderAxisRail(); state.view.draw();
    };
    renderLegend();
  }

  function renderLegend() {
    var box = $('legend');
    box.innerHTML = '';
    var cm = state.view.colourMap;
    if (!cm) return;
    if (cm.kind === 'categorical') {
      cm.summary.categories.slice(0, CorpusView.PALETTE.length).forEach(function (c) {
        var s = el('span', 'swatch');
        s.style.background = cm.map[c];
        var w = el('span', 'legend-item');
        w.appendChild(s); w.appendChild(document.createTextNode(CorpusView.truncate(c, 22)));
        box.appendChild(w);
      });
      if (cm.summary.categories.length > CorpusView.PALETTE.length) {
        box.appendChild(el('span', 'legend-item', '+' +
          (cm.summary.categories.length - CorpusView.PALETTE.length) + ' more in grey'));
      }
    } else {
      var w2 = el('span', 'legend-item', CorpusView.formatNumber(cm.summary.min) + ' → ' + CorpusView.formatNumber(cm.summary.max));
      box.appendChild(w2);
      CorpusView.PALETTE.forEach(function (p) { var s = el('span', 'swatch'); s.style.background = p; box.appendChild(s); });
    }
    var g = el('span', 'legend-item');
    var gs = el('span', 'swatch'); gs.style.background = CorpusView.GREY;
    g.appendChild(gs); g.appendChild(document.createTextNode('absent on the colour column'));
    box.appendChild(g);
  }

  // ---------------------------------------------------------------- pointer

  function wirePointer(view) {
    var canvas = $('pc-overlay');
    var drag = null;
    function local(e) {
      var r = canvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    }
    canvas.addEventListener('mousedown', function (e) {
      var p = local(e);
      var a = view.axisAt(p.x);
      if (a >= 0 && p.y >= view.geom.bandTop - 6 && p.y <= view.geom.bandBottom + 6) {
        drag = { axis: a, y0: p.y, y1: p.y };
      }
    });
    canvas.addEventListener('mousemove', function (e) {
      var p = local(e);
      if (drag) {
        drag.y1 = p.y;
        view.paintOverlay();
        var g = view.geom;
        var ctx = view.octx;
        ctx.save();
        ctx.fillStyle = 'rgba(110,168,255,0.18)';
        ctx.strokeStyle = '#6ea8ff';
        var x = g.xs[drag.axis];
        var top = Math.min(drag.y0, drag.y1), bot = Math.max(drag.y0, drag.y1);
        ctx.fillRect(x - 16, top, 32, bot - top);
        ctx.strokeRect(x - 16.5, top + 0.5, 33, bot - top - 1);
        ctx.restore();
        return;
      }
      var hit = view.pick(p.x, p.y);
      if (hit !== view.hover) {
        view.hover = hit;
        view.paintOverlay();
        showHoverInfo(hit);
      }
    });
    window.addEventListener('mouseup', function (e) {
      if (!drag) return;
      var p = local(e);
      drag.y1 = p.y;
      var s = view.summaries[drag.axis];
      var top = Math.min(drag.y0, drag.y1), bot = Math.max(drag.y0, drag.y1);
      if (bot - top < 3) { view.setBrush(s.name, null); }
      else if (s.kind === 'numeric') {
        var hi = view.valueAt(drag.axis, top), lo = view.valueAt(drag.axis, bot);
        view.setBrush(s.name, { lo: Math.min(lo, hi), hi: Math.max(lo, hi) });
      } else {
        var terms = [];
        s.categories.forEach(function (c) {
          var y = view.yOf(s, c);
          if (y != null && y >= top - 1 && y <= bot + 1) terms.push(c);
        });
        view.setBrush(s.name, terms.length ? { terms: terms } : null);
      }
      drag = null;
      renderAxisCaptions();
    });
    canvas.addEventListener('click', function (e) {
      var p = local(e);
      if (view.axisAt(p.x) >= 0 && p.y >= view.geom.bandTop - 6 && p.y <= view.geom.bandBottom + 6) return;
      var hit = view.pick(p.x, p.y);
      if (hit >= 0) selectRecording(hit);
    });
    canvas.addEventListener('mouseleave', function () {
      view.hover = -1; view.paintOverlay(); showHoverInfo(-1);
    });
  }

  function showHoverInfo(index) {
    var box = $('hoverinfo');
    if (index < 0) { box.textContent = ''; return; }
    var r = state.rows[index];
    box.textContent = r.stem + '  ·  ' + r.verdict + '  ·  click to open';
  }

  // ---------------------------------------------------------------- the list

  function renderList() {
    var box = $('list');
    box.innerHTML = '';
    var shown = 0;
    for (var i = 0; i < state.rows.length && shown < 400; i++) {
      if (state.view.selected[i] !== 1) continue;
      shown++;
      var r = state.rows[i];
      var item = el('button', 'list-item' + (i === state.current ? ' current' : ''));
      item.appendChild(el('span', 'li-verdict ' + r.verdict, r.verdict));
      item.appendChild(el('span', 'li-stem', r.stem));
      (function (idx) { item.onclick = function () { selectRecording(idx); }; })(i);
      box.appendChild(item);
    }
    if (state.view.selectedCount > shown) {
      box.appendChild(el('div', 'list-more', '… ' + (state.view.selectedCount - shown).toLocaleString() + ' more; brush to narrow'));
    }
  }

  // ---------------------------------------------------------------- the recording view

  async function selectRecording(index) {
    state.current = index;
    state.view.focus = index;
    state.view.paintOverlay();
    renderList();
    var row = state.rows[index];
    $('rec-panel').hidden = false;
    $('rec-title').textContent = row.stem;
    $('rec-loading').hidden = false;
    $('rec-loading').textContent = state.cache ? 'decoding…' : 'reading this recording’s binary blocks…';
    var t0 = performance.now();
    var blockRow;
    try {
      blockRow = await readBlocks(index);
    } catch (err) {
      $('rec-loading').textContent = 'could not read the blocks: ' + err.message;
      return;
    }
    var merged = Object.assign({}, row, blockRow);
    var decoded;
    try {
      decoded = SchemaDecode.decodeRow(merged);
    } catch (err) {
      $('rec-loading').textContent = 'the blocks do not decode against schema_version ' +
        SchemaDecode.SCHEMA_VERSION + ': ' + err.message;
      return;
    }
    var ms = performance.now() - t0;
    $('rec-loading').hidden = true;
    renderDecision(merged, ms);
    if (!state.recView) {
      state.recView = new RecordingView($('rec-canvas'));
      wireRecordingPointer(state.recView);
    }
    state.recView.setRecording(merged, decoded);
    if (decoded.timeScaleS == null) {
      $('rec-canvas').hidden = true;
      $('rec-nostream').hidden = false;
      $('rec-nostream').textContent =
        'No stream: time_scale_s is null, so there is no time axis and nothing to draw. ' +
        'verdict ' + merged.verdict + (merged.grounds ? ', ground ' + merged.grounds : '') + '.';
    } else {
      $('rec-canvas').hidden = false;
      $('rec-nostream').hidden = true;
      state.recView.paint();
    }
    renderTranscript(merged, decoded);
    renderMeasurements(merged, decoded);
    $('rec-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (state.caching && !state.cache) startWarm();
  }

  function startWarm() {
    var bar = $('warm');
    bar.hidden = false;
    warmCache(function (i, n, name) {
      bar.textContent = name
        ? 'keeping the binary blocks in memory so the next selection is instant — ' + i + '/' + n + ' (' + name + ')'
        : 'binary blocks cached; selections are now instant';
      if (!name) setTimeout(function () { bar.hidden = true; }, 2500);
    });
  }

  function wireRecordingPointer(view) {
    var canvas = $('rec-canvas');
    canvas.addEventListener('mousemove', function (e) {
      var r = canvas.getBoundingClientRect();
      var hit = view.hitAt(e.clientX - r.left, e.clientY - r.top);
      if (hit !== view.hover) { view.hover = hit; view.paint(); }
      highlightWord(hit && hit.wordIndex != null ? hit.wordIndex : -1);
    });
    canvas.addEventListener('mouseleave', function () {
      if (view.hover) { view.hover = null; view.paint(); }
      highlightWord(-1);
    });
  }

  function field(dl, key, value, cls) {
    dl.appendChild(el('dt', null, key));
    var dd = el('dd', cls || null);
    if (value == null) { dd.className = (cls || '') + ' null'; dd.textContent = 'null'; }
    else dd.textContent = String(value);
    dl.appendChild(dd);
  }

  function renderDecision(row, ms) {
    var dl = $('rec-decision');
    dl.innerHTML = '';
    field(dl, 'verdict', row.verdict, 'verdict ' + row.verdict);
    field(dl, 'release', row.release);
    field(dl, 'grounds', row.grounds);
    field(dl, 'route_state', row.route_state);
    field(dl, 'flags_n', row.flags_n);
    field(dl, 'flag_nodes', row.flag_nodes && row.flag_nodes.length ? row.flag_nodes.join(', ') : (row.flag_nodes ? '[] none' : null));
    ['airway', 'speech', 'voice', 'quality'].forEach(function (b) {
      field(dl, 'conformance_' + b, row['conformance_' + b]);
    });
    ['airway', 'speech', 'voice'].forEach(function (b) { field(dl, 'route_' + b, row['route_' + b]); });
    field(dl, 'participant', row.participant);
    field(dl, 'session', row.session);
    field(dl, 'task', row.task);
    field(dl, 'declared_family', row.declared_family);
    field(dl, 'run_dir', row.run_dir);
    field(dl, 'duration_s (source)', row.duration_s == null ? null : row.duration_s.toFixed(4));
    field(dl, 'duration_conditioned_s', row.duration_conditioned_s == null ? null : row.duration_conditioned_s.toFixed(4));
    field(dl, 'time_scale_s (the time denominator)', row.time_scale_s == null ? null : row.time_scale_s.toFixed(4));
    field(dl, 'sampling_rate', row.sampling_rate);
    field(dl, 'wave_peak', row.wave_peak == null ? null : row.wave_peak.toFixed(5));
    field(dl, 'floor_dbfs', row.floor_dbfs == null ? null : row.floor_dbfs.toFixed(3));
    field(dl, 'spans_unrowed_n', row.spans_unrowed_n);
    field(dl, 'malformed_store_lines', row.malformed_store_lines);
    field(dl, 'pii_findings_n', row.pii_findings_n);
    $('rec-timing').textContent = 'blocks read and decoded in ' + ms.toFixed(0) + ' ms' +
      (state.cache ? ' (from the in-memory cache)' : ' (direct read)');
  }

  function renderTranscript(row, blocks) {
    var box = $('rec-transcript');
    box.innerHTML = '';
    var note = $('rec-transcript-note');
    if (blocks.asrWords == null) {
      note.textContent = 'null — no consensus_transcript measurement exists for this recording.';
      note.className = 'note absent';
      return;
    }
    if (!blocks.asrWords.length) {
      note.textContent = 'present and empty — a consensus transcript with no words.';
      note.className = 'note';
      return;
    }
    var marks = blocks.piiMarks;
    var byWord = {};
    if (marks) {
      marks.forEach(function (m) {
        blocks.asrWords.forEach(function (w) {
          if (w.t1 > m.t0 - 1e-6 && w.t0 < m.t1 + 1e-6) {
            (byWord[w.index] = byWord[w.index] || []).push(m.category);
          }
        });
      });
    }
    var cats = {};
    (marks || []).forEach(function (m) { cats[m.category] = (cats[m.category] || 0) + 1; });
    note.className = 'note' + (marks == null ? ' absent' : '');
    note.textContent = marks == null
      ? 'PII: null — the scan did not run, so this transcript is NOT known to be clean.'
      : marks.length
        ? 'PII: ' + marks.length + ' mark(s) — ' +
          Object.keys(cats).map(function (c) { return c + '×' + cats[c]; }).join(', ')
        : 'PII: scanned, nothing found.';
    blocks.asrWords.forEach(function (w) {
      var span = el('span', 'w' + (byWord[w.index] ? ' pii' : '') +
        (w.outcome ? ' o-' + w.outcome : ' o-unknown'), w.text);
      span.dataset.i = w.index;
      span.title = (w.outcome || 'outcome 255') + ' · ' + w.t0.toFixed(3) + '–' + w.t1.toFixed(3) + ' s' +
        (byWord[w.index] ? ' · PII ' + byWord[w.index].join(', ') : '');
      if (byWord[w.index]) {
        var tag = el('sup', 'pii-tag', byWord[w.index].join('/'));
        span.appendChild(tag);
      }
      box.appendChild(span);
      box.appendChild(document.createTextNode(' '));
    });
  }

  function highlightWord(index) {
    var prev = $('rec-transcript').querySelector('.w.hot');
    if (prev) prev.classList.remove('hot');
    if (index < 0) return;
    var n = $('rec-transcript').querySelector('.w[data-i="' + index + '"]');
    if (n) { n.classList.add('hot'); }
  }

  function renderMeasurements(row, blocks) {
    var body = $('rec-measures');
    body.innerHTML = '';
    function tr(name, value, n, note, absent) {
      var r = el('tr', absent ? 'absent-row' : null);
      r.appendChild(el('td', 'mname', name));
      var v = el('td', 'mvalue' + (absent ? ' null' : ''));
      v.textContent = absent ? 'not measured' : value;
      r.appendChild(v);
      r.appendChild(el('td', 'mn', n == null ? '—' : String(n)));
      r.appendChild(el('td', 'mnote', note || ''));
      body.appendChild(r);
    }
    SchemaAxes.SCALAR_MEASUREMENTS.forEach(function (name) {
      var v = row['m_' + name];
      var n = row['m_' + name + '_n'];
      tr(name, v == null ? '' : formatScalar(v), n,
        v == null ? 'null ⟺ _n == 0' : n > 1 ? 'arithmetic mean of ' + n + ' readings' : 'the single reading',
        v == null);
    });
    SchemaAxes.VECTOR_MEASUREMENTS.forEach(function (name) {
      var v = row['m_' + name];
      var n = row['m_' + name + '_n'];
      if (v == null) { tr(name, '', n, 'null ⟺ _n == 0', true); return; }
      var text = v.map(function (x) { return x == null ? '—' : formatScalar(x); }).join('  ');
      var unrealised = v.filter(function (x) { return x == null; }).length;
      tr(name, text, n, v.length + ' syllable positions' +
        (unrealised ? ', ' + unrealised + ' not realised (— , not 0)' : ''));
    });
    SchemaAxes.MATRIX_MEASUREMENTS.forEach(function (name) {
      var m = blocks.matrix;
      var n = row['m_' + name + '_n'];
      if (m == null) { tr(name, '', n, 'null ⟺ _n == 0', true); return; }
      var text = m.rows.map(function (r) {
        return r.map(function (x) { return x == null ? '—' : formatScalar(x); }).join(' ');
      }).join('   |   ');
      tr(name, text, n, m.rows.length + ' repetitions × ' + m.width + ' syllable positions, row-major');
    });
    SchemaAxes.CATEGORICAL_MEASUREMENTS.forEach(function (name) {
      var v = row['m_' + name];
      var n = row['m_' + name + '_n'];
      var sentinel = SchemaAxes.SENTINEL_ONLY.indexOf(name) >= 0;
      if (v == null) { tr(name, '', n, 'null ⟺ _n == 0', true); return; }
      tr(name, v.join(', '), n, sentinel
        ? 'presence flag: the sentinel only, fired ' + n + '×'
        : v.length + ' distinct term(s) over ' + n + ' rejection(s)');
    });
  }

  function formatScalar(v) {
    if (v == null) return '—';
    if (Number.isInteger(v)) return String(v);
    var a = Math.abs(v);
    if (a >= 1e5 || (a > 0 && a < 1e-4)) return v.toExponential(3);
    return v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
  }

  // ---------------------------------------------------------------- landing

  function wireLanding() {
    var input = $('picker');
    input.onchange = function () { if (input.files[0]) openFile(input.files[0]).catch(fail); };
    var drop = document.body;
    ['dragover', 'drop'].forEach(function (k) {
      drop.addEventListener(k, function (e) { e.preventDefault(); });
    });
    drop.addEventListener('drop', function (e) {
      var f = e.dataTransfer.files[0];
      if (f) openFile(f).catch(fail);
    });
    $('cache-toggle').onchange = function () {
      state.caching = $('cache-toggle').checked;
      if (!state.caching) { state.cache = null; }
    };
  }

  function fail(err) {
    status('failed: ' + (err && err.message ? err.message : String(err)), 'bad');
    if (window.console) window.console.error(err);
  }

  window.addEventListener('DOMContentLoaded', wireLanding);
  window.__viewerState = state;
})();
