// The corpus view: one polyline per recording over up to ten axes.
//
// The drawing rule: a line is BROKEN wherever its value on an axis is null. Absence is never a
// position on the value scale; it is a dashed stub toward a separated rail, and a count on that
// rail. See specs/20260922-compact-recording-vectors/views.md.

'use strict';

var CorpusView = (function () {
  var PAD = { top: 74, bottom: 58, left: 118, right: 40 };
  var RAIL_GAP = 26;
  var RAIL_HEIGHT = 16;
  var STUB = 22;
  var CHUNK = 6000;

  var PALETTE = [
    '#6ea8ff', '#ffb454', '#8ce99a', '#ff8fa3', '#c4a7ff',
    '#4fd1c5', '#f6e05e', '#fc8181', '#9ae6b4', '#b794f4',
  ];
  var GREY = '#7a8291';

  function CorpusView(canvas, overlay) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.ctx = canvas.getContext('2d');
    this.octx = overlay.getContext('2d');
    this.rows = [];
    this.axes = [];
    this.summaries = [];
    this.colourBy = 'verdict';
    this.colourMap = null;
    this.opacity = null;
    this.showStubs = true;
    this.brushes = {};
    this.selected = new Uint8Array(0);
    this.selectedCount = 0;
    this.hover = -1;
    this.focus = -1;
    this.geom = null;
    this.onSelectionChange = null;
    this.onPick = null;
    this._raf = null;
    this._paintToken = 0;
  }

  CorpusView.prototype.setRows = function (rows) {
    this.rows = rows;
    this.selected = new Uint8Array(rows.length);
    this.selected.fill(1);
    this.selectedCount = rows.length;
  };

  CorpusView.prototype.setAxes = function (names) {
    this.axes = names.slice(0, SchemaAxes.MAX_AXES);
    this.resummarise();
  };

  /** Scale for one axis: 'linear' or 'log'. Refused silently when the domain reaches zero. */
  CorpusView.prototype.setScale = function (name, scale) {
    this.scales = this.scales || {};
    this.scales[name] = scale;
    this.resummarise();
  };

  CorpusView.prototype.resummarise = function () {
    var self = this;
    this.scales = this.scales || {};
    this.summaries = this.axes.map(function (n) {
      return SchemaAxes.summarise(n, self.rows, { scale: self.scales[n] });
    });
    Object.keys(this.brushes).forEach(function (k) {
      if (self.axes.indexOf(k) < 0) delete self.brushes[k];
    });
    this.buildColourMap();
    this.applyBrushes();
  };

  CorpusView.prototype.buildColourMap = function () {
    if (!this.colourBy) { this.colourMap = null; return; }
    var col = SchemaAxes.BY_NAME[this.colourBy];
    if (!col || !col.assignable) { this.colourMap = null; return; }
    var s = SchemaAxes.summarise(this.colourBy, this.rows);
    this.colourSummary = s;
    if (s.kind === 'categorical') {
      var map = {};
      s.categories.slice(0, PALETTE.length).forEach(function (c, i) { map[c] = PALETTE[i]; });
      this.colourMap = { kind: 'categorical', map: map, summary: s };
    } else {
      this.colourMap = { kind: 'numeric', summary: s };
    }
  };

  CorpusView.prototype.colourOf = function (row) {
    if (!this.colourMap) return PALETTE[0];
    var v = SchemaAxes.readValue(this.colourMap.summary.col, row);
    if (v == null) return GREY;
    if (this.colourMap.kind === 'categorical') return this.colourMap.map[v] || GREY;
    var p = SchemaAxes.position(this.colourMap.summary, v);
    if (p == null) return GREY;
    var i = Math.min(PALETTE.length - 1, Math.max(0, Math.round(p * (PALETTE.length - 1))));
    return PALETTE[i];
  };

  // ------------------------------------------------------------------ geometry

  CorpusView.prototype.layout = function () {
    var w = this.canvas.clientWidth;
    var h = this.canvas.clientHeight;
    var n = this.axes.length;
    var xs = [];
    for (var i = 0; i < n; i++) {
      xs.push(n === 1 ? (PAD.left + w - PAD.right) / 2
        : PAD.left + (i * (w - PAD.left - PAD.right)) / (n - 1));
    }
    var bandTop = PAD.top;
    var railY = h - PAD.bottom;
    var bandBottom = railY - RAIL_GAP - RAIL_HEIGHT;
    this.geom = { w: w, h: h, xs: xs, bandTop: bandTop, bandBottom: bandBottom, railY: railY };
    this.cacheYs();
    return this.geom;
  };

  /**
   * One Float64Array of y per axis, rebuilt on every layout or axis change.
   * NaN is the absent marker, because a typed array cannot hold null. Every reader of these
   * arrays must test for it; `vertices` is the one that converts it back to null.
   */
  CorpusView.prototype.cacheYs = function () {
    var g = this.geom;
    var span = g.bandBottom - g.bandTop;
    this.ys = [];
    for (var a = 0; a < this.summaries.length; a++) {
      var s = this.summaries[a];
      var out = new Float64Array(this.rows.length);
      for (var i = 0; i < this.rows.length; i++) {
        var p = SchemaAxes.position(s, SchemaAxes.readValue(s.col, this.rows[i]));
        out[i] = p == null ? NaN : g.bandBottom - p * span;
      }
      this.ys.push(out);
    }
    this.ysFor = this.summaries;
  };

  /** The cached y, or NaN. Falls back to a direct read when the cache is not built yet. */
  CorpusView.prototype.yAt = function (a, index) {
    if (this.ys && this.ysFor === this.summaries && this.ys[a]) return this.ys[a][index];
    var s = this.summaries[a];
    var y = this.yOf(s, SchemaAxes.readValue(s.col, this.rows[index]));
    return y == null ? NaN : y;
  };

  /** The y a value occupies, or null when the value is absent. */
  CorpusView.prototype.yOf = function (summary, value) {
    var p = SchemaAxes.position(summary, value);
    if (p == null) return null;
    var g = this.geom;
    return g.bandBottom - p * (g.bandBottom - g.bandTop);
  };

  // ------------------------------------------------------------------ brushing

  CorpusView.prototype.applyBrushes = function () {
    var self = this;
    var active = Object.keys(this.brushes);
    var count = 0;
    for (var i = 0; i < this.rows.length; i++) {
      var keep = true;
      for (var a = 0; a < active.length; a++) {
        var name = active[a];
        var b = self.brushes[name];
        var s = self.summaries[self.axes.indexOf(name)];
        if (!s) continue;
        var v = SchemaAxes.readValue(s.col, self.rows[i]);
        if (v == null) { if (b.absent !== 'only' && b.absent !== 'include') { keep = false; break; } continue; }
        if (b.absent === 'only') { keep = false; break; }
        if (s.kind === 'numeric') {
          if (b.lo != null && (v < b.lo || v > b.hi)) { keep = false; break; }
        } else if (b.terms && b.terms.length) {
          if (b.terms.indexOf(v) < 0) { keep = false; break; }
        }
      }
      self.selected[i] = keep ? 1 : 0;
      if (keep) count++;
    }
    this.selectedCount = count;
    if (this.onSelectionChange) this.onSelectionChange(count);
  };

  CorpusView.prototype.setBrush = function (name, brush) {
    if (brush == null) delete this.brushes[name];
    else this.brushes[name] = brush;
    this.applyBrushes();
    this.draw();
  };

  CorpusView.prototype.clearBrushes = function () {
    this.brushes = {};
    this.applyBrushes();
    this.draw();
  };

  // ------------------------------------------------------------------ painting

  CorpusView.prototype.draw = function () {
    var self = this;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame(function () { self._raf = null; self.paint(); });
  };

  CorpusView.prototype.paint = function () {
    var dpr = window.devicePixelRatio || 1;
    var c = this.canvas;
    var w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
      this.overlay.width = c.width; this.overlay.height = c.height;
    }
    var ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.axes.length) return;
    this.layout();
    this.paintChrome(ctx);

    var token = ++this._paintToken;
    var self = this;
    var i = 0;
    var alpha = this.opacity != null ? this.opacity
      : Math.max(0.015, Math.min(0.3, 1800 / Math.max(1, this.selectedCount)));

    // Unselected mass first, as one dim colour; then selected, coloured.
    function step() {
      if (token !== self._paintToken) return;
      var end = Math.min(self.rows.length, i + CHUNK);
      self.paintRange(ctx, i, end, alpha);
      i = end;
      if (i < self.rows.length) requestAnimationFrame(step);
      else { self.paintAxisMarks(ctx); self.paintOverlay(); }
    }
    step();
  };

  /**
   * One recording's vertices, one per axis, in axis order.
   * `y` is null exactly where the value is absent; the rail y is never substituted here.
   */
  CorpusView.prototype.vertices = function (row) {
    var g = this.geom;
    var i = row.__i;
    var cached = typeof i === 'number' && this.rows[i] === row &&
      this.ys && this.ysFor === this.summaries;
    var out = [];
    for (var a = 0; a < this.summaries.length; a++) {
      var s = this.summaries[a];
      var y;
      var v;
      if (cached) {
        y = this.ys[a][i];
        if (Number.isNaN(y)) y = null;
        v = undefined;
      } else {
        v = SchemaAxes.readValue(s.col, row);
        y = this.yOf(s, v);
      }
      out.push({ axis: a, x: g.xs[a], y: y, value: v, absent: y == null });
    }
    return out;
  };

  /**
   * The segments drawn for one recording, and its stubs to the absent rail.
   * A segment runs only between two consecutive axes that BOTH carry a value; an absent value
   * yields a stub instead, and never a segment.
   */
  CorpusView.prototype.segmentsFor = function (verts) {
    var g = this.geom;
    var railY = g.railY - RAIL_HEIGHT / 2;
    var segments = [];
    var stubs = [];
    for (var i = 0; i + 1 < verts.length; i++) {
      var a = verts[i], b = verts[i + 1];
      if (!a.absent && !b.absent) segments.push([a.x, a.y, b.x, b.y]);
      else if (!a.absent && b.absent) stubs.push([a.x, a.y, b.x - STUB, railY]);
      else if (a.absent && !b.absent) stubs.push([a.x + STUB, railY, b.x, b.y]);
    }
    return { segments: segments, stubs: stubs };
  };

  CorpusView.prototype.paintRange = function (ctx, from, to, alpha) {
    var rows = this.rows;
    var byColour = {};
    var dimPath = new Path2D();
    var stubPath = new Path2D();
    var anyDim = false;
    for (var r = from; r < to; r++) {
      var row = rows[r];
      var chosen = this.selected[r] === 1;
      var path;
      if (chosen) {
        var col = this.colourOf(row);
        path = byColour[col] || (byColour[col] = new Path2D());
      } else { path = dimPath; anyDim = true; }
      var drawn = this.segmentsFor(this.vertices(row));
      for (var k = 0; k < drawn.segments.length; k++) {
        var s2 = drawn.segments[k];
        path.moveTo(s2[0], s2[1]); path.lineTo(s2[2], s2[3]);
      }
      if (chosen && this.showStubs) {
        for (var q = 0; q < drawn.stubs.length; q++) {
          var t2 = drawn.stubs[q];
          stubPath.moveTo(t2[0], t2[1]); stubPath.lineTo(t2[2], t2[3]);
        }
      }
    }
    ctx.lineWidth = 1;
    ctx.lineJoin = 'round';
    if (anyDim) { ctx.globalAlpha = Math.min(0.05, alpha * 0.4); ctx.strokeStyle = '#3a4150'; ctx.stroke(dimPath); }
    ctx.globalAlpha = Math.min(0.12, alpha * 0.5);
    ctx.strokeStyle = '#6b7280';
    ctx.setLineDash([2, 3]);
    ctx.stroke(stubPath);
    ctx.setLineDash([]);
    ctx.globalAlpha = alpha;
    var keys = Object.keys(byColour);
    for (var k = 0; k < keys.length; k++) { ctx.strokeStyle = keys[k]; ctx.stroke(byColour[keys[k]]); }
    ctx.globalAlpha = 1;
  };

  CorpusView.prototype.paintChrome = function (ctx) {
    var g = this.geom;
    ctx.save();
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    for (var a = 0; a < this.summaries.length; a++) {
      var s = this.summaries[a];
      var x = g.xs[a];
      ctx.strokeStyle = '#4b5361';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, g.bandTop); ctx.lineTo(x, g.bandBottom); ctx.stroke();
      // the break between the value band and the absent rail
      ctx.strokeStyle = '#6b7280';
      ctx.beginPath();
      ctx.moveTo(x - 5, g.bandBottom + 7); ctx.lineTo(x + 5, g.bandBottom + 3);
      ctx.moveTo(x - 5, g.bandBottom + 13); ctx.lineTo(x + 5, g.bandBottom + 9);
      ctx.stroke();
    }
    ctx.restore();
  };

  CorpusView.prototype.paintAxisMarks = function (ctx) {
    var g = this.geom;
    ctx.save();
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textBaseline = 'middle';
    for (var a = 0; a < this.summaries.length; a++) {
      var s = this.summaries[a];
      var x = g.xs[a];
      var brushed = this.countPresent(a);
      // the absent rail: a bar whose width is the share of the drawn set that has no value here
      var share = this.selectedCount ? brushed.absent / this.selectedCount : 0;
      var full = 74;
      ctx.fillStyle = '#242a35';
      ctx.fillRect(x - full / 2, g.railY - RAIL_HEIGHT, full, RAIL_HEIGHT);
      ctx.fillStyle = brushed.absent ? '#8a6d3b' : '#2f3542';
      ctx.fillRect(x - full / 2, g.railY - RAIL_HEIGHT, full * share, RAIL_HEIGHT);
      ctx.strokeStyle = '#5a6373';
      ctx.strokeRect(x - full / 2 + 0.5, g.railY - RAIL_HEIGHT + 0.5, full - 1, RAIL_HEIGHT - 1);
      ctx.fillStyle = brushed.absent ? '#e8c07d' : '#6b7280';
      ctx.textAlign = 'center';
      ctx.fillText('absent ' + brushed.absent.toLocaleString(), x, g.railY - RAIL_HEIGHT / 2);

      // scale ticks
      ctx.textAlign = 'right';
      ctx.fillStyle = '#9aa3b2';
      if (s.kind === 'numeric') {
        for (var t = 0; t <= 4; t++) {
          var frac = t / 4;
          var y = g.bandBottom - frac * (g.bandBottom - g.bandTop);
          var val = SchemaAxes.valueOf(s, frac);
          ctx.fillText(formatNumber(val), x - 6, y);
          ctx.strokeStyle = '#3a4150';
          ctx.beginPath(); ctx.moveTo(x - 4, y); ctx.lineTo(x, y); ctx.stroke();
        }
      } else {
        var cats = s.categories;
        var span = g.bandBottom - g.bandTop;
        var step = cats.length > 1 ? span / (cats.length - 1) : 0;
        var every = Math.max(1, Math.ceil(13 / Math.max(step, 1)));
        for (var ci = 0; ci < cats.length; ci += every) {
          var yy = g.bandTop + ci * step;
          ctx.fillText(truncate(cats[ci], 16), x - 6, yy);
          ctx.strokeStyle = '#3a4150';
          ctx.beginPath(); ctx.moveTo(x - 4, yy); ctx.lineTo(x, yy); ctx.stroke();
        }
        if (cats.length > 1 && every > 1) {
          ctx.fillStyle = '#6b7280';
          ctx.textAlign = 'center';
          ctx.fillText(cats.length.toLocaleString() + ' categories', x, g.bandTop - 10);
        }
      }
    }
    ctx.restore();
  };

  /** Present/absent counts on one axis over the currently drawn set. */
  CorpusView.prototype.countPresent = function (a) {
    var s = this.summaries[a];
    var cached = this.ys && this.ysFor === this.summaries && this.ys[a];
    var present = 0, absent = 0;
    for (var i = 0; i < this.rows.length; i++) {
      if (this.selected[i] !== 1) continue;
      var missing = cached
        ? Number.isNaN(this.ys[a][i])
        : SchemaAxes.readValue(s.col, this.rows[i]) == null;
      if (missing) absent++; else present++;
    }
    return { present: present, absent: absent };
  };

  // ------------------------------------------------------------------ overlay: one line, whole

  CorpusView.prototype.paintOverlay = function () {
    var dpr = window.devicePixelRatio || 1;
    var ctx = this.octx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.overlay.width / dpr, this.overlay.height / dpr);
    if (this.focus >= 0) this.paintOne(ctx, this.focus, '#ffffff', 2.2);
    if (this.hover >= 0 && this.hover !== this.focus) this.paintOne(ctx, this.hover, '#ffd166', 1.6);
  };

  /**
   * One recording, drawn whole. Where a value is absent the segment is dashed and the vertex
   * is a hollow marker on the rail, so a single line stays traceable without the rail ever
   * reading as a value.
   */
  CorpusView.prototype.paintOne = function (ctx, index, colour, width) {
    var g = this.geom;
    if (!g) return;
    var row = this.rows[index];
    var pts = this.vertices(row).map(function (v) {
      return { x: v.x, y: v.absent ? g.railY - RAIL_HEIGHT / 2 : v.y, absent: v.absent };
    });
    ctx.save();
    ctx.lineWidth = width;
    ctx.strokeStyle = colour;
    for (var i = 0; i + 1 < pts.length; i++) {
      ctx.setLineDash(pts[i].absent || pts[i + 1].absent ? [3, 3] : []);
      ctx.globalAlpha = pts[i].absent || pts[i + 1].absent ? 0.5 : 1;
      ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[i + 1].x, pts[i + 1].y); ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    for (var j = 0; j < pts.length; j++) {
      ctx.beginPath();
      if (pts[j].absent) {
        ctx.strokeStyle = colour;
        ctx.rect(pts[j].x - 4, pts[j].y - 4, 8, 8);
        ctx.stroke();
      } else {
        ctx.fillStyle = colour;
        ctx.arc(pts[j].x, pts[j].y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  };

  // ------------------------------------------------------------------ hit testing

  /** The drawn recording whose polyline passes nearest (x, y), or -1. */
  CorpusView.prototype.pick = function (x, y) {
    var g = this.geom;
    if (!g || this.summaries.length < 1) return -1;
    var a0 = 0;
    for (var a = 0; a + 1 < g.xs.length; a++) if (x >= g.xs[a]) a0 = a;
    var a1 = Math.min(a0 + 1, g.xs.length - 1);
    var x0 = g.xs[a0], x1 = g.xs[a1];
    var t = x1 === x0 ? 0 : (x - x0) / (x1 - x0);
    var best = -1, bestD = 9;
    var sel = this.selected;
    for (var i = 0; i < this.rows.length; i++) {
      if (sel[i] !== 1) continue;
      var y0 = this.yAt(a0, i);
      var y1 = this.yAt(a1, i);
      // NaN is the absent marker in the cache, and NaN comparisons are false, so an
      // absent endpoint can never win: the segment is not drawn, so it is not pickable.
      var d = Math.abs(y0 + (y1 - y0) * t - y);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  };

  /** Which axis a pointer x is within grabbing distance of, or -1. */
  CorpusView.prototype.axisAt = function (x) {
    var g = this.geom;
    if (!g) return -1;
    for (var a = 0; a < g.xs.length; a++) if (Math.abs(x - g.xs[a]) < 22) return a;
    return -1;
  };

  /** The value a y within the band stands for on axis a, under that axis's own scale. */
  CorpusView.prototype.valueAt = function (a, y) {
    var g = this.geom;
    var p = (g.bandBottom - y) / (g.bandBottom - g.bandTop);
    return SchemaAxes.valueOf(this.summaries[a], Math.max(0, Math.min(1, p)));
  };

  function formatNumber(v) {
    if (!Number.isFinite(v)) return '—';
    var a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e5 || a < 1e-3) return v.toExponential(1);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }

  function truncate(s, n) {
    s = String(s);
    return s.length <= n ? s : s.slice(0, n - 1) + '…';
  }

  CorpusView.PALETTE = PALETTE;
  CorpusView.GREY = GREY;
  CorpusView.formatNumber = formatNumber;
  CorpusView.truncate = truncate;
  CorpusView.RAIL_HEIGHT = RAIL_HEIGHT;
  return CorpusView;
})();

if (typeof module !== 'undefined' && module.exports) module.exports = CorpusView;
