// The recording view: one recording drawn from its compact vectors.
//
// Every time on this canvas is decoded against time_scale_s. duration_s is the source
// recording's duration and is reported in the header, never used as a denominator.

'use strict';

var RecordingView = (function () {
  var LEFT = 104;
  var RIGHT = 16;
  var GAP = 10;

  var LANE_COLOURS = {
    E: '#6ea8ff', C: '#4fd1c5', A: '#ffb454', S: '#8ce99a', G: '#5a6373',
    AIRWAY: '#ff8fa3', SPEECH: '#ffb454', VOICE: '#c4a7ff', REDACT: '#e8c07d',
  };
  var OUTCOME_COLOURS = { agreement: '#8ce99a', variant: '#ffb454', insertion: '#ff8fa3' };
  var UNKNOWN_COLOUR = '#7a8291';

  function RecordingView(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.row = null;
    this.blocks = null;
    this.hits = [];
    this.hover = null;
  }

  RecordingView.prototype.setRecording = function (row, blocks) {
    this.row = row;
    this.blocks = blocks;
    this.hover = null;
  };

  RecordingView.prototype.plan = function () {
    var b = this.blocks;
    var lanes = [];
    function push(key, label, height, absentNote) {
      lanes.push({ key: key, label: label, height: height, absentNote: absentNote || null });
    }
    push('wave', 'waveform', 72, b.wave ? null : 'no stream decoded');
    push('env', 'envelope dBFS', 72, b.envelope ? null : 'energy_envelope absent');
    push('cont', 'continuity', 54, b.continuity ? null : 'continuity_trace absent');
    push('spans', 'spans E C A S G', 5 * 15 + 6, b.spans ? null : 'PREPROCESS wrote no verdict');
    push('labels', 'top label', 3 * 17 + 6, b.spanLabels ? null : 'no classifier scored a span');
    push('squim', 'SQUIM', 3 * 24 + 6, b.spanSquim ? null : 'no span SQUIM');
    push('asr', 'consensus ASR', 30, b.asrWords ? null : 'no consensus_transcript');
    push('pii', 'PII marks', 26, b.piiMarks ? null : 'the PII scan did not run');
    push('branch', 'branch lanes', 4 * 17 + 6, b.branchLanes ? null : 'routing wrote no branch_decision');
    var y = 34;
    lanes.forEach(function (l) { l.y = y; y += l.height + GAP; });
    return { lanes: lanes, height: y + 24 };
  };

  RecordingView.prototype.paint = function () {
    var b = this.blocks;
    if (!b) return;
    var plan = this.plan();
    var dpr = window.devicePixelRatio || 1;
    var c = this.canvas;
    var w = c.clientWidth;
    c.style.height = plan.height + 'px';
    var h = plan.height;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    var ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    this.hits = [];

    var ts = b.timeScaleS;
    var x0 = LEFT, x1 = w - RIGHT;
    var self = this;
    function X(t) { return ts ? x0 + (t / ts) * (x1 - x0) : x0; }
    this.X = X;
    this.plot = { x0: x0, x1: x1, ts: ts };

    this.paintTimeAxis(ctx, plan, x0, x1, ts);

    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textBaseline = 'middle';
    plan.lanes.forEach(function (lane) {
      ctx.fillStyle = '#9aa3b2';
      ctx.textAlign = 'right';
      ctx.fillText(lane.label, LEFT - 8, lane.y + 9);
      ctx.fillStyle = '#161a22';
      ctx.fillRect(x0, lane.y, x1 - x0, lane.height);
      if (lane.absentNote) {
        // An absent producer is said, never drawn as an empty result.
        ctx.fillStyle = '#2a2018';
        ctx.fillRect(x0, lane.y, x1 - x0, lane.height);
        ctx.fillStyle = '#e8c07d';
        ctx.textAlign = 'left';
        ctx.fillText('absent — ' + lane.absentNote, x0 + 8, lane.y + lane.height / 2);
        return;
      }
      self['paint_' + lane.key](ctx, lane, X);
    });
    this.paintHover(ctx, plan);
  };

  RecordingView.prototype.paintTimeAxis = function (ctx, plan, x0, x1, ts) {
    ctx.save();
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#6b7280';
    ctx.strokeStyle = '#2a3140';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    var span = ts || 1;
    var step = niceStep(span / 8);
    for (var t = 0; t <= span + 1e-9; t += step) {
      var x = x0 + (t / span) * (x1 - x0);
      ctx.beginPath(); ctx.moveTo(x, 28); ctx.lineTo(x, plan.height - 22); ctx.stroke();
      ctx.fillText(t.toFixed(step < 1 ? 1 : 0) + 's', x, plan.height - 18);
    }
    ctx.textAlign = 'left';
    ctx.fillStyle = '#9aa3b2';
    ctx.fillText('time_scale_s = ' + (ts == null ? '—' : ts.toFixed(4) + ' s'), x0, 12);
    ctx.restore();
  };

  RecordingView.prototype.paint_wave = function (ctx, lane, X) {
    var wave = this.blocks.wave;
    var peak = this.row.wave_peak;
    var mid = lane.y + lane.height / 2;
    var half = lane.height / 2 - 3;
    ctx.strokeStyle = '#2a3140';
    ctx.beginPath(); ctx.moveTo(this.plot.x0, mid); ctx.lineTo(this.plot.x1, mid); ctx.stroke();
    ctx.fillStyle = LANE_COLOURS.E;
    var n = wave.points;
    for (var i = 0; i < n; i++) {
      var xa = X((i / n) * this.plot.ts), xb = X(((i + 1) / n) * this.plot.ts);
      var ya = mid - (wave.max[i] / peak) * half;
      var yb = mid - (wave.min[i] / peak) * half;
      ctx.fillRect(xa, Math.min(ya, yb), Math.max(1, xb - xa), Math.max(1, Math.abs(yb - ya)));
    }
    this.axisLabels(ctx, lane, [[1, '+' + peak.toFixed(3)], [0.5, '0'], [0, '-' + peak.toFixed(3)]]);
  };

  RecordingView.prototype.paint_env = function (ctx, lane, X) {
    var env = this.blocks.envelope;
    var lo = SchemaDecode.ENVELOPE_DBFS_RANGE[0], hi = SchemaDecode.ENVELOPE_DBFS_RANGE[1];
    var self = this;
    function Y(v) { return lane.y + lane.height - ((v - lo) / (hi - lo)) * lane.height; }
    ctx.strokeStyle = LANE_COLOURS.C;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (var i = 0; i < env.length; i++) {
      var x = X(((i + 0.5) / env.length) * this.plot.ts);
      if (i === 0) ctx.moveTo(x, Y(env[i])); else ctx.lineTo(x, Y(env[i]));
    }
    ctx.stroke();
    var floor = this.row.floor_dbfs;
    if (floor != null) {
      ctx.strokeStyle = '#e8c07d';
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(this.plot.x0, Y(floor)); ctx.lineTo(this.plot.x1, Y(floor)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#e8c07d';
      ctx.textAlign = 'left';
      ctx.fillText('floor ' + floor.toFixed(1) + ' dBFS', this.plot.x0 + 6, Y(floor) - 7);
    }
    this.axisLabels(ctx, lane, [[1, '0'], [0.5, '-50'], [0, '-100']]);
    ctx.fillStyle = '#5a6373';
    ctx.textAlign = 'left';
    ctx.fillText('256 buckets, max per bucket', this.plot.x0 + 6, lane.y + 8);
  };

  RecordingView.prototype.paint_cont = function (ctx, lane, X) {
    var c = this.blocks.continuity;
    var lo = SchemaDecode.CONTINUITY_RANGE[0], hi = SchemaDecode.CONTINUITY_RANGE[1];
    function Y(v) { return lane.y + lane.height - ((v - lo) / (hi - lo)) * lane.height; }
    ctx.strokeStyle = '#8ce99a';
    ctx.beginPath();
    for (var i = 0; i < c.length; i++) {
      var x = X(((i + 0.5) / c.length) * this.plot.ts);
      if (i === 0) ctx.moveTo(x, Y(c[i])); else ctx.lineTo(x, Y(c[i]));
    }
    ctx.stroke();
    this.axisLabels(ctx, lane, [[1, '1.05'], [0, '0']]);
    ctx.fillStyle = '#5a6373';
    ctx.textAlign = 'left';
    ctx.fillText('256 buckets, mean per bucket', this.plot.x0 + 6, lane.y + 8);
  };

  RecordingView.prototype.paint_spans = function (ctx, lane, X) {
    var rows = SchemaDecode.SPAN_ROWS;
    var self = this;
    var h = 15;
    rows.forEach(function (code, i) {
      var y = lane.y + 3 + i * h;
      ctx.fillStyle = '#9aa3b2';
      ctx.textAlign = 'right';
      ctx.fillText(code, LEFT - 2, y + h / 2);
      ctx.strokeStyle = '#212734';
      ctx.beginPath(); ctx.moveTo(self.plot.x0, y + h - 0.5); ctx.lineTo(self.plot.x1, y + h - 0.5); ctx.stroke();
    });
    this.blocks.spans.forEach(function (s) {
      var y = lane.y + 3 + s.rowIndex * h;
      var xa = X(s.t0), xb = X(s.t1);
      ctx.fillStyle = LANE_COLOURS[s.row];
      ctx.globalAlpha = 0.75;
      ctx.fillRect(xa, y + 2, Math.max(1.5, xb - xa), h - 5);
      ctx.globalAlpha = 1;
      self.hits.push({
        x0: xa, x1: Math.max(xa + 2, xb), y0: y, y1: y + h,
        text: 'span ' + s.index + ' · row ' + s.row + ' (' + s.rowName + ') · ' +
          s.t0.toFixed(3) + '–' + s.t1.toFixed(3) + ' s',
      });
    });
    if (this.row.spans_unrowed_n) {
      ctx.fillStyle = '#e8c07d';
      ctx.textAlign = 'left';
      ctx.fillText(this.row.spans_unrowed_n + ' span(s) carry no row code and are drawn nowhere',
        this.plot.x0 + 6, lane.y + lane.height - 6);
    }
  };

  RecordingView.prototype.paint_labels = function (ctx, lane, X) {
    var self = this;
    var spans = this.blocks.spans || [];
    var h = 17;
    SchemaDecode.CLASSIFIERS.forEach(function (name, i) {
      var y = lane.y + 3 + i * h;
      ctx.fillStyle = '#9aa3b2';
      ctx.textAlign = 'right';
      ctx.fillText(name, LEFT - 2, y + h / 2);
    });
    this.blocks.spanLabels.forEach(function (l) {
      var span = spans[l.spanIndex];
      if (!span) return;
      var ci = l.classifier == null ? -1 : SchemaDecode.CLASSIFIERS.indexOf(l.classifier);
      var y = lane.y + 3 + (ci < 0 ? SchemaDecode.CLASSIFIERS.length - 1 : ci) * h;
      var xa = X(span.t0), xb = X(span.t1);
      ctx.globalAlpha = 0.25 + 0.7 * l.score;
      ctx.fillStyle = l.classifier == null ? UNKNOWN_COLOUR : '#6ea8ff';
      ctx.fillRect(xa, y + 2, Math.max(1.5, xb - xa), h - 5);
      ctx.globalAlpha = 1;
      if (xb - xa > 46) {
        ctx.save();
        ctx.beginPath(); ctx.rect(xa, y, xb - xa, h); ctx.clip();
        ctx.fillStyle = '#0d1016';
        ctx.textAlign = 'left';
        ctx.fillText(l.name, xa + 3, y + h / 2);
        ctx.restore();
      }
      self.hits.push({
        x0: xa, x1: Math.max(xa + 2, xb), y0: y, y1: y + h,
        text: (l.classifier || 'classifier 255 (not yamnet/hear/ast)') + ' · ' + l.name +
          ' · score ' + l.score.toFixed(3) + ' · span ' + l.spanIndex,
      });
    });
  };

  RecordingView.prototype.paint_squim = function (ctx, lane, X) {
    var self = this;
    var spans = this.blocks.spans || [];
    var h = 24;
    var metrics = [['stoi', 'stoi'], ['pesq', 'pesq'], ['si_sdr', 'si_sdr']];
    metrics.forEach(function (m, i) {
      var y = lane.y + 3 + i * h;
      var range = SchemaDecode.SQUIM_RANGES[m[0]];
      ctx.fillStyle = '#9aa3b2';
      ctx.textAlign = 'right';
      ctx.fillText(m[1], LEFT - 2, y + h / 2);
      ctx.fillStyle = '#5a6373';
      ctx.textAlign = 'left';
      ctx.fillText(range[0] + '–' + range[1], self.plot.x1 + 2, y + h / 2);
      ctx.strokeStyle = '#212734';
      ctx.beginPath(); ctx.moveTo(self.plot.x0, y + h - 0.5); ctx.lineTo(self.plot.x1, y + h - 0.5); ctx.stroke();
    });
    this.blocks.spanSquim.forEach(function (s) {
      var span = spans[s.spanIndex];
      if (!span) return;
      var xa = X(span.t0), xb = X(span.t1);
      metrics.forEach(function (m, i) {
        var y = lane.y + 3 + i * h;
        var range = SchemaDecode.SQUIM_RANGES[m[0]];
        var frac = (s[m[0]] - range[0]) / (range[1] - range[0]);
        var barH = Math.max(1, frac * (h - 5));
        ctx.fillStyle = ['#4fd1c5', '#c4a7ff', '#8ce99a'][i];
        ctx.globalAlpha = 0.8;
        ctx.fillRect(xa, y + h - 3 - barH, Math.max(1.5, xb - xa), barH);
        ctx.globalAlpha = 1;
        self.hits.push({
          x0: xa, x1: Math.max(xa + 2, xb), y0: y, y1: y + h,
          text: 'span ' + s.spanIndex + ' · ' + m[0] + ' ' + s[m[0]].toFixed(3) +
            ' over [' + range[0] + ', ' + range[1] + ']',
        });
      });
    });
  };

  RecordingView.prototype.paint_asr = function (ctx, lane, X) {
    var self = this;
    // Bars are acoustic extents. Word text lives in the transcript panel; it is not fitted
    // to a bar, because a word's text length has nothing to do with its duration.
    this.blocks.asrWords.forEach(function (word) {
      var xa = X(word.t0), xb = X(word.t1);
      ctx.fillStyle = word.outcome == null ? UNKNOWN_COLOUR : OUTCOME_COLOURS[word.outcome];
      ctx.globalAlpha = 0.85;
      ctx.fillRect(xa, lane.y + 5, Math.max(1.2, xb - xa - 0.6), lane.height - 12);
      ctx.globalAlpha = 1;
      self.hits.push({
        x0: xa, x1: Math.max(xa + 2, xb), y0: lane.y, y1: lane.y + lane.height,
        wordIndex: word.index,
        text: '"' + word.text + '" · ' + (word.outcome || 'outcome 255') + ' · ' +
          word.t0.toFixed(3) + '–' + word.t1.toFixed(3) + ' s',
      });
    });
    if (!this.blocks.asrWords.length) {
      ctx.fillStyle = '#6b7280';
      ctx.textAlign = 'left';
      ctx.fillText('present and empty — a consensus transcript with no words', this.plot.x0 + 6, lane.y + lane.height / 2);
    }
  };

  RecordingView.prototype.paint_pii = function (ctx, lane, X) {
    var self = this;
    var marks = this.blocks.piiMarks;
    if (!marks.length) {
      ctx.fillStyle = '#8ce99a';
      ctx.textAlign = 'left';
      ctx.fillText('scanned, nothing found (pii_findings_n = ' + this.row.pii_findings_n + ')',
        this.plot.x0 + 6, lane.y + lane.height / 2);
      return;
    }
    marks.forEach(function (m) {
      var xa = X(m.t0), xb = X(m.t1);
      ctx.fillStyle = '#ff8fa3';
      ctx.globalAlpha = 0.6;
      ctx.fillRect(xa, lane.y + 4, Math.max(2, xb - xa), lane.height - 8);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#ff8fa3';
      ctx.strokeRect(xa + 0.5, lane.y + 4.5, Math.max(2, xb - xa) - 1, lane.height - 9);
      if (xb - xa > 40) {
        ctx.save();
        ctx.beginPath(); ctx.rect(xa, lane.y, xb - xa, lane.height); ctx.clip();
        ctx.fillStyle = '#12151c';
        ctx.textAlign = 'left';
        ctx.fillText(m.category, xa + 3, lane.y + lane.height / 2);
        ctx.restore();
      }
      self.hits.push({
        x0: xa, x1: Math.max(xa + 2, xb), y0: lane.y, y1: lane.y + lane.height,
        text: 'PII ' + m.category + ' · ' + m.t0.toFixed(3) + '–' + m.t1.toFixed(3) + ' s',
      });
    });
  };

  RecordingView.prototype.paint_branch = function (ctx, lane, X) {
    var self = this;
    var h = 17;
    SchemaDecode.LANES.forEach(function (name, i) {
      var y = lane.y + 3 + i * h;
      ctx.fillStyle = '#9aa3b2';
      ctx.textAlign = 'right';
      ctx.fillText(name, LEFT - 2, y + h / 2);
      ctx.strokeStyle = '#212734';
      ctx.beginPath(); ctx.moveTo(self.plot.x0, y + h - 0.5); ctx.lineTo(self.plot.x1, y + h - 0.5); ctx.stroke();
    });
    this.blocks.branchLanes.forEach(function (l) {
      var y = lane.y + 3 + l.laneIndex * h;
      var xa = X(l.t0), xb = X(l.t1);
      ctx.fillStyle = LANE_COLOURS[l.lane];
      ctx.globalAlpha = 0.7;
      ctx.fillRect(xa, y + 2, Math.max(1.5, xb - xa), h - 5);
      ctx.globalAlpha = 1;
      if (xb - xa > 44) {
        ctx.save();
        ctx.beginPath(); ctx.rect(xa, y, xb - xa, h); ctx.clip();
        ctx.fillStyle = '#12151c';
        ctx.textAlign = 'left';
        ctx.fillText(l.role, xa + 3, y + h / 2);
        ctx.restore();
      }
      self.hits.push({
        x0: xa, x1: Math.max(xa + 2, xb), y0: y, y1: y + h,
        text: l.lane + ' · ' + l.role + ' · ' + l.t0.toFixed(3) + '–' + l.t1.toFixed(3) + ' s',
      });
    });
  };

  RecordingView.prototype.axisLabels = function (ctx, lane, marks) {
    ctx.save();
    ctx.fillStyle = '#5a6373';
    ctx.textAlign = 'right';
    marks.forEach(function (m) {
      ctx.fillText(m[1], LEFT - 8, lane.y + lane.height - m[0] * lane.height + (m[0] === 1 ? 5 : m[0] === 0 ? -5 : 0));
    });
    ctx.restore();
  };

  RecordingView.prototype.paintHover = function (ctx, plan) {
    if (!this.hover) return;
    var hit = this.hover;
    ctx.save();
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    var pad = 6;
    var tw = ctx.measureText(hit.text).width;
    var bx = Math.min(hit.x0, this.canvas.clientWidth - tw - 2 * pad - 4);
    var by = Math.max(2, hit.y0 - 20);
    ctx.fillStyle = 'rgba(8,10,14,0.94)';
    ctx.strokeStyle = '#4b5361';
    ctx.fillRect(bx, by, tw + 2 * pad, 18);
    ctx.strokeRect(bx + 0.5, by + 0.5, tw + 2 * pad - 1, 17);
    ctx.fillStyle = '#e6e9ef';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(hit.text, bx + pad, by + 9);
    ctx.restore();
  };

  /** The topmost drawn rectangle under (x, y), or null. */
  RecordingView.prototype.hitAt = function (x, y) {
    for (var i = this.hits.length - 1; i >= 0; i--) {
      var h = this.hits[i];
      if (x >= h.x0 - 1 && x <= h.x1 + 1 && y >= h.y0 && y <= h.y1) return h;
    }
    return null;
  };

  function niceStep(raw) {
    var pow = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-6))));
    var n = raw / pow;
    var m = n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10;
    return m * pow;
  }

  RecordingView.LANE_COLOURS = LANE_COLOURS;
  RecordingView.OUTCOME_COLOURS = OUTCOME_COLOURS;
  RecordingView.niceStep = niceStep;
  return RecordingView;
})();

if (typeof module !== 'undefined' && module.exports) module.exports = RecordingView;
