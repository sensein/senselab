// The axis model for the corpus view: which columns can carry an axis, what kind of
// axis each one is, and where a value sits on it.
//
// Three properties this module exists to hold:
//   * null is absent, never a position on the value scale. `position` returns null and the
//     caller must break the line rather than place it;
//   * a categorical axis is ordered categories, not numbers;
//   * an `m_<name>` axis is the mean of `m_<name>_n` readings, and the axis says so.

'use strict';

var SchemaAxes = (function () {
  var SCALAR_MEASUREMENTS = [
    'breath_coverage_fraction',
    'breath_peak_over_floor_db',
    'cough_peak_over_floor_db',
    'ddk_ppg_period_dispersion',
    'ddk_repetition_count_from_ppg_decode',
    'ddk_syllable_rate_from_envelope_modulation_hz',
    'ddk_syllable_rate_from_ppg_decode_hz',
    'expected_sequence_repeat_fraction',
    'extent_dominant_speaker_share',
    'extent_secondary_source_s',
    'extent_source_active_s',
    'extent_speaker_count',
    'glide_extent_semitones',
    'interruptions',
    'pause_fraction_of_response',
    'phonation_onset_to_offset_s',
    'source_content_coverage',
    'speech_rate_from_consensus_words_per_s',
    'train_fraction_of_recording',
    'verbatim_overlap_fraction',
    'voiced_duration_s',
  ];
  var VECTOR_MEASUREMENTS = ['ddk_position_realised_mass'];
  var MATRIX_MEASUREMENTS = ['ddk_cv_instrument_reading'];
  var CATEGORICAL_MEASUREMENTS = [
    'carrier_rejected',
    'category_membership',
    'defines_its_cue',
    'effort_absolute',
    'measured_route',
    'phonation_extent',
    'repetition_rule',
    'route',
    'source_overlap',
    'sweep_extent',
  ];
  // Nine of the ten categoricals only ever carry this one term, so for those the column is a
  // presence flag and `_n` is the number of times it fired. carrier_rejected is the real one.
  var SENTINEL = 'NOT_SEPARABLE_BY_THIS_DESIGN';
  var SENTINEL_ONLY = CATEGORICAL_MEASUREMENTS.filter(function (n) {
    return n !== 'carrier_rejected';
  });

  // VERDICT's gates, in the registry's declaration order. Each is a reading, a bound the fold
  // resolved for this recording's task group, and the side of the bound that passes.
  var GATES = [
    ['production_min_s', 'carrier_duration_s', 'at_least', 's'],
    ['voiced_fraction_min', 'carrier_voiced_fraction', 'at_least', null],
    ['f0_spread_max_semitones', 'carrier_f0_spread_semitones', 'at_most', 'semitones'],
    ['continuity_min', 'carrier_continuity', 'at_least', null],
    ['dominant_segment_min_fraction', 'sweep_dominant_fraction', 'at_least', null],
    ['monotone_tolerance_semitones', 'sweep_monotone_reversal_semitones', 'at_most', 'semitones'],
    ['expected_tokens_matched_min', 'expected_tokens_matched', 'at_least', null],
    ['omissions_max', 'expected_tokens_omitted', 'at_most', null],
    ['response_min_s', 'response_duration_s', 'at_least', 's'],
    ['coverage_min', 'source_content_coverage', 'at_least', null],
    ['dominant_speaker_share_min', 'extent_dominant_speaker_share', 'at_least', null],
    ['items_min', 'items_produced', 'at_least', null],
    ['events_min', 'airway_events_found', 'at_least', null],
    ['repetitions_min', 'ddk_repetitions_found', 'at_least', null],
    ['repeat_overlap_min', null, 'at_least', null],
    ['echo_overlap_max', null, 'at_most', null],
    ['verbatim_overlap_max', null, 'at_most', null],
    ['gap_off_task_min_s', null, 'at_least', 's'],
    ['interval_max_s', null, 'at_most', 's'],
    ['score_min', null, 'at_least', null],
    ['train_min_s', null, 'at_least', 's'],
    ['rate_prominence_min', null, 'at_least', null],
  ];

  // Orders for the closed vocabularies. Anything not listed orders by frequency.
  var ORDERINGS = {
    verdict: ['pass', 'flag', 'discard'],
    release: ['releasable', 'nothing_to_redact', 'withheld', 'not_assessed'],
    conformance_airway: ['true', 'undetermined', 'false'],
    conformance_speech: ['true', 'undetermined', 'false'],
    conformance_voice: ['true', 'undetermined', 'false'],
    conformance_quality: ['true', 'undetermined', 'false'],
    route_state: ['routed', 'unexplained', 'empty', 'declined', 'unavailable'],
    route_airway: ['routed', 'declined', 'unavailable'],
    route_speech: ['routed', 'declined', 'unavailable'],
    route_voice: ['routed', 'declined', 'unavailable'],
  };
  GATES.forEach(function (g) { ORDERINGS['gate_' + g[0] + '_passed'] = ['true', 'undetermined', 'false']; });

  function column(spec) {
    return {
      name: spec.name,
      label: spec.label || spec.name,
      kind: spec.kind,
      group: spec.group,
      assignable: spec.assignable !== false,
      reason: spec.reason || null,
      unit: spec.unit || null,
      nullMeans: spec.nullMeans || null,
      countColumn: spec.countColumn || null,
      reduction: spec.reduction || null,
      sizeOf: spec.sizeOf || null,
      boundColumn: spec.boundColumn || null,
      op: spec.op || null,
      gate: spec.gate || null,
    };
  }

  var IDENTITY = [
    column({ name: 'participant', kind: 'categorical', group: 'identity', nullMeans: 'the stem carries no sub-' }),
    column({ name: 'task', kind: 'categorical', group: 'identity', nullMeans: 'the stem carries no task-' }),
    column({ name: 'verdict', kind: 'categorical', group: 'decision', nullMeans: 'never null' }),
    column({ name: 'session', kind: 'categorical', group: 'identity', nullMeans: 'the stem carries no ses-' }),
    column({ name: 'declared_family', kind: 'categorical', group: 'identity', nullMeans: 'nothing was declared' }),
    column({ name: 'release', kind: 'categorical', group: 'decision', nullMeans: 'the fold wrote none' }),
    column({ name: 'release_ground', kind: 'categorical', group: 'decision', nullMeans: 'REDACT itself decided' }),
    column({ name: 'grounds', kind: 'categorical', group: 'decision', nullMeans: 'nothing was discarded' }),
    column({ name: 'route_state', kind: 'categorical', group: 'decision', nullMeans: 'the fold wrote none' }),
    column({ name: 'conformance_airway', kind: 'categorical', group: 'branch', nullMeans: 'AIRWAY wrote no branch report' }),
    column({ name: 'conformance_speech', kind: 'categorical', group: 'branch', nullMeans: 'SPEECH wrote no branch report' }),
    column({ name: 'conformance_voice', kind: 'categorical', group: 'branch', nullMeans: 'VOICE wrote no branch report' }),
    column({ name: 'conformance_quality', kind: 'categorical', group: 'branch', nullMeans: 'QUALITY wrote no branch report' }),
    column({ name: 'route_airway', kind: 'categorical', group: 'branch', nullMeans: 'routing wrote no state' }),
    column({ name: 'route_speech', kind: 'categorical', group: 'branch', nullMeans: 'routing wrote no state' }),
    column({ name: 'route_voice', kind: 'categorical', group: 'branch', nullMeans: 'routing wrote no state' }),
    column({ name: 'duration_s', kind: 'numeric', group: 'stream', unit: 's', nullMeans: 'the file never decoded' }),
    column({
      name: 'duration_conditioned_s', kind: 'numeric', group: 'stream', unit: 's',
      nullMeans: 'PREPROCESS wrote no stream',
    }),
    column({
      name: 'time_scale_s', kind: 'numeric', group: 'stream', unit: 's',
      nullMeans: 'neither duration is known',
    }),
    column({ name: 'sampling_rate', kind: 'numeric', group: 'stream', unit: 'Hz', nullMeans: 'no conditioned stream' }),
    column({ name: 'wave_peak', kind: 'numeric', group: 'stream', nullMeans: 'no stream decoded' }),
    column({ name: 'floor_dbfs', kind: 'numeric', group: 'stream', unit: 'dBFS', nullMeans: 'energy_envelope is absent' }),
    column({ name: 'flags_n', kind: 'count', group: 'decision', nullMeans: 'never null' }),
    column({
      name: 'pii_findings_n', kind: 'count', group: 'decision',
      nullMeans: 'the PII scan did not run — 0 means scanned and clean',
    }),
    column({ name: 'spans_unrowed_n', kind: 'count', group: 'stream', nullMeans: 'PREPROCESS did not run' }),
    column({ name: 'malformed_store_lines', kind: 'count', group: 'stream', nullMeans: 'never null' }),
    column({ name: 'schema_version', kind: 'count', group: 'stream', nullMeans: 'never null' }),
    column({
      name: 'flag_nodes', kind: 'set', group: 'decision', assignable: false,
      reason: 'a set of node names, not a value — put its size on the axis, or filter by a term',
      sizeOf: 'flag_nodes',
    }),
    column({ name: 'flag_nodes.size', kind: 'count', group: 'decision', label: 'flag_nodes · size', sizeOf: 'flag_nodes' }),
  ];

  function measurementColumns() {
    var out = [];
    SCALAR_MEASUREMENTS.forEach(function (n) {
      out.push(column({
        name: 'm_' + n, kind: 'numeric', group: 'measurement', reduction: 'mean',
        countColumn: 'm_' + n + '_n', nullMeans: 'the graph wrote no reading',
      }));
      out.push(column({
        name: 'm_' + n + '_n', kind: 'count', group: 'measurement count',
        nullMeans: 'never null; 0 means the graph ran and wrote nothing',
      }));
    });
    VECTOR_MEASUREMENTS.forEach(function (n) {
      out.push(column({
        name: 'm_' + n, kind: 'vector', group: 'measurement', assignable: false,
        reason: 'a vector, one value per syllable position, each null where the position was not realised — no single number to place on an axis',
        countColumn: 'm_' + n + '_n',
      }));
      out.push(column({ name: 'm_' + n + '_n', kind: 'count', group: 'measurement count' }));
    });
    MATRIX_MEASUREMENTS.forEach(function (n) {
      out.push(column({
        name: 'm_' + n, kind: 'matrix', group: 'measurement', assignable: false,
        reason: 'a matrix, repetitions by syllable position, flattened row-major — no single number to place on an axis',
        countColumn: 'm_' + n + '_n',
      }));
      out.push(column({ name: 'm_' + n + '_n', kind: 'count', group: 'measurement count' }));
      out.push(column({ name: 'm_' + n + '_width', kind: 'count', group: 'measurement count' }));
    });
    CATEGORICAL_MEASUREMENTS.forEach(function (n) {
      var sentinelOnly = SENTINEL_ONLY.indexOf(n) >= 0;
      out.push(column({
        name: 'm_' + n, kind: 'set', group: 'measurement', assignable: false,
        reason: sentinelOnly
          ? 'this column only ever carries ' + SENTINEL + ', so it is a presence flag — put its size or its _n on the axis'
          : 'a set of distinct gate names, not a value — put its size or its _n on the axis, or filter by a term',
        sizeOf: 'm_' + n,
        countColumn: 'm_' + n + '_n',
      }));
      out.push(column({
        name: 'm_' + n + '.size', kind: 'count', group: 'measurement',
        label: 'm_' + n + ' · distinct terms',
        sizeOf: 'm_' + n,
      }));
      out.push(column({
        name: 'm_' + n + '_n', kind: 'count', group: 'measurement count',
        nullMeans: 'never null; 0 means the graph ran and wrote nothing',
      }));
    });
    return out;
  }

  // One column group per gate: the reading VERDICT read, the bound it resolved for this
  // recording's task group, and what it made of the two. The reading is the axis; the bound
  // rides along on `boundColumn` so the corpus view can draw it as a reference line.
  function gateColumns() {
    var out = [];
    GATES.forEach(function (g) {
      var name = g[0], reading = g[1], op = g[2], unit = g[3];
      var asks = op === 'at_least' ? 'at least' : 'at most';
      out.push(column({
        name: 'gate_' + name, kind: 'numeric', group: 'gate reading', unit: unit,
        label: name + ' · ' + (reading || 'located reading'),
        boundColumn: 'gate_' + name + '_bound', op: op, gate: name,
        nullMeans: reading
          ? 'this gate was not applied, or nothing carried ' + reading
          : 'this gate is applied where its finding is located, not by VERDICT',
      }));
      out.push(column({
        name: 'gate_' + name + '_bound', kind: 'numeric', group: 'gate bound', unit: unit,
        label: name + ' · bound (' + asks + ')', gate: name,
        nullMeans: 'no layer named a bound for this recording’s task group',
      }));
      out.push(column({
        name: 'gate_' + name + '_passed', kind: 'categorical', group: 'gate outcome',
        label: name + ' · passed', gate: name,
        nullMeans: 'this gate was not applied to this recording',
      }));
    });
    return out;
  }

  var GATE_SUMMARY = [
    column({ name: 'gate_group', kind: 'categorical', group: 'gate', nullMeans: 'no task group was resolved' }),
    column({ name: 'gate_family', kind: 'categorical', group: 'gate', nullMeans: 'no task group was resolved' }),
    column({ name: 'gate_node', kind: 'categorical', group: 'gate', nullMeans: 'no task group was resolved' }),
    column({ name: 'gate_applied_n', kind: 'count', group: 'gate', nullMeans: 'never null; 0 means no group resolved' }),
    column({ name: 'gate_flagging_n', kind: 'count', group: 'gate', nullMeans: 'never null' }),
    column({ name: 'gate_failed_n', kind: 'count', group: 'gate', nullMeans: 'never null; 0 means nothing refused' }),
    column({
      name: 'gate_undetermined_n', kind: 'count', group: 'gate',
      nullMeans: 'never null; 0 means every applied gate could be answered',
    }),
    column({
      name: 'gate_failed_names', kind: 'set', group: 'gate', assignable: false,
      reason: 'a set of gate names, not a value — put its size on the axis, or filter by a term',
      sizeOf: 'gate_failed_names',
    }),
    column({
      name: 'gate_failed_names.size', kind: 'count', group: 'gate',
      label: 'gate_failed_names · size', sizeOf: 'gate_failed_names',
    }),
    column({
      name: 'gate_flagged_names', kind: 'set', group: 'gate', assignable: false,
      reason: 'a set of gate names, not a value — put its size on the axis, or filter by a term',
      sizeOf: 'gate_flagged_names',
    }),
    column({
      name: 'gate_flagged_names.size', kind: 'count', group: 'gate',
      label: 'gate_flagged_names · size', sizeOf: 'gate_flagged_names',
    }),
  ];

  var SPEAKER_AND_LEVEL = [
    column({ name: 'separated_n', kind: 'count', group: 'speakers', nullMeans: 'never null; 0 means separation did not run' }),
    column({ name: 'secondary_extent_n', kind: 'count', group: 'speakers', nullMeans: 'never null' }),
    column({
      name: 'secondary_extent_s', kind: 'numeric', group: 'speakers', unit: 's',
      nullMeans: 'no secondary-speaker run was located',
    }),
    column({
      name: 'solo_extent_s', kind: 'numeric', group: 'speakers', unit: 's',
      nullMeans: 'no solo run was located',
    }),
    column({
      name: 'extent_dominant_speaker_share_min', kind: 'numeric', group: 'speakers',
      label: 'extent_dominant_speaker_share · worst extent',
      nullMeans: 'no task extent carried a diarized segment',
    }),
    column({
      name: 'residual_rms_dbfs', kind: 'numeric', group: 'level', unit: 'dBFS',
      nullMeans: 'PREPROCESS wrote no residual decomposition',
    }),
    column({
      name: 'residual_peak_dbfs', kind: 'numeric', group: 'level', unit: 'dBFS',
      nullMeans: 'PREPROCESS wrote no residual decomposition',
    }),
    column({
      name: 'enhanced_rms_dbfs', kind: 'numeric', group: 'level', unit: 'dBFS',
      nullMeans: 'PREPROCESS wrote no residual decomposition',
    }),
    column({
      name: 'enhanced_over_residual_rms_db', kind: 'numeric', group: 'level', unit: 'dB',
      label: 'enhanced over residual · RMS',
      nullMeans: 'PREPROCESS wrote no residual decomposition',
    }),
    column({
      name: 'enhanced_over_residual_rms_fitted_db', kind: 'numeric', group: 'level', unit: 'dB',
      label: 'enhanced over residual · RMS, gain-fitted',
      nullMeans: 'PREPROCESS wrote no residual decomposition',
    }),
  ];

  var CATALOGUE = IDENTITY.concat(GATE_SUMMARY, SPEAKER_AND_LEVEL, gateColumns(), measurementColumns());
  var BY_NAME = {};
  CATALOGUE.forEach(function (c) { BY_NAME[c.name] = c; });

  // Owner-directed: participant, task, verdict first. Then the columns that actually
  // discriminate over this corpus — see specs/20260922-compact-recording-vectors/views.md.
  var DEFAULT_AXES = [
    'participant',
    'task',
    'verdict',
    'duration_s',
    'conformance_airway',
    'conformance_speech',
    'conformance_voice',
    'flags_n',
    'pii_findings_n',
    'release',
  ];
  var MAX_AXES = 10;

  /** Every parquet column the corpus view must read: catalogue columns minus derived ones. */
  function corpusColumns() {
    var seen = {};
    CATALOGUE.forEach(function (c) {
      var name = c.sizeOf && c.name.indexOf('.size') > 0 ? c.sizeOf : c.name;
      if (c.kind === 'vector' || c.kind === 'matrix') return;
      seen[name] = true;
    });
    seen['stem'] = true;
    seen['flag_nodes'] = true;
    return Object.keys(seen).sort();
  }

  /** The value an axis reads for one row: null stays null, `.size` reads a list length. */
  function readValue(col, row) {
    if (col.name.indexOf('.size') > 0 && col.sizeOf) {
      var list = row[col.sizeOf];
      return list == null ? null : list.length;
    }
    var v = row[col.name];
    if (v === undefined) return null;
    if (typeof v === 'number' && !Number.isFinite(v)) return null;
    return v;
  }

  function compareCategories(name, counts) {
    var order = ORDERINGS[name];
    var terms = Object.keys(counts);
    if (order) {
      var known = order.filter(function (t) { return counts[t] !== undefined; });
      var rest = terms.filter(function (t) { return order.indexOf(t) < 0; })
        .sort(function (a, b) { return counts[b] - counts[a] || (a < b ? -1 : 1); });
      return known.concat(rest);
    }
    return terms.sort(function (a, b) { return counts[b] - counts[a] || (a < b ? -1 : 1); });
  }

  /**
   * Summarise one column over the rows an axis will draw.
   * Returns {col, kind, present, absent, min, max, categories, counts, folded, foldedMax}.
   * `absent` is the count of nulls and is never folded into the value scale.
   * `options` may carry `order` ('frequency' | 'alphabetical') and `scale`
   * ('linear' | 'log'); log is refused rather than silently ignored when the domain
   * reaches zero or below.
   */
  function summarise(name, rows, options) {
    var opts = typeof options === 'string' ? { order: options } : (options || {});
    var order = opts.order;
    var col = BY_NAME[name];
    if (!col) throw new Error('no column ' + name + ' in the catalogue');
    if (!col.assignable) throw new Error(name + ' is not assignable to an axis: ' + col.reason);
    var present = 0;
    var absent = 0;
    var min = Infinity;
    var max = -Infinity;
    var counts = {};
    var folded = 0;
    var foldedMax = 0;
    var numeric = col.kind === 'numeric' || col.kind === 'count';
    for (var i = 0; i < rows.length; i++) {
      var row = rows[i];
      var v = readValue(col, row);
      if (v == null) { absent++; continue; }
      present++;
      if (numeric) {
        if (v < min) min = v;
        if (v > max) max = v;
      } else {
        counts[v] = (counts[v] || 0) + 1;
      }
      if (col.countColumn) {
        var n = row[col.countColumn];
        if (n != null && n > 1) { folded++; if (n > foldedMax) foldedMax = n; }
      }
    }
    var summary = {
      col: col,
      name: name,
      kind: numeric ? 'numeric' : 'categorical',
      present: present,
      absent: absent,
      total: rows.length,
      folded: folded,
      foldedMax: foldedMax,
    };
    if (numeric) {
      if (present === 0) { summary.min = 0; summary.max = 1; summary.empty = true; }
      else if (min === max) { summary.min = min - 0.5; summary.max = max + 0.5; summary.flat = true; }
      else { summary.min = min; summary.max = max; }
      summary.logCapable = summary.min > 0;
      summary.scale = opts.scale === 'log' && summary.logCapable ? 'log' : 'linear';
      if (summary.scale === 'log') {
        summary.logMin = Math.log10(summary.min);
        summary.logMax = Math.log10(summary.max);
      }
    } else {
      summary.counts = counts;
      summary.categories = order === 'alphabetical'
        ? Object.keys(counts).sort()
        : compareCategories(name, counts);
      summary.index = {};
      summary.categories.forEach(function (c, i) { summary.index[c] = i; });
    }
    return summary;
  }

  /**
   * Where a value sits in the axis's value band, as a fraction 0 (bottom) to 1 (top).
   * Returns null for an absent value — the caller must not place it on the band.
   */
  function position(summary, value) {
    if (value == null) return null;
    if (summary.kind === 'numeric') {
      if (typeof value !== 'number' || !Number.isFinite(value)) return null;
      if (summary.max === summary.min) return 0.5;
      if (summary.scale === 'log') {
        return (Math.log10(value) - summary.logMin) / (summary.logMax - summary.logMin);
      }
      return (value - summary.min) / (summary.max - summary.min);
    }
    var i = summary.index[value];
    if (i === undefined) return null;
    if (summary.categories.length === 1) return 0.5;
    return 1 - i / (summary.categories.length - 1);
  }

  /** The value a fraction of the value band stands for: the inverse of `position`. */
  function valueOf(summary, fraction) {
    if (summary.kind !== 'numeric') {
      var i = Math.round((1 - fraction) * Math.max(0, summary.categories.length - 1));
      return summary.categories[i];
    }
    if (summary.scale === 'log') {
      return Math.pow(10, summary.logMin + fraction * (summary.logMax - summary.logMin));
    }
    return summary.min + fraction * (summary.max - summary.min);
  }

  /** One line of prose under the axis title saying what the axis is showing. */
  function caption(summary) {
    var bits = [];
    if (summary.kind === 'numeric') {
      bits.push((summary.col.unit ? summary.col.unit : 'numeric') +
        (summary.scale === 'log' ? ' · log10 scale' : ''));
    } else {
      bits.push(summary.categories.length + ' ordered categories');
    }
    if (summary.col.reduction === 'mean') {
      bits.push(
        summary.folded > 0
          ? 'mean · ' + summary.folded.toLocaleString() + ' of ' + summary.present.toLocaleString() +
            ' fold >1 reading (max ' + summary.foldedMax + ')'
          : 'mean · every reading is a single measurement'
      );
    }
    if (summary.absent > 0) {
      bits.push(summary.absent.toLocaleString() + ' absent');
    }
    return bits.join(' · ');
  }

  return {
    SCALAR_MEASUREMENTS: SCALAR_MEASUREMENTS,
    VECTOR_MEASUREMENTS: VECTOR_MEASUREMENTS,
    MATRIX_MEASUREMENTS: MATRIX_MEASUREMENTS,
    CATEGORICAL_MEASUREMENTS: CATEGORICAL_MEASUREMENTS,
    SENTINEL: SENTINEL,
    SENTINEL_ONLY: SENTINEL_ONLY,
    GATES: GATES,
    ORDERINGS: ORDERINGS,
    CATALOGUE: CATALOGUE,
    BY_NAME: BY_NAME,
    DEFAULT_AXES: DEFAULT_AXES,
    MAX_AXES: MAX_AXES,
    corpusColumns: corpusColumns,
    readValue: readValue,
    summarise: summarise,
    position: position,
    valueOf: valueOf,
    caption: caption,
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = SchemaAxes;
