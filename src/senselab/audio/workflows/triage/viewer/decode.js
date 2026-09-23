// Decoders for the binary blocks of recording_vectors.parquet, schema_version 1.
// Byte layouts and enumerations mirror senselab.audio.workflows.triage.recording_vectors;
// specs/20260922-compact-recording-vectors/schema.md is the contract.
//
// Two rules hold throughout:
//   * a null block decodes to null (absent) and a zero-length block to [] (present, empty);
//   * an enumeration code outside its table is a schema violation and throws, except where
//     the schema names 255 a documented sentinel, which decodes to null.

'use strict';

var SchemaDecode = (function () {
  var SCHEMA_VERSION = 3;
  var TIME_SCALE = 65535;
  var TRACE_POINTS = 256;
  var UNKNOWN_CODE = 255;

  var SPAN_ROWS = ['E', 'C', 'A', 'S', 'G'];
  var SPAN_ROW_NAMES = {
    E: 'envelope amplitude',
    C: 'continuity',
    A: 'ASR',
    S: 'normalised amplitude',
    G: 'gap',
  };
  var CLASSIFIERS = ['yamnet', 'hear', 'ast'];
  var WORD_OUTCOMES = ['agreement', 'variant', 'insertion'];
  var LANES = ['AIRWAY', 'SPEECH', 'VOICE', 'REDACT'];

  var ENVELOPE_DBFS_RANGE = [-100.0, 0.0];
  var CONTINUITY_RANGE = [0.0, 1.05];
  var SCORE_RANGE = [0.0, 1.0];
  var SQUIM_RANGES = { stoi: [0.0, 1.0], pesq: [1.0, 4.5], si_sdr: [-10.0, 30.0] };

  var RECORD_SIZES = {
    spans: 5,
    span_labels: 4,
    span_squim: 5,
    asr_words: 5,
    pii_marks: 4,
    branch_lanes: 5,
  };

  /** Seconds from a uint16 time code. time_scale_s, never duration_s, is the denominator. */
  function decodeTime(code, timeScaleS) {
    return (code / TIME_SCALE) * timeScaleS;
  }

  /** A reading from a uint8 code over its declared range. */
  function decodeValue(code, low, high) {
    return low + (code / 255) * (high - low);
  }

  function viewOf(bytes) {
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  function recordCount(bytes, name) {
    var size = RECORD_SIZES[name];
    if (bytes.byteLength % size !== 0) {
      throw new Error(
        bytes.byteLength + ' bytes is not a whole number of ' + size + '-byte ' + name + ' records'
      );
    }
    return bytes.byteLength / size;
  }

  function fixedLength(bytes, want, name) {
    if (bytes.byteLength !== want) {
      throw new Error(name + ' is ' + bytes.byteLength + ' bytes, not the fixed ' + want);
    }
  }

  function parallelLength(bytes, names, count, blockName, columnName) {
    if (names == null) {
      throw new Error(columnName + ' is null while ' + blockName + ' is present');
    }
    if (names.length !== count) {
      throw new Error(
        columnName + ' has ' + names.length + ' elements for ' + count + ' ' + blockName + ' records'
      );
    }
  }

  function enumName(table, code, what) {
    if (code === UNKNOWN_CODE) return null;
    if (code < 0 || code >= table.length) {
      throw new Error(what + ' code ' + code + ' is outside ' + JSON.stringify(table));
    }
    return table[code];
  }

  function strictEnumName(table, code, what) {
    if (code < 0 || code >= table.length) {
      throw new Error(what + ' code ' + code + ' is outside ' + JSON.stringify(table));
    }
    return table[code];
  }

  /**
   * The conditioned waveform: 256 (min, max) pairs over [-wave_peak, +wave_peak].
   * Returns {min, max, points} or null.
   */
  function decodeWaveMinmax(bytes, wavePeak) {
    if (bytes == null) return null;
    fixedLength(bytes, TRACE_POINTS * 2, 'wave_minmax');
    if (wavePeak == null) throw new Error('wave_minmax is present but wave_peak is null');
    var view = viewOf(bytes);
    var lo = new Float64Array(TRACE_POINTS);
    var hi = new Float64Array(TRACE_POINTS);
    for (var i = 0; i < TRACE_POINTS; i++) {
      lo[i] = decodeValue(view.getUint8(2 * i), -wavePeak, wavePeak);
      hi[i] = decodeValue(view.getUint8(2 * i + 1), -wavePeak, wavePeak);
    }
    return { min: lo, max: hi, points: TRACE_POINTS };
  }

  /** A fixed 256-point uint8 trace over [low, high]. Returns Float64Array or null. */
  function decodeTrace(bytes, low, high, name) {
    if (bytes == null) return null;
    fixedLength(bytes, TRACE_POINTS, name || 'trace');
    var out = new Float64Array(TRACE_POINTS);
    for (var i = 0; i < TRACE_POINTS; i++) out[i] = decodeValue(bytes[i], low, high);
    return out;
  }

  function decodeEnvelopeDbfs(bytes) {
    return decodeTrace(bytes, ENVELOPE_DBFS_RANGE[0], ENVELOPE_DBFS_RANGE[1], 'env_dbfs');
  }

  function decodeContinuity(bytes) {
    return decodeTrace(bytes, CONTINUITY_RANGE[0], CONTINUITY_RANGE[1], 'continuity');
  }

  /** The seconds a fixed-trace bucket covers: [i/256, (i+1)/256) of time_scale_s. */
  function bucketExtent(i, timeScaleS) {
    return [(i / TRACE_POINTS) * timeScaleS, ((i + 1) / TRACE_POINTS) * timeScaleS];
  }

  /** The five-row span lane: <BHH per record. Returns [{index,row,rowIndex,rowName,t0,t1}] or null. */
  function decodeSpans(bytes, timeScaleS) {
    if (bytes == null) return null;
    var n = recordCount(bytes, 'spans');
    var view = viewOf(bytes);
    var out = [];
    for (var i = 0; i < n; i++) {
      var o = i * 5;
      var rowIndex = view.getUint8(o);
      var row = strictEnumName(SPAN_ROWS, rowIndex, 'span row');
      out.push({
        index: i,
        rowIndex: rowIndex,
        row: row,
        rowName: SPAN_ROW_NAMES[row],
        t0: decodeTime(view.getUint16(o + 1, true), timeScaleS),
        t1: decodeTime(view.getUint16(o + 3, true), timeScaleS),
      });
    }
    return out;
  }

  /** Top classifier label per span: <HBB. Returns [{spanIndex,classifier,score,name}] or null. */
  function decodeSpanLabels(bytes, names, spanCount) {
    if (bytes == null) return null;
    var n = recordCount(bytes, 'span_labels');
    parallelLength(bytes, names, n, 'span_labels', 'span_label_name');
    var view = viewOf(bytes);
    var out = [];
    for (var i = 0; i < n; i++) {
      var o = i * 4;
      var spanIndex = view.getUint16(o, true);
      if (spanCount != null && spanIndex >= spanCount) {
        throw new Error('span_labels record ' + i + ' indexes span ' + spanIndex + ' of ' + spanCount);
      }
      out.push({
        spanIndex: spanIndex,
        classifier: enumName(CLASSIFIERS, view.getUint8(o + 2), 'classifier'),
        score: decodeValue(view.getUint8(o + 3), SCORE_RANGE[0], SCORE_RANGE[1]),
        name: names[i],
      });
    }
    return out;
  }

  /** SQUIM per span: <HBBB. Returns [{spanIndex,stoi,pesq,si_sdr}] or null. */
  function decodeSpanSquim(bytes, spanCount) {
    if (bytes == null) return null;
    var n = recordCount(bytes, 'span_squim');
    var view = viewOf(bytes);
    var out = [];
    for (var i = 0; i < n; i++) {
      var o = i * 5;
      var spanIndex = view.getUint16(o, true);
      if (spanCount != null && spanIndex >= spanCount) {
        throw new Error('span_squim record ' + i + ' indexes span ' + spanIndex + ' of ' + spanCount);
      }
      out.push({
        spanIndex: spanIndex,
        stoi: decodeValue(view.getUint8(o + 2), SQUIM_RANGES.stoi[0], SQUIM_RANGES.stoi[1]),
        pesq: decodeValue(view.getUint8(o + 3), SQUIM_RANGES.pesq[0], SQUIM_RANGES.pesq[1]),
        si_sdr: decodeValue(view.getUint8(o + 4), SQUIM_RANGES.si_sdr[0], SQUIM_RANGES.si_sdr[1]),
      });
    }
    return out;
  }

  /** The consensus ASR lane: <HHB. Returns [{index,t0,t1,outcome,text}] or null. */
  function decodeAsrWords(bytes, texts, timeScaleS) {
    if (bytes == null) return null;
    var n = recordCount(bytes, 'asr_words');
    parallelLength(bytes, texts, n, 'asr_words', 'asr_word_text');
    var view = viewOf(bytes);
    var out = [];
    for (var i = 0; i < n; i++) {
      var o = i * 5;
      out.push({
        index: i,
        t0: decodeTime(view.getUint16(o, true), timeScaleS),
        t1: decodeTime(view.getUint16(o + 2, true), timeScaleS),
        outcome: enumName(WORD_OUTCOMES, view.getUint8(o + 4), 'word outcome'),
        text: texts[i],
      });
    }
    return out;
  }

  /** Detected PII: <HH. Returns [{index,t0,t1,category}] or null — null means never scanned. */
  function decodePiiMarks(bytes, categories, timeScaleS) {
    if (bytes == null) return null;
    var n = recordCount(bytes, 'pii_marks');
    parallelLength(bytes, categories, n, 'pii_marks', 'pii_category');
    var view = viewOf(bytes);
    var out = [];
    for (var i = 0; i < n; i++) {
      var o = i * 4;
      out.push({
        index: i,
        t0: decodeTime(view.getUint16(o, true), timeScaleS),
        t1: decodeTime(view.getUint16(o + 2, true), timeScaleS),
        category: categories[i],
      });
    }
    return out;
  }

  /** The branch-proposal lanes: <BHH. Returns [{lane,laneIndex,t0,t1,role}] or null. */
  function decodeBranchLanes(bytes, roles, timeScaleS) {
    if (bytes == null) return null;
    var n = recordCount(bytes, 'branch_lanes');
    parallelLength(bytes, roles, n, 'branch_lanes', 'branch_lane_role');
    var view = viewOf(bytes);
    var out = [];
    for (var i = 0; i < n; i++) {
      var o = i * 5;
      var laneIndex = view.getUint8(o);
      out.push({
        laneIndex: laneIndex,
        lane: strictEnumName(LANES, laneIndex, 'branch lane'),
        t0: decodeTime(view.getUint16(o + 1, true), timeScaleS),
        t1: decodeTime(view.getUint16(o + 3, true), timeScaleS),
        role: roles[i],
      });
    }
    return out;
  }

  /**
   * The flattened DDK matrix, rebuilt as n / width rows of width columns.
   * Returns {rows, width} or null; inner nulls are preserved.
   */
  function decodeMatrix(values, width) {
    if (values == null) return null;
    if (!width || width <= 0) throw new Error('matrix width ' + width + ' is not positive');
    if (values.length % width !== 0) {
      throw new Error(values.length + ' values is not a whole number of rows of ' + width);
    }
    var rows = [];
    for (var o = 0; o < values.length; o += width) rows.push(values.slice(o, o + width));
    return { rows: rows, width: width };
  }

  /**
   * Every block of one row, decoded together, with the cross-block indices checked.
   * `row` is the parquet row as an object; binary columns are Uint8Array or null.
   */
  function decodeRow(row) {
    var ts = row.time_scale_s;
    if (ts == null) {
      return {
        timeScaleS: null,
        wave: null,
        envelope: null,
        continuity: null,
        spans: null,
        spanLabels: null,
        spanSquim: null,
        asrWords: null,
        piiMarks: null,
        branchLanes: null,
        matrix: null,
      };
    }
    var spans = decodeSpans(row.spans, ts);
    var spanCount = spans == null ? null : spans.length;
    return {
      timeScaleS: ts,
      wave: decodeWaveMinmax(row.wave_minmax, row.wave_peak),
      envelope: decodeEnvelopeDbfs(row.env_dbfs),
      continuity: decodeContinuity(row.continuity),
      spans: spans,
      spanLabels: decodeSpanLabels(row.span_labels, row.span_label_name, spanCount),
      spanSquim: decodeSpanSquim(row.span_squim, spanCount),
      asrWords: decodeAsrWords(row.asr_words, row.asr_word_text, ts),
      piiMarks: decodePiiMarks(row.pii_marks, row.pii_category, ts),
      branchLanes: decodeBranchLanes(row.branch_lanes, row.branch_lane_role, ts),
      matrix: decodeMatrix(row.m_ddk_cv_instrument_reading, row.m_ddk_cv_instrument_reading_width),
    };
  }

  return {
    SCHEMA_VERSION: SCHEMA_VERSION,
    TIME_SCALE: TIME_SCALE,
    TRACE_POINTS: TRACE_POINTS,
    UNKNOWN_CODE: UNKNOWN_CODE,
    SPAN_ROWS: SPAN_ROWS,
    SPAN_ROW_NAMES: SPAN_ROW_NAMES,
    CLASSIFIERS: CLASSIFIERS,
    WORD_OUTCOMES: WORD_OUTCOMES,
    LANES: LANES,
    ENVELOPE_DBFS_RANGE: ENVELOPE_DBFS_RANGE,
    CONTINUITY_RANGE: CONTINUITY_RANGE,
    SCORE_RANGE: SCORE_RANGE,
    SQUIM_RANGES: SQUIM_RANGES,
    RECORD_SIZES: RECORD_SIZES,
    decodeTime: decodeTime,
    decodeValue: decodeValue,
    bucketExtent: bucketExtent,
    decodeWaveMinmax: decodeWaveMinmax,
    decodeTrace: decodeTrace,
    decodeEnvelopeDbfs: decodeEnvelopeDbfs,
    decodeContinuity: decodeContinuity,
    decodeSpans: decodeSpans,
    decodeSpanLabels: decodeSpanLabels,
    decodeSpanSquim: decodeSpanSquim,
    decodeAsrWords: decodeAsrWords,
    decodePiiMarks: decodePiiMarks,
    decodeBranchLanes: decodeBranchLanes,
    decodeMatrix: decodeMatrix,
    decodeRow: decodeRow,
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = SchemaDecode;
