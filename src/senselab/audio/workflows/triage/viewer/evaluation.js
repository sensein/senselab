// The "evaluate triage" summary: trimmable recordings, raw quality issues and what enhancement
// resolved, clipping and its re-assessment, and other speakers inside the task extent. Each count
// names the facet and term that select its recordings, so the panel can hand the selection to the
// facet model and the keyboard steps through it.

'use strict';

var TriageEvaluation = (function () {
  var QUALITY_CHECKS = ['noise_floor', 'low_snr', 'clipping', 'dropout'];
  var CLIP_STATES = ['kept_inconsistent', 'kept_consistent', 'withdrawn_only', 'none'];
  var SIGNALS = ['diarization', 'separation', 'reviewer'];

  function countScalar(rows, name, term) {
    var n = 0;
    for (var i = 0; i < rows.length; i++) if (rows[i][name] != null && String(rows[i][name]) === term) n++;
    return n;
  }

  function countSet(rows, name, term) {
    var n = 0;
    for (var i = 0; i < rows.length; i++) {
      var list = rows[i][name];
      if (list == null) continue;
      for (var j = 0; j < list.length; j++) if (String(list[j]) === term) { n++; break; }
    }
    return n;
  }

  function scalarTerms(rows, name) {
    var seen = Object.create(null);
    var out = [];
    for (var i = 0; i < rows.length; i++) {
      var v = rows[i][name];
      if (v == null) continue;
      var t = String(v);
      if (!seen[t]) { seen[t] = true; out.push(t); }
    }
    return out.sort();
  }

  /** Sum of `sumName` over the rows whose `name` is `term`; null when no such row carries it. */
  function sumWhere(rows, name, term, sumName) {
    var total = null;
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      if (r[name] == null || String(r[name]) !== term || r[sumName] == null) continue;
      total = (total || 0) + Number(r[sumName]);
    }
    return total;
  }

  function item(label, count, facet, term, seconds) {
    return { label: label, count: count, facet: facet, term: term, seconds: seconds == null ? null : seconds };
  }

  /**
   * The panel's sections over a set of rows.
   *
   * @param {Array<Object>} rows the decoded parquet rows.
   * @returns {Array<{title: string, items?: Array<Object>, rows?: Array<Object>}>} the sections; an
   *   item's `facet` and `term` select its recordings, and `seconds` totals its trim or clip seconds.
   */
  function summarise(rows) {
    var trim = {
      title: 'task extent: trimmable',
      items: [
        item('trimmable', countScalar(rows, 'trimmable', 'true'), 'trimmable', 'true',
          sumWhere(rows, 'trimmable', 'true', 'trim_s')),
        item('not trimmable', countScalar(rows, 'trimmable', 'false'), 'trimmable', 'false',
          sumWhere(rows, 'trimmable', 'false', 'trim_s')),
      ],
    };
    var quality = {
      title: 'raw quality → enhanced',
      header: ['check', 'flagged on raw', 'resolved in enhanced', 'still flagged'],
      rows: QUALITY_CHECKS.map(function (check) {
        return {
          check: check,
          raw: item(check + ' on raw', countSet(rows, 'q_raw_issues', check), 'q_raw_issues', check),
          resolved: item(check + ' resolved', countSet(rows, 'q_resolved_by_enhanced', check), 'q_resolved_by_enhanced', check),
          unresolved: item(check + ' still flagged', countSet(rows, 'q_unresolved', check), 'q_unresolved', check),
        };
      }),
    };
    var clipping = {
      title: 'clipping and its re-assessment',
      items: CLIP_STATES.map(function (state) {
        return item(state.replace(/_/g, ' '), countScalar(rows, 'clip_state', state), 'clip_state', state,
          state === 'none' ? null : sumWhere(rows, 'clip_state', state, 'clip_s'));
      }),
    };
    var speakers = {
      title: 'another speaker inside the task extent',
      items: SIGNALS.map(function (signal) {
        return item(signal + ' says so', countSet(rows, 'ms_signals', signal), 'ms_signals', signal);
      }).concat(scalarTerms(rows, 'ms_agreement').map(function (term) {
        return item('agreement: ' + term, countScalar(rows, 'ms_agreement', term), 'ms_agreement', term);
      })),
    };
    return [trim, quality, clipping, speakers];
  }

  return {
    QUALITY_CHECKS: QUALITY_CHECKS,
    CLIP_STATES: CLIP_STATES,
    SIGNALS: SIGNALS,
    summarise: summarise,
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = TriageEvaluation;
