// Stepping through the current selection from the keyboard: which row a key moves to.

'use strict';

var SelectionKeys = (function () {
  var ACTIONS = { j: 'next', ArrowDown: 'next', k: 'prev', ArrowUp: 'prev', Home: 'first', End: 'last' };
  var HELP = 'j / ↓ next · k / ↑ previous · Home / End first / last — within the current selection';

  function actionFor(event) {
    if (event.altKey || event.ctrlKey || event.metaKey) return null;
    var t = event.target;
    if (t && (t.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName || ''))) return null;
    return ACTIONS[event.key] || null;
  }

  function selectedIndices(selected) {
    var out = [];
    for (var i = 0; i < selected.length; i++) if (selected[i] === 1) out.push(i);
    return out;
  }

  // Where one action moves from `current`, over the selected rows in row order. `clamped` says the
  // move was asked past an end and stayed there; `index` is -1 only where nothing is selected.
  function step(selected, current, action) {
    var order = selectedIndices(selected);
    var total = order.length;
    if (!total) return { index: -1, position: 0, total: 0, clamped: false };
    var at = order.indexOf(current);
    var to;
    if (action === 'first') to = 0;
    else if (action === 'last') to = total - 1;
    else if (at >= 0) to = action === 'next' ? Math.min(total - 1, at + 1) : Math.max(0, at - 1);
    else if (action === 'next') {
      // The open recording is outside the selection: the next selected one after it, else the last.
      to = order.findIndex(function (i) { return i > current; });
      if (to < 0) to = total - 1;
    } else {
      to = -1;
      for (var p = total - 1; p >= 0; p--) if (order[p] < current) { to = p; break; }
      if (to < 0) to = current < 0 ? total - 1 : 0;
    }
    var clamped = at >= 0 && to === at && (action === 'next' || action === 'prev');
    return { index: order[to], position: to + 1, total: total, clamped: clamped };
  }

  // Where one row stands in the selection, for the readout when a row is opened by a click.
  function locate(selected, index) {
    var order = selectedIndices(selected);
    return { index: index, position: order.indexOf(index) + 1, total: order.length, clamped: false };
  }

  function describe(move) {
    if (move.total === 0) return 'nothing selected';
    if (move.position === 0) return 'not in the selection (' + move.total.toLocaleString() + ' selected)';
    var where = move.position.toLocaleString() + ' of ' + move.total.toLocaleString();
    if (!move.clamped) return where;
    return where + (move.position === 1 ? ' · first in the selection' : ' · last in the selection');
  }

  return { ACTIONS: ACTIONS, HELP: HELP, actionFor: actionFor, step: step, locate: locate, describe: describe };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = SelectionKeys;
