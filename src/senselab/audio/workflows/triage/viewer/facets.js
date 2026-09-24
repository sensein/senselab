// Faceted selection over the corpus: narrow the drawn set by what a recording *is*, not by where
// its line sits on an axis.
//
// Within one facet the chosen values are a union; across facets they are an intersection; and the
// whole facet mask intersects the axis brushes. Absence is a value, not a silent drop: every facet
// carries an explicit absent bucket with its own count.
//
// Two counts are reported per value and they answer different questions. `total` is the count over
// the whole corpus and never moves. `available` is the count over the set every *other* facet and
// the brushes admit, so a sibling value keeps a meaningful number after its neighbour is chosen.
//
// See specs/20260924-recording-vectors-facets/facets.md.

'use strict';

var SchemaFacets = (function () {
  var ABSENT = '(absent)';

  /** Facet groups, in the order the panel lists them. */
  var GROUP_ORDER = ['decision', 'branch', 'identity', 'gate', 'gate outcome'];

  /** Columns never offered as a facet, with the reason the panel shows. */
  var REFUSED = {
    participant: '1,527 levels, one per person - a list, not a facet',
    session: 'one level per session; put participant on an axis instead',
  };

  /**
   * Every column the facet panel offers, in panel order.
   *
   * A categorical scalar faces by equality; a `set` column faces by membership, which is what its
   * own catalogue entry already points at ("filter by a term").
   *
   * @returns {Array<{col: Object, mode: string}>} the offered columns.
   */
  function catalogue() {
    var out = [];
    SchemaAxes.CATALOGUE.forEach(function (col) {
      if (REFUSED[col.name]) return;
      if (col.name.indexOf('.size') > 0) return;
      if (col.kind === 'categorical') out.push({ col: col, mode: 'scalar' });
      else if (col.kind === 'set') out.push({ col: col, mode: 'set' });
    });
    var rank = function (g) {
      var i = GROUP_ORDER.indexOf(g);
      return i < 0 ? GROUP_ORDER.length : i;
    };
    return out.map(function (f, i) { return { f: f, i: i }; })
      .sort(function (a, b) { return rank(a.f.col.group) - rank(b.f.col.group) || a.i - b.i; })
      .map(function (x) { return x.f; });
  }

  var CATALOGUE = catalogue();
  var BY_NAME = {};
  CATALOGUE.forEach(function (f) { BY_NAME[f.col.name] = f; });

  /** The facets the panel opens expanded: the ones carrying a decision the reader acts on. */
  var DEFAULT_OPEN = ['verdict', 'task', 'release', 'conformance_speech'];

  /**
   * A facet model over one set of rows.
   *
   * @param {Array<Object>} rows every recording, in corpus order.
   */
  function FacetModel(rows) {
    this.rows = rows;
    this.encodings = {};
    this.selection = {};
    this.base = null;
    this._masks = null;
  }

  /**
   * Build, once, the coded form of one facet column.
   *
   * A scalar becomes an `Int32Array` of term indices with -1 for absent; a set becomes one posting
   * list of row indices per term. Both are built lazily, because encoding all of the offered
   * columns up front over a 62,548-row corpus costs more than every interaction that follows.
   *
   * @param {string} name the column.
   * @returns {Object} the encoding.
   * @throws {Error} when the column is not one the panel offers.
   */
  FacetModel.prototype.encode = function (name) {
    if (this.encodings[name]) return this.encodings[name];
    var f = BY_NAME[name];
    if (!f) throw new Error('no facet column ' + name);
    var rows = this.rows;
    var terms = [];
    var index = {};
    var totals = [];
    var enc;
    var i;
    var j;

    function termIndex(t) {
      var k = index[t];
      if (k === undefined) { k = terms.length; terms.push(t); totals.push(0); index[t] = k; }
      return k;
    }

    if (f.mode === 'scalar') {
      var codes = new Int32Array(rows.length);
      var absent = 0;
      for (i = 0; i < rows.length; i++) {
        var v = rows[i][name];
        if (v == null) { codes[i] = -1; absent++; continue; }
        var k = termIndex(String(v));
        codes[i] = k;
        totals[k]++;
      }
      enc = { mode: 'scalar', codes: codes, terms: terms, totals: totals, absent: absent, index: index };
    } else {
      var lists = {};
      var absentRows = [];
      for (i = 0; i < rows.length; i++) {
        var list = rows[i][name];
        if (list == null) { absentRows.push(i); continue; }
        for (j = 0; j < list.length; j++) {
          var kk = termIndex(String(list[j]));
          (lists[kk] || (lists[kk] = [])).push(i);
          totals[kk]++;
        }
      }
      enc = {
        mode: 'set',
        postings: terms.map(function (_, t) { return Int32Array.from(lists[t] || []); }),
        terms: terms,
        totals: totals,
        absent: absentRows.length,
        absentRows: Int32Array.from(absentRows),
        index: index,
      };
    }
    this.encodings[name] = enc;
    return enc;
  };

  /**
   * The base mask the facets narrow: the rows the axis brushes admit.
   *
   * @param {Uint8Array|null} mask one byte per row, or null for every row.
   */
  FacetModel.prototype.setBase = function (mask) {
    this.base = mask || null;
  };

  /**
   * The chosen values of one facet.
   *
   * @param {string} name the column.
   * @returns {Array<string>} the values, as written.
   */
  FacetModel.prototype.chosen = function (name) {
    var sel = this.selection[name];
    return sel ? sel.slice() : [];
  };

  /**
   * Add or remove one value from one facet.
   *
   * @param {string} name the column.
   * @param {string} term the value, or `ABSENT`.
   * @returns {Array<string>} what is chosen afterwards.
   */
  FacetModel.prototype.toggle = function (name, term) {
    var sel = this.selection[name] || [];
    var at = sel.indexOf(term);
    sel = at < 0 ? sel.concat([term]) : sel.slice(0, at).concat(sel.slice(at + 1));
    if (sel.length) this.selection[name] = sel;
    else delete this.selection[name];
    this._masks = null;
    return this.chosen(name);
  };

  /**
   * Drop one facet's choices, or every facet's.
   *
   * @param {string|null} name the column, or null for all of them.
   */
  FacetModel.prototype.clear = function (name) {
    if (name == null) this.selection = {};
    else delete this.selection[name];
    this._masks = null;
  };

  /**
   * The facets currently narrowing anything.
   *
   * @returns {Array<string>} their column names, sorted.
   */
  FacetModel.prototype.activeNames = function () {
    var self = this;
    return Object.keys(this.selection).filter(function (n) { return self.selection[n].length; }).sort();
  };

  /**
   * How many values are chosen across every facet.
   *
   * @returns {number} the count.
   */
  FacetModel.prototype.chosenCount = function () {
    var self = this;
    return this.activeNames().reduce(function (n, k) { return n + self.selection[k].length; }, 0);
  };

  /**
   * The rows one facet admits on its own.
   *
   * A scalar reads its code array once; a set walks only the posting lists of the chosen terms, so
   * choosing a rare term costs that term's size rather than the corpus's.
   *
   * @param {string} name the column.
   * @returns {Uint8Array} one byte per row.
   */
  FacetModel.prototype.passMask = function (name) {
    var enc = this.encode(name);
    var sel = this.selection[name] || [];
    var out = new Uint8Array(this.rows.length);
    var wantAbsent = sel.indexOf(ABSENT) >= 0;
    var i;
    var j;
    if (enc.mode === 'scalar') {
      var want = new Uint8Array(enc.terms.length);
      for (i = 0; i < sel.length; i++) {
        var k = enc.index[sel[i]];
        if (k !== undefined) want[k] = 1;
      }
      for (i = 0; i < out.length; i++) {
        var c = enc.codes[i];
        out[i] = c < 0 ? (wantAbsent ? 1 : 0) : want[c];
      }
    } else {
      for (i = 0; i < sel.length; i++) {
        var kk = enc.index[sel[i]];
        if (kk === undefined) continue;
        var p = enc.postings[kk];
        for (j = 0; j < p.length; j++) out[p[j]] = 1;
      }
      if (wantAbsent) for (j = 0; j < enc.absentRows.length; j++) out[enc.absentRows[j]] = 1;
    }
    return out;
  };

  /** Every active facet's own mask, computed once per selection change. */
  FacetModel.prototype._ensure = function () {
    if (this._masks) return this._masks;
    var self = this;
    var names = this.activeNames();
    this._masks = { names: names, per: names.map(function (n) { return self.passMask(n); }) };
    return this._masks;
  };

  /**
   * The facet mask over the corpus.
   *
   * @param {string} [excluding] leave this facet out, which is what makes a sibling value's count
   *   meaningful after its neighbour is chosen.
   * @returns {Uint8Array|null} one byte per row, or null when nothing is narrowing.
   */
  FacetModel.prototype.mask = function (excluding) {
    var m = this._ensure();
    var used = [];
    var a;
    var i;
    for (a = 0; a < m.names.length; a++) if (m.names[a] !== excluding) used.push(m.per[a]);
    if (!used.length) return null;
    var out = new Uint8Array(this.rows.length);
    out.set(used[0]);
    for (a = 1; a < used.length; a++) {
      var p = used[a];
      for (i = 0; i < out.length; i++) if (!p[i]) out[i] = 0;
    }
    return out;
  };

  /**
   * How many rows a mask keeps, the base mask included.
   *
   * @param {Uint8Array|null} mask the facet mask, or null.
   * @returns {number} the count.
   */
  FacetModel.prototype.countMask = function (mask) {
    var base = this.base;
    var n = 0;
    for (var i = 0; i < this.rows.length; i++) {
      if (base && !base[i]) continue;
      if (mask && !mask[i]) continue;
      n++;
    }
    return n;
  };

  /**
   * How many rows the brushes alone admit: the denominator the facets narrow.
   *
   * @returns {number} the count.
   */
  FacetModel.prototype.before = function () {
    return this.countMask(null);
  };

  /**
   * How many rows survive every facet and the brushes: the denominator after.
   *
   * @returns {number} the count.
   */
  FacetModel.prototype.after = function () {
    return this.countMask(this.mask());
  };

  /**
   * One facet's values, each with both counts and whether it is chosen.
   *
   * Sorted by the column's declared ordering where it has one, so `pass < flag < discard` reads as
   * severity; by `available` otherwise. The absent bucket is always last and always present, so a
   * null can be selected rather than silently carried.
   *
   * @param {string} name the column.
   * @returns {Object} `{name, label, mode, nullMeans, chosen, values}`.
   */
  FacetModel.prototype.values = function (name) {
    var f = BY_NAME[name];
    var enc = this.encode(name);
    var sel = this.selection[name] || [];
    var admit = this.mask(name);
    var base = this.base;
    var avail = new Int32Array(enc.terms.length);
    var availAbsent = 0;
    var i;
    var j;

    if (enc.mode === 'scalar') {
      for (i = 0; i < this.rows.length; i++) {
        if (base && !base[i]) continue;
        if (admit && !admit[i]) continue;
        var c = enc.codes[i];
        if (c < 0) availAbsent++;
        else avail[c]++;
      }
    } else {
      for (i = 0; i < enc.postings.length; i++) {
        var p = enc.postings[i];
        var n = 0;
        for (j = 0; j < p.length; j++) {
          var r = p[j];
          if (base && !base[r]) continue;
          if (admit && !admit[r]) continue;
          n++;
        }
        avail[i] = n;
      }
      for (j = 0; j < enc.absentRows.length; j++) {
        var rr = enc.absentRows[j];
        if (base && !base[rr]) continue;
        if (admit && !admit[rr]) continue;
        availAbsent++;
      }
    }

    var values = enc.terms.map(function (t, k) {
      return { term: t, label: t, total: enc.totals[k], available: avail[k], chosen: sel.indexOf(t) >= 0, absent: false };
    });
    var order = SchemaAxes.ORDERINGS[name];
    if (order) {
      values.sort(function (a, b) {
        var ia = order.indexOf(a.term);
        var ib = order.indexOf(b.term);
        if (ia < 0) ia = order.length;
        if (ib < 0) ib = order.length;
        return ia - ib || b.available - a.available || (a.term < b.term ? -1 : 1);
      });
    } else {
      values.sort(function (a, b) {
        return b.available - a.available || b.total - a.total || (a.term < b.term ? -1 : 1);
      });
    }
    values.push({
      term: ABSENT,
      label: 'absent',
      total: enc.absent,
      available: availAbsent,
      chosen: sel.indexOf(ABSENT) >= 0,
      absent: true,
    });
    return {
      name: name,
      label: f.col.label,
      mode: f.mode,
      nullMeans: f.col.nullMeans,
      chosen: sel.slice(),
      values: values,
    };
  };

  return {
    ABSENT: ABSENT,
    CATALOGUE: CATALOGUE,
    BY_NAME: BY_NAME,
    GROUP_ORDER: GROUP_ORDER,
    REFUSED: REFUSED,
    DEFAULT_OPEN: DEFAULT_OPEN,
    FacetModel: FacetModel,
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = SchemaFacets;
