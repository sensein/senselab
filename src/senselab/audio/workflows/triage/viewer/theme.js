// The page's theme: system, light or dark, and the canvas colours read from the CSS tokens.
//
// Every colour the canvases draw is a `--c-<name>` custom property in styles.css, defined for both
// themes; `color(name)` reads the one in force. Nothing here is a colour literal except FALLBACK,
// which is the dark theme's values for a host with no stylesheet (node, a test).

'use strict';

var Theme = (function () {
  var MODES = ['system', 'light', 'dark'];
  var STORAGE_KEY = 'recording-vectors-theme';
  var SERIES_N = 10;

  var TOKENS = [
    'ground', 'panel', 'panel-2', 'ink', 'ink-2', 'ink-3', 'accent', 'warn', 'warn-bg', 'bad', 'good',
    'teal', 'orange', 'violet', 'muted', 'unknown', 'grid', 'rule', 'label-bg', 'tip-bg', 'tip-line',
    'dim', 'rail', 'rail-bg', 'absent-fill', 'brush-fill', 'brush-bad-fill', 'brush-bad-line',
    'brush-bad-ink', 'focus', 'hover',
  ];
  for (var s = 0; s < SERIES_N; s++) TOKENS.push('series-' + s);

  var FALLBACK = {
    ground: '#0d1016', panel: '#12161f', 'panel-2': '#161a22', ink: '#e6e9ef', 'ink-2': '#9aa3b2',
    'ink-3': '#6b7280', accent: '#6ea8ff', warn: '#e8c07d', 'warn-bg': '#2a2018', bad: '#ff8fa3',
    good: '#8ce99a', teal: '#4fd1c5', orange: '#ffb454', violet: '#c4a7ff', muted: '#5a6373',
    unknown: '#7a8291', grid: '#2a3140', rule: '#212734', 'label-bg': '#12151c',
    'tip-bg': 'rgba(8,10,14,0.94)', 'tip-line': '#4b5361', dim: '#3a4150', rail: '#2f3542',
    'rail-bg': '#242a35', 'absent-fill': '#8a6d3b', 'brush-fill': 'rgba(110,168,255,0.18)',
    'brush-bad-fill': 'rgba(200,80,80,0.10)', 'brush-bad-line': '#d06060', 'brush-bad-ink': '#e08a8a',
    focus: '#ffffff', hover: '#ffd166',
    'series-0': '#6ea8ff', 'series-1': '#ffb454', 'series-2': '#8ce99a', 'series-3': '#ff8fa3',
    'series-4': '#c4a7ff', 'series-5': '#4fd1c5', 'series-6': '#f6e05e', 'series-7': '#fc8181',
    'series-8': '#9ae6b4', 'series-9': '#b794f4',
  };

  var cache = {};
  var listeners = [];

  function root() { return typeof document !== 'undefined' ? document.documentElement : null; }

  function color(name) {
    if (name in cache) return cache[name];
    var el = root();
    var value = '';
    if (el && typeof getComputedStyle === 'function') {
      value = getComputedStyle(el).getPropertyValue('--c-' + name).trim();
    }
    cache[name] = value || FALLBACK[name];
    return cache[name];
  }

  function series() {
    var out = [];
    for (var i = 0; i < SERIES_N; i++) out.push(color('series-' + i));
    return out;
  }

  function next(mode) {
    var i = Math.max(0, MODES.indexOf(mode));
    return MODES[(i + 1) % MODES.length];
  }

  function stored() {
    try {
      var v = window.localStorage.getItem(STORAGE_KEY);
      return MODES.indexOf(v) >= 0 ? v : 'system';
    } catch (e) {
      return 'system';
    }
  }

  function mode() {
    var el = root();
    var set = el ? el.getAttribute('data-theme') : null;
    return set === 'light' || set === 'dark' ? set : 'system';
  }

  function changed() {
    cache = {};
    listeners.forEach(function (fn) { fn(mode()); });
  }

  function apply(chosen) {
    var el = root();
    if (el) {
      if (chosen === 'light' || chosen === 'dark') el.setAttribute('data-theme', chosen);
      else el.removeAttribute('data-theme');
    }
    try { window.localStorage.setItem(STORAGE_KEY, chosen); } catch (e) { /* storage unavailable */ }
    changed();
  }

  function onChange(fn) { listeners.push(fn); }

  function wire(button) {
    function label() {
      var m = mode();
      button.textContent = 'theme: ' + m;
      button.title = 'colour theme: ' + m + ' (click for ' + next(m) + ')';
    }
    button.addEventListener('click', function () { apply(next(mode())); label(); });
    if (typeof window !== 'undefined' && window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      var onSystem = function () { if (mode() === 'system') changed(); };
      if (mq.addEventListener) mq.addEventListener('change', onSystem);
      else if (mq.addListener) mq.addListener(onSystem);
    }
    label();
  }

  return {
    MODES: MODES, TOKENS: TOKENS, FALLBACK: FALLBACK, STORAGE_KEY: STORAGE_KEY, SERIES_N: SERIES_N,
    color: color, series: series, next: next, stored: stored, mode: mode, apply: apply,
    onChange: onChange, wire: wire,
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = Theme;
