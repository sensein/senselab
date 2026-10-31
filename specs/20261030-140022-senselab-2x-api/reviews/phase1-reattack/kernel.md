# H8c surface K findings (commit b42d3cf0, macOS arm64, py3.12). Probes: p1..p7.py in this dir (run inside a worktree with uv run).

## XK-1 (high) Recording/Audio identity derivation and source-stamp check bypassed by `from_attributes=True`
Invariant: "forged identity is refused" (Recording docstring; ADR-009). `_derive_identity` and `_stamp_source` are mode before/wrap validators that act only `if isinstance(data, Mapping)`.
Repro:
    ns = types.SimpleNamespace(groups=dict(aud.groups), anchor=aud.anchor, identity="content:"+"0"*64)
    Audio.model_validate(ns, from_attributes=True).identity   # accepted; same for Recording
    # RegularGroup: SimpleNamespace(..., source=Path(p), source_stamp=SourceStamp.of(p).model_copy(update={"frames":10**9}))
    RegularGroup.model_validate(ns, from_attributes=True).frames  # 1000000000, never checked against the file
Observed: forged identity `content:000...` accepted; forged stamp accepted. Expected: ValidationError (as the dict path gives). Not a regression of a closed A-row that I know of.

## XK-2 (medium) StaleSource does not fire on an in-place rewrite that keeps size/mtime/inode/device
Invariant: "StaleSource fires on any source change". The docstring of `SourceStamp.checked` admits this for the memo path; the cold path is no better because the stamp's `header` covers only frames/channels/rate/format/subtype, not samples.
Repro: Audio.from_path(p); rewrite 100 samples with soundfile (same size) then `os.utime(p, ns=(old_atime, old_mtime))` (what `cp -p`, `rsync -t`, tar restore do).
 - `g.read()` returns the new samples, no error (memo path).
 - clear `_VERIFIED`/`_ROOTS` (= fresh process, unpickled copy): `g.read()` still succeeds with new data; `Audio.identity` still the old content root. A deepcopy in the cold state raises ValidationError (identity mismatch), not StaleSource.
Held: mtime change, rename-over (new inode), size change, deletion (FileNotFoundError, not StaleSource) are all detected.

## XK-3 (medium) FrozenMapping is re-initialisable: `m.__init__(...)` replaces contents under a stale digest
Repro: `rec.groups.__init__({"other": g})` changes Recording.groups (identity now != identity_of(groups)); `lk.params.__init__({"a":5})` changes LogicalKey.params while `lk.digest()` is unchanged (re-validating the dump gives a different digest); same for ExecutionContext.numerics/driver, License.verified, TaskSpec ports. Expected: immutable. (`object.__setattr__` in `__init__` with no re-entry guard.)

## XK-4 (medium/low) Frozen immutability holds only against normal attribute assignment
All reproduced:
 - `Span.model_config["frozen"]=False` (class flag flip) then `s.end_ns=-100` succeeds, yields end<start, no validation.
 - `s._canonical = other.canonical_digest()` (PrivateAttr is settable on a frozen model): `s4==s3` and equal hashes for different spans. `lk._digest = "f"*64` forges `LogicalKey.digest()`.
 - `s.__setstate__({...})` and `s.__dict__["end_ns"]=-1` mutate and bypass validation (deepcopy/model_copy then re-validate and refuse, so the damage is local).
Held: setattr, setflags/flags.writeable/as_strided on frozen arrays, pickle/copy/deepcopy/model_copy re-validation, model_construct.

## XK-5 (medium) N-3 confirmed and broader: stored provenance records are mutable and silently diverge from their id
Repro: `attrs={"ev":{"n":[1,2,3]}}; ext=[0.0,1.0]; eid=s.entity(prov_type="span",extent=ext,attributes=attrs); attrs["ev"]["n"].append(4); ext[1]=99.0`
 - stored Entity.attributes == {'ev': {'n': [1,2,3,4]}}, extent [0.0, 99.0] (extent is not copied at all); recomputing the id gives a different id.
 - `s.get_entity(eid).attributes["x"]=1` mutates the "append-only" store of a frozen dataclass; `fingerprint()` is blind to it (ids only). Same for Activity.parameters.
 - `ProvStore.merge` keeps the last record per id, so two stores holding the same id with different (mutated) content merge silently.
Answer to N-3: yes it breaks the invariant "id = digest of content" users rely on (ids are never re-verified on read either).

## XK-6 (medium) N-1 confirmed: write_jsonl is lossy; ids cannot be recomputed and the docstring promise is false
Repro: entity attributes with Path, tuple, nan/-inf, Enum, set, bytes, ndarray(2000).
 - `default=str` writes ndarray as "[   0    1    2 ... 1997 1998 1999]" (numpy summarisation): two different 2000-element arrays write byte-identical lines (modulo id). Data loss, not only type loss.
 - Path->str, Enum->"E.A", set->"{1, 2}", bytes->"b'ab'", tuple->list; NaN/-Infinity written as bare literals (not valid JSON).
 - read_jsonl then `ProvStore("r").entity(**read-back)` yields a different id. Docstring says "A store written by write_jsonl always reads back unchanged."

## XK-7 (low) write accepts what read refuses
`entity(prov_type="bogus")`, `agent(agent_type="bogus", ...)`, `environment(kind="bogus", ...)` are accepted at write; the written file raises ValueError on read_jsonl. Also `entity(attributes=[1])` raises AttributeError, extent is unvalidated (("a", nan) accepted).

## XK-8 (low) N-2 measured: element-wise ndarray normalisation
float64 arange: 1e6 elements 1.49 s, 110 MB peak (tracemalloc); 1e7 elements 15.46 s (~1.1 GB extrapolated); sha256 of the raw bytes 3.6 ms at 1e6 (about 400x). Linear, not quadratic. Real cost for any pipeline that records a feature matrix as an attribute; no guard or typed refusal on size.

## XK-9 (low) Totality gaps in `normalised`
 - `10**5000` raises a plain ValueError from int->str (4300-digit limit), not UnrecordableValue.
 - Empty arrays of refused dtypes are accepted (complex, datetime64, bytes, structured, object) although any element raises; empty structured arrays of equal itemsize collide: `normalised(np.zeros(0,dtype=[('a','i4'),('b','i4')]))==normalised(np.zeros(0,dtype=[('x','i8')]))` (both dtype name "void64"). Within the letter of the ADR equivalence list ("equal in dtype name, shape, values") but contradicts "structured refused".
 - PureWindowsPath("a\\b") and PurePosixPath("a/b") share a digest (as_posix); not on the ADR list.

## XK-10 (low) Untyped RecursionError at the LogicalKey params boundary
`LogicalKey(params={"a": <list nested 5000>})` and a self-referential dict raise RecursionError (in `_thaw`, before jcs's depth/cycle guard), not ValidationError. Nesting 65..~900 gives the typed error.

## XK-11 (low) Value-equal labels, unequal values
Intervals/Events label arrays keep their numpy width: `np.array(["a"],dtype="U1")` vs `"U3"` hold the same labels but different canonical_digest/eq; `"a\x00"` is silently stored as `"a"` (numpy NUL strip) yet unequal to `"a"`. Label identity depends on the longest string of the source array.

## XK-12 (low) Smaller looseness
 - Audio.from_array accepts NaN/inf and silently overflows float64 1e300 to inf (RuntimeWarning only); SignalArray does not check finiteness.
 - FiniteFloat is lax: `TimeMap(drift_ppm="5")` and `True` accepted (ns fields are strict). Non-Formatted models ignore unknown keys in model_validate while `model_copy(update=)` refuses them.
 - `TimeMap.map` can return a value outside int64 (`offset_ns=2**63-1`, `map(10)` = 2**63+9).
 - ModelRef(source="url", id="http://h/x#sha256=", ref="") does not round-trip through str/parse.
 - License.verified date accepts ISO week form "2026-W01-1" (py3.12 fromisoformat), not YYYY-MM-DD.
 - `identity.EQUIVALENCE` is a plain mutable module dict; assigning a class makes `acceptable(..., "equivalent")` true across contexts.
 - ExecutionContext/LogicalKey: Audio anchor clock_id is not checked against identity (Audio(anchor=TimeAnchor(clock_id="zzz")) accepted).

## Held (tried, no break)
- Provenance: `$`-key escaping and token injectivity ($set/$f/$int/$path across escaped keys); NFC duplicate keys; int boundary 2**53; non-finite floats; depth 62/63 ok, 64+ typed error incl. sets, ndarray-in-list token depth, object arrays, 600-deep Enum chains; cycles via lists and via Enum values; shallow wide dict 1e5.
- numpy: float16, big-endian == little-endian, non-contiguous, negative-stride, 0-d, uint64>2**53 ($int), longdouble/bytes/masked/matrix/memmap refused, non-empty complex/datetime/structured refused.
- Subclasses of str/int/float/bytes/dict/list (values and keys) refused; Path subclasses and Enum with unhashable value (list/dict) recorded; IntFlag combos; Decimal NaN/sNaN.
- Signals: int16/int32 PCM from_array bit-equal to WAV read (incl. extremes, 1-D vs 2-D, negative strides); Rate/Nanoseconds strict (bool/float/str/uint64>int64 refused); Span end<start, drift (-1e6,1e6); Formatted major mismatch, trailing-newline and uppercase tags refused; copy/deepcopy/pickle/model_copy re-validate; model_copy(update=) invalid refused; dict-path and TypeAdapter forged identity refused.
- Identity v2: params canonicalisation (1==1.0, NFC, order, tuple/list), non-JSON refused typed (except XK-10); regexes reject trailing newline/uppercase/unicode digits; pinned-model rules; context host/protocol excluded from fingerprint and context not in key; reuse policy coerced strictly ("EXACT", None, True, bytes refused); equivalent with empty EQUIVALENCE behaves as exact.
