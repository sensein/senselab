# H8c surface C findings (canonical encoding, hashing, tagged digests), commit b42d3cf0, macOS arm64

## XC-1 (medium) NFC depends on the Unicode database, so digests differ between Python 3.10 and 3.12
Invariant: ADR-006 "text: UTF-8 NFC" + cross-version stability (same value, same digest across Python 3.10/3.12).
`jcs_dumps` normalises with `unicodedata.normalize`, whose result for a string containing a combining mark
assigned after Unicode 13 depends on the interpreter's database (3.10 = 13.0, 3.12 = 15.0).
Repro (same script, two interpreters):
    s = "a\U0001e08f̣"   # U+1E08F is ccc=230 in Unicode 15, unassigned (ccc 0) in 13
    tagged_digest("value@1", s)
    py3.10: b2afac1f48eaad58...   py3.12: 5d304d324ad08a7c...   (also differs as a dict key)
Expected: one digest. Observed: two; NFC gives [a, 1E08F, 0323] on 3.10 and [1EA1, 1E08F] on 3.12.
Not a regression of a closed A-row (no pinned test). Fix direction: vendor a frozen NFC table or refuse
code points unassigned in the pinned Unicode version.

## XC-2 (medium) `value@1`: an array and its sha256 hex string have the same digest and compare equal
Invariant: ADR-006 value@1 "an array becomes the sha256 of its canonical .npy"; equivalence list does not
admit array == str.
Repro (core env):
    class M(Frozen): x: Any
    a = np.arange(4, dtype="<f8"); sha = sha256(canonical_npy_bytes(a)).hexdigest()
    M(x=a) == M(x=sha)                       # True
    M(x=a).canonical_digest() == M(x=sha).canonical_digest()   # True
Observed: equal; expected unequal (the array is replaced by an untagged string, unlike `$nd`/`$array` tokens
used by provenance and store). Matters for Any/union fields. Not a regression.

## XC-3 (medium) `Frozen._canonical_value` has no cycle/depth guard: RecursionError, not the shared ValueError
Invariant: `canonical.entered` docstring "Every encoder that recurses into JSON containers enters each one
through this, so a cycle or excess nesting raises the same error wherever values are encoded"; H-R rulings on
the shared guard.
Repro:
    c = []; c.append(c); M(x=c)            # constructs fine
    M(x=c).canonical_digest()              # RecursionError (traceback: _canonical_value x ~1000)
    M(x=deep(5000)).canonical_digest()     # RecursionError; deep(100) gives ValueError("deeper than 64")
Also hits __eq__/__hash__ since they go through canonical_digest. Expected: ValueError via `entered`.
Related to round-1 line 75 (RecursionError on cycle) but in a path the fix did not cover.

## XC-4 (medium) canonical .npy does not canonicalise NaN bit patterns, -0.0, or non-0/1 bool bytes
Invariant: ADR-006 equivalence list "every NaN, whatever its sign or payload, in the forms that admit NaN";
`-0.0`, `0.0` equal; canonical_npy_bytes "byte-identical across ... platform".
Repro (3.10/np1.23 and 3.12/np2 identical):
    nan_pos = np.array([np.nan]); nan_neg = np.array([0xFFF8000000000000], "<u8").view("<f8")
    canonical_npy_bytes(nan_pos) != canonical_npy_bytes(nan_neg)         # distinct digests
    np.array([np.inf]) - np.array([np.inf])  -> 0x7FF8... here (arm64); x86 yields 0xFFF8... (sign set)
    so the same computation hashes differently across platforms (audio@2, value@1, output@2 all hash the npy).
    canonical_npy_bytes(np.array([0.0])) != canonical_npy_bytes(np.array([-0.0]))
    canonical_npy_bytes(np.frombuffer(bytes([2]), dtype=bool)) != canonical_npy_bytes(np.array([True]))
Expected: one form per logical value (at minimum NaN -> canonical quiet NaN; bool -> 0/1). The x86 vs arm
difference was inferred from the sign-bit behaviour of the architectures, only the arm result was run here.

## XC-5 (low) `jcs_dumps` shared DAG is expanded as a tree: 64 references give unbounded work
`x=[]; for _ in range(22): x=[x,x]` -> jcs_dumps takes 7.8 s and writes 21 MB; depth 64 is 2**64 items.
Depth guard bounds nesting, not size. Same in store `_encode`, provenance `normalised`. Cheap-input DoS.

## XC-6 (low) `jcs_loads` accepts what its doc says it refuses or cannot have been written by `jcs_dumps`
- `jcs_loads("1e400")` -> `inf` (doc: ValueError on Infinity; only the literal is refused, overflow is not).
- duplicate keys: `{"a":1,"a":2}` -> `{'a': 2}` silently.
- `jcs_loads("["*100000)` -> RecursionError, not ValueError; `jcs_loads("-0.0")` -> `-0.0` (non-canonical input accepted).

## XC-7 (low) sets: NFC-equal / NaN elements are not merged
`normalised({"é","é"})` -> `{"$set":["é","é"]}` vs `{"$set":["é"]}` for one element; same for two
distinct NaN objects. ADR-006: "two strings ... equal after NFC share every digest" is violated for sets.

## XC-8 (low) doc drift
ADR-006 lists 8 tags; `DIGEST_TAGS` has 9 (`provenance-id@2` is registered, not in the ADR line).

## Held (tried, no break)
- RFC 8785 number vectors (0, -0, 5e-324, max double, 2^53, 9.999999999999997e+22, 1e+23, 999999999999999700000,
  1e+21, 9.999999999999997e-7, 0.000001, 333333333.33333325 ...): all exact; 1e-7 -> 1e-7, -0.0 -> 0.
- I-JSON range: 2^53-1 ok, 2^53 and -2^53 refused for int; float 2^53 formats as 9007199254740992; non-finite refused.
- Escaping: U+0000-1F as \u00xx / \b\t\n\f\r, `"` `\\`, `/` and DEL literal, U+2028/2029 raw (matches JCS).
- Lone surrogates (value and key) refused with UnicodeEncodeError (a ValueError); valid pair ok.
- UTF-16 key order (U+1F600 before U+E000 and U+FFFF; RFC 8785 sorted-keys example) correct.
- Exact-type refusal of subclasses, numpy longdouble/complex/datetime64/bytes_, non-str keys, masked/matrix arrays.
- numpy scalars (float32/float16/int64/uint64 over range/bool_/str_) identical under np1.23 and np2.5.
- npy: BE vs LE, F vs C, strided views, 0-d, empty, every b/i/u/f/U dtype, object/complex/M8/S/longdouble refused;
  full probe output diffed between py3.10+np1.23 and py3.12+np2.5: identical.
- Tags: every tag ASCII with no NUL, payload JCS never starts with 0x00, so tag||0x00||payload is unambiguous; no
  tag is a prefix of another.
- `$`-key escaping vs `$int` token: {"$int":"5"} vs 5**30, {"$$int":..} all distinct.
- depth 64 passes, 65 refused in provenance, value@1 (list nesting), jcs; shared (non-cyclic) child not a cycle.
- Unicode DB: single-code-point NFC identical 3.10 vs 3.12 (only combining-class reordering differs, XC-1).
