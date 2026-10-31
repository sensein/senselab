# Wire protocol v2

The protocol between a host (`WorkerClient` in core) and a worker (`senselab_worker.serve`) in an
isolated env. `WIRE_VERSION = 2`. Closes R-37; implements I-2 (design §4.12) and the typed faults of
ADR-009.

**v2 against v1.** A response is one line per item result, then a `done` line, instead of one line
holding every result. The line budget therefore applies per item, and one large result can no
longer fail its batch-mates (R-68). Every message is checked by one validator on both sides, and
the error path is total (R-92). There is no migration: a v1 worker and a v2 host refuse each other
at the handshake.

## Framing

- One JSON object per line, UTF-8, terminated by `\n`. Wire JSON is not canonical (JCS is used
  only where something is hashed). It is written without `NaN` or `Infinity`; non-finite values
  travel in arrays. Numpy scalars are written as their `.item()`.
- The channel is private in both directions. The host starts
  `python -m senselab_worker.launch <backend_module>` with pipes on fd 0 and fd 1. Before the hello
  and before any backend code is imported, the launcher duplicates fd 0 and fd 1 to private,
  non-inheritable descriptors and serves the protocol on those; it then points fd 0 and `sys.stdin`
  at `/dev/null` and fd 1 and `sys.stdout` at stderr. A backend's prints cannot reach the protocol
  stream, and a child it starts with the default stdin (`subprocess.run(["ffmpeg", ...])`) reads
  `/dev/null`, not requests. A child made by `fork` without `exec` still holds the private
  descriptors. If fd 0 is closed when the worker starts, the launcher first opens it on
  `/dev/null`, so the channel reads end of file and the worker exits 0 after its hello. The
  backend module is imported during load.
- stderr is never parsed. The host drains it continuously and keeps the last 100 lines, which it
  attaches to every fault it raises itself.
- A line longer than 16 MiB (`MAX_LINE_BYTES`, newline included), or one that is not a JSON
  object, is a protocol violation. The budget is per line, and so per item result; there is no
  aggregate budget for a response.
- **Strict values.** The reader (`parse_line`) refuses, as a protocol violation: a key repeated
  within one object; a number that is `NaN`, `Infinity` or overflows to infinity (`1e999`); and
  nesting deeper than 64 containers (`MAX_DEPTH`, counting the line's own object). The writer
  (`dump_line`) refuses the same depth, as do `to_wire` and `jcs_dumps` (through the shared
  `canonical.entered` guard, which also refuses a container that contains itself), so neither
  side writes what the other refuses and no encoder or decoder recurses without bound.
- **One validator.** `validate_message(msg, kind)` is the schema of every message kind (`hello`,
  `init`, `ready`, `request`, `result`, `done`, `fatal`, `stop`). The writer calls it, through
  `dump_line(msg, kind)`, before every send, and the reader after every parse. Field types are
  exact: an integer is `type(v) is int`, so `true` and `1.0` are not integers; ids are strings of
  valid Unicode (no lone surrogates) of at most 1024 characters (`MAX_ID_CHARS`).
- Unknown keys inside a known message are ignored. A line that is not a valid message of the
  kind expected at that point is a protocol violation.
- One request is outstanding at a time. The host sends the next request only after it has read
  the response to the previous one; a second concurrent `run` on one client raises
  `ConcurrentUse` and sends nothing.

## Messages

| Direction | Kind | Shape |
|---|---|---|
| W → H | hello (first line) | `{"hello": {"wire": 2, "runtime": "<sha256>", "python": "3.10.14", "numpy": "1.23.5", "pid": 123}}`; `wire` and `pid` are ints, the others strings |
| H → W | init | `{"init": {"array_dir": "<abs path>", ...}}` |
| W → H | ready | `{"ready": {"load_s": <int or float>}}` |
| H → W | request | `{"id": "<uuid>", "items": [{"id": "<item>", "inputs": {...}}], "params": {...}}`; item ids unique |
| W → H | result (one per item) | `{"id": "<uuid>", "item": "<item>", "outputs": {...}}` or `{"id": "<uuid>", "item": "<item>", "error": {...}}`; exactly one of `outputs` and `error`; the error is a complete error record |
| W → H | done | `{"id": "<uuid>", "done": <number of items>}` |
| W → H | fatal | `{"error": <any>, "fatal": true}` |
| H → W | stop | `{"stop": true}` |

## Handshake

1. The worker writes the hello as soon as it starts.
   - `runtime` is `runtime_identity()`: take the `*.py` files under the `senselab_worker` package
     root, as POSIX paths relative to that root, sorted by path. Concatenate, per file, the
     UTF-8 path, `\0`, the file bytes, `\0`; the hex sha256 of the concatenation is the identity.
   - `python`, `numpy` and `pid` are recorded in the execution context (ADR-002 §2); they do not
     gate the connection.
2. Lines that are not JSON objects before the first JSON object (a `.pth` file or
   `sitecustomize` printing at interpreter start, before the launcher claims fd 1) are logged and
   discarded. The first JSON object must be a valid hello (`validate_message`, so `"wire": true`
   or `1.0` is refused). The host accepts only if `wire` is the int `2` and `runtime` equals its
   own. Otherwise, or if the first JSON object is not a valid hello, or none
   arrives within 60 s, it sends `{"stop": true}` and raises `ProtocolMismatch` (an
   `infrastructure` fault) naming both values. Nothing else is decoded.
3. The host sends `init`. `array_dir` is reserved; every other key is the backend's.
4. The worker calls `load(init)` and answers `ready` with the load time in seconds, or `fatal`
   and exits non-zero.

## Requests and responses

- `id` is a UUID string; every reply line carries the same `id`. Item ids are unique within a
  request.
- **Reply.** The worker writes one `result` line per item, in item order, each naming its `item`,
  then `{"id": <request>, "done": <n>}` with `n` the number of items. A request with no items is
  answered by `done` alone. The host reads exactly that sequence; a line with the wrong `id`,
  `item`, kind or count is a protocol violation.
- **Per-item budget.** An item whose result line would exceed `MAX_LINE_BYTES` is sent as that
  item's error, `{"type": "ResponseTooLarge", "fault": "resource", "gap": false, ...}`. Its
  batch-mates are unaffected. Large data travels as arrays.
- **Totality.** A failure in one item never fails its batch-mates or the worker. Building an
  item's result line is total: any exception from encoding its outputs (`to_wire`) or framing the
  line (`dump_line`) becomes that item's error record, and an error record always fits the budget
  (its text fields are bounded and ids are at most 1024 characters). `serve` loops until stop or
  end of input; nothing an item or `handle` does ends it, except `KeyboardInterrupt` and
  `SystemExit`. This is pinned by a property test that drives the real `serve` with hostile
  `handle` behaviours (`test_serve_is_total_fuzz`).
- **Backend contract.** The backend module defines
  `load: Callable[[dict], object]` and `handle: Callable[[object, list[dict], dict], list[dict]]`;
  the launcher serves them.
  - `load(init)` returns the state, once.
  - `handle(state, items, params)` gets the decoded items (`{"id", "inputs"}`) and returns one dict
    per item, in item order: `{"outputs": {...}}`, or `{"error": exc}` with an exception instance
    for a per-item failure. `serve` adds the ids and classifies each `exc` by the fault table.
  - If `handle` raises any exception other than `KeyboardInterrupt` and `SystemExit`, including
    a `BaseException` subclass, every item gets that error, classified.
  - If it returns anything but exactly a `list` of the right length whose entries are exactly
    `dict`s with one of `outputs` (exactly a `dict`) and `error` (an exception instance), every
    item gets an `infrastructure` error of type `ProtocolError`.
  - Outputs are encoded by `to_wire`, which accepts exact types only: `dict` with `str` keys,
    `list`, `tuple`, the scalars `canonical_scalar` accepts (so a numpy scalar is its `.item()`),
    and exactly `numpy.ndarray`. Anything else, a non-finite float, a string that is not valid
    Unicode, a cycle or nesting past `MAX_DEPTH` is a `WireValueError`, a typed `content` fault.
- A line the worker cannot parse as a valid `init`, `request` or `stop` message is answered with
  `fatal` (`ProtocolError`), and the worker exits `EXIT_FATAL`. In particular a message with a
  top-level `stop` key whose value is not exactly `true` is such a violation, not a stop: it is
  answered, never taken as a silent exit.

## Arrays

```json
{"__array__": {"path": "<array_dir>/<sha256>.npy", "sha256": "<hex>", "dtype": "<f4", "shape": [2, 16000]}}
```

- An object whose only key is `__array__` is an array reference; it may appear anywhere inside
  `init`, `inputs`, `params` or `outputs`. Every numpy array travels this way; none is inlined.
- The file is canonical `.npy` (ADR-006): version 1.0, C order, little-endian. The writer
  converts non-contiguous and big-endian arrays to that form, so `dtype` is always `<…` or `|…`.
  `sha256` is over the file bytes and names the file.
- Object and structured dtypes are refused with a `content` error (`WireValueError`): on the worker,
  for that item; on the host, before the request is sent.
- The writer writes to a temporary name `.<uuid4 hex>.tmp` in `array_dir` and renames, so a
  reader never sees a partial file and equal arrays share one file.
- The reader rejects a path that is not `<sha256>.npy` directly in `array_dir`. It opens the file
  with `O_NOFOLLOW | O_NONBLOCK` and requires a regular file, so a symlink or a FIFO is an
  `ArrayIntegrityError` and never blocks. It reads in chunks against the request deadline,
  recomputes the sha256 and checks `dtype` and `shape`. It loads the array into memory; it does
  not memory-map.
- **Decode failure scope.** On either side, a missing, unreadable or mismatched array fails only
  the item that references it, with an `infrastructure` error of type `ArrayIntegrityError`; on
  the worker, that item is not passed to `handle`. The same failure in `params` fails every item
  of the request; in `init` it is `fatal`.
- **Lifetime and ownership.** The host creates its own `array_dir` before spawning, with
  `mkdtemp(prefix="senselab-arrays-")` under a parent the caller may name; it never adopts an
  existing directory. It deletes only the names the runtime writes there (`<sha256>.npy` and
  `.<uuid4 hex>.tmp`), through the one deletion helper `remove_owned`, which removes a matching
  direct child without following symlinks, and a directory only if its name is a uuid4 hex (a
  staging directory the runtime made); a directory named like an array file is left. After each request it deletes every such file except
  those referenced by `init`, including files written for items that failed to encode. On close
  it deletes the rest and removes `array_dir` if it is then empty; any other name is left, with
  the directory, and logged. Cleanup failures are logged and never replace a result. The worker
  never deletes.

## Errors and faults

```json
{"type": "ContentFault", "message": "...", "fault": "content", "traceback": "...", "gap": false}
```

- **Typed at the source (ADR-009).** A backend states a fault's class by raising
  `ContentFault`, `ResourceFault` or `InfrastructureFault` (or a subclass). These decide the class,
  in that precedence when an exception is more than one (`InfrastructureFault`, then
  `ResourceFault`, then `ContentFault`), and the record has `"gap": false`. `ProtocolError`,
  `ArrayIntegrityError` are `InfrastructureFault`s; `WireValueError` is a `ContentFault`;
  `ResponseTooLarge` is a `ResourceFault`.
- **Fallback.** Every other exception is classified by the table, by `isinstance`, first match
  wins, and its record always has `"gap": true`, whatever class it gets:

| Exception | `fault` |
|---|---|
| `ValueError`, `TypeError`, `KeyError` | `content` |
| `MemoryError`, `TimeoutError`, any class whose MRO has a class named `OutOfMemoryError` | `resource` |
| anything else | `infrastructure` |

- `gap: true` marks a classification gap; the host logs it. A gap is not a verdict on the item: a
  `KeyError` from a backend bug is `content` by the table, but its gap says the class was guessed
  (R-83; the executor's rule that a gap fault is never cached is Phase 3).
- **The record is total** (`error_record`). `type` is the class name; `message` is `str(exc)`,
  else `repr(exc)`, else `<unprintable TypeName>`; `traceback` is the last 8 frames, formatted
  under the same guard. Each text field has every code point UTF-8 cannot encode, such as a lone
  surrogate from a `surrogateescape` filename, backslash-escaped, and is cut to 65 536 characters
  (`MAX_TEXT_CHARS`), so a record always encodes and always fits the line budget.
- Faults the host raises itself: `ProtocolMismatch`, `ProtocolError`, `WorkerLost` and
  `WorkerFatal` are `infrastructure`; a host timeout is `resource` (`TimeoutError`). Each fails
  every item of the request in flight, and the worker is discarded.
- **Fatal.** On a `fatal` line, during load or a request, the host raises
  `WorkerFatal(RuntimeError)` carrying the worker's error object; its fault is `infrastructure`,
  whatever the error object's `fault` says. The decode of a fatal's error object is total: `type`
  and `message` are kept when they are strings (else placeholders say what was received), and a
  missing or unknown `fault`, or a `gap` that is not a bool, makes the record `infrastructure`
  with `gap: true`. A fatal never loses the worker's message to a `ProtocolError`.

## Timeouts and shutdown

- Load and request timeouts are arguments to `WorkerClient` (default: none). One deadline covers
  each phase end to end: sending `init` and receiving `ready`; sending a request (by non-blocking
  writes, so a worker that has stopped reading cannot hold the host), receiving its response and
  reading its arrays. On expiry the host raises `TimeoutError` and stops the worker.
- **Process group.** The worker is started in a new session, so it leads its own process group,
  and the group is the unit of kill. The host notices the worker's exit without reaping it
  (`waitid(WNOWAIT)` on Linux, a kqueue `NOTE_EXIT` on macOS) and reaps it only after the group has
  been signalled, so the group id cannot be reused by another process while the host may still
  signal it. A worker whose exit has not been observed is held; after its exit, whether it is still
  unreaped is asked of `waitid(WNOHANG|WNOWAIT)` (Linux) or `ps -o ppid=` within 2 s (macOS). If it
  was reaped by someone else, or that cannot be told, the group is not signalled, which is logged,
  and every wait stays bounded. Signal errors other than "no such process" (and, on macOS, EPERM
  for a group whose only member is the zombie leader) are logged. Stopping
  the worker: on close, write `{"stop": true}` if the pipe takes it at once and wait for the worker
  to exit, leaving 3 s (`TERM_GRACE_S`) of the 10 s stop window (`STOP_GRACE_S`); then send SIGTERM
  and SIGCONT to the group and wait, for the rest of that window, until no live member remains;
  then SIGKILL the group and wait up to 5 s (`KILL_GRACE_S`); then stop the pipe readers within
  2 s (`JOIN_GRACE_S`). Pipes are closed only after that. `close()` returns within the sum of the
  three (`SHUTDOWN_BOUND_S`), also when called from another thread during a `run`, which then
  raises `WorkerLost`. A host fault skips the stop line.
- **Limit.** A descendant that calls `setsid` (or `setpgid`) leaves the group and is not reached;
  if it holds the worker's stdout or stderr, the host leaves those pipes open and logs it rather
  than block.
- **Worker lost.** EOF on the worker's stdout, or the exit of the worker process (after the
  lines it wrote are read), while a reply is awaited, including after some of a request's result
  lines and before its `done`, is `WorkerLost`, carrying the exit code as `returncode` and the
  stderr tail. A partial last line is discarded, never parsed. A forked descendant holding the
  channel open does not delay it.
- **Stop.** On `{"stop": true}` (exactly `true`), or EOF on stdin, the worker exits 0 without
  replying.

## Example

```text
W→H {"hello": {"wire": 2, "runtime": "9f2c…e1", "python": "3.10.14", "numpy": "1.23.5", "pid": 4711}}
H→W {"init": {"array_dir": "/tmp/sl-w-4711", "model": "fake"}}
W→H {"ready": {"load_s": 0.042}}
H→W {"id": "6d1f0c3a-5b7e-4c2d-9a10-3e8f2b7c4d55", "items": [{"id": "a", "inputs": {"x": {"__array__": {"path": "/tmp/sl-w-4711/3b5d…7a.npy", "sha256": "3b5d…7a", "dtype": "<f4", "shape": [1, 16000]}}}}, {"id": "b", "inputs": {}}], "params": {"faults": {"fail_items": ["b"]}}}
W→H {"id": "6d1f0c3a-5b7e-4c2d-9a10-3e8f2b7c4d55", "item": "a", "outputs": {"value": "c0ff…ee"}}
W→H {"id": "6d1f0c3a-5b7e-4c2d-9a10-3e8f2b7c4d55", "item": "b", "error": {"type": "ContentFault", "message": "fake failure for item b", "fault": "content", "traceback": "…", "gap": false}}
W→H {"id": "6d1f0c3a-5b7e-4c2d-9a10-3e8f2b7c4d55", "done": 2}
H→W {"stop": true}
```
