# Wire protocol v1

The protocol between a host (`WorkerClient` in core) and a worker (`senselab_worker.serve`) in an
isolated env. `WIRE_VERSION = 1`. Closes R-37; implements I-2 (design §4.12) and the fault table of
ADR-009.

## Framing

- One JSON object per line, UTF-8, terminated by `\n`. Wire JSON is not canonical (JCS is used
  only where something is hashed). It is written without `NaN` or `Infinity`; non-finite values
  travel in arrays. Numpy scalars are written as their `.item()`.
- Host → worker on the worker's stdin. Worker → host on a private duplicate of the worker's fd 1.
  At startup, before the hello and before any backend code is imported, the worker points fd 1 and
  `sys.stdout` at stderr, so library prints cannot reach the protocol stream.
- stderr is never parsed. The host drains it continuously and keeps the last 100 lines, which it
  attaches to every fault it raises itself.
- A line longer than 16 MiB, or one that is not a JSON object, is a protocol violation.
- Unknown keys inside a known message are ignored. A line with no recognised top-level key is a
  protocol violation.
- One request is outstanding at a time. The host sends the next request only after it has read
  the response to the previous one.

## Messages

| Direction | Message | Shape |
|---|---|---|
| W → H | hello (first line) | `{"hello": {"wire": 1, "runtime": "<sha256>", "python": "3.10.14", "numpy": "1.23.5", "pid": 123}}` |
| H → W | init | `{"init": {"array_dir": "<abs path>", ...}}` |
| W → H | ready | `{"ready": {"load_s": float}}` |
| H → W | request | `{"id": "<uuid>", "items": [{"id": "<item>", "inputs": {...}}], "params": {...}}` |
| W → H | response | `{"id": "<uuid>", "results": [{"id": "<item>", "outputs": {...}} \| {"id": "<item>", "error": {...}}]}`; each entry has exactly one of `outputs` and `error` |
| W → H | fatal | `{"error": {...}, "fatal": true}` |
| H → W | stop | `{"stop": true}` |

## Handshake

1. The worker writes the hello as soon as it starts.
   - `runtime` is `runtime_identity()`: take the `*.py` files under the `senselab_worker` package
     root, as POSIX paths relative to that root, sorted by path. Concatenate, per file, the
     UTF-8 path, `\0`, the file bytes, `\0`; the hex sha256 of the concatenation is the identity.
   - `python`, `numpy` and `pid` are recorded in the execution context (ADR-002 §2); they do not
     gate the connection.
2. The host accepts only if `wire` and `runtime` both equal its own. Otherwise, or if the first
   line is not a hello, or none arrives within 60 s, it sends `{"stop": true}` and raises
   `ProtocolMismatch` (an `infrastructure` fault) naming both values. Nothing else is decoded.
3. The host sends `init`. `array_dir` is reserved; every other key is the backend's.
4. The worker calls `load(init)` and answers `ready` with the load time in seconds, or `fatal`
   and exits non-zero.

## Requests and responses

- `id` is a UUID string; the response carries the same `id`. Item ids are unique within a request.
- `results` has exactly one entry per item, in item order, with matching ids. A failure in one
  item never fails its batch-mates.
- **Backend contract.** The worker module calls
  `serve(load: Callable[[dict], object], handle: Callable[[object, list[dict], dict], list[dict]])`.
  - `load(init)` returns the state, once.
  - `handle(state, items, params)` gets the decoded items (`{"id", "inputs"}`) and returns one dict
    per item, in item order: `{"outputs": {...}}`, or `{"error": exc}` with an exception instance
    for a per-item failure. `serve` adds the ids and classifies each `exc` by the fault table.
  - If `handle` raises, every item gets that error, classified.
  - If it returns the wrong count, or an entry without exactly one of `outputs` and `error`,
    every item gets an `infrastructure` error of type `ProtocolError`.
- A line the worker cannot parse as a request is answered with `fatal`, and the worker exits.
- The host validates `id`, length, order and item ids. A mismatch is a protocol violation.

## Arrays

```json
{"__array__": {"path": "<array_dir>/<sha256>.npy", "sha256": "<hex>", "dtype": "<f4", "shape": [2, 16000]}}
```

- An object whose only key is `__array__` is an array reference; it may appear anywhere inside
  `init`, `inputs`, `params` or `outputs`. Every numpy array travels this way; none is inlined.
- The file is canonical `.npy` (ADR-006): version 1.0, C order, little-endian. The writer
  converts non-contiguous and big-endian arrays to that form, so `dtype` is always `<…` or `|…`.
  `sha256` is over the file bytes and names the file.
- Object and structured dtypes are refused with a `content` error (`TypeError`): on the worker,
  for that item; on the host, before the request is sent.
- The writer writes to a temporary name in `array_dir` and renames, so a reader never sees a
  partial file and equal arrays share one file.
- The reader rejects a path outside `array_dir`, recomputes the sha256 and checks `dtype` and
  `shape`. It loads the array into memory; it does not memory-map.
- **Decode failure scope.** On either side, a missing, unreadable or mismatched array fails only
  the item that references it, with an `infrastructure` error of type `ArrayIntegrityError`; on
  the worker, that item is not passed to `handle`. The same failure in `params` fails every item
  of the request; in `init` it is `fatal`.
- **Lifetime.** The host creates `array_dir` before spawning and removes it on close. After
  decoding a response, the host deletes the files referenced by that request and response, except
  those referenced by `init`. The worker never deletes.

## Errors and faults

```json
{"type": "ValueError", "message": "...", "fault": "content", "traceback": "...", "gap": false}
```

- `type` is the exception's class name; `traceback` is the last 8 frames.
- **Precedence.** `ArrayIntegrityError` and `ProtocolError` are `infrastructure` by definition and
  are decided before the table. Every other exception is classified by the table, by
  `isinstance`, first match wins:

| Exception | `fault` |
|---|---|
| `ValueError`, `TypeError`, `KeyError` | `content` |
| `MemoryError`, `TimeoutError`, any class whose MRO has a class named `OutOfMemoryError` | `resource` |
| anything else | `infrastructure`, with `"gap": true` |

- `gap: true` marks a classification gap; the host reports it.
- Faults the host raises itself: `ProtocolMismatch`, `ProtocolError`, `WorkerLost` and
  `WorkerFatal` are `infrastructure`; a host timeout is `resource` (`TimeoutError`). Each fails
  every item of the request in flight, and the worker is discarded.
- **Fatal.** On a `fatal` line, during load or a request, the host raises
  `WorkerFatal(RuntimeError)` carrying the worker's error object; its fault is `infrastructure`,
  whatever the error object's `fault` says.

## Timeouts and shutdown

- Load and request timeouts are arguments to `WorkerClient` (default: none). On expiry the host
  kills the worker.
- **Worker lost.** EOF on the worker's stdout, or its exit, while a reply is awaited is
  `WorkerLost`, with the exit code and stderr tail.
- **Stop.** On `{"stop": true}`, or EOF on stdin, the worker exits 0 without replying. The host
  waits 10 s, then sends SIGKILL.

## Example

```text
W→H {"hello": {"wire": 1, "runtime": "9f2c…e1", "python": "3.10.14", "numpy": "1.23.5", "pid": 4711}}
H→W {"init": {"array_dir": "/tmp/sl-w-4711", "model": "fake"}}
W→H {"ready": {"load_s": 0.042}}
H→W {"id": "6d1f0c3a-5b7e-4c2d-9a10-3e8f2b7c4d55", "items": [{"id": "a", "inputs": {"x": {"__array__": {"path": "/tmp/sl-w-4711/3b5d…7a.npy", "sha256": "3b5d…7a", "dtype": "<f4", "shape": [1, 16000]}}}}, {"id": "b", "inputs": {}}], "params": {"faults": {"fail_items": ["b"]}}}
W→H {"id": "6d1f0c3a-5b7e-4c2d-9a10-3e8f2b7c4d55", "results": [{"id": "a", "outputs": {"value": "c0ff…ee"}}, {"id": "b", "error": {"type": "ValueError", "message": "fake failure for item b", "fault": "content", "traceback": "…", "gap": false}}]}
H→W {"stop": true}
```
