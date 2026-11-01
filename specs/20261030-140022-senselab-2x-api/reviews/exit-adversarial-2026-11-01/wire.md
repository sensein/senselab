# Phase 1 exit attack: worker and wire (commit c54e9712)

Probes: `/p1..p12.py` and `<scratch>`, run from the main `.venv` with PYTHONPATH on the scratch worktree.

## W-1 (medium) An int over 4300 digits in one item's inputs fails the whole request with ProtocolError, not that item alone
- Invariant: `WorkerClient.run` docstring and wire-protocol.md "an item whose inputs have no wire form fails alone, with its content error"; R-68 totality (H5).
- Reproduction: `<scratch>`. Client on an echo backend, `run([{"id":"a","inputs":{"n":10**4400}},{"id":"b","inputs":{"n":1}}], {})`.
- Observed: `run` raises `ProtocolError: request message has no wire form: ValueError: Exceeds the limit (4300 digits)`. Item b never runs. `to_wire` accepts the int (`canonical_scalar` passes any `int`, `charge(1)`); `dump_line` then raises in `_json_bytes`, outside the per-item `try` in `_run`. The same int in `params` or `init` gives `ProtocolError` where the docstring says `ValueError`.
- Worker side (`<scratch>`): a backend output holding `10**5000` becomes an item error `ProtocolError`, `fault: infrastructure`, `gap: false`. It is a content fault and is stamped authoritative. An executor will treat it as machinery failure.
- Expected: `to_wire` refuses the int as `WireValueError` (content), so the item fails alone. The CPython limit is `sys.get_int_max_str_digits`, and parse_line already refuses these on read, so writer and reader disagree.

## W-2 (medium) Array references multiply memory: every reference to one file is a separate full private copy
- Invariant: ADR-009 resource faults / P-34 bounded work; the line limit exists so that "wire form fits a line"; `_Encoder` shares repeated values and charges only the reference size for them.
- Reproduction: `<scratch>`. `a = np.zeros(2**17)` (1 MiB); inputs `{"a": [a]*100}`.
- Observed: request line 25,561 bytes, 1,049,000 bytes on disk (one file); `from_wire` returns 100 arrays, 104,857,600 bytes, 100 distinct buffers. Amplification is bounded only by the line limit, about 95k references per 16 MiB line, so a 1 GiB array repeated that often asks for about 95 TiB. Realistic form: one waveform shared by many items (`_respond` decodes every item before calling `handle`). The host does the same on results.
- Expected: one buffer per distinct sha within a message (shared read-only or copy-on-write view), or a cap on total decoded bytes per message that yields a `resource` fault for the item.

## W-3 (low-medium) A leaked client alive at interpreter exit takes about 15 s to finalise and logs false errors
- Invariant: wire-protocol.md "`__del__` runs every step not yet completed"; graces describe a stop that ends when the worker exits.
- Reproduction: `<scratch>` with no argument (client created, never closed, no `del`): `time` 15.2 s, log `signal SIGTERM to worker process group N failed: Operation not permitted` (also SIGCONT, SIGKILL) and `worker process group N still has members after SIGKILL` (logged at error). With `del c` first it takes 0.18 s.
- Cause: at finalisation the daemon reaper thread cannot run, so `_exited` is never set; the already-exited worker is waited on for stop minus term, then term, then kill. `killpg` gives EPERM on the zombie group and the code reads it as "still has members".
- Expected: `_terminate` or `_leader_held` should detect the exit without the thread (a `waitid(WNOHANG|WNOWAIT)` poll) so a stopped worker ends the wait at once and EPERM on a zombie group is not an error.

## W-4 (low-medium) `ShutdownGraces` is unvalidated, and `client.graces` is a plain mutable attribute; a non-finite or oversize grace breaks every shutdown path
- Invariant: "close and every host fault stop the group and release the client within `graces.bound`" (class docstring, wire-protocol.md "Timeouts and shutdown"); "graces fixed per client".
- Reproduction: `<scratch> "wc.ShutdownGraces(stop=float('nan'), ...)"`, and `join=float('nan')`, `stop=float('inf')`, `kill=1e20`.
- Observed: `close()` raises `ValueError: Invalid value NaN` (nan) or `OverflowError: timestamp out of range` (inf, 1e20) from `_acquire_shutdown`, before any step; the "finishing an interrupted shutdown" pass fails the same way. The client stays unstopped (`_stopped` and `_released` false), the array directory is left, and `__del__` retries and fails again. The worker was killed only by `_kill_group_now`, which runs without a lock.
- `stop=-1` is accepted silently (clamped to 0). `client.graces = ShutdownGraces(...)` can also be reassigned on a live client, so the bound can change under a shutdown in flight (read from `self.graces` in several places).
- Expected: `__post_init__` validation (finite, at least 0, `term <= stop`), and `graces` held privately per client.
- Left behind by my probes: about five empty `senselab-arrays-*` dirs in `$TMPDIR` (I used the default `array_parent` for the W-4 runs).

## W-5 (low-medium) A read failure of an array file becomes ArrayIntegrityError, losing its class and its transience
- Invariant: ADR-009 (MemoryError is resource); ADR-003 amendment "one classifier decides transient; storage faults say retry only when the cause is transient"; R-61 principle that an I/O error is not corruption.
- Reproduction: `<scratch>` (patches `os.read` to raise).
- Observed: EIO, ESTALE and MemoryError each give `ArrayIntegrityError` (`_read_array` catches every `Exception`) with fault `infrastructure`, `is_transient(e)` False although `is_transient(e.__cause__)` is True for EIO and ESTALE. `is_transient` returns False for any non-`OSError` without looking at `__cause__`. A host reading a large result array that cannot be allocated reports "bad array reference: MemoryError" as infrastructure, not resource.
- Expected: only a completed read that fails the sha, dtype, shape or header checks is `ArrayIntegrityError`; `OSError` and `MemoryError` keep their types (or are chained into a class that `is_transient` follows).

## W-6 (low) `is_transient` recurses without a bound on a cause cycle, and reads SSL errnos as OS errnos
- Invariant: totality principle (R-68); "OSError not blanket-transient".
- Reproduction `<scratch>` and `<scratch>`.
- Observed: `a=OSError("a"); b=OSError("b"); a.__cause__=b; b.__cause__=a` gives `RecursionError` from `is_transient(a)`; so does a `URLError` whose `reason` is itself. `ssl.SSLEOFError` (errno 8) is permanent, `ssl.SSLError(5)` (SSL_ERROR_SYSCALL, which equals EIO) is transient, `ssl.SSLError(4)` would read as EINTR; `SSLError.errno` holds an SSL code, not an errno.
- Expected: a visited set (or depth cap) and an `ssl.SSLError` branch before the errno table. HTTP 408 and 425 are permanent; state that in the data or the docstring.

## W-7 (low) Array reference checks are lax about types and extra keys
- Invariant: "field types are exact" (`validate_message` docstring), strict wire values (R-92).
- Observed (`<scratch>`): a reference with `shape: [4.0]` or `shape: [true]` is accepted for a shape `[4]` / `[1]` array (Python `==`), and extra keys in the reference are ignored. `dtype` and `sha256` are exact. Harmless alone, but the one validator for the reference is `_read_array`, not `validate_message`.
- Expected: `type(n) is int` per shape element, and exactly the four keys.

## Not findings (probed, see "held")

## duplicates
- `_disown` and `_release` carry the same four-line "set `_wake_r`/`_wake_w` to -1 and close" loop; `_Pump.close` and the wake-pipe pair belong with one `_close_wake()`.
- Three SIGKILL-the-group sites: `_kill_group_now`, `_signal_group_until_gone`, the `except` in `_stop_group`; they differ only in the held/exited condition.
- `_send` and `_write_stop` both write to `proc.stdin` under `_channel_lock` with the same closed check; `_write_stop` is a degenerate `_send` without the selector.
- `_FAULTS` (codec) and `{f.value for f in FaultClass}` (`_fatal_record`) are the same set built twice; `_fatal_record` re-validates the fields `_result` validates.
- `_read_file` (codec) and `hashing._read_chunks` and `digest.file_digest` are three chunked-read loops over a file; `file_digest` and `bytes_digest` both end in `hashlib.sha256` through separate `import hashlib # noqa` stubs, and `npy_digest(canonical_npy_bytes)` and `_write_array`'s `bytes_digest("npy@1", npy_bytes)` hash the same bytes by two paths.
- Depth/height walkers: `codec._height`, `_Encoder.encode` height tracking and `canonical._Writer` with `Walk` all enforce `MAX_DEPTH`; the first two are near-identical memoised walks.
- `remove_owned` and `WorkerClient._remove_arrays` both decide ownership by name pattern (`_remove_arrays` calls `remove_owned` twice with two patterns).
- `serve._send` and `serve._write` / `dump_line`; `_is_stop` and `codec._stop` re-validate the same key.

## held
- Line limit: boundary lines of MAX-1, MAX, MAX+1 bytes in both directions through a live worker (`<scratch>`): MAX accepted, MAX+1 becomes `ResponseTooLarge` (item) or `ValueTooLarge` (request), worker stays alive. Pump limit `MAX+1` is consistent with `dump_line`.
- `parse_line` (`<scratch>`): 1M-deep lists and objects, BOM, duplicate keys through escapes, lone surrogates in keys and strings, 1e999, trailing data, two objects, NUL and raw control characters all refused as `ProtocolError` in under 0.1 s; 64-container nesting accepted, 65 refused.
- Array round trips (`<scratch>`): empty, 0-d, zero-size dims, bool, f16, U, Fortran-order, non-contiguous, read-only, NaN/inf/-0 are exact, writable and aligned; subclasses (masked, matrix, memmap), object, structured, complex, datetime, bytes, longdouble are refused as content faults. Big-endian input is normalised to little-endian by design.
- fd 0-2 claim (direct launch): fd 0 closed, fd 2 closed, both closed all give the hello and exit 0 on stop; fd 1 closed writes the hello to /dev/null and exits 0, no crash.
- Fork owner guard (`<scratch>`): in a forked child `run` gives `WorkerLost`, `close` and `del` touch nothing; the parent's worker, its run and its array directory are unaffected, also after a child `sys.exit` and a multiprocessing fork child. (Observation: `alive` is True in the child.)
- Constructor failure paths (`<scratch>`): a load that ignores SIGTERM with a SIGTERM-ignoring grandchild times out in 3.1 s against a bound of 5, leaving no directory or process; a missing backend module gives `WorkerFatal` and cleans up.
- Race by design noted, not reproduced: `_Pump._lock` held across a fork by `finish()` would hang the child's `_disown`; the lock is held only around a 1-byte write to a private pipe.
