"""Optional adapter: build a ranking signal table from ``audio_analysis`` outputs.

The ranker is signal-source-agnostic (its input is a generic per-item signal
table — ``signal-table.parquet.md``). This convenience adapter pivots a set of
``audio_analysis`` per-axis uncertainty parquets (each carrying ``start``,
``end``, ``aggregated_uncertainty``) into a per-segment signal table, one signal
column per axis. Segment ``item_id`` is ``"<source_audio>#<start>-<end>"``.

Note: higher uncertainty = worse, so a metric over these signals typically uses
negative weights (or ``lower_is_better``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _segment_id(source_audio: str, start: float, end: float) -> str:
    return f"{source_audio}#{start:.2f}-{end:.2f}"


def harvest_from_axis_parquets(
    axis_parquets: dict[str, Path | str],
    source_audio: str,
    out: Path | str,
) -> Path:
    """Pivot ``{signal_name: axis_parquet_path}`` into one segment signal table.

    Each axis parquet must have ``start``, ``end``, and ``aggregated_uncertainty``
    columns. Segments are keyed by ``(start, end)``; a signal missing for a
    segment is written as NaN. Returns the output parquet path.
    """
    seg_keys: list[tuple[float, float]] = []
    seg_index: dict[tuple[float, float], int] = {}
    per_signal: dict[str, dict[tuple[float, float], float]] = {}

    for signal_name, path in axis_parquets.items():
        table = pq.read_table(path)
        starts = table.column("start").to_pylist()
        ends = table.column("end").to_pylist()
        vals = table.column("aggregated_uncertainty").to_pylist()
        bucket: dict[tuple[float, float], float] = {}
        for s, e, v in zip(starts, ends, vals, strict=True):
            key = (float(s), float(e))
            if key not in seg_index:
                seg_index[key] = len(seg_keys)
                seg_keys.append(key)
            bucket[key] = float("nan") if v is None else float(v)
        per_signal[signal_name] = bucket

    item_ids = [_segment_id(source_audio, s, e) for (s, e) in seg_keys]
    data: dict[str, list] = {
        "item_id": item_ids,
        "unit": ["segment"] * len(seg_keys),
        "source_audio": [source_audio] * len(seg_keys),
        "start": [s for (s, _) in seg_keys],
        "end": [e for (_, e) in seg_keys],
    }
    for signal_name, bucket in per_signal.items():
        data[signal_name] = [bucket.get(key, np.nan) for key in seg_keys]

    table = pa.table(data)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return out
