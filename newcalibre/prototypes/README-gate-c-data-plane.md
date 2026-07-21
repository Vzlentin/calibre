# PROTOTYPE — Gate C data plane

This throwaway logic prototype answers [Choose the Gate C data-plane
architecture](https://github.com/Vzlentin/calibre/issues/435). It does not define
production interfaces.

Run it from `newcalibre/`:

```bash
uv run python prototypes/gate_c_data_plane.py
```

It exercises the M5 time shape (1,941 days, 64 origins, horizon 28) at 1,000
and 10,000 series. After each action it prints the complete relevant state:
history rows, streamed/resolved/pending ledger rows, hot-index memory, conformal
partition count, and terminal checksums.

## Candidates

### A — scan-shaped frames and scalar state

- Keep the immutable panel in a repeated-key long pandas frame.
- Copy a filtered frame for each history cutoff.
- Keep pending ledger rows in one frame, scan and copy it at each origin, and
  concatenate each forecast batch.
- Call conformal state once per partition and keep Python objects per state row.

This is an optimistic columnar stand-in for the current skeleton. The current
object-row snapshots add more allocation than this candidate.

### B — indexed columnar batches

- Encode series and calendar labels once. Keep values in a calendar-aligned
  columnar plane with a presence bitmap. A history request is a two-axis view;
  an incremental adapter receives only the newly admissible rows.
- Append immutable Arrow-shaped forecast batches. Sort each batch by target
  time, index its contiguous unresolved spans, and append resolutions as a
  side stream. Do not rewrite forecast history.
- Route conformal rows with stable integer partition IDs. Give each method one
  batch per origin and keep method-owned state in columnar arrays. Persistence
  still exposes independently addressable partition rows.
- Convert to pandas and string labels only at ingestion, plugin, and export
  adapters.

The intended deep module hides representation and indexing. The engine should
ask it for a history view, newly admissible actuals, a ledger append receipt,
or the next due batch. It should not know about Arrow, NumPy layouts, target
buckets, or partition encodings. The exact interface remains the question of
[Define the indexed streaming engine
interfaces](https://github.com/Vzlentin/calibre/issues/438).

## Limits

- Timings are directional local measurements, not Gate C evidence and not a
  claim against the 15-minute budget.
- The prototype uses M5's dense daily panel. The presence bitmap preserves the
  meaning of absent/missing observations, but this prototype does not measure a
  highly sparse calendar.
- It omits reconciliation math, persistence I/O, Ray, and plugin conversion
  cost. Those do not affect the representation decision tested here.
- The synthetic conformal state is a running statistic plus an adaptive level.
  It measures batch-routing and state-layout overhead, not a method's
  statistical result.
