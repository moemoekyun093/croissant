"""Sequential pickle format for fully materialized 128-D training batches.

The preparation process resolves every stochastic/data-dependent decision
once: query order, sampled positive, hard negatives, candidate de-duplication,
and the complete multi-positive mask.  A training process consequently reads
one record, transfers its tensors to the GPU, and trains; it never looks up a
query/table/string or reconstructs labels from IDs.

Records are dumped one after another rather than as one enormous Python list.
The file is still a normal deterministic pickle stream, but reading it keeps
only one batch resident at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
import queue
import threading
from typing import Iterable, Iterator, TypeVar

import torch


FORMAT_NAME = "croissant_prepared_batches"
FORMAT_VERSION = 3
T = TypeVar("T")


@dataclass
class MaterializedBatch:
    """GPU-ready tensors reconstructed from one packed pickle record."""

    query_features: torch.Tensor
    query_mask: torch.Tensor
    cell_features: torch.Tensor
    header_features: torch.Tensor
    row_mask: torch.Tensor
    col_mask: torch.Tensor
    cell_mask: torch.Tensor
    positive_mask: torch.Tensor


@dataclass
class PreparedBatch:
    query_texts: tuple[str, ...]         # audit only; never used by training
    candidate_table_ids: tuple[str, ...]  # audit only; never used by training
    query_features: torch.Tensor       # [Bq, L, r], stored float16 on CPU
    query_mask: torch.Tensor           # [Bq, L], bool
    cell_features: torch.Tensor        # [sum_t Nt*Mt, r], packed float16 CPU
    cell_scatter: torch.Tensor         # [sum_t Nt*Mt], int32 flat destinations
    header_features: torch.Tensor      # [sum_t Nt, r], packed float16 CPU
    header_scatter: torch.Tensor       # [sum_t Nt], int32 flat destinations
    row_mask: torch.Tensor             # [Bt, M], bool
    col_mask: torch.Tensor             # [Bt, N], bool
    cell_mask: torch.Tensor            # [Bt, N, M], bool
    positive_mask: torch.Tensor        # [Bq, Bt], bool; final labels

    def validate(self, projection_dim: int | None = None) -> None:
        q, qm = self.query_features, self.query_mask
        x, h = self.cell_features, self.header_features
        rm, cm, xm, pm = self.row_mask, self.col_mask, self.cell_mask, self.positive_mask
        if q.ndim != 3 or qm.shape != q.shape[:2]:
            raise ValueError(f"bad prepared query shapes: Q={tuple(q.shape)}, mask={tuple(qm.shape)}")
        if len(self.query_texts) != q.shape[0]:
            raise ValueError("audit query-text count does not match query tensor")
        if x.ndim != 2 or h.ndim != 2 or x.shape[1] != h.shape[1]:
            raise ValueError(f"bad packed table/header shapes: X={tuple(x.shape)}, H={tuple(h.shape)}")
        if self.cell_scatter.shape != (x.shape[0],) or self.header_scatter.shape != (h.shape[0],):
            raise ValueError("packed feature and scatter-index lengths differ")
        if self.cell_scatter.dtype != torch.int32 or self.header_scatter.dtype != torch.int32:
            raise TypeError("packed scatter indices must be int32 on disk")
        if rm.ndim != 2 or cm.ndim != 2 or rm.shape[0] != cm.shape[0]:
            raise ValueError("prepared row/column masks do not describe one table batch")
        if len(self.candidate_table_ids) != cm.shape[0]:
            raise ValueError("audit table-ID count does not match candidate tensor")
        if xm.shape != (cm.shape[0], cm.shape[1], rm.shape[1]) or pm.shape != (q.shape[0], cm.shape[0]):
            raise ValueError("prepared cell/positive masks do not match their tensors")
        if self.cell_scatter.numel() and (
            int(self.cell_scatter.min()) < 0 or int(self.cell_scatter.max()) >= xm.numel()
        ):
            raise ValueError("packed cell scatter index is out of bounds")
        if self.header_scatter.numel() and (
            int(self.header_scatter.min()) < 0 or int(self.header_scatter.max()) >= cm.numel()
        ):
            raise ValueError("packed header scatter index is out of bounds")
        if q.shape[-1] != x.shape[-1]:
            raise ValueError("query and table prepared widths differ")
        if projection_dim is not None and q.shape[-1] != projection_dim:
            raise ValueError(
                f"prepared width {q.shape[-1]} does not match metadata width {projection_dim}"
            )
        if not torch.all(pm.any(dim=1)):
            raise ValueError("a prepared query has no positive candidate")
        for name in ("query_mask", "row_mask", "col_mask", "cell_mask", "positive_mask"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must be bool")

    def materialize(
        self, device: str | torch.device, dtype: torch.dtype = torch.float32
    ) -> MaterializedBatch:
        """One packed transfer + scatter; no semantic/data lookup occurs."""
        device = torch.device(device)
        q = self.query_features.to(device=device, dtype=dtype, non_blocking=True)
        query_mask = self.query_mask.to(device=device, non_blocking=True)
        row_mask = self.row_mask.to(device=device, non_blocking=True)
        col_mask = self.col_mask.to(device=device, non_blocking=True)
        cell_mask = self.cell_mask.to(device=device, non_blocking=True)
        positive_mask = self.positive_mask.to(device=device, non_blocking=True)

        bt, max_n, max_m = cell_mask.shape
        r = self.query_features.shape[-1]
        packed_cells = self.cell_features.to(device=device, dtype=dtype, non_blocking=True)
        packed_headers = self.header_features.to(device=device, dtype=dtype, non_blocking=True)
        cells_flat = torch.zeros(bt * max_n * max_m, r, device=device, dtype=dtype)
        headers_flat = torch.zeros(bt * max_n, r, device=device, dtype=dtype)
        cells_flat.index_copy_(
            0, self.cell_scatter.to(device=device, dtype=torch.long, non_blocking=True), packed_cells
        )
        headers_flat.index_copy_(
            0, self.header_scatter.to(device=device, dtype=torch.long, non_blocking=True), packed_headers
        )
        return MaterializedBatch(
            query_features=q,
            query_mask=query_mask,
            cell_features=cells_flat.view(bt, max_n, max_m, r),
            header_features=headers_flat.view(bt, max_n, r),
            row_mask=row_mask,
            col_mask=col_mask,
            cell_mask=cell_mask,
            positive_mask=positive_mask,
        )


class PreparedBatchWriter:
    """Write a metadata header followed by independently pickled batches."""

    def __init__(self, path: str, metadata: dict):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._file = open(path, "wb")
        header = dict(metadata)
        header.update({"format": FORMAT_NAME, "format_version": FORMAT_VERSION})
        pickle.dump(header, self._file, protocol=pickle.HIGHEST_PROTOCOL)
        self.count = 0

    def write(self, batch: PreparedBatch) -> None:
        batch.validate()
        pickle.dump(batch, self._file, protocol=pickle.HIGHEST_PROTOCOL)
        self.count += 1

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "PreparedBatchWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_prepared_metadata(path: str) -> dict:
    with open(path, "rb") as f:
        metadata = pickle.load(f)
    _validate_metadata(metadata, path)
    return metadata


def iter_prepared_batches(path: str, validate: bool = True) -> Iterator[PreparedBatch]:
    """Stream records sequentially; no batch/query/table lookup is performed."""
    with open(path, "rb") as f:
        metadata = pickle.load(f)
        _validate_metadata(metadata, path)
        projection_dim = int(metadata["projection_dim"])
        while True:
            try:
                batch = pickle.load(f)
            except EOFError:
                return
            if not isinstance(batch, PreparedBatch):
                raise TypeError(f"{path!r} contains a non-PreparedBatch record")
            if validate:
                batch.validate(projection_dim)
            yield batch


def prefetch_iterable(iterable: Iterable[T], depth: int = 2) -> Iterator[T]:
    """Read/unpickle future CPU batches while the GPU trains on this one.

    Prepared shards normally live on NAS.  Without prefetching, each step
    waits synchronously for ``pickle.load`` before launching any GPU work.
    A bounded daemon thread overlaps that I/O/deserialization with the
    current forward/backward pass while retaining only ``depth`` additional
    batches in memory.  Record order and tensor contents are unchanged.
    """
    if depth <= 0:
        yield from iterable
        return

    records: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=depth)

    def produce() -> None:
        try:
            for value in iterable:
                records.put(("value", value))
        except BaseException as error:
            records.put(("error", error))
        finally:
            records.put(("done", None))

    worker = threading.Thread(target=produce, name="prepared-batch-prefetch", daemon=True)
    worker.start()
    while True:
        kind, payload = records.get()
        if kind == "value":
            yield payload  # type: ignore[misc]
        elif kind == "error":
            raise payload  # type: ignore[misc]
        else:
            return


def _validate_metadata(metadata: object, path: str) -> None:
    if not isinstance(metadata, dict):
        raise TypeError(f"{path!r} has no prepared-batch metadata dictionary")
    if metadata.get("format") != FORMAT_NAME:
        raise ValueError(f"{path!r} is not a {FORMAT_NAME} file")
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported prepared-batch version {metadata.get('format_version')!r}; "
            f"expected {FORMAT_VERSION}"
        )
    if "projection_dim" not in metadata:
        raise ValueError(f"{path!r} metadata has no projection_dim")
