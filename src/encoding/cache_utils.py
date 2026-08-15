"""Compact on-disk storage for the frozen-feature caches (table_cache.pt,
frozen_cache.pt, text cache, query cache).

These caches store FROZEN backbone outputs -- deterministic features that
are only ever consumed downstream (projected, scored, or fed to a trainable
stack). Storing them on disk in float16 instead of float32 halves their
footprint with negligible effect (the values get L2-normalized / projected
anyway), which matters a lot at corpus scale: bert/tapas's full table cache
alone can be tens of GB in float32. The in-memory copy stays float32 -- only
the disk copy is halved -- and everything is restored to float32 on load, so
the models still consume float32 exactly as before.

downcast_cache/upcast_cache walk dicts/lists/tuples recursively and only
touch FLOATING tensors, so non-float entries (e.g. a query cache's long
token-id tensors) pass through untouched. upcast_cache is also a no-op on
tensors that are already float32, so caches written by the OLD float32
format still load correctly -- no migration needed.
"""

from __future__ import annotations

import torch


def downcast_cache(obj):
    """Recursively cast floating tensors to float16 for compact storage;
    leave non-float tensors and any other objects untouched."""
    if isinstance(obj, torch.Tensor):
        return obj.half() if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: downcast_cache(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(downcast_cache(v) for v in obj)
    return obj


def upcast_cache(obj):
    """Inverse of downcast_cache for loading: restore float16 tensors to
    float32 (what the models consume). No-op on already-float32 tensors, so
    old float32 cache files still load unchanged."""
    if isinstance(obj, torch.Tensor):
        return obj.float() if obj.dtype == torch.float16 else obj
    if isinstance(obj, dict):
        return {k: upcast_cache(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(upcast_cache(v) for v in obj)
    return obj
