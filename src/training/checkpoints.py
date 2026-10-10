"""Checkpoint and cache file helpers."""

import glob
import os


def latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Same natural/version-sort convention as scripts/run_pilot.sh's
    shell logic, reimplemented in Python -- picks the highest-epoch
    checkpoint_epoch*.pt in a PretrainTrainer checkpoint_dir."""
    paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch*.pt"))
    if not paths:
        return None

    def epoch_num(path: str) -> int:
        name = os.path.basename(path)
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else -1

    return max(paths, key=epoch_num)


def safe_load_cache(load_fn, path: str, label: str) -> bool:
    """Wraps a *.load_*cache(path) call so a corrupted cache file can't
    crash the whole training launch. Root cause seen in practice: a
    training process killed mid torch.save() (e.g. SIGKILL from a
    relaunch/restart) leaves a truncated .pt file -- torch.load then
    raises a RuntimeError ("PytorchStreamReader failed locating file ...
    internal miniz error") deep inside torch's own unpickler, which
    previously propagated straight out of this script before training
    even started (a corrupted CACHE file was blocking an otherwise-fine
    run). On a load failure: renames the bad file aside to
    '<path>.corrupted' (so it's not silently lost/inspectable later, and
    so it stops being picked up as "exists" on the NEXT launch too) and
    continues with an empty cache -- exactly as if the file had never
    existed, just slower on the first pass while it rebuilds. Returns
    True if the load actually succeeded.
    """
    try:
        load_fn(path)
        return True
    except Exception as e:
        corrupted_path = path + ".corrupted"
        print(
            f"WARNING: failed to load {label} cache from {path!r} "
            f"({type(e).__name__}: {e}) -- treating as if it doesn't "
            f"exist and continuing with an empty cache. Moving the bad "
            f"file to {corrupted_path!r} so it isn't picked up again."
        )
        try:
            os.replace(path, corrupted_path)
        except OSError as move_err:
            print(f"  (also failed to move the corrupted file aside: {move_err})")
        return False
