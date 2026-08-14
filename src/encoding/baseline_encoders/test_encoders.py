"""
Smoke test: run every encoder on a toy table and confirm the output shapes
match the TableEncoding contract. Not a correctness/quality test -- just
verifies each encoder wires together and runs end-to-end.

Usage:
    python test_encoders.py            # test all encoders
    python test_encoders.py bert tabbie  # test a subset
"""
import sys
import time

from src.encoding.baseline_encoders import ENCODER_REGISTRY

HEADERS = ["Player", "Team", "Points"]
ROWS = [
    ["LeBron James", "Lakers", "27.1"],
    ["Nikola Jokic", "Nuggets", "26.4"],
    ["Luka Doncic", "Mavericks", "32.4"],
]
CAPTION = "2023-24 NBA scoring leaders"


def run_one(name: str, cls) -> None:
    print(f"\n=== {name} ===")
    t0 = time.time()
    encoder = cls()
    out = encoder.encode(HEADERS, ROWS, caption=CAPTION)
    dt = time.time() - t0

    n_rows, n_cols = len(ROWS), len(HEADERS)
    assert out.cell_embeddings.shape[:2] == (n_rows, n_cols), out.cell_embeddings.shape
    assert out.row_embeddings.shape[0] == n_rows, out.row_embeddings.shape
    assert out.col_embeddings.shape[0] == n_cols, out.col_embeddings.shape
    assert out.table_embedding.dim() == 1, out.table_embedding.shape

    print(f"cell_embeddings: {tuple(out.cell_embeddings.shape)}")
    print(f"row_embeddings:  {tuple(out.row_embeddings.shape)}")
    print(f"col_embeddings:  {tuple(out.col_embeddings.shape)}")
    print(f"table_embedding: {tuple(out.table_embedding.shape)}")
    print(f"OK in {dt:.2f}s")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(ENCODER_REGISTRY.keys())
    failures = []
    for name in targets:
        try:
            run_one(name, ENCODER_REGISTRY[name])
        except Exception as e:  # noqa: BLE001
            failures.append((name, e))
            print(f"FAILED: {name}: {e}")

    print("\n" + "=" * 40)
    if failures:
        print(f"{len(failures)} encoder(s) failed: {[f[0] for f in failures]}")
        sys.exit(1)
    print(f"All {len(targets)} encoders passed.")
