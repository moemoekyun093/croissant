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

import torch

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

    if name == "turl":
        # TURL contextualizes one already-pooled entity/cell node per cell,
        # not every word piece inside a cell.  Metadata remains token-level.
        # This catches a regression to the old token-level cell attention
        # implementation without depending on any model output values.
        attn_bias, metadata_ids, cell_token_ids = encoder._build_visibility(
            HEADERS, ROWS, CAPTION
        )
        n_cells = len(ROWS) * len(HEADERS)
        assert attn_bias.shape == (len(metadata_ids) + n_cells,) * 2
        assert sum(len(row) for row in cell_token_ids) == len(ROWS)
        assert all(len(cell) >= 1 for row in cell_token_ids for cell in row)

        caption_tokens = encoder.tokenizer(CAPTION, add_special_tokens=False)["input_ids"]
        header0_idx = 1 + len(caption_tokens)
        first_cell_idx = len(metadata_ids)
        other_column_cell_idx = first_cell_idx + 1
        # Caption/[CLS] is global, while header 0 only sees column 0.
        assert attn_bias[0, other_column_cell_idx].item() == 0.0
        assert torch.isneginf(attn_bias[header0_idx, other_column_cell_idx]).item()
        assert attn_bias[header0_idx, first_cell_idx].item() == 0.0

        # Batched TURL must preserve the exact per-table visibility result
        # and input order even when it internally sorts differently-sized
        # tables into dynamic microbatches. eval() disables dropout so the
        # single-table and batched paths are directly comparable.
        encoder.eval()
        small_headers = ["City", "Country"]
        small_rows = [["Helsinki", "Finland"]]
        batch_inputs = [
            (HEADERS, ROWS, CAPTION),
            (small_headers, small_rows, "Cities"),
        ]
        with torch.no_grad():
            singles = [encoder(*table) for table in batch_inputs]
            batched = encoder.forward_batch(batch_inputs)
        assert len(batched) == len(singles)
        for single, combined in zip(singles, batched):
            assert torch.allclose(single.cell_embeddings, combined.cell_embeddings, atol=1e-5)
            assert torch.allclose(single.row_embeddings, combined.row_embeddings, atol=1e-5)
            assert torch.allclose(single.col_embeddings, combined.col_embeddings, atol=1e-5)
            assert torch.allclose(single.table_embedding, combined.table_embedding, atol=1e-5)

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
