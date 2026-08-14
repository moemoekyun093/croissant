"""
Bisects exactly where NaN first appears, instead of guessing further.

Two passes:
  1. Scan every cell in the slice directly through try_parse_numeric --
     confirms (or disproves) that the nan/inf string fix is actually
     catching everything, independent of the rest of the model.
  2. Run each table through CellEncoder.encode_column stage by stage,
     checking for NaN immediately after cell encoding -- if it's already
     NaN there, the problem is in CellEncoder, not in the attention
     stages downstream.

Usage:
    python -m scripts.debug_nan --corpus_jsonl /path/to/corpus.jsonl --n_tables 200
"""

import argparse

import torch

from src.data.corpus_loader import iter_tables_from_jsonl
from src.encoding.cell_encoder import CellEncoder
from src.encoding.cell_type import detect_cell_type, try_parse_numeric, CellType


def scan_all_cells_for_bad_numerics(tables) -> None:
    print("[pass 1] scanning every cell's try_parse_numeric result directly...")

    bad_found = 0
    for table in tables:
        for col in table.columns:
            for cell in col.cells:
                value = try_parse_numeric(cell)
                if value is None:
                    continue
                if value != value or value in (float("inf"), float("-inf")):
                    bad_found += 1
                    print(
                        f"  [BAD] table_id={table.table_id} col={col.header!r} "
                        f"cell={cell!r} -> parsed={value}"
                    )

    if bad_found == 0:
        print("  [pass 1] no bad numeric parses found -- try_parse_numeric itself is clean")
    else:
        print(f"  [pass 1] found {bad_found} bad parse(s) -- fix did not fully catch these")


def scan_cell_encoder_outputs(cell_encoder: CellEncoder, tables) -> None:
    print("\n[pass 2] running CellEncoder.encode_column per table/column, checking for NaN...")

    first_bad = None

    for table in tables:
        for col in table.columns:
            out = cell_encoder.encode_column(col)

            if torch.isnan(out).any() or torch.isinf(out).any():
                print(
                    f"  [NaN/Inf FOUND] table_id={table.table_id} "
                    f"table_name={table.table_name!r} col={col.header!r}"
                )
                # identify which specific cell(s) within this column
                for i, cell in enumerate(col.cells):
                    cell_type = detect_cell_type(cell)
                    row_out = out[i]
                    if torch.isnan(row_out).any() or torch.isinf(row_out).any():
                        print(
                            f"      cell[{i}] type={cell_type.value} "
                            f"raw={cell!r} -> embedding has NaN/Inf"
                        )
                if first_bad is None:
                    first_bad = (table.table_id, col.header)

    if first_bad is None:
        print("  [pass 2] no NaN/Inf found in any CellEncoder output -- problem is downstream")
    else:
        print(f"\n  first bad output at table_id={first_bad[0]} col={first_bad[1]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl", required=True)
    parser.add_argument("--n_tables", type=int, default=200)
    parser.add_argument("--embed_dim", type=int, default=32)
    args = parser.parse_args()

    tables = []
    for table in iter_tables_from_jsonl(args.corpus_jsonl):
        tables.append(table)
        if len(tables) >= args.n_tables:
            break

    print(f"loaded {len(tables)} tables\n")

    scan_all_cells_for_bad_numerics(tables)

    cell_encoder = CellEncoder(text_model_name="bert-base-uncased", output_dim=args.embed_dim)
    cell_encoder.eval()
    with torch.no_grad():
        scan_cell_encoder_outputs(cell_encoder, tables)