"""
Prints a random sample of values from one column of one table, plus
overall null/distinct-value statistics for that column across the WHOLE
table -- lets you directly check whether a column flagged as a
"collapse attractor" (see analyze_column_collapse.py) is heavily
null/sparse, or has very few distinct values.

IMPORTANT: validates the requested column name against the table's REAL
schema before querying. SQLite has a documented, dangerous quirk: a
double-quoted identifier that doesn't match any real column silently
falls back to being treated as a STRING LITERAL instead of raising an
error -- e.g. SELECT "Closed Date" FROM posts, if no column named
exactly "Closed Date" exists, silently returns the literal text
'Closed Date' for every row instead of failing. This script queries the
real schema first (via a LIMIT 0 SELECT * and cursor.description) and
refuses to proceed on anything but an exact or explicit case-insensitive
match, rather than risk this silent failure mode.

Usage (paste the table name directly as printed in eval output, e.g.
'codebase_community#sep#posts'):
    python -m scripts.inspect_random_column_values \
        --bird_db_root /mnt/nas/ayane/tables/dev_database \
        --full_table_name "codebase_community#sep#posts" \
        --column_name "Closed Date" \
        --sample_size 20
"""

import argparse
import sqlite3


def get_sqlite_path(db_root: str, db_id: str) -> str:
    return f"{db_root}/{db_id}/{db_id}.sqlite"


def resolve_column_name(cur: sqlite3.Cursor, table_name: str, requested: str) -> str:
    """
    Returns the REAL column name to use, validated against the table's
    actual schema -- never trusts the requested name blind. Raises
    SystemExit with the full real column list if no match is found,
    rather than risk SQLite's silent string-literal fallback.
    """

    cur.execute(f'SELECT * FROM "{table_name}" LIMIT 0')
    real_columns = [desc[0] for desc in cur.description]

    if requested in real_columns:
        return requested

    case_insensitive_matches = [c for c in real_columns if c.lower() == requested.lower()]
    if len(case_insensitive_matches) == 1:
        resolved = case_insensitive_matches[0]
        print(
            f"[WARNING] requested column {requested!r} doesn't exactly match the "
            f"schema -- using case-insensitive match {resolved!r} instead"
        )
        return resolved

    raise SystemExit(
        f"No column named {requested!r} found in table {table_name!r}.\n"
        f"Real columns in this table: {real_columns}\n"
        f"(This check exists specifically to avoid SQLite's silent "
        f"string-literal fallback for unmatched double-quoted identifiers.)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird_db_root", required=True)
    parser.add_argument(
        "--full_table_name", default=None,
        help="'db_id#sep#table_name', as printed directly in eval output",
    )
    parser.add_argument("--db_id", default=None)
    parser.add_argument("--table_name", default=None, help="the ORIGINAL sqlite table name")
    parser.add_argument("--column_name", required=True)
    parser.add_argument("--sample_size", type=int, default=20)
    args = parser.parse_args()

    if args.full_table_name:
        if "#sep#" not in args.full_table_name:
            raise SystemExit("--full_table_name must be in 'db_id#sep#table_name' form")
        db_id, table_name = args.full_table_name.split("#sep#", 1)
    elif args.db_id and args.table_name:
        db_id, table_name = args.db_id, args.table_name
    else:
        raise SystemExit("provide either --full_table_name, or both --db_id and --table_name")

    sqlite_path = get_sqlite_path(args.bird_db_root, db_id)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    real_column_name = resolve_column_name(cur, table_name, args.column_name)

    print(f"table: {db_id}.{table_name}")
    print(f"column requested: {args.column_name!r}  ->  resolved: {real_column_name!r}")
    print()

    cur.execute(f'SELECT COUNT(*), COUNT("{real_column_name}") FROM "{table_name}"')
    total_rows, non_null_rows = cur.fetchone()
    null_rows = total_rows - non_null_rows
    null_pct = 100 * null_rows / total_rows if total_rows > 0 else 0

    print(f"total rows:      {total_rows}")
    print(f"non-null rows:   {non_null_rows}")
    print(f"null rows:       {null_rows}  ({null_pct:.1f}% null)")

    cur.execute(f'SELECT COUNT(DISTINCT "{real_column_name}") FROM "{table_name}"')
    distinct_count = cur.fetchone()[0]
    print(f"distinct non-null values: {distinct_count}")

    if total_rows > 0:
        print(f"distinct/total ratio: {distinct_count / total_rows:.3f}  (near 0 = very repetitive/low-info column)")
    print()

    cur.execute(
        f'SELECT "{real_column_name}" FROM "{table_name}" ORDER BY RANDOM() LIMIT ?',
        (args.sample_size,),
    )
    sample = [row[0] for row in cur.fetchall()]
    conn.close()

    print(f"random sample of {len(sample)} values (NULL shown explicitly):")
    for v in sample:
        display = "NULL" if v is None else repr(v)
        print(f"  {display}")