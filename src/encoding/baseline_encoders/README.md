# table_encoders

Six table-encoding baselines re-implemented against modern `transformers`
(no old `pytorch-pretrained-bert` / `allennlp` / `torch_geometric`
dependencies), sharing one output contract so they're interchangeable in a
retrieval pipeline.

| Model | Structural mechanism | Header handling (exact, per reference repo) | Reference repo |
|---|---|---|---|
| `BertTableEncoder` | none — full self-attention over the flattened table | arbitrary `"header : value"` per cell (BERT has no paper-defined table format) | n/a (vanilla `transformers.BertModel`) |
| `TabbieTableEncoder` | alternating row-Transformer / column-Transformer, averaged each layer | header is literal **row 0** of the table, run through the identical row/column transformer — never mixed into a data cell's own text | [SFIG611/tabbie](https://github.com/SFIG611/tabbie) |
| `StruBertTableEncoder` | separate row-view / column-view BERT encoders + horizontal/vertical self-attention, fused at the end (late interaction) | every cell serialized as `"[header] [type] [value]"`, `type` = inferred `real`/`text` (TaBERT/StruBERT's `get_cell_input` template) | [medtray/StruBERT](https://github.com/medtray/StruBERT) |
| `TapasTableEncoder` | single linearized sequence + structural position ids (segment/column/row/rank) | headers are the pandas `DataFrame` column names, handled natively by `TapasTokenizer` | native `transformers.TapasModel` |
| `TurlTableEncoder` | visibility-matrix masked attention (row/column/global only) | caption/[CLS] is global; header tokens are column-local metadata nodes; every cell is one mean-pooled word-embedding node before contextualization — never concatenated into cell text | [sunlab-osu/TURL](https://github.com/sunlab-osu/TURL) |
| `HyTrelTableEncoder` | hypergraph message passing (row/column/table hyperedges), permutation-invariant | header text directly initializes that **column's hyperedge** (not the cell/node); row hyperedges init from a fixed `[ROW]` token, table hyperedge from the caption | [awslabs/hypergraph-tabular-lm](https://github.com/awslabs/hypergraph-tabular-lm) |

All six take the *same* input and return the *same* output shape:

```python
from table_encoders import TabbieTableEncoder

encoder = TabbieTableEncoder(model_name="bert-base-uncased")
out = encoder.encode(
    headers=["Player", "Team", "Points"],
    rows=[
        ["LeBron James", "Lakers", "27.1"],
        ["Nikola Jokic", "Nuggets", "26.4"],
    ],
    caption="2023-24 NBA scoring leaders",   # optional
)

out.cell_embeddings   # [n_rows, n_cols, dim]
out.row_embeddings    # [n_rows, dim]
out.col_embeddings    # [n_cols, dim]
out.table_embedding   # [dim]
```

Every encoder subclasses `BaseTableEncoder` (`common.py`) and is an
`nn.Module`, so they can be trained (fine-tuned / contrastively trained for
your query-table or table-table similarity objective) exactly like any
other PyTorch model — `encode()` is just an inference-mode convenience
wrapper around `forward()`.

## Why these particular re-implementations (not the original repos as-is)

- The original TABBIE/StruBERT/TURL repos pin very old `transformers` /
  `pytorch-pretrained-bert` versions and `allennlp`, which conflicts with a
  modern retrieval stack (and, in TABBIE's case, needs `torch==1.5.1+cu101`).
  Re-implementing the *architecture* against current `transformers` avoids
  a dependency-hell subproject in your repo.
- HyTrel's reference repo depends on `torch_geometric` purely to do
  node<->hyperedge attention pooling. At table scale (dozens–hundreds of
  cells) that's dense padded multi-head attention in disguise, so
  `hytrel.py` reimplements the same computation without the extra
  build/CUDA-wheel dependency.
- TAPAS is already natively supported in `transformers`, so `tapas_encoder.py`
  is a thin wrapper, not a reimplementation — it just reconstructs
  cell/row/col/table embeddings from TAPAS's token-level output the way the
  [Observatory paper/repo](https://github.com/superctj/observatory) does.

**Caveat:** none of these load the original papers' pretrained checkpoints
(those aren't published for `bert-base-uncased`-compatible modern
`transformers`, except TAPAS). They're architecturally faithful, randomly
initialized (aside from reusing BERT's pretrained token embeddings /
backbone where the paper does the same), and meant to be **trained
end-to-end on your own contrastive query-table / table-table objective** —
exactly like you'd train your own encoder, just with each one's structural
inductive bias baked in. If you want closer parity with the papers' own
numbers, each file's docstring explains exactly which part of the
architecture is exact vs. simplified.

## Installation

```bash
pip install torch transformers pandas
```

(`pandas` is only needed for `TapasTableEncoder`, which builds a DataFrame
for `TapasTokenizer`.)

## Suggested layout in your repo

Given your experiment plan references "Table Encoder Baselines" as one of
the pluggable components tested against your architecture, I'd drop this in
as a sibling package to your retrieval code, e.g.:

```
your_repo/
├── retrieval/                  # your query-table / table-table similarity code
│   ├── encoders/
│   │   ├── table_encoders/     # <- this package, unmodified
│   │   ├── your_encoder.py     # the model you're actually proposing
│   │   └── __init__.py
│   ├── joiners/                 # Starmie / DeepJoin baselines go here, same pattern
│   ├── retrievers/              # JAR / ARM / REAR baselines
│   └── train.py
├── experiments/
│   └── encoder_benchmark.py     # component-level test: each baseline vs SOTA benchmark
└── data/
```

Two integration points worth wiring up explicitly, since they map directly
onto open items in your plan:

1. **Column vs. cell-level similarity ablation** — every encoder already
   exposes both `cell_embeddings` and `col_embeddings`, so you can swap
   which one feeds your similarity module without touching encoder code.
2. **Header incorporation** — deliberately *not* centralized in `common.py`,
   because each paper handles it differently (see the table above) and a
   single shared `serialize_cell()` would have hidden that as a fake
   abstraction. Each encoder file owns a small, clearly-labeled
   header-handling function/section instead (`_serialize_cell` in
   `bert_baseline.py` and `strubert.py`, the row-0 construction in
   `tabbie.py`, the column-hyperedge init in `hytrel.py`). If you want a
   header-weighted vs. unweighted ablation, that's the one function to
   branch on per encoder — not a shared one to edit once, since "how
   headers are used" is itself part of what you're comparing across
   baselines.

## Testing

```bash
python test_encoders.py                 # all 6
python test_encoders.py bert tabbie      # subset
```

This is a wiring smoke test (shapes only), not a quality benchmark — plug
these into your Table Encoder Baselines benchmarking step (per your plan's
"Test components individually" experiment) to get real numbers.
