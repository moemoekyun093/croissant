"""Evaluate a prepared-model checkpoint on prepared val/test artifacts."""

from __future__ import annotations

import argparse
import json

import torch

from src.models.prepared_table_encoder import (
    PreparedQueryEncoder,
    PreparedTabbieEncoder,
    PreparedTableEncoder,
    PreparedTurlEncoder,
)
from src.scoring.multi_score import MultiScorer
from src.training.prepared_evaluator import evaluate_prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query_batch_size", type=int, default=32)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--encoder", choices=("ours", "tabbie", "turl"), default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--channel_mix_hidden_dim", type=int, default=None)
    parser.add_argument("--tabbie_ffn_hidden_dim", type=int, default=None)
    parser.add_argument("--turl_ffn_hidden_dim", type=int, default=None)
    parser.add_argument("--turl_attention_budget", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    try:
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint["metadata"]
    config = checkpoint.get("model_config", {})
    encoder = args.encoder or checkpoint.get("encoder") or config.get("encoder")
    if encoder not in ("ours", "tabbie", "turl"):
        parser.error("checkpoint has no encoder metadata; pass --encoder")
    dim = int(metadata["projection_dim"])
    num_layers = args.num_layers or config.get("num_layers", 3)
    num_heads = args.num_heads or config.get("num_heads", 8)

    if encoder == "ours":
        table_model = PreparedTableEncoder(
            embed_dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            channel_mix_hidden_dim=(
                args.channel_mix_hidden_dim
                or config.get("channel_mix_hidden_dim", 512)
            ),
            nonlinearity=config.get("nonlinearity", "sigmoid"),
        ).to(device)
    elif encoder == "tabbie":
        table_model = PreparedTabbieEncoder(
            embed_dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden_dim=(
                args.tabbie_ffn_hidden_dim
                or config.get("tabbie_ffn_hidden_dim", 512)
            ),
            max_rows=int(metadata["max_rows"]) + 1,
            max_columns=int(metadata["max_columns"]),
        ).to(device)
    else:
        table_model = PreparedTurlEncoder(
            embed_dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden_dim=(
                args.turl_ffn_hidden_dim
                or config.get("turl_ffn_hidden_dim", 512)
            ),
            attention_budget=(
                args.turl_attention_budget
                or config.get("turl_attention_budget", 2_000_000)
            ),
        ).to(device)
    query_model = PreparedQueryEncoder(dim, dim).to(device)
    scorer = MultiScorer().to(device)
    table_model.load_state_dict(checkpoint["table_model_state_dict"])
    query_model.load_state_dict(checkpoint["query_model_state_dict"])
    scorer.load_state_dict(checkpoint["scorer_state_dict"])

    metrics = evaluate_prepared(
        args.prepared_dir,
        table_model,
        query_model,
        scorer,
        device,
        metadata,
        query_batch_size=args.query_batch_size,
        progress=lambda message: print(message, flush=True),
    )
    result = {
        "checkpoint": args.checkpoint,
        "prepared_dir": args.prepared_dir,
        "encoder": encoder,
        **metrics,
    }
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)


if __name__ == "__main__":
    main()
