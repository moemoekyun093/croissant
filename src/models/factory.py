"""Builds the table model for a given --encoder choice."""

from src.encoding.baseline_encoders.adapter import build_baseline_model
from src.encoding.cell_encoder import CellEncoder
from src.models.table_encoder import TableEncoder


def build_table_model(args):
    if args.encoder == "ours":
        cell_encoder = CellEncoder(
            text_model_name=args.text_model_name,
            output_dim=args.embed_dim,
            text_max_length=args.text_max_length,
            text_trainable=args.text_trainable,
            text_max_batch_size=args.text_max_batch_size,
            header_mode=args.header_mode,
        )
        return TableEncoder(
            cell_encoder,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            nonlinearity=args.nonlinearity,
            channel_mix_hidden_dim=args.channel_mix_hidden_dim,
            num_heads=args.num_heads,
            table_microbatch_cell_budget=getattr(args, "table_microbatch_cell_budget", None),
            table_microbatch_max_tables=getattr(args, "table_microbatch_max_tables", None),
        )
    return build_baseline_model(
        args.encoder,
        embed_dim=args.embed_dim,
        model_name=args.model_name,
        num_layers=args.num_layers,
        tabbie_ffn_hidden_dim=args.tabbie_ffn_hidden_dim,
        strubert_ffn_hidden_dim=getattr(args, "strubert_ffn_hidden_dim", None),
        turl_attention_budget=getattr(args, "turl_attention_budget", None),
        table_microbatch_cell_budget=getattr(args, "table_microbatch_cell_budget", None),
        table_microbatch_max_tables=getattr(args, "table_microbatch_max_tables", None),
        device=args.device,
    )
