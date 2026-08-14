"""
Embeds text cell values into R^k using a pretrained BERT-style model.

Pooling strategy: [CLS] token from the last hidden state, then a linear
projection down (or up) to the shared output_dim, so text and numeric
embeddings land in the same-sized space before going to the aggregator.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEmbedder(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        max_length: int = 32,
        trainable: bool = False,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        # Always a trainable linear map, even if hidden_size == output_dim --
        # matching CLIP's design: the backbone's native space and the
        # shared comparison space aren't assumed to be the same thing just
        # because they're the same size.
        self.proj = nn.Linear(hidden_size, output_dim, bias=False)

    def forward(self, cells: list[str]) -> torch.Tensor:
        """
        cells: list of N raw text strings
        returns: [N, output_dim]
        """

        if len(cells) == 0:
            return torch.empty(0, self.proj.out_features)

        device = next(self.encoder.parameters()).device

        encoded = self.tokenizer(
            cells,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        outputs = self.encoder(**encoded)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [N, hidden_size]

        return self.proj(cls_embeddings)