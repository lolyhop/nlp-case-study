from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class BertForTokenClassificationWithPOSFeatures(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self,
        config: AutoConfig,
        pos_feature_type: str,
        pos_feature_dim: Optional[int] = None,
        num_pos_tags: Optional[int] = None,
        pos_embed_dim: Optional[int] = None,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pos_feature_type = pos_feature_type
        self.num_pos_tags = num_pos_tags
        self.pos_embed_dim = pos_embed_dim

        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if pos_feature_type in {"one_hot", "logits"}:
            if pos_feature_dim is None:
                raise ValueError("pos_feature_dim must be provided for one_hot/logits")
            self.pos_feature_dim = pos_feature_dim
            self.pos_embedding = None

        elif pos_feature_type == "trainable_embed":
            if num_pos_tags is None or pos_embed_dim is None:
                raise ValueError(
                    "num_pos_tags and pos_embed_dim must be provided for trainable_embed"
                )
            self.pos_embedding = nn.Embedding(num_pos_tags, pos_embed_dim)
            self.pos_feature_dim = pos_embed_dim

        else:
            raise ValueError(f"Unsupported pos_feature_type: {pos_feature_type}")

        self.classifier = nn.Linear(
            config.hidden_size + self.pos_feature_dim,
            config.num_labels,
        )

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        pos_features: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        sequence_output = self.dropout(outputs.last_hidden_state)

        if self.pos_feature_type in {"one_hot", "logits"}:
            if pos_features is None:
                raise ValueError("pos_features must be provided for one_hot/logits")
            pos_repr = pos_features

        elif self.pos_feature_type == "trainable_embed":
            if pos_ids is None:
                raise ValueError("pos_ids must be provided for trainable_embed")

            pos_ids_clamped = pos_ids.clone()
            pos_ids_clamped[pos_ids_clamped < 0] = 0
            pos_repr = self.pos_embedding(pos_ids_clamped)

            ignore_mask = (pos_ids == -100).unsqueeze(-1)
            pos_repr = pos_repr.masked_fill(ignore_mask, 0.0)

        else:
            raise ValueError(f"Unsupported pos_feature_type: {self.pos_feature_type}")

        combined = torch.cat([sequence_output, pos_repr], dim=-1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForTokenClassificationWithMidPOSInjection(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self,
        config: AutoConfig,
        num_pos_tags: int,
        pos_embed_dim: int,
        inject_layer_idx: Optional[int] = None,
    ):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.num_pos_tags = num_pos_tags
        self.pos_embed_dim = pos_embed_dim

        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pos_embedding = nn.Embedding(num_pos_tags, pos_embed_dim)
        self.pos_proj = nn.Linear(pos_embed_dim, config.hidden_size)
        self.pos_layernorm = nn.LayerNorm(config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.inject_layer_idx = (
            inject_layer_idx
            if inject_layer_idx is not None
            else config.num_hidden_layers - 2
        )

        self._current_pos_repr = None

        self.bert.encoder.layer[self.inject_layer_idx].register_forward_pre_hook(
            self._inject_pos_hook,
            with_kwargs=True,
        )

        self.post_init()

    def _inject_pos_hook(self, module, args, kwargs):
        if self._current_pos_repr is None:
            return args, kwargs

        if len(args) > 0:
            hidden_states = args[0]
            new_hidden_states = self.pos_layernorm(
                hidden_states + self._current_pos_repr
            )
            new_args = (new_hidden_states,) + args[1:]
            return new_args, kwargs

        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
            kwargs["hidden_states"] = self.pos_layernorm(
                hidden_states + self._current_pos_repr
            )
            return args, kwargs

        raise RuntimeError("Could not find hidden_states in BertLayer forward inputs")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        if pos_ids is None:
            raise ValueError("pos_ids must be provided for mid-layer POS injection")

        pos_ids_clamped = pos_ids.clone()
        pos_ids_clamped[pos_ids_clamped < 0] = 0

        pos_repr = self.pos_embedding(pos_ids_clamped)
        pos_repr = self.pos_proj(pos_repr)

        ignore_mask = (pos_ids == -100).unsqueeze(-1)
        pos_repr = pos_repr.masked_fill(ignore_mask, 0.0)

        self._current_pos_repr = pos_repr

        try:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                **kwargs,
            )
        finally:
            self._current_pos_repr = None

        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
