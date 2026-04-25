from __future__ import annotations

from typing import Optional

from src.constants import NER_LABELS_COLUMN, TOKENS_COLUMN


def tokenize_and_align_labels(
    examples,
    tokenizer,
    max_length: int,
    labels_column: str = NER_LABELS_COLUMN,
):
    tokenized = tokenizer(
        examples[TOKENS_COLUMN],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )

    aligned_labels = []
    for batch_index, word_labels in enumerate(examples[labels_column]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(int(word_labels[word_id]))
            else:
                label_ids.append(-100)

            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def build_one_hot_feature(
    raw_pos_id: int,
    pos_id_map: dict[int, int],
    num_pos_tags: int,
) -> list[float]:
    mapped_id = pos_id_map[int(raw_pos_id)]
    vec = [0.0] * num_pos_tags
    vec[mapped_id] = 1.0
    return vec


def tokenize_and_align_features(
    examples,
    tokenizer,
    max_length: int,
    pos_feature_type: str,
    pos_column: str,
    pos_id_map: Optional[dict[int, int]] = None,
    pos_feature_dim: Optional[int] = None,
    ner_labels_column: str = NER_LABELS_COLUMN,
):
    tokenized = tokenizer(
        examples[TOKENS_COLUMN],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )

    aligned_ner_labels = []
    aligned_pos_payload = []

    for batch_index, (word_ner_labels, word_pos_values) in enumerate(
        zip(examples[ner_labels_column], examples[pos_column])
    ):
        word_ids = tokenized.word_ids(batch_index=batch_index)

        previous_word_id = None
        ner_label_ids = []

        if pos_feature_type in {"one_hot", "logits"}:
            pos_payload = []
        elif pos_feature_type == "trainable_embed":
            pos_payload = []
        else:
            raise ValueError(f"Unsupported pos_feature_type: {pos_feature_type}")

        for word_id in word_ids:
            if word_id is None:
                ner_label_ids.append(-100)

                if pos_feature_type in {"one_hot", "logits"}:
                    pos_payload.append([0.0] * pos_feature_dim)
                else:
                    pos_payload.append(-100)

            elif word_id != previous_word_id:
                ner_label_ids.append(int(word_ner_labels[word_id]))

                if pos_feature_type == "one_hot":
                    pos_payload.append(
                        build_one_hot_feature(
                            raw_pos_id=int(word_pos_values[word_id]),
                            pos_id_map=pos_id_map,
                            num_pos_tags=pos_feature_dim,
                        )
                    )
                elif pos_feature_type == "logits":
                    pos_payload.append([float(x) for x in word_pos_values[word_id]])
                elif pos_feature_type == "trainable_embed":
                    pos_payload.append(pos_id_map[int(word_pos_values[word_id])])
                else:
                    raise ValueError(
                        f"Unsupported pos_feature_type: {pos_feature_type}"
                    )

            else:
                ner_label_ids.append(-100)

                if pos_feature_type in {"one_hot", "logits"}:
                    pos_payload.append([0.0] * pos_feature_dim)
                else:
                    pos_payload.append(-100)

            previous_word_id = word_id

        aligned_ner_labels.append(ner_label_ids)
        aligned_pos_payload.append(pos_payload)

    tokenized["labels"] = aligned_ner_labels

    if pos_feature_type in {"one_hot", "logits"}:
        tokenized["pos_features"] = aligned_pos_payload
    elif pos_feature_type == "trainable_embed":
        tokenized["pos_ids"] = aligned_pos_payload

    return tokenized
