
import torch
from data.sft import SFTCollator


def test_sft_collator_pads_variable_length_examples():
    collator = SFTCollator(
        pad_token_id=0,
        ignore_index=-100,
    )

    examples = [
        {
            "input_ids": torch.tensor([10, 11, 42, 20]),
            "labels": torch.tensor([-100, -100, -100, 20]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
        },
        {
            "input_ids": torch.tensor([30, 42, 40, 41, 42, 50]),
            "labels": torch.tensor([-100, -100, 40, 41, 42, 50]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
        },
    ]

    batch = collator(examples)

    expected_input_ids = torch.tensor([
        [10, 11, 42, 20, 0, 0],
        [30, 42, 40, 41, 42, 50],
    ])

    expected_labels = torch.tensor([
        [-100, -100, -100, 20, -100, -100],
        [-100, -100, 40, 41, 42, 50],
    ])

    expected_attention_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ])

    assert torch.equal(
        batch["input_ids"],
        expected_input_ids,
    )

    assert torch.equal(
        batch["labels"],
        expected_labels,
    )

    assert torch.equal(
        batch["attention_mask"],
        expected_attention_mask,
    )


def test_sft_collator_preserves_already_padded_example():
    collator = SFTCollator(
        pad_token_id=0,
        ignore_index=-100,
    )

    examples = [
        {
            "input_ids": torch.tensor([10, 11, 42, 20]),
            "labels": torch.tensor([-100, -100, -100, 20]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
        },
        {
            "input_ids": torch.tensor([30, 42, 40, 41]),
            "labels": torch.tensor([-100, -100, 40, 41]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
        },
    ]

    batch = collator(examples)

    expected_input_ids = torch.tensor([
        [10, 11, 42, 20],
        [30, 42, 40, 41],
    ])

    expected_labels = torch.tensor([
        [-100, -100, -100, 20],
        [-100, -100, 40, 41],
    ])

    expected_attention_mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])

    assert torch.equal(batch["input_ids"], expected_input_ids)
    assert torch.equal(batch["labels"], expected_labels)
    assert torch.equal(
        batch["attention_mask"],
        expected_attention_mask,
    )


def test_sft_collator_respects_custom_ignore_index():
    collator = SFTCollator(
        pad_token_id=0,
        ignore_index=-999,
    )

    examples = [
        {
            "input_ids": torch.tensor([10, 20]),
            "labels": torch.tensor([-999, 20]),
            "attention_mask": torch.tensor([1, 1]),
        },
        {
            "input_ids": torch.tensor([30, 40, 50]),
            "labels": torch.tensor([-999, 40, 50]),
            "attention_mask": torch.tensor([1, 1, 1]),
        },
    ]

    batch = collator(examples)

    expected_labels = torch.tensor([
        [-999, 20, -999],
        [-999, 40, 50],
    ])

    assert torch.equal(
        batch["labels"],
        expected_labels,
    )


def test_sft_collator_returns_correct_batch_shapes():
    collator = SFTCollator(
        pad_token_id=0,
        ignore_index=-100,
    )

    examples = [
        {
            "input_ids": torch.tensor([10, 20, 30]),
            "labels": torch.tensor([-100, 20, 30]),
            "attention_mask": torch.tensor([1, 1, 1]),
        },
        {
            "input_ids": torch.tensor([40, 50]),
            "labels": torch.tensor([-100, 50]),
            "attention_mask": torch.tensor([1, 1]),
        },
        {
            "input_ids": torch.tensor([60, 70, 80, 90]),
            "labels": torch.tensor([-100, 70, 80, 90]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
        },
    ]

    batch = collator(examples)

    assert batch["input_ids"].shape == (3, 4)
    assert batch["labels"].shape == (3, 4)
    assert batch["attention_mask"].shape == (3, 4)