import torch

from data.sft import SFTProcessor


class FakeTokenizer:
    """
    Deterministic tokenizer for testing SFTProcessor.

    We deliberately avoid a real tokenizer here so that we know
    exactly what token IDs should be produced.
    """

    def __call__(self, text, return_tensors = None):
        mapping = {
            "abc": [10, 11, 12],
            "de": [20, 21],
        }

        return {
            "input_ids": torch.tensor(
                [mapping[text]],
                dtype=torch.long,
            )
        }


def test_sft_processor():

    tokenizer = FakeTokenizer()

    processor = SFTProcessor(
        tokenizer=tokenizer,
        assistant_token_id=42,
        ignore_index=-100,
    )

    example = {
        "prompt": "abc",
        "response": "de",
    }

    processed = processor.process(example)

    expected_input_ids = torch.tensor(
        [10, 11, 12, 42, 20, 21],
        dtype=torch.long,
    )

    expected_labels = torch.tensor(
        [-100, -100, -100, -100, 20, 21],
        dtype=torch.long,
    )

    expected_attention_mask = torch.tensor(
        [1, 1, 1, 1, 1, 1],
        dtype=torch.long,
    )

    assert torch.equal(
        processed["input_ids"],
        expected_input_ids,
    )

    assert torch.equal(
        processed["labels"],
        expected_labels,
    )

    assert torch.equal(
        processed["attention_mask"],
        expected_attention_mask,
    )


def test_sft_processor_respects_custom_ignore_index():

    tokenizer = FakeTokenizer()

    processor = SFTProcessor(
        tokenizer=tokenizer,
        assistant_token_id=42,
        ignore_index=-999,
    )

    example = {
        "prompt": "abc",
        "response": "de",
    }

    processed = processor.process(example)

    expected_labels = torch.tensor(
        [-999, -999, -999, -999, 20, 21],
        dtype=torch.long,
    )

    assert torch.equal(
        processed["labels"],
        expected_labels,
    )


def test_sft_processor_output_lengths_match():

    tokenizer = FakeTokenizer()

    processor = SFTProcessor(
        tokenizer=tokenizer,
        assistant_token_id=42,
    )

    example = {
        "prompt": "abc",
        "response": "de",
    }

    processed = processor.process(example)

    assert len(processed["input_ids"]) == 6
    assert len(processed["labels"]) == 6
    assert len(processed["attention_mask"]) == 6
