import torch

from data.dpo import DPOProcessor


class FakeTokenizer:
    def __init__(self):
        self.vocab = {
            "What": 1,
            "is": 2,
            "2": 3,
            "+": 4,
            "equals": 5,
            "4": 6,
            "3": 7,
        }

    def __call__(self, text, return_tensors=None):
        ids = [
            self.vocab[word]
            for word in text.split()
        ]

        return {
            "input_ids": torch.tensor([ids], dtype=torch.long)
        }


def test_dpo_processor():
    tokenizer = FakeTokenizer()

    processor = DPOProcessor(
        tokenizer=tokenizer,
        assistant_token_id=42,
        ignore_index=-100,
    )

    example = {
        "prompt": "What is 2 +",
        "chosen": "equals 4",
        "rejected": "equals 3",
    }

    result = processor.process(example)

    # Prompt:
    # What is 2 +
    # -> [1, 2, 3, 4]
    #
    # Chosen:
    # equals 4
    # -> [5, 6]
    #
    # Rejected:
    # equals 3
    # -> [5, 7]

    expected_chosen_input_ids = torch.tensor([
        1, 2, 3, 4, 42, 5, 6
    ])

    expected_chosen_labels = torch.tensor([
        -100, -100, -100, -100, -100, 5, 6
    ])

    expected_chosen_attention_mask = torch.tensor([
        1, 1, 1, 1, 1, 1, 1
    ])

    expected_rejected_input_ids = torch.tensor([
        1, 2, 3, 4, 42, 5, 7
    ])

    expected_rejected_labels = torch.tensor([
        -100, -100, -100, -100, -100, 5, 7
    ])

    expected_rejected_attention_mask = torch.tensor([
        1, 1, 1, 1, 1, 1, 1
    ])

    torch.testing.assert_close(
        result["chosen_input_ids"],
        expected_chosen_input_ids,
    )

    torch.testing.assert_close(
        result["chosen_labels"],
        expected_chosen_labels,
    )

    torch.testing.assert_close(
        result["chosen_attention_mask"],
        expected_chosen_attention_mask,
    )

    torch.testing.assert_close(
        result["rejected_input_ids"],
        expected_rejected_input_ids,
    )

    torch.testing.assert_close(
        result["rejected_labels"],
        expected_rejected_labels,
    )

    torch.testing.assert_close(
        result["rejected_attention_mask"],
        expected_rejected_attention_mask,
    )