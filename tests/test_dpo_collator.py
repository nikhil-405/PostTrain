import torch

from data.dpo import DPOCollator


def test_dpo_collator():
    collator = DPOCollator(
        pad_token_id=0,
        ignore_index=-100,
    )

    examples = [
        {
            "chosen_input_ids": torch.tensor([1, 2, 42, 5]),
            "chosen_labels": torch.tensor([-100, -100, -100, 5]),
            "chosen_attention_mask": torch.tensor([1, 1, 1, 1]),

            "rejected_input_ids": torch.tensor([1, 2, 42, 6, 7]),
            "rejected_labels": torch.tensor([-100, -100, -100, 6, 7]),
            "rejected_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        },

        {
            "chosen_input_ids": torch.tensor([1, 2, 3, 42, 8, 9]),
            "chosen_labels": torch.tensor([-100, -100, -100, -100, 8, 9]),
            "chosen_attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),

            "rejected_input_ids": torch.tensor([1, 2, 42, 10]),
            "rejected_labels": torch.tensor([-100, -100, -100, 10]),
            "rejected_attention_mask": torch.tensor([1, 1, 1, 1]),
        },
    ]

    result = collator(examples)

    # --------------------------------------------------
    # CHOSEN
    # --------------------------------------------------

    # Maximum chosen length = 6
    expected_chosen_input_ids = torch.tensor([
        [1, 2, 42, 5, 0, 0],
        [1, 2, 3, 42, 8, 9],
    ])

    expected_chosen_labels = torch.tensor([
        [-100, -100, -100, 5, -100, -100],
        [-100, -100, -100, -100, 8, 9],
    ])

    expected_chosen_attention_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ])

    # --------------------------------------------------
    # REJECTED
    # --------------------------------------------------

    # Maximum rejected length = 5
    expected_rejected_input_ids = torch.tensor([
        [1, 2, 42, 6, 7],
        [1, 2, 42, 10, 0],
    ])

    expected_rejected_labels = torch.tensor([
        [-100, -100, -100, 6, 7],
        [-100, -100, -100, 10, -100],
    ])

    expected_rejected_attention_mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
    ])

    # --------------------------------------------------
    # ASSERTIONS
    # --------------------------------------------------

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