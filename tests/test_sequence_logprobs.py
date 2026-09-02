import torch

from objectives.dpo import DPOLoss


def test_sequence_logprobs():
    loss_fn = DPOLoss()

    logits = torch.tensor([[
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 1.0],
        [3.0, 1.0, 2.0],
        [1.0, 3.0, 2.0],
    ]])

    labels = torch.tensor([[
        0, 2, 1, 0
    ]])

    result = loss_fn.get_sequence_logprobs(logits, labels)

    expected = torch.tensor([
        -0.985058
    ])

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_sequence_logprobs_ignore_index():
    loss_fn = DPOLoss()

    logits = torch.tensor([[
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 1.0],
        [3.0, 1.0, 2.0],
        [1.0, 3.0, 2.0],
    ]])

    labels = torch.tensor([[
        0, 2, -100, 0
    ]])

    result = loss_fn.get_sequence_logprobs(
        logits,
        labels,
        ignore_index=-100,
    )

    expected = torch.tensor([
        -0.815212
    ])

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_sequence_logprobs_batch():
    loss_fn = DPOLoss()

    logits = torch.tensor([
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 1.0],
            [3.0, 1.0, 2.0],
        ],
        [
            [3.0, 1.0, 2.0],
            [1.0, 3.0, 2.0],
            [2.0, 4.0, 1.0],
        ],
    ])

    labels = torch.tensor([
        [0, 2, 1],
        [1, 0, 1],
    ])

    result = loss_fn.get_sequence_logprobs(logits, labels)

    expected = torch.tensor([
        -0.577452,
        -0.815212,
    ])

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )