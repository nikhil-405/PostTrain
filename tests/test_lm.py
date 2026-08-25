import torch
import torch.nn.functional as F

from objectives.lm import LMLoss

def test_lm_loss_known_value():
    """
    Sequence: a b c a
    Token IDs: 0 1 2 0
    """

    logits = torch.tensor([
        [
            [0.0, 2.0, 0.0],  # a -> b
            [0.0, 0.0, 2.0],  # b -> c
            [2.0, 0.0, 0.0],  # c -> a
            [0.0, 0.0, 0.0],  # no target
        ]
    ])

    labels = torch.tensor([
        [0, 1, 2, 0]
    ])

    loss_fn = LMLoss()

    actual = loss_fn.get_loss(logits, labels)

    expected = -torch.log(
        torch.exp(torch.tensor(2.0))
        / (torch.exp(torch.tensor(2.0)) + 2.0)
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_lm_loss_better_predictions_have_lower_loss():
    """
    A model assigning high logits to the correct next tokens
    should have a lower loss than one assigning high logits
    to incorrect tokens.
    """

    labels = torch.tensor([
        [0, 1, 2, 0]
    ])

    good_logits = torch.tensor([
        [
            [0.0, 5.0, 0.0],  # a -> b
            [0.0, 0.0, 5.0],  # b -> c
            [5.0, 0.0, 0.0],  # c -> a
            [0.0, 0.0, 0.0],
        ]
    ])

    bad_logits = torch.tensor([
        [
            [5.0, 0.0, 0.0],  # a -> a (wrong)
            [5.0, 0.0, 0.0],  # b -> a (wrong)
            [0.0, 5.0, 0.0],  # c -> b (wrong)
            [0.0, 0.0, 0.0],
        ]
    ])

    loss_fn = LMLoss()

    good_loss = loss_fn.get_loss(good_logits, labels)
    bad_loss = loss_fn.get_loss(bad_logits, labels)

    assert good_loss < bad_loss


def test_lm_loss_ignore_index():
    """
    Sequence: a b c a

    Labels:

        [a, b, -100, a]

    After shifting:

        prediction from a  -> b       included
        prediction from ab -> -100    ignored
        prediction from abc -> a      included

    Therefore only TWO predictions contribute to the loss.
    """

    logits = torch.tensor([
        [
            [0.0, 2.0, 0.0],      # a -> b
            [100.0, 100.0, 100.0], # ab -> c, ignored
            [2.0, 0.0, 0.0],      # abc -> a
            [0.0, 0.0, 0.0],
        ]
    ])

    labels = torch.tensor([
        [0, 1, -100, 0]
    ])

    loss_fn = LMLoss()

    actual = loss_fn.get_loss(logits, labels)

    expected = -torch.log(
        torch.exp(torch.tensor(2.0))
        / (torch.exp(torch.tensor(2.0)) + 2.0)
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_lm_loss_ignore_index_matches_manual_selection():
    """
    Verify that ignore_index=-100 produces the same loss as
    manually removing the ignored prediction.
    """

    logits = torch.tensor([
        [
            [0.0, 2.0, 0.0],  # target b
            [0.0, 0.0, 5.0],  # target ignored
            [2.0, 0.0, 0.0],  # target a
            [0.0, 0.0, 0.0],
        ]
    ])

    labels = torch.tensor([
        [0, 1, -100, 0]
    ])

    loss_fn = LMLoss()

    actual = loss_fn.get_loss(logits, labels)

    # After shifting, the ignored prediction is the second one.
    selected_logits = torch.tensor([
        [0.0, 2.0, 0.0],
        [2.0, 0.0, 0.0],
    ])

    selected_labels = torch.tensor([
        1,
        0,
    ])

    expected = F.cross_entropy(
        selected_logits,
        selected_labels,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_lm_loss_batching():
    """
    Verify that multiple sequences in a batch are handled correctly.
    """

    logits = torch.tensor([
        [
            [0.0, 2.0, 0.0],  # a -> b
            [0.0, 0.0, 2.0],  # b -> c
            [0.0, 0.0, 0.0],
        ],
        [
            [2.0, 0.0, 0.0],  # c -> a
            [0.0, 2.0, 0.0],  # a -> b
            [0.0, 0.0, 0.0],
        ],
    ])

    labels = torch.tensor([
        [0, 1, 2],
        [2, 0, 1],
    ])

    loss_fn = LMLoss()

    actual = loss_fn.get_loss(logits, labels)

    expected = -torch.log(
        torch.exp(torch.tensor(2.0))
        / (torch.exp(torch.tensor(2.0)) + 2.0)
    )

    assert torch.allclose(actual, expected, atol=1e-6)