import torch
from objectives import GRPOLoss

def test_compute_advantages_basic():
    loss_fn = GRPOLoss()

    rewards = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
    ])

    advantages = loss_fn.compute_advantages(rewards)

    # Advantages should have the same shape as rewards.
    assert advantages.shape == rewards.shape

    # Each group should be normalized to mean ~0
    torch.testing.assert_close(
        advantages.mean(dim=1),
        torch.tensor([0.0]),
        atol=1e-6,
        rtol=1e-6,
    )

    # With unbiased=False, normalized advantages should have std ~1
    torch.testing.assert_close(
        advantages.std(dim=1, unbiased=False),
        torch.tensor([1.0]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_compute_advantages_groups_are_independent():
    loss_fn = GRPOLoss()

    rewards = torch.tensor([
        [1.0, 2.0, 3.0],
        [100.0, 200.0, 300.0],
    ])

    advantages = loss_fn.compute_advantages(rewards)

    # Both groups have the same relative structure:
    #
    # [1, 2, 3]
    # [100, 200, 300]
    #
    # Therefore their normalized advantages should be identical.
    torch.testing.assert_close(
        advantages[0],
        advantages[1],
        atol=1e-6,
        rtol=1e-6,
    )


def test_compute_advantages_preserves_group_order():
    loss_fn = GRPOLoss()

    rewards = torch.tensor([
        [1.0, 2.0, 3.0],
    ])

    advantages = loss_fn.compute_advantages(rewards)

    # Higher reward → higher advantage
    assert advantages[0, 0] < advantages[0, 1]
    assert advantages[0, 1] < advantages[0, 2]


def test_compute_advantages_constant_rewards():
    loss_fn = GRPOLoss()

    rewards = torch.tensor([
        [5.0, 5.0, 5.0, 5.0],
    ])

    advantages = loss_fn.compute_advantages(rewards)

    # Mean-centering gives zero everywhere.
    # epsilon prevents division by zero.
    expected = torch.zeros_like(rewards)

    torch.testing.assert_close(
        advantages,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )