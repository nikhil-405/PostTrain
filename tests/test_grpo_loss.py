import torch
from objectives.grpo import GRPOLoss

def test_grpo_loss_basic():
    loss_fn = GRPOLoss(beta=1.0, epsilon=0.2)

    policy_logprobs = torch.tensor([
        [
            [-0.5, -0.5],
            [-1.0, -1.0],
        ]
    ])

    old_logprobs = torch.tensor([
        [
            [-0.5, -0.5],
            [-1.0, -1.0],
        ]
    ])

    reference_logprobs = torch.tensor([
        [
            [-0.5, -0.5],
            [-1.0, -1.0],
        ]
    ])

    advantages = torch.tensor([
        [1.0, -1.0]
    ])

    mask = torch.ones(1, 2, 2)

    result = loss_fn.get_loss(
        policy_logprobs,
        old_logprobs,
        reference_logprobs,
        advantages,
        mask,
    )

    # ratio = 1 everywhere
    # KL = 0 everywhere
    # policy objective = advantage
    #
    # advantages = [1, -1]
    # mean objective = 0
    # policy loss = -0 = 0
    #
    # total loss = 0
    expected = torch.tensor(0.0)

    torch.testing.assert_close(result, expected)


def test_grpo_loss_clipping():
    loss_fn = GRPOLoss(beta=0.0, epsilon=0.2)

    # Current policy is 2x as likely as old policy.
    policy_logprobs = torch.log(torch.tensor([
        [
            [2.0],
            [2.0],
        ]
    ]))

    old_logprobs = torch.log(torch.tensor([
        [
            [1.0],
            [1.0],
        ]
    ]))

    reference_logprobs = policy_logprobs.clone()

    advantages = torch.tensor([
        [1.0, 1.0]
    ])

    mask = torch.ones(1, 2, 1)

    result = loss_fn.get_loss(
        policy_logprobs,
        old_logprobs,
        reference_logprobs,
        advantages,
        mask,
    )

    # ratio = 2
    # clipped ratio = 1.2
    #
    # objective = min(2 * 1, 1.2 * 1)
    #           = 1.2
    #
    # policy loss = -1.2
    expected = torch.tensor(-1.2)

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_grpo_loss_mask():
    loss_fn = GRPOLoss(beta=0.0, epsilon=0.2)

    # ratio = 1 everywhere
    policy_logprobs = torch.tensor([
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    ])

    old_logprobs = torch.tensor([
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    ])

    reference_logprobs = policy_logprobs.clone()

    advantages = torch.tensor([
        [1.0, 1.0]
    ])

    # Only two tokens are valid.
    mask = torch.tensor([
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    ])

    result = loss_fn.get_loss(
        policy_logprobs,
        old_logprobs,
        reference_logprobs,
        advantages,
        mask,
    )

    # Both valid tokens have:
    #
    # ratio = 1
    # advantage = 1
    # objective = 1
    #
    # mean = 1
    # policy loss = -1
    expected = torch.tensor(-1.0)

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_grpo_loss_kl_penalty():
    loss_fn = GRPOLoss(beta=1.0, epsilon=0.2)

    policy_logprobs = torch.tensor([
        [
            [0.0],
        ]
    ])

    old_logprobs = policy_logprobs.clone()

    # reference - policy = 1
    reference_logprobs = torch.tensor([
        [
            [1.0],
        ]
    ])

    advantages = torch.tensor([
        [0.0]
    ])

    mask = torch.ones(1, 1, 1)

    result = loss_fn.get_loss(
        policy_logprobs,
        old_logprobs,
        reference_logprobs,
        advantages,
        mask,
    )

    # x = reference - policy = 1
    #
    # KL = exp(x) - x - 1
    #    = e - 2
    #
    # policy loss = 0 because advantage = 0
    expected = torch.tensor(torch.e - 2)

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )