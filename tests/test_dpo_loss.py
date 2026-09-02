import torch

from objectives.dpo import DPOLoss


def test_dpo_loss():
    loss_fn = DPOLoss(beta=1.0)

    policy_chosen = torch.tensor([-1.0])
    policy_rejected = torch.tensor([-2.0])

    reference_chosen = torch.tensor([-1.5])
    reference_rejected = torch.tensor([-1.5])

    result = loss_fn.get_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
    )

    expected = torch.tensor(0.3132617)

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_dpo_loss_with_beta():
    loss_fn = DPOLoss(beta=2.0)

    policy_chosen = torch.tensor([-1.0])
    policy_rejected = torch.tensor([-2.0])

    reference_chosen = torch.tensor([-1.5])
    reference_rejected = torch.tensor([-1.5])

    result = loss_fn.get_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
    )

    expected = torch.tensor(0.126928)

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_dpo_loss_batch():
    loss_fn = DPOLoss(beta=1.0)

    policy_chosen = torch.tensor([
        -1.0,
        -2.0,
    ])

    policy_rejected = torch.tensor([
        -2.0,
        -1.0,
    ])

    reference_chosen = torch.tensor([
        -1.5,
        -1.5,
    ])

    reference_rejected = torch.tensor([
        -1.5,
        -1.5,
    ])

    result = loss_fn.get_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
    )

    expected = torch.tensor(0.8132617)

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )