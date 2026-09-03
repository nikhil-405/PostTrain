import copy

import torch
import torch.nn as nn

from trainers.dpo_trainer import DPOTrainer
from objectives.dpo import DPOLoss

class TinyModel(nn.Module):
    """
    Tiny causal LM-like model.

    Produces logits of shape [B, T, V].
    """

    def __init__(self, vocab_size=5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 4)
        self.linear = nn.Linear(4, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        logits = self.linear(x)

        return type(
            "Output",
            (),
            {"logits": logits}
        )()


def make_batch():
    return {
        "chosen_input_ids": torch.tensor([
            [1, 2, 3, 4],
            [1, 2, 3, 0],
        ]),

        "chosen_labels": torch.tensor([
            [-100, -100, 3, 4],
            [-100, -100, 3, -100],
        ]),

        "chosen_attention_mask": torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ]),

        "rejected_input_ids": torch.tensor([
            [1, 2, 4, 3],
            [1, 2, 4, 0],
        ]),

        "rejected_labels": torch.tensor([
            [-100, -100, 4, 3],
            [-100, -100, 4, -100],
        ]),

        "rejected_attention_mask": torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ]),
    }


def test_get_sequence_logprobs():

    torch.manual_seed(42)

    model = TinyModel(vocab_size=5)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    loss_fn = DPOLoss(beta=1.0)

    trainer = DPOTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
    )

    input_ids = torch.tensor([
        [1, 2, 3, 4],
    ])

    labels = torch.tensor([
        [-100, -100, 3, 4],
    ])

    attention_mask = torch.tensor([
        [1, 1, 1, 1],
    ])

    result = trainer._get_sequence_logprobs(
        model,
        input_ids,
        labels,
        attention_mask,
    )

    # Manually compute the expected value.
    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits

        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]

        log_probs = torch.log_softmax(
            shifted_logits,
            dim=-1
        )

        # Only labels 3 and 4 should contribute.
        expected = (
            log_probs[0, 1, 3]
            + log_probs[0, 2, 4]
        ).reshape(1)

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_get_sequence_logprobs_ignores_padding():

    torch.manual_seed(42)

    model = TinyModel(vocab_size=5)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    loss_fn = DPOLoss(beta=1.0)

    trainer = DPOTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
    )

    input_ids = torch.tensor([
        [1, 2, 3, 0],
    ])

    labels = torch.tensor([
        [-100, -100, 3, -100],
    ])

    attention_mask = torch.tensor([
        [1, 1, 1, 0],
    ])

    result = trainer._get_sequence_logprobs(
        model,
        input_ids,
        labels,
        attention_mask,
    )

    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits

        expected = torch.log_softmax(
            logits[:, 1, :],
            dim=-1
        )[:, 3]

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_train_step():

    torch.manual_seed(42)

    model = TinyModel(vocab_size=5)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1
    )

    loss_fn = DPOLoss(beta=1.0)

    trainer = DPOTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
    )

    batch = make_batch()

    initial_policy_parameters = [
        param.detach().clone()
        for param in trainer.policy_model.parameters()
    ]

    initial_reference_parameters = [
        param.detach().clone()
        for param in trainer.reference_model.parameters()
    ]

    loss = trainer.train_step(batch)

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))

    # --------------------------------------------------
    # Policy should have changed.
    # --------------------------------------------------

    policy_changed = any(
        not torch.equal(before, after)
        for before, after in zip(
            initial_policy_parameters,
            trainer.policy_model.parameters(),
        )
    )

    assert policy_changed

    # --------------------------------------------------
    # Reference model MUST NOT have changed.
    # --------------------------------------------------

    reference_changed = any(
        not torch.equal(before, after)
        for before, after in zip(
            initial_reference_parameters,
            trainer.reference_model.parameters(),
        )
    )

    assert not reference_changed


def test_reference_model_is_independent_copy():

    torch.manual_seed(42)

    model = TinyModel(vocab_size=5)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1
    )

    loss_fn = DPOLoss(beta=1.0)

    trainer = DPOTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
    )

    # They must not be the same object.
    assert trainer.policy_model is not trainer.reference_model

    # But initially they should have identical parameters.
    for policy_param, reference_param in zip(
        trainer.policy_model.parameters(),
        trainer.reference_model.parameters(),
    ):
        torch.testing.assert_close(
            policy_param,
            reference_param,
        )


def test_reference_model_has_no_gradients():

    torch.manual_seed(42)

    model = TinyModel(vocab_size=5)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1
    )

    loss_fn = DPOLoss(beta=1.0)

    trainer = DPOTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
    )

    batch = make_batch()

    trainer.train_step(batch)

    for parameter in trainer.reference_model.parameters():
        assert parameter.requires_grad is False
        assert parameter.grad is None