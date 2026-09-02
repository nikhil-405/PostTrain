import torch
import torch.nn as nn

from trainers.trainer import Trainer
from objectives.lm import LMLoss


class TinyLM(nn.Module):
    """
    causal LM for testing the Trainer API.

    input_ids: [B, T]
    logits:    [B, T, V]
    """

    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)

        return type(
            "Output",
            (),
            {"logits": logits}
        )()


class TinyDataset(torch.utils.data.Dataset):
    """
    Dataset containing the same simple sequence repeatedly.

    Vocabulary:
        a = 0
        b = 1
        c = 2

    Sequence:
        a b c a b c

    The model should learn the repeating pattern:
        a -> b
        b -> c
        c -> a
    """

    def __init__(self, num_examples=32):
        self.input_ids = torch.tensor(
            [0, 1, 2, 0, 1, 2]
        )

        self.num_examples = num_examples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids.clone(),
            "attention_mask": torch.ones_like(self.input_ids),
            "labels": self.input_ids.clone(),
        }


def test_trainer_reduces_loss():
    torch.manual_seed(42)

    vocab_size = 3
    hidden_size = 16

    model = TinyLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05,
    )

    loss_fn = LMLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
    )

    dataset = TinyDataset(num_examples=32)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )

    # Measure loss before training.
    batch = next(iter(dataloader))

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        initial_loss = loss_fn.get_loss(
            outputs.logits,
            batch["labels"],
        ).item()

    # Train.
    history = trainer.train(
        dataloader=dataloader,
        epochs=20,
        verbose=0,
    )

    final_loss = history["loss"][-1]

    assert final_loss < initial_loss


def test_trainer_updates_parameters():
    torch.manual_seed(42)

    model = TinyLM(
        vocab_size=3,
        hidden_size=8,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05,
    )

    loss_fn = LMLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
    )

    dataset = TinyDataset(num_examples=8)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
    )

    # Save parameters before training.
    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    batch = next(iter(dataloader))

    trainer.train_step(batch)

    parameters_after = list(model.parameters())

    # At least one parameter should have changed.
    changed = [
        not torch.equal(before, after.detach())
        for before, after in zip(
            parameters_before,
            parameters_after,
        )
    ]

    assert any(changed)