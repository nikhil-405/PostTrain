import torch

class Trainer: 
    def __init__(self, model, optimizer, loss_fn, device = ""): 
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        if (device == ""):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def train_step(self, batch): 
        """
        batch: {input_ids, labels, attention_mask}
        """
        batch = {key: value.to(self.device) for key, value in batch.items()}
        x = {key: value for key, value in batch.items() if key != "labels"}

        # Forward pass
        self.optimizer.zero_grad()
        logits = self.model(**x).logits ## FIXED
        loss = self.loss_fn.get_loss(logits, batch['labels'])

        # Backprop & update
        loss.backward()
        self.optimizer.step()  
        
        return loss.item()

    def train(self, dataloader, epochs, verbose = 1):
        self.model.to(self.device)
        self.model.train()

        history = {"loss": []}

        for epoch in range(1, epochs+1):
            loss = 0
            for batch in dataloader:
                step_loss = self.train_step(batch)
                loss += step_loss

            loss /= len(dataloader)
            history["loss"].append(loss)

            if verbose: 
                print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f}")

        return history
