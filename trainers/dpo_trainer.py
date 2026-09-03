import copy
import torch
import torch.nn.functional as F

class DPOTrainer:
    def __init__(self, model, optimizer, loss_fn, device = "", ignore_index = -100): 
        self.policy_model = model
        self.reference_model = copy.deepcopy(model)
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.ignore_index = ignore_index

        if (device == ""):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def _get_sequence_logprobs(self, model, input_ids, labels, attention_mask): 
        logits = model(input_ids = input_ids, attention_mask = attention_mask).logits

        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]

        mask = shifted_labels != self.ignore_index
        safe_labels = shifted_labels.masked_fill(~mask, 0)

        log_probs = F.log_softmax(shifted_logits, dim = -1)

        token_logprobs = log_probs.gather(dim = -1, index = safe_labels.unsqueeze(-1)).squeeze(-1) * mask
        sequence_logprobs = torch.sum(token_logprobs, dim = -1)
        return sequence_logprobs
        
    def train_step(self, batch): 
        """
        batch: {input_ids, labels, attention_mask} * 2
        (for rejected as well as chosen)
        """
        batch = {key: value.to(self.device) for key, value in batch.items()}

        self.optimizer.zero_grad()

        policy_chosen_seq_logprobs = self._get_sequence_logprobs(self.policy_model, batch["chosen_input_ids"], batch["chosen_labels"], batch["chosen_attention_mask"])

        policy_rej_seq_logprobs = self._get_sequence_logprobs(self.policy_model, batch["rejected_input_ids"], batch["rejected_labels"], batch["rejected_attention_mask"])

        with torch.no_grad():
            ref_chosen_seq_logprobs = self._get_sequence_logprobs(self.reference_model, batch["chosen_input_ids"], batch["chosen_labels"], batch["chosen_attention_mask"])
            
            ref_rej_seq_logprobs = self._get_sequence_logprobs(self.reference_model, batch["rejected_input_ids"], batch["rejected_labels"], batch["rejected_attention_mask"])

        loss = self.loss_fn.get_loss(policy_chosen_seq_logprobs, policy_rej_seq_logprobs, ref_chosen_seq_logprobs, ref_rej_seq_logprobs)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, dataloader, epochs, verbose = 1):
            self.policy_model.to(self.device)
            self.policy_model.train()

            self.reference_model.to(self.device)
            self.reference_model.eval()
    
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