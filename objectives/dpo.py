import torch
import torch.nn.functional as F
from .base import BaseLoss

# https://arxiv.org/pdf/2305.18290
class DPOLoss(BaseLoss):
    def __init__(self, beta = 1):
        self.beta = beta

    def get_sequence_logprobs(self, logits, labels, ignore_index = -100):
        """
        logits: [B, T, V] <-- for every timestep t, we have V numbers, logits (not probs) 
        labels: [B, T] 
        ignore_index = -100 <-- do not consider this label for loss computation
        """
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]

        mask = shifted_labels != ignore_index # this is [B, T]
        safe_labels = shifted_labels.masked_fill(~mask, 0)

        log_probs = F.log_softmax(shifted_logits, dim = -1)
        chosen_logprobs = log_probs.gather(dim = -1, index = safe_labels.unsqueeze(-1)).squeeze(-1) * mask

        return torch.sum(chosen_logprobs, dim = -1)

    def get_loss(
            self, policy_chosen_logprobs, policy_rejected_logprobs,
            reference_chosen_logprobs, reference_rejected_logprobs):
    
            chosen_relative = policy_chosen_logprobs - reference_chosen_logprobs
            rejected_relative = policy_rejected_logprobs - reference_rejected_logprobs
    
            preference_margin = chosen_relative - rejected_relative
    
            return F.softplus(-self.beta * preference_margin).mean()