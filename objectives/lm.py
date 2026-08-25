import torch
import torch.nn.functional as F
from .base import BaseLoss

class LMLoss(BaseLoss):
    def __init__(self, temperature = 1):
        self.temperature = temperature

    def get_loss(self, logits, labels, ignore_index = -100):
        """
        logits: [B, T, V] <-- for every timestep t, we have V numbers, logits (not probs) 
        labels: [B, T] 
        ignore_index = -100 <-- do not consider this label for loss computation
        """
        # b, t, v = logits.shape

        # Causal Shifting: The easiest way to think about it is: drop the last logit and drop the first label.
        shifted_logits = logits[:, :-1, :] / self.temperature
        shifted_labels = labels[:, 1:]

        # now we unbatch (so to say) and treat all of them as single examples
        unrolled_logits = torch.flatten(shifted_logits, start_dim = 0, end_dim = -2)
        unrolled_labels = torch.flatten(shifted_labels, start_dim = 0, end_dim = -1)
        
        return F.cross_entropy(unrolled_logits, unrolled_labels, ignore_index = ignore_index)