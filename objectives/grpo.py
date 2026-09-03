import torch
import torch.nn.functional as F
from .base import BaseLoss

class GRPOLoss(BaseLoss):
    def __init__(self, beta = 1.0, epsilon = 0.2):
        self.beta = beta
        self.epsilon = epsilon

    def get_loss(self, policy_logprobs, old_logprobs, reference_logprobs, advantages, mask): 
        r = torch.exp(policy_logprobs - old_logprobs)
        r_clip = torch.clip(r, 1-self.epsilon, 1 + self.epsilon)

        adv_broadcasted = torch.unsqueeze(advantages, dim = -1)
        objective = torch.minimum(adv_broadcasted * r, r_clip * adv_broadcasted)
        policy_loss = - torch.sum(objective * mask) / torch.sum(mask)

        log_ratio_ref = reference_logprobs - policy_logprobs
        kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
        kl_loss = torch.sum(kl*mask) / torch.sum(mask)
        return policy_loss + self.beta * kl_loss