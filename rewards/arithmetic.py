import torch
from .base import RewardFunction

class ArithmeticReward(RewardFunction):
    def __call__(self, prompts, responses, answers):
        """
        prompts:   [B]
        responses: [B, G]
        answers:   [B]

        Returns:
            rewards: [B, G]
        """
        rewards = []

        for prompt, prompt_responses, answer in zip(prompts, responses, answers):
            group_rewards = []

            for response in prompt_responses:
                if answer in response:
                    group_rewards.append(1.0)
                else:
                    group_rewards.append(0.0)

            rewards.append(group_rewards)

        return torch.tensor(rewards, dtype=torch.float32)