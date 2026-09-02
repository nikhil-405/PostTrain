import torch

class SFTProcessor:
    def __init__(self, tokenizer, assistant_token_id = 42, ignore_index = -100):
        self.tokenizer = tokenizer
        self.assistant_token_id = assistant_token_id
        self.ignore_index = ignore_index

    def process(self, example): 
        """
        This is LM Objective with response-only labels

        We get examples in the form of an array 
        {
            "prompt": "What is 2 + 2?",
            "response": "2 + 2 equals 4."
        }
        """

        prompt_ids = self.tokenizer(example["prompt"], return_tensors = 'pt')['input_ids'][0]
        response_ids = self.tokenizer(example["response"], return_tensors = 'pt')['input_ids'][0]

        n = len(prompt_ids) + 1 # 1 is for the assistant token

        input_ids = torch.cat([prompt_ids, torch.tensor([self.assistant_token_id], dtype = torch.long), response_ids], dim = 0)
        labels = torch.cat([torch.full((n, ), self.ignore_index, dtype = torch.long), response_ids], dim = 0)
        attention_mask = torch.ones_like(labels, dtype = torch.long)

        processed_example = {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": attention_mask
                            }

        return processed_example