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

class SFTCollator:
    def __init__(self, pad_token_id, ignore_index):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, examples): 
        """
        We receive a list of examples in dictionary format (concatenated output of multiple SFTProcessor.process() calls)
        """
        max_len = max([len(d["input_ids"]) for d in examples])

        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []

        for example in examples: 
            length_diff = max_len - len(example["input_ids"])

            input_id = example["input_ids"]
            label = example["labels"]
            attention_mask = example["attention_mask"]

            if (length_diff != 0):
                input_id = torch.cat([input_id, torch.full((length_diff, ), self.pad_token_id, dtype = torch.long)], dim = 0)
                label = torch.cat([label, torch.full((length_diff, ), self.ignore_index, dtype = torch.long)], dim = 0)
                attention_mask = torch.cat([attention_mask, torch.full((length_diff, ), 0, dtype = torch.long)], dim = 0)

            padded_input_ids.append(input_id)
            padded_labels.append(label)
            padded_attention_masks.append(attention_mask)

        stacked_inputs = torch.stack(padded_input_ids)
        stacked_labels = torch.stack(padded_labels)
        stacked_masks = torch.stack(padded_attention_masks)

        return {
            "input_ids": stacked_inputs, 
            "labels": stacked_labels,
            "attention_mask": stacked_masks
        }