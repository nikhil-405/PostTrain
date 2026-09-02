import torch

class DPOProcessor:
    def __init__(self, tokenizer, assistant_token_id = 42, ignore_index = -100):
        self.tokenizer = tokenizer
        self.assistant_token_id = assistant_token_id
        self.ignore_index = ignore_index

    def process(self, example): 
        """
        This is DPOObjective with response-only labels

        We get examples in the form of an array 
        {
            "prompt": "What is 2 + 2?",
            "chosen": "2 + 2 equals 4.",
            "rejected": "2 + 2 equals 3."
        }
        """
        # For chosen response
        prompt_ids = self.tokenizer(example["prompt"], return_tensors = 'pt')['input_ids'][0]
        chosen_response_ids = self.tokenizer(example["chosen"], return_tensors = 'pt')['input_ids'][0]

        n = len(prompt_ids) + 1 # 1 is for the assistant token

        chosen_input_ids = torch.cat([prompt_ids, torch.tensor([self.assistant_token_id], dtype = torch.long), chosen_response_ids], dim = 0)
        chosen_labels = torch.cat([torch.full((n, ), self.ignore_index, dtype = torch.long), chosen_response_ids], dim = 0)
        chosen_attention_mask = torch.ones_like(chosen_labels, dtype = torch.long)


        # For rejected response
        prompt_ids = self.tokenizer(example["prompt"], return_tensors = 'pt')['input_ids'][0]
        rejected_response_ids = self.tokenizer(example["rejected"], return_tensors = 'pt')['input_ids'][0]

        n = len(prompt_ids) + 1 # 1 is for the assistant token

        rejected_input_ids = torch.cat([prompt_ids, torch.tensor([self.assistant_token_id], dtype = torch.long), rejected_response_ids], dim = 0)
        rejected_labels = torch.cat([torch.full((n, ), self.ignore_index, dtype = torch.long), rejected_response_ids], dim = 0)
        rejected_attention_mask = torch.ones_like(rejected_labels, dtype = torch.long)


        processed_example = {
                        "chosen_input_ids": chosen_input_ids,
                        "chosen_labels": chosen_labels,
                        "chosen_attention_mask": chosen_attention_mask,

                        "rejected_input_ids": rejected_input_ids,
                        "rejected_labels": rejected_labels,
                        "rejected_attention_mask": rejected_attention_mask
                            }

        return processed_example

class DPOCollator: 
    def __init__(self, pad_token_id, ignore_index):
            self.pad_token_id = pad_token_id
            self.ignore_index = ignore_index

    def __call__(self, examples): 
        # For chosen
        chosen_max_len = max([len(d["chosen_input_ids"]) for d in examples])
        rejected_max_len = max([len(d["rejected_input_ids"]) for d in examples])

        chosen_padded_input_ids = []
        chosen_padded_labels = []
        chosen_padded_attention_masks = []

        rejected_padded_input_ids = []
        rejected_padded_labels = []
        rejected_padded_attention_masks = []

        for example in examples: 
            chosen_length_diff = chosen_max_len - len(example["chosen_input_ids"])
            rejected_length_diff = rejected_max_len - len(example["rejected_input_ids"])

            chosen_input_id = example["chosen_input_ids"]
            chosen_label = example["chosen_labels"]
            chosen_attention_mask = example["chosen_attention_mask"]

            rejected_input_id = example["rejected_input_ids"]
            rejected_label = example["rejected_labels"]
            rejected_attention_mask = example["rejected_attention_mask"]

            if (chosen_length_diff != 0):
                chosen_input_id = torch.cat([chosen_input_id, torch.full((chosen_length_diff, ), self.pad_token_id, dtype = torch.long)], dim = 0)
                chosen_label = torch.cat([chosen_label, torch.full((chosen_length_diff, ), self.ignore_index, dtype = torch.long)], dim = 0)
                chosen_attention_mask = torch.cat([chosen_attention_mask, torch.full((chosen_length_diff, ), 0, dtype = torch.long)], dim = 0)

            chosen_padded_input_ids.append(chosen_input_id)
            chosen_padded_labels.append(chosen_label)
            chosen_padded_attention_masks.append(chosen_attention_mask)

            if (rejected_length_diff != 0):
                    rejected_input_id = torch.cat([rejected_input_id, torch.full((rejected_length_diff, ), self.pad_token_id, dtype = torch.long)], dim = 0)
                    rejected_label = torch.cat([rejected_label, torch.full((rejected_length_diff, ), self.ignore_index, dtype = torch.long)], dim = 0)
                    rejected_attention_mask = torch.cat([rejected_attention_mask, torch.full((rejected_length_diff, ), 0, dtype = torch.long)], dim = 0)

            rejected_padded_input_ids.append(rejected_input_id)
            rejected_padded_labels.append(rejected_label)
            rejected_padded_attention_masks.append(rejected_attention_mask)


        chosen_stacked_inputs = torch.stack(chosen_padded_input_ids)
        chosen_stacked_labels = torch.stack(chosen_padded_labels)
        chosen_stacked_masks = torch.stack(chosen_padded_attention_masks)

        rejected_stacked_inputs = torch.stack(rejected_padded_input_ids)
        rejected_stacked_labels = torch.stack(rejected_padded_labels)
        rejected_stacked_masks = torch.stack(rejected_padded_attention_masks)

        return {
            "chosen_input_ids": chosen_stacked_inputs, 
            "chosen_labels": chosen_stacked_labels,
            "chosen_attention_mask": chosen_stacked_masks,

            "rejected_input_ids": rejected_stacked_inputs, 
            "rejected_labels": rejected_stacked_labels,
            "rejected_attention_mask": rejected_stacked_masks
        }