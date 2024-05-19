import torch.nn as nn
from transformers import GPT2Model

class FunctionCallModel(nn.Module):
    def __init__(self, num_classes):
        super(FunctionCallModel, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        logits = self.classifier(last_hidden_state[:, -1, :])
        return logits
