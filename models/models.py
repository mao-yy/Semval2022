from torch import nn
from transformers import AutoModelForSequenceClassification

# https://discuss.huggingface.co/t/how-do-i-change-the-classification-head-of-a-model/4720/19


class TwitterClassifier(nn.Module):
    def __init__(self, model_path, num_labels=2):
        super(TwitterClassifier, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        logits = self.bert(**x)["logits"]
        return logits
