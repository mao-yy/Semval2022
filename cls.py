from data import datasets
from models import models
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch import nn

BATCH_SIZE = 12
EPOCH = 10
LR = 5e-6
'''
bert-base-uncased

  epoch=10 Accuracy: 86.5%, Avg loss: 0.531606 
  epoch=2  Accuracy: 85.6%, Avg loss: 0.385264 
emoji is not installed, thus not converting emoticons or emojis into text. Please install emoji: pip3 install emoji

ardiffnlp/bertweet-base-irony  Accuracy: 84.1%, Avg loss: 0.533593  without emjo

'''
# bert-base-uncased
# vinai/bertweet-large  70.3%   0.59
# roberta-large         64.7%   0.61
# cardiffnlp/twitter-roberta-base-sentiment 71.4%   0.61
# finiteautomata/bertweet-base-sentiment-analysis   64.4%   0.64
# cardiffnlp/bertweet-base-emotion  68.9%   0.62
# cardiffnlp/twitter-roberta-base-irony 66.7%   0.63
# cardiffnlp/bertweet-base-irony    66.7%   0.59
pretrained_model = "cardiffnlp/twitter-roberta-base-sentiment"

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print("Using device: %s" % device)

train_dataset, test_dataset = datasets.load_datasets("data/csv/train.En.csv")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


def train(tokenizer, train_dataloader, test_dataloader, model, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    for epoch in range(EPOCH):
        model.train()
        print(f"epoch #{epoch}:")
        for batch, data in enumerate(train_dataloader):
            X = data["text"]
            y = data["label"]
            X = tokenizer(X, padding=True, return_tensors="pt")
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(X["input_ids"])
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        evaluate(tokenizer, test_dataloader, model, loss_fn)


def evaluate(tokenizer, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            X = data["text"]
            y = data["label"]
            X = tokenizer(X, padding=True, return_tensors="pt")
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct) :>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = models.TwitterClassifier(pretrained_model).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(tokenizer, train_dataloader, test_dataloader, model, loss_fn, optimizer)
    evaluate(tokenizer, test_dataloader, model, loss_fn)
