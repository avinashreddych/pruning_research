# %%
import torch
import pandas as pd
from model import ClassificationModel
from dataset import IMDBDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics import Accuracy


df = pd.read_csv("../raw_data/IMDB Dataset.csv")

train_dataset = IMDBDataset(df[:35000])
validation_dataset = IMDBDataset(df[35000:40000])
test_dataset = IMDBDataset(df[40000:])


train_dataloader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)

validation_dataloader = DataLoader(
    dataset=validation_dataset, batch_size=5, shuffle=True
)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=5, shuffle=True)


nn = ClassificationModel()

optimizer = torch.optim.Adam(nn.parameters(), lr=0.001)

accuracy = Accuracy()


def training_loop():
    for i, batch in enumerate(train_dataloader):
        if i == 4:
            break
        optimizer.zero_grad()
        x, y = batch
        y = y.to(torch.float)
        y_hat = nn(x)
        loss = F.binary_cross_entropy(y_hat.flatter(), y)
        print(
            i,
            loss,
        )

        loss.backward()
        optimizer.step()


# torch.save(nn.state_dict(), "imdb_classifier.pt")

# %%
def test_loop():
    total_accuracy = 0
    for i, batch in enumerate(train_dataloader):
        x, y = batch
        y_hat = nn(x)
        acc = accuracy(y_hat, y)
        print(acc)
        total_accuracy = acc * 5

    print(total_accuracy / len(train_dataset))


# to reload the model

# model = ClassificationModel()
# model.load_state_dict(torch.load("imdb_classifier.pt"))

# %%
