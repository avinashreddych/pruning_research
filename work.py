# %%
import pandas as pd
from model import ClassificationModel
from dataset import IMDBDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
import torch

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
# %%
a = iter(train_dataloader)
x, y = next(a)
# %%
o = nn(x)
# %%
