import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torchmetrics
from torch.utils.data import DataLoader
from dataset import IMDBDataset


df = pd.read_csv("../raw_data/IMDB Dataset.csv")


train_df = df[:40000]
validation_df = df[40000:45000]
test_df = df[45000:]

class ClassifierLight(pl.LightningModule):
    def __init__(self, model, learning_rate, batch_size):
        super().__init__()
        self.model = model
        self.accuracy = torchmetrics.Accuracy()
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def training_step(self, batch, batch_index):
        x, y = batch
        y = y.to(torch.float)
        y_hat = self.model(x)
        loss = F.binary_cross_entropy(y_hat.flatten(), y.float())
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.float)
        y_hat = self.model(x)
        loss = F.binary_cross_entropy(y_hat.flatten(), y.float())
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        accuracy = self.accuracy(y_hat.flatten(), y)
        self.log(
            "test_step",
            accuracy,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        train_dataset = IMDBDataset(input_dataframe=train_df)
        return DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        validation_dataset = IMDBDataset(input_dataframe=validation_df)
        return DataLoader(dataset=validation_dataset, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        test_dataset = IMDBDataset(input_dataframe=test_df)
        return DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)