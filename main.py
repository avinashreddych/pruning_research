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


class ClassifierLight(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.model(x)
        y_hat.squeeze_()
        loss = F.cross_entropy(y_hat, y.float())
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat.squeeze_()
        loss = F.cross_entropy(y_hat, y.float())
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat.squeeze_()
        loss = F.cross_entropy(y_hat, y.float())
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# %%

model = ClassifierLight(nn)

trainer = pl.Trainer(max_epochs=2)

# %%
trainer.fit(model, train_dataloader)
# %%
