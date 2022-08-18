# %%
import pandas as pd
from model import ClassificationModel
from dataset import IMDBDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint

df = pd.read_csv("../raw_data/IMDB Dataset.csv")


checkpoint_callback = ModelCheckpoint(dirpath="./saved_models")

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
        self.accuracy = torchmetrics.Accuracy()

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
        return torch.optim.Adam(self.model.parameters(), lr=0.02)


# %%

lightning_nn = ClassifierLight(nn)

trainer = pl.Trainer(max_epochs=2, callbacks=[checkpoint_callback])

# %%
trainer.fit(lightning_nn, train_dataloader)


# torch.save(lightning_nn.state_dict(), "imdb_classifier.pt")

# %%
