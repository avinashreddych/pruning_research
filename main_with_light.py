# %%
import pandas as pd
from model import ClassificationModel
from dataset import IMDBDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F

from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_module import ClassifierLight


checkpoint_callback = ModelCheckpoint(dirpath="./saved_models")


nn = ClassificationModel()


# %%


# %%

lightning_nn = ClassifierLight(nn)

trainer = pl.Trainer(
    max_epochs=15,
    accelerator="auto",
    strategy="ddp",
    callbacks=[checkpoint_callback],
    auto_lr_find=True,
    auto_scale_batch_size="binsearch",
)

# %%
trainer.fit(lightning_nn)


# torch.save(lightning_nn.state_dict(), "imdb_classifier.pt")

# %%
