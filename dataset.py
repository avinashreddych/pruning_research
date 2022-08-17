import pandas as pd
from torch.utils.data import Dataset


def clean(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df.replace(to_replace="<br /><br />", value="", regex=True, inplace=True)

    input_df["label"] = input_df["sentiment"].apply(filter)
    input_df.drop(columns=["sentiment"], inplace=True)
    return input_df


def filter(label: str):
    if label == "positive":
        return 1
    else:
        return 0


class IMDBDataset(Dataset):
    tokenizer_path = "../models/tokenizer"

    def __init__(self, input_df: pd.DataFrame) -> None:
        super().__init__()
        self.input_df = clean(input_df=input_df)

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, index):
        item = self.input_df.loc[index]
        return (item.review, item.label)


if __name__ == "__main__":
    pass
