import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

class JBVDataModule(pl.LightningDataModule):
    """
    JBVDataModule requires:
      X_train, y_train, X_test, y_test, X_predict is optional. If X_predict doesnt exist, we predict on X_test, y_test
      Splits X_train and y_train into a train and validation set
    """
    def __init__(self, X_train, y_train, X_test, y_test, X_predict=None, batch_size=32, num_workers=0, val_size=0.2, random_state=42):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_predict = X_predict
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.random_state = random_state

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage=None):
        """
        Uses random_split to create train/val with manual_seed for reproducibility.
        """

        full_dataset = TensorDataset(torch.tensor(self.X_train.values, dtype=torch.float), torch.tensor(self.y_train.values.ravel(), dtype=torch.float)
        )

        val_len = int(len(full_dataset) * self.val_size)
        train_len = len(full_dataset) - val_len

        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, val_len], generator=generator)

        self.test_dataset = TensorDataset(torch.tensor(self.X_test.values, dtype=torch.float), torch.tensor(self.y_test.values.ravel(), dtype=torch.float),)

        if self.X_predict is not None:
            self.predict_dataset = TensorDataset(torch.tensor(self.X_predict.values, dtype=torch.float),)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        if self.predict_dataset is None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
