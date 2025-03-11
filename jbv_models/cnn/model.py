import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, R2Score

class CNNRegressor(pl.LightningModule):
    """
    CNNRegressor is a convolutional neural network for regression tasks.

    Args:
        input_size (int): Number of features (channels) per time step.
        seq_len (int): The length of the input sequences.
        output_size (int, optional): The size of the output. Defaults to 1.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-5.

    Returns:
        torch.Tensor: A tensor containing the regression prediction for each input sample,
                      with shape [batch_size].
    """
    def __init__(self, input_size, seq_len, output_size=1,
                 learning_rate=1e-3, weight_decay=1e-5):
        super(CNNRegressor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Output shape: (batch, 256, 1)
            nn.Flatten(),            # (batch, 256)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
        self.loss_fn = nn.MSELoss()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.test_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()
        
    def forward(self, x):
        # Expect x to have shape (batch, seq_len, input_size); transpose to (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        out = self.model(x)
        return out.squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.train_mae.update(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_mae", self.train_mae, prog_bar=True, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        self.train_mae.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.val_mae.update(y_hat, y)
        self.val_r2.update(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_mae", self.val_mae.compute())
        self.log("val_r2", self.val_r2.compute())
        self.val_mae.reset()
        self.val_r2.reset()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.test_mae.update(y_hat, y)
        self.test_r2.update(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        self.log("test_mae", self.test_mae.compute())
        self.log("test_r2", self.test_r2.compute())
        self.test_mae.reset()
        self.test_r2.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        return optimizer