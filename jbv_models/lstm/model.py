import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, R2Score

class Attention(nn.Module):
    def __init__(self, lstm_output_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(lstm_output_dim, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        weighted = lstm_out * weights  # (batch, seq_len, lstm_output_dim)
        representation = weighted.sum(dim=1)  # (batch, lstm_output_dim)
        return representation

class LSTMRegressor(pl.LightningModule):
    """
    LSTMRegressor model with an attention mechanism for sequence regression tasks.

    Args:
        input_dim (int): Number of features per time step in the input sequence.
        hidden_dim (int, optional): Number of hidden units in the LSTM layers. Defaults to 32.
        num_layers (int, optional): Number of stacked LSTM layers. Defaults to 2.
        bidirectional (bool, optional): If True, uses a bidirectional LSTM. Defaults to True.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        dropout (float, optional): Dropout probability for the LSTM and fully-connected layers. Defaults to 0.3.
        weight_decay (float, optional): Weight decay (L2 penalty) for the optimizer. Defaults to 5e-4.
        fc_hidden_dim (int, optional): Number of hidden units in the fully-connected layer following the attention mechanism. Defaults to 64.

    Returns:
        torch.Tensor: A tensor containing the regression predictions for each input sample,
                      with the output dimension squeezed (i.e., shape [batch_size]).
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, bidirectional=True,
                 lr=1e-3, dropout=0.3, weight_decay=5e-4, fc_hidden_dim=64):
        super(LSTMRegressor, self).__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.attention = Attention(lstm_output_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, 1)
        )
        
        self.loss_fn = nn.MSELoss()
        
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.test_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_output_dim)
        attn_out = self.attention(lstm_out)  # (batch, lstm_output_dim)
        out = self.fc_layers(attn_out)       # (batch, 1)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
