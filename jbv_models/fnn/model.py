import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, R2Score

class FNNRegressor(pl.LightningModule):
    """
    FNNRegressor is a feed-forward neural network for regression tasks.

    Args:
        input_size (int): The size of the input feature vector (number of features).
        output_size (int, optional): The size of the output. Defaults to 1.
        hidden_sizes (list, optional): List containing the sizes of hidden layers. Defaults to [128, 64, 32].
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay (L2 regularization) for the optimizer. Defaults to 1e-5.

    Returns:
        torch.Tensor: A tensor containing the regression prediction for each input sample,
                      with shape [batch_size].
    """
    def __init__(self, input_size, input_multiple, output_size=1, hidden_dim=[128, 64, 32],
                 learning_rate=1e-3, weight_decay=1e-5):
        super(FNNRegressor, self).__init__()
        self.save_hyperparameters()
        self.loss_fn = nn.SmoothL1Loss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        layers = []
        layers.append(nn.Flatten())  # Ensure the input is a flat vector.
        prev_size = input_size * input_multiple
        for hidden in hidden_dim:
            layers.append(nn.Linear(prev_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
        
        self.model.apply(self.init_weights)
        
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.test_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        return self.model(x).squeeze(-1)
    
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
