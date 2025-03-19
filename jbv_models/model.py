import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, R2Score, MeanSquaredError

class FNNRegressor(pl.LightningModule):
    """
    Args:
        input_size (int): The size of the input feature vector.
        output_size (int, optional): The size of the output, defaults to 1. Should not be modified, regression model.
        hidden_sizes (list, optional): List of hidden layer sizes.
        learning_rate (float, optional): Learning rate for the optimizer.
        weight_decay (float, optional): Weight decay for the optimizer.
        activation (str, optional): Options: "relu", "silu", "leaky_relu", defaults to "relu".
        dropout (float, optional): Dropout rate to apply after each activation, defaults is 0.1.
        loss_fn_name (str, optional): Options: "huber", "mse", "l1", "smoothl1", defaults is "huber".
    """
    def __init__(self, 
                 input_size, 
                 output_size=1, 
                 hidden_sizes=[256,128,64,32],
                 learning_rate=1e-3, 
                 weight_decay=1e-5, 
                 activation="relu", 
                 dropout=0.1,
                 loss_fn_name="huber"):
        super(FNNRegressor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        layers = []
        prev_size = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "silu":
                layers.append(nn.SiLU())
            elif activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
        
        self.model.apply(self.init_weights)
        
        self._set_loss_fn(loss_fn_name)

        self.train_mae = MeanAbsoluteError()
        self.train_mse = MeanSquaredError()
        self.train_r2 = R2Score()

        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_r2 = R2Score()

        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score()

    
    def _set_loss_fn(self, loss_fn_name):
        if loss_fn_name == "huber":
            self.loss_fn = nn.HuberLoss(delta=1.0)
        elif loss_fn_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn_name == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_fn_name == "smoothl1":
            self.loss_fn = nn.SmoothL1Loss(beta=1.0)
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn_name}")
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        return self.model(x).squeeze(-1)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.train_mae.update(y_hat, y)
        self.train_mse.update(y_hat, y)
        self.train_r2.update(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_mae", self.train_mae, prog_bar=True, on_epoch=True)
        self.log("train_r2", self.train_r2, prog_bar=True, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        self.train_mae.reset()
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.val_mae.update(y_hat, y)
        self.val_mse.update(y_hat, y)
        self.val_r2.update(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        val_mae = self.val_mae.compute()
        val_mse = self.val_mse.compute()
        val_r2  = self.val_r2.compute()

        self.log("val_mae", val_mae)
        self.log("val_mse", val_mse)
        self.log("val_r2", val_r2)

        self.val_mae.reset()
        self.val_mse.reset()
        self.val_r2.reset()
    
    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.test_mae.update(y_hat, y)
        self.test_mse.update(y_hat, y)
        self.test_r2.update(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        self.print("Note: These metrics are on scaled data."
                   "Compute on inverse scaled data as a final stage.")
        self.log("test_mae", self.test_mae.compute())
        self.log("test_mse", self.test_mse.compute())
        self.log("test_r2", self.test_r2.compute())
        self.test_mae.reset()
        self.test_r2.reset()
        self.test_mse.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        return optimizer
