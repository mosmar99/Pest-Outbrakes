import pytorch_lightning as pl
import torch

class PrintSampleCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        train_loader = trainer.datamodule.train_dataloader()
        batch = next(iter(train_loader))
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x = x.to(pl_module.device)
        y_hat = pl_module(x)
        y_true = y.cpu().tolist()
        y_pred = y_hat.cpu().tolist()

        print("\n=== Sample at epoch end (First 5 Samples) ===")
        header = "{:<8} {:<15} {:<15}".format("Index", "Ground Truth", "Prediction")
        print(header)
        print("-" * 40)
        for i in range(min(5, len(y_true))):
            row = "{:<8} {:<15} {:<15}".format(i, y_true[i], y_pred[i])
            print(row)
