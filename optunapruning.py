import pytorch_lightning as pl
import optuna

class CustomOptunaPruningCallback(pl.Callback):
    def __init__(self, trial, monitor="val_mse"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    # (Optional) Satisfy Lightning's check for state_dict
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        # Grab the metric we want to monitor from trainer.callback_metrics
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return  # Nothing to prune on yet

        # Report the metric to Optuna
        self.trial.report(float(current_score), step=trainer.current_epoch)

        # If the metric is getting worse, Optuna can prune
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned(
                f"Trial was pruned at epoch {trainer.current_epoch}"
            )
