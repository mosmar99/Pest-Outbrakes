# tools/train_net.py

import yaml
import pytorch_lightning as pl
import pandas as pd

from jbv_models.data.datamodule import AgricultureDataModule
from jbv_models.lstm.model import LSTMRegressor
from tools.callbacks import PrintSampleCallback

def get_model_class(model_type: str):
    """Helper to map a model 'type' string to an actual model class."""
    if model_type == "LSTMRegressor":
        return LSTMRegressor
    # add more
    raise ValueError(f"Unknown model type: {model_type}")

def do_train(df, config_path="configs/config.yaml"):
    """
    - Loads the YAML config.
    - Dynamically sets feature_cols if needed.
    - Instantiates DataModule, Model, Trainer from config.
    - Runs .fit() and .test().
    """
    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    target_col = 'varde'
    # exclude columns you don't want as features
    exclude = {target_col, 'graderingsdatum', 'geometry'}
    feature_cols = [col for col in df.columns if col not in exclude]
    group_cols = ['geometry'] if 'geometry' in df.columns else None

    data_module_cfg = config["data_module"]
    data_module = AgricultureDataModule(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_cols=group_cols,
        **data_module_cfg
    )

    model_cfg = config["model"]
    model_type = model_cfg.pop("type")
    model_cls = get_model_class(model_type)
    model = model_cls(
        input_dim=len(feature_cols),
        **model_cfg
    )

    trainer_cfg = config["trainer"]
    callbacks = []
    for cb_name in trainer_cfg.pop("callbacks", []):
        if cb_name == "PrintSampleCallback":
            callbacks.append(PrintSampleCallback())

    trainer = pl.Trainer(**trainer_cfg, callbacks=callbacks)

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
