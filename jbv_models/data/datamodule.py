import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from .dataset import AgricultureDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class AgricultureDataModule(pl.LightningDataModule):
    def __init__(self, df, feature_cols, target_col, group_cols=None, seq_len=5,
                 batch_size=32, num_workers=0, val_split=0.2, val_geo_frac=0.1, test_geo_frac=0.2,
                 split_by_geo=True, scale_features=True):
        """
        Args:
            df (pd.DataFrame): The entire dataset.
            feature_cols (list): List of feature column names.
            target_col (str): The target column name.
            group_cols (list or None): Columns to group by (for sequence creation). For example: ['geometry'].
            seq_len (int): Length of each sequence.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for DataLoaders.
            val_split (float): Fraction of data for validation if not doing geographical split.
            val_geo_frac (float): Fraction of unique geographical groups to reserve for validation.
            test_geo_frac (float): Fraction of unique geographical groups to reserve for testing.
            split_by_geo (bool): Whether to perform a geographical split.
            scale_features (bool): Whether to scale the features.
        """
        super().__init__()
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.group_cols = group_cols
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.val_geo_frac = val_geo_frac
        self.test_geo_frac = test_geo_frac
        self.split_by_geo = split_by_geo
        self.scale_features = scale_features

    def setup(self, stage=None):
        if 'graderingsdatum' in self.df.columns:
            self.df['graderingsdatum'] = pd.to_datetime(self.df['graderingsdatum'])
            self.df = self.df.sort_values('graderingsdatum')
        
        if self.split_by_geo and self.group_cols is not None:
            # Pure geographical split
            unique_groups = self.df[self.group_cols[0]].unique()
            np.random.shuffle(unique_groups)  # randomize order of groups
            num_groups = len(unique_groups)
            num_test = int(num_groups * self.test_geo_frac)
            num_val = int(num_groups * self.val_geo_frac)
            test_groups = unique_groups[:num_test]
            val_groups = unique_groups[num_test:num_test+num_val]
            train_groups = unique_groups[num_test+num_val:]
            
            train_df = self.df[self.df[self.group_cols[0]].isin(train_groups)]
            val_df = self.df[self.df[self.group_cols[0]].isin(val_groups)]
            test_df = self.df[self.df[self.group_cols[0]].isin(test_groups)]
            
            # Scale features consistently over the entire dataset.
            if self.scale_features:
                scaler = StandardScaler().fit(train_df[self.feature_cols])
                train_df[self.feature_cols] = scaler.transform(train_df[self.feature_cols])
                val_df[self.feature_cols]   = scaler.transform(val_df[self.feature_cols])
                test_df[self.feature_cols]  = scaler.transform(test_df[self.feature_cols])
            
            self.train_dataset = AgricultureDataset(train_df, self.feature_cols, self.target_col,
                                                      seq_len=self.seq_len, group_cols=self.group_cols)
            self.val_dataset = AgricultureDataset(val_df, self.feature_cols, self.target_col,
                                                    seq_len=self.seq_len, group_cols=self.group_cols)
            self.test_dataset = AgricultureDataset(test_df, self.feature_cols, self.target_col,
                                                     seq_len=self.seq_len, group_cols=self.group_cols)
        else:
            # If not splitting purely by geography, fall back to a time-based split.
            dataset = AgricultureDataset(self.df, self.feature_cols, self.target_col,
                                          seq_len=self.seq_len, group_cols=self.group_cols)
            total = len(dataset)
            val_size = int(total * self.val_split)
            train_size = total - val_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
            self.test_dataset = self.val_dataset  # For demonstration.

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)
