# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd

class AgricultureDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len=5, group_cols=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            feature_cols (list): List of feature column names.
            target_col (str): The target column name.
            seq_len (int): Number of weeks used as input to predict the next week.
            group_cols (list or None): Columns to group by (e.g., ['geometry'] or ['groda', 'skadegorare']).
                Sequences are built within each group.
        """
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.sequences = []
        
        if group_cols is not None:
            groups = df.groupby(group_cols)
            for _, group in groups:
                group = group.sort_values('graderingsdatum')
                data = group[feature_cols].to_numpy(dtype=float)
                targets = group[target_col].to_numpy(dtype=float)
                if len(data) > seq_len:
                    for i in range(len(data) - seq_len):
                        x_seq = data[i:i+seq_len]
                        y_value = targets[i+seq_len]
                        self.sequences.append((x_seq, y_value))
        else:
            df = df.sort_values('graderingsdatum')
            data = df[feature_cols].to_numpy(dtype=float)
            targets = df[target_col].to_numpy(dtype=float)
            if len(data) > seq_len:
                for i in range(len(data) - seq_len):
                    x_seq = data[i:i+seq_len]
                    y_value = targets[i+seq_len]
                    self.sequences.append((x_seq, y_value))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x_seq, y_value = self.sequences[idx]
        return (torch.tensor(x_seq, dtype=torch.float),
                torch.tensor(y_value, dtype=torch.float))
