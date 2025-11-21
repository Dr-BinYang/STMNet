import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.uea import Normalizer, interpolate_missing
import warnings

warnings.filterwarnings('ignore')


class Dataset_Weather(Dataset):
    """
    Weather dataset loader for time series forecasting
    Handles data loading, scaling, and temporal feature engineering
    """

    def __init__(self, root_path, data_path, flag='train', size=None,
                 scale=True, freq='h', timeenc=1):
        """
        Initialize weather dataset

        Args:
            root_path: Root directory path
            data_path: Data file path
            flag: Data split type ('train', 'val', 'test')
            size: Tuple of (sequence_length, prediction_length)
            scale: Whether to scale the data
            freq: Frequency for time features ('h' for hourly)
            timeenc: Time encoding method (0: manual, 1: time_features)
        """
        # Sequence and prediction lengths
        self.seq_len = size[0]
        self.pred_len = size[1]

        # Validate split type
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # Configuration parameters
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path

        # Load and process data
        self.__read_data__()

    def __read_data__(self):
        """
        Load and preprocess weather data
        Handles data splitting, scaling, and temporal feature extraction
        """
        # Initialize standard scaler
        self.scaler = StandardScaler()

        # Load raw data from CSV file
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Reorder columns to put date first
        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        # Calculate dataset splits (70% train, 20% test, 10% validation)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # Define boundaries for each dataset split
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        # Select boundaries based on dataset type
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Extract feature columns (exclude date column)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        # Apply scaling if enabled
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Process temporal features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        # Time feature encoding methods
        if self.timeenc == 0:
            # Manual feature extraction: month, day, weekday, hour, minute
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)  # Bin minutes
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # Automated time feature extraction
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # Store processed data
        self.data_x = data[border1:border2]  # Input features
        self.data_y = data[border1:border2]  # Target values (same as input for forecasting)
        self.data_stamp = data_stamp  # Temporal features

    def __getitem__(self, index):
        """
        Get a single sample from the dataset

        Args:
            index: Index of the sample

        Returns:
            tuple: (input_sequence, target_sequence, input_temporal_features, target_temporal_features)
        """
        # Calculate sequence boundaries
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        # Extract sequences
        seq_x = self.data_x[s_begin:s_end]  # Input sequence
        seq_y = self.data_y[r_begin:r_end]  # Target sequence
        seq_x_mark = self.data_stamp[s_begin:s_end]  # Input temporal features
        seq_y_mark = self.data_stamp[r_begin:r_end]  # Target temporal features

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Get the total number of samples in the dataset

        Returns:
            int: Number of valid samples (considering sequence and prediction lengths)
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1