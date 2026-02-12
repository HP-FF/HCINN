# -*- coding:utf-8 -*-
# author: Haipeng Feng
# software: PyCharm
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import DataLoader as TorchLoader


class DataLoader_DIY:
    def __init__(self, file_path, file_path_val, deep_max):
        self.file_path = file_path
        self.file_path_val = file_path_val
        self.data = self.load_meter(deep_max)

    def normalization(self, data_arr, max_value, min_value):
        """
        Normalize the data
        """
        new_data = [(data - min_value) / (max_value - min_value) for data in data_arr]
        return np.array(new_data, dtype=np.float32)

    def load_meter(self, deep_max):
        """
        Load the training and validation data and perform normalization processing.
        """
        train_excel = pd.read_excel(self.file_path)
        train_index_time = train_excel['index'].tolist()  # Time series normalization
        val_excel = pd.read_excel(self.file_path_val)
        val_index_time = val_excel['index'].tolist()

        max_day = max(val_index_time)
        train_time_nor = self.normalization(train_index_time, max_day, 0)
        val_time_nor = self.normalization(val_index_time, max_day, 0)

        clos = train_excel.columns
        clo_name = []

        temperature_list = []
        temperature_list_val = []

        for index, clo in enumerate(clos):
            if index == 0 or index == 1:
                continue
            else:
                temperature_list.append(train_excel[clo].tolist())
                temperature_list_val.append(val_excel[clo].tolist())
                clo_name.append(clo)

        deep_meter = []
        deep_meter_val = []
        for x_index in clo_name:
            deep_meter.append(self.str_to_number(x_index) / deep_max * torch.ones(len(train_index_time), 1).float())
            deep_meter_val.append(self.str_to_number(x_index) / deep_max * torch.ones(len(val_index_time), 1).float())

        return train_time_nor, val_time_nor, temperature_list, temperature_list_val, deep_meter, deep_meter_val

    def str_to_number(self, s):
        """
        Convert the deep index string to a number.
        """
        match = re.search(r'\d+(\.\d+)?', s)
        if match:
            num_str = match.group()
            return float(num_str) if '.' in num_str else int(num_str)
        return None
