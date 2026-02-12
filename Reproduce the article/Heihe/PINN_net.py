# -*- coding:utf-8 -*-
# author:Haipeng Feng
# software: PyCharm

import math
import numpy as np
import torch
import torch.nn as nn


def time_mapping(t, sc_day):
    # Normalized time conversion to actual days
    day = t * sc_day
    day_of_year = torch.remainder(day, 365)
    normalized_day = day_of_year / 365

    # Fundamental Seasonal Characteristics
    seasonal_base = 2 * np.pi * normalized_day

    # 1. year
    annual_sin = torch.sin(seasonal_base)
    annual_cos = torch.cos(seasonal_base)

    # 2. half-year
    semi_annual_sin = torch.sin(2 * seasonal_base)
    semi_annual_cos = torch.cos(2 * seasonal_base)

    # 3. season
    quarterly_sin = torch.sin(4 * seasonal_base)
    quarterly_cos = torch.cos(4 * seasonal_base)

    # 4. month
    monthly_sin = torch.sin(12 * seasonal_base)
    monthly_cos = torch.cos(12 * seasonal_base)

    # Integration of All Seasonal Characteristics
    seasonal_features = torch.cat([
        annual_sin, annual_cos,
        semi_annual_sin, semi_annual_cos,
        quarterly_sin, quarterly_cos,
        monthly_sin, monthly_cos
    ], dim=1)

    return seasonal_features


class TimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, t):
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        div_term = div_term.to(t.device)

        pe = torch.zeros(t.shape[0], self.d_model).to(t.device)
        pe[:, 0::2] = torch.sin(t * div_term)
        pe[:, 1::2] = torch.cos(t * div_term)

        return pe


class SpatioTemporalEncoderK(nn.Module):
    def __init__(self, d_model: int, time_period: float = 365.0):
        super().__init__()
        self.spatial_enc = nn.Linear(1, d_model // 2)
        self.time_enc = SeasonalTimeEncodingk(d_model // 2, period=time_period)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.spatial_enc(x)
        t_enc = self.time_enc(t)
        return torch.cat([t_enc, x_enc], dim=1)


class SeasonalTimeEncodingk(nn.Module):
    def __init__(self, d_model: int, period: float = 365.0):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        self.period = period
        self.d_model = d_model
        self.n_seasonal = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t * self.period
        seasonal_pe = torch.zeros(t.shape[0], self.n_seasonal, device=t.device)
        seasonal_pe[:, 0::2] = torch.sin(2 * math.pi * t_scaled / self.period)
        seasonal_pe[:, 1::2] = torch.cos(2 * math.pi * t_scaled / self.period)

        return seasonal_pe


class PhysMLP(nn.Module):
    def __init__(self, device, d_model=128, time_encoding_dim=64):
        super().__init__()
        self.time_encoder = TimeEncoding(time_encoding_dim)

        # The dimension is the time-encoding dimension plus the spatial dimension.
        self.layer = nn.Sequential(
            nn.Linear(time_encoding_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 1)
        ).to(device)

    def forward(self, t, x):
        t_encoded = self.time_encoder(t)

        combined = torch.cat([t_encoded, x], dim=1)

        u = self.layer(combined)
        return u


class PINN(nn.Module):
    def __init__(self, device):
        super(PINN, self).__init__()
        self.main_net = PhysMLP(device).to(device)

    def forward(self, t, x):
        return self.main_net(t, x)


class ParameterK(nn.Module):
    def __init__(self, device, d_model=128):
        super().__init__()

        self.encoder = SpatioTemporalEncoderK(d_model)
        self.layer = nn.Sequential(
            nn.Linear(9, 64), nn.Tanh(),
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1),
        ).to(device)

    def forward(self, t, x, sc_day):
        t_mapping = time_mapping(t, sc_day)
        out = self.layer(torch.cat([t_mapping, x], dim=1))
        out = torch.nn.functional.softplus(out)
        return out


class ParameterV(nn.Module):
    def __init__(self, device, d_model=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(device)

    def forward(self, t, x, sc_day):
        t_mapping = time_mapping(t, sc_day)
        out = self.net(torch.cat([t_mapping, x], dim=1))
        out = torch.nn.functional.softplus(out)
        return out


class Parameter_vsm_beta(nn.Module):
    def __init__(self, device, d_model=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        ).to(device)

    def forward(self, t, x):
        out = self.net(x)
        vsm = torch.sigmoid(out[:, 0])
        beta = torch.exp(out[:, 1])
        return vsm, beta


if __name__ == "__main__":
    print('0')
