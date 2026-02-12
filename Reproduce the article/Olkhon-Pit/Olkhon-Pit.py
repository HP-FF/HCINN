# -*- coding:utf-8 -*-
# author: Haipeng Feng
# software: PyCharm


import os.path
import re
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import MaxNLocator

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
plt.rcParams['svg.fonttype'] = 'none'


def cal_bias(T,P):
    temp_list = []
    for i in range(len(T)):
        temp = T[i]-P[i]
        temp_list.append(temp)

    bias = np.array(temp_list).mean()

    return bias

def normalization_total(data_total):

    max_value = max(data_total)
    min_value = 0

    new_data = []

    for data in data_total:
        temp = (data-min_value)/(max_value-min_value)
        new_data.append(temp)

    return np.array(new_data)

def plot_under_mae(temp,model,time_arr,out_path):
    x = temp/meter * torch.ones(time_arr.shape[0],1).float().to(device)
    t = torch.from_numpy(time_arr).float().reshape(time_arr.shape[0],1).to(device)
    with torch.no_grad():
        u_pred = model(t,x).cpu().numpy().reshape(730)

    plt.figure(figsize=(15, 3))

    bias = cal_bias(temperature_temp_cm[365:],u_pred[365:])

    temperature_smooth = gaussian_filter1d(temperature_temp_cm,sigma=8)
    temperature_smooth_val = gaussian_filter1d(u_pred, sigma=8)
    plt.plot(time_arr*729,  temperature_smooth, label='Observation', color='blue')
    plt.plot(time_arr*729, temperature_smooth_val,label='Predicted', color='red')  # linestyle='dashed'

    plt.xlabel('Simulation Days', fontsize=14)
    plt.ylabel('Temperature (℃)'.format(str(temp)), fontsize=14)
    plt.title('Heat Conduction Equation Solution at x={}M'.format(str(temp)), fontsize=16)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    print("at x={}M, bias:{}".format(str(temp),str(round(bias,1))))
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.axvline(365)
    plt.xlim(left=0)
    plt.show()
    # plt.savefig(out_path,dpi=500)

def str_to_number(s):

    match = re.search(r'\d+(\.\d+)?', s)
    if match:
        num_str = match.group()
        return float(num_str) if '.' in num_str else int(num_str)
    return None


if __name__ == "__main__":
    model = torch.load(r"./Olkhon-Pit.pth",weights_only=False)
    true_excel = pd.read_excel(r"./val.xlsx")
    train_excel = pd.read_excel(r"./train.xlsx")
    Total_excel = pd.read_excel(r"./Total.xlsx")

    colname = true_excel.columns

    meter = 3.65
    model.eval()
    with torch.no_grad():

        for index,temp_number in enumerate(colname):
            if index == 0 or index == 1:
                continue
            else:
                temperature_temp_cm = Total_excel['{}'.format(temp_number)].tolist()

                time_Total = Total_excel['index'].tolist()
                time_nor = normalization_total(time_Total)

                out = temp_number + '.svg'
                plot_under_mae(str_to_number(temp_number), model, time_nor,out)


