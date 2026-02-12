# -*- coding:utf-8 -*-
# author: Haipeng Feng
# software: PyCharm


import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from Data_Loader import DataLoader_DIY
from PINN_net import PINN, ParameterK, ParameterV, Parameter_vsm_beta
from Loss import loss_cal
from Sampling import PINNSampler1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
np.random.seed(3407)
torch.manual_seed(3407)



'''
# Supervising training - can eliminate to enhance training speed
'''

# def draw():
#     if epoch % 100 == 0:
#         print(
#             f'Epoch {epoch}, Loss: {loss.item()},f_loss:{f_loss.item()},bc_loss_up:{bc_loss_up.item()} ,bc_loss_low:{bc_loss_low.item()} , ic_loss:{ic_loss.item()} , da'
#             f'ta_loss:{data_loss.item()},val_Loss:{data_loss_val}')
#
#         time_draw = external.cpu() * 1459
#
#         ax.cla()
#         ax.plot(ep, losses, color='black')
#         ax.plot(ep, losses_val, color='red')
#         ax.autoscale_view()
#         ax.set_title('loss', fontsize=14)
#         ax.axvline(730)
#
#         # Up boundary
#         jimi_up = 0  # 索引0米
#         x0_val = 0 / meter * torch.ones(1460, 1).to(device)
#         outputs0 = model_T(external, x0_val)
#         # 训练集真实情况
#         T0 = torch.cat((T_true[jimi_up], T_true_val[jimi_up]))
#         ax2.cla()
#         ax2.plot(time_draw, T0.cpu(), label='0cm', color='grey')
#         # 预测值
#         ax2.plot(time_draw, outputs0.cpu().detach().numpy(), label='p_0m', color='red')
#         ax2.set_title('0m', fontsize=14)
#         ax2.axvline(730)
#
#         # lower boundary
#         jimi_down = 11  # 下边界
#         x300_val = 3.0 / meter * torch.ones(1460, 1).to(device)
#         # outputs300_train = model(t_ture, x300_v)
#         # outputs300_val = model(t_ture_val,x300_v)
#         # outputs300 = model(torch.cat((t_ture,t_ture_val)),x300_v)
#         outputs300 = model_T(external, x300_val)
#         ax3.cla()
#         # 训练集真实情况
#         T300 = torch.cat((T_true[jimi_down], T_true_val[jimi_down]))
#         ax3.plot(time_draw, T300.cpu(), label='300cm', color='grey')
#         # 预测值
#         T_val300 = outputs300
#         # plt.plot(torch.cat((t_ture, t_ture_val)).cpu().numpy(), T_val.cpu().detach(), label='p_300m', color='red')
#         ax3.plot(time_draw, T_val300.cpu().detach().numpy(), label='p_300m', color='red')
#         ax3.set_title('3m', fontsize=14)
#         ax3.axvline(730)
#
#         jimi_105 = 6  # 1m
#         x105_v = 1.0 / meter * torch.ones(1460, 1).to(device)
#         outputs105 = model_T(external, x105_v)
#         # 训练集真实情况
#         T105 = torch.cat((T_true[jimi_105], T_true_val[jimi_105]))
#         ax5.cla()
#         ax5.plot(time_draw, T105.cpu(), label='0cm', color='grey')
#         # 预测值
#         T105_val = outputs105
#         ax5.plot(time_draw, T105_val.cpu().detach().numpy(), label='p_0m', color='red')
#         ax5.set_title('1m', fontsize=14)
#         ax5.axvline(730)
#
#         jimi245 = 10  # 2.5
#         x0_v = 2.5 / meter * torch.ones(1460, 1).to(device)
#         outputs245 = model_T(external, x0_v)
#         # 训练集真实情况
#         T245 = torch.cat((T_true[jimi245], T_true_val[jimi245]))
#         ax6.cla()
#         ax6.plot(time_draw, T245.cpu(), label='0cm', color='grey')
#         # 预测值
#         T_val245 = outputs245
#         ax6.plot(time_draw, T_val245.cpu().detach().numpy(), label='p_0m', color='red')
#         ax6.set_title('2.5m', fontsize=14)
#         # 分界线
#         ax6.axvline(730)
#
#         # 创建 (x, t) 网格
#         # t_plot = torch.linspace(0, 1, 1459).reshape(-1, 1).to(device)
#         x01_v = 2.5 / meter * torch.ones(1460, 1).to(device)
#         V_01 = model_V(external, x01_v, (day - 1))
#         # 绘制三维曲面图
#         ax4.cla()
#         # ax4.plot(external.cpu() * 1460, u_pred_3m.cpu().numpy() * (time_scale / (space_scale ** 2)))
#         ax4.plot(time_draw, V_01.cpu().detach().numpy() * 86400)
#         ax4.relim()  # 重新计算坐标范围
#         ax4.autoscale_view()
#         ax4.set_xlabel('x', fontsize=12)
#         ax4.set_ylabel('t', fontsize=12)
#         ax4.set_title('245v', fontsize=14)
#         ax4.axvline(730)
#
#         k = model_K(external, x01_v, (day - 1))
#         ax7.cla()
#         ax7.plot(time_draw, k.cpu().detach().numpy() * 86400, color='black')
#         ax7.autoscale_view()
#         ax7.set_title('2.5k', fontsize=14)
#
#         out = model_vsm_beta(external, x01_v)
#         ax8.cla()
#         ax8.plot(time_draw, out[0].cpu().detach().numpy(), color='black')
#         ax8.autoscale_view()
#         ax8.set_title('vsm2.5', fontsize=14)
#
#         ax9.cla()
#         ax9.autoscale_view()
#         ax9.set_title('point', fontsize=14)
#         ax9.scatter(init_points[:, 0], init_points[:, 1],
#                     c='red', label='初始条件点', alpha=0.6)
#         ax9.scatter(bc_points[:, 0], bc_points[:, 1],
#                     c='green', label='边界条件点', alpha=0.6)
#         ax9.scatter(collocation_points[:, 0], collocation_points[:, 1],
#                     c='blue', label='内部配点', alpha=0.6, s=10)
#
#         plt.pause(0.1)


def normalization(data_arr, max_value, min_value):
    new_data = [(data - min_value) / (max_value - min_value) for data in data_arr]
    return np.array(new_data, dtype=np.float32)


day = 1460
meter = 3
L = 334  # unit MJ/m3

# load datasets
file_path = "./data/train.xlsx"  # train
file_path_val = "./data/val.xlsx"  # val

# Return the normalized training time, validation time, training actual,
# validation actual temperature, and deep normalization.
data_loader = DataLoader_DIY(file_path, file_path_val, deep_max=meter)
Total = pd.read_excel("./data/Total.xlsx")

# Divide the data
t_ture = torch.from_numpy(data_loader.data[0].reshape(730, 1)).to(device)  # Time index of the training dataset
t_ture_val = torch.from_numpy(data_loader.data[1].reshape(730, 1)).to(device)  # Time index of the validation dataset.
T_true = torch.tensor(np.stack(data_loader.data[2])).float().to(device)  # The temperature of the training dataset.
T_true_val = torch.tensor(np.stack(data_loader.data[3])).float().to(
    device)  # The temperature of the validation dataset.
deep_train = torch.stack(data_loader.data[4]).to(device)  # The normalized depth of the training dataset.
deep_val = torch.stack(data_loader.data[5]).to(device)  # The normalized depth of the validation dataset.

external_index = Total['index'].tolist()
external = torch.from_numpy(normalization(external_index, max(external_index), 0).reshape(1460, 1)).to(device)

bc_up = np.array(Total['0m']).reshape(1460, 1)  # Up boundary
bc_down = np.array(Total['3m']).reshape(1460, 1)  # Down boundary
t_xup_bc = torch.from_numpy(bc_up).float().to(device)
t_xlow_bc = torch.from_numpy(bc_down).float().to(device)

losses = []
losses_val = []
ep = []
alp_list = []
num_epochs = 10000
best_pd_err = 99999

# PINN sampling
num_collocation = 6000  # Sampling in domain of definition
num_bc = 1460  # boundary sampling
num_init = 1460  # initial condition
sampler = PINNSampler1D(base_seed=3407)
init_points, bc_points, collocation_points = sampler.initial_sampling(
    n_collocation=num_collocation, n_init=num_init, n_bc=num_bc
)
pt_all_zeros = torch.zeros((num_collocation, 1)).to(device)

# model
model_T = PINN(device)
model_K = ParameterK(device).to(device)
model_V = ParameterV(device).to(device)
model_vsm_beta = Parameter_vsm_beta(device).to(device)

optimizer = optim.Adam(
    list(model_T.parameters())
    + list(model_K.parameters())
    + list(model_V.parameters())
    + list(model_vsm_beta.parameters()),
    lr=1e-3)

# loss function
loss_init_def = loss_cal()

# Supervising training - can eliminate to enhance training speed
# plt.ion()
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(331)
# ax2 = fig.add_subplot(332)
# ax3 = fig.add_subplot(333)
# ax4 = fig.add_subplot(334)
# ax5 = fig.add_subplot(335)
# ax6 = fig.add_subplot(336)
# ax7 = fig.add_subplot(337)
# ax8 = fig.add_subplot(338)
# ax9 = fig.add_subplot(339)

# set power value
w_physics = 5
w_bc = 50
w_ic = 15
w_data = 20

for epoch in tqdm(range(num_epochs), desc="Training"):
    # Sampling
    if epoch % 10 == 0 and epoch > 0:
        init_points, bc_points, collocation_points = sampler.initial_sampling(
            n_collocation=num_collocation, n_init=num_init, n_bc=num_bc
        )

    model_T.train()
    model_K.train()
    model_V.train()
    optimizer.zero_grad()
    ep.append(epoch)

    # physical loss
    f_loss = loss_init_def.loss_PDE(model_T, model_K, model_V, L, collocation_points, pt_all_zeros, model_vsm_beta,
                                    num_collocation, day, meter)

    # boundary loss
    bc_loss_up = loss_init_def.loss_BC_up(model_T, external, t_xup_bc)
    bc_loss_low = loss_init_def.loss_BC_low(model_T, external, t_xlow_bc)

    # initial loss
    ic_loss = loss_init_def.initial_loss(model_T, init_points, num_init)

    # data loss
    data_loss = 0.0
    for index, x_train in enumerate(deep_train[1:-1]):
        u_pred = model_T(t_ture, x_train)
        target = T_true[index + 1].reshape(T_true.size(1), 1)
        data_loss = data_loss + loss_init_def.loss_function(u_pred, target)

    # Balancing losses
    loss = (w_physics * f_loss +
            w_bc * (bc_loss_up + bc_loss_low) +
            w_ic * ic_loss +
            w_data * data_loss)

    loss.backward()
    optimizer.step()
    losses.append(loss.item())  # record

    # val
    model_T.eval()
    model_K.eval()
    model_V.eval()
    with torch.no_grad():

        data_loss_val = 0.0

        for index_val, x_val in enumerate(deep_val):
            u_pred_val = model_T(t_ture_val, x_val)
            target_val = T_true_val[index_val].reshape(T_true_val.size(1), 1)
            data_loss_val = data_loss_val + loss_init_def.loss_function(u_pred_val, target_val)

        losses_val.append(data_loss_val.item())

        if data_loss_val < best_pd_err:
            torch.save(model_T, "./model/modelT_best.pth")
            torch.save(model_K, "./model/modelK_best.pth")
            torch.save(model_V, "./model/modelV_best.pth")
            torch.save(model_vsm_beta, "./model/model_vsm_beta_best.pth")

            print('best_epoch:{}'.format(epoch))
            best_pd_err = data_loss_val

        if epoch % 100 == 0:
            print(
                f'Epoch {epoch}, Loss: {loss.item()},f_loss:{f_loss.item()},bc_loss_up:{bc_loss_up.item()} ,bc_loss_low:{bc_loss_low.item()} , ic_loss:{ic_loss.item()} , da'
                f'ta_loss:{data_loss.item()},val_Loss:{data_loss_val}')

        # Supervising training - can eliminate to enhance training speed. Of course, you can use other tools to supervise.
        # draw()

# save
torch.save(model_T, "./model/modelT_latest.pth")
torch.save(model_K, "./model/modelK_latest.pth")
torch.save(model_V, "./model/modelV_latest.pth")
torch.save(model_vsm_beta, "./model/model_vsm_beta_latest.pth")
print("Over")
