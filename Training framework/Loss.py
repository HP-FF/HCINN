import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def f_init(x):
    return 0.04343 + (-6.38866 - 0.04343) / (1 + torch.pow((x / 0.50643), 1.81846))  # initial temperature. x-->(0,1)


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        diff = pred[mask] - target[mask]
        return torch.mean(diff ** 2)


def get_FT_index(temperature):
    freeze_mask = temperature < 0  # frozen zone
    thaw_mask = ~freeze_mask  # thawing zone
    return freeze_mask, thaw_mask


def smooth_theta_u(T, theta_bar, beta, T_f=0.0, k=10.0):
    """
    parameters:
        T         : Temperature
        theta_bar : Average volumetric water content during the thawing period (constant or tensor)
        T_f       : Freezing point (unit: °C), default: 0
        beta      : Fitting parameters (controlling slope)
        k         : tanh Control parameter (regulating the rate of smooth transition)

    Return:
        theta_u   : unfrozen water content
    """
    frac = torch.clamp((T_f - T) / 273.15, min=0.0)

    smooth_step = 0.5 * (1.0 - torch.tanh(k * (T - T_f)))  # close to 1：frozen. close to 0：thawing

    theta_u = theta_bar * (1.0 - smooth_step * (frac ** beta))
    return theta_u


class loss_cal(nn.Module):
    def __init__(self):
        super(loss_cal, self).__init__()
        self.loss_function = MaskedMSELoss()

    def loss_BC_up(self, model_T, t_bc, g1_bcup):
        x_up = torch.zeros_like(t_bc).to(device)
        u_up = model_T(t_bc, x_up)
        loss_up = self.loss_function(u_up, g1_bcup)
        return loss_up

    def loss_BC_low(self, model_T, t_bc, g2_bclow):
        x_low = torch.ones_like(t_bc).to(device)
        u_low = model_T(t_bc, x_low)
        loss_low = self.loss_function(u_low, g2_bclow)
        return loss_low

    def initial_loss(self, model_T, point_ic, numpoint):
        t_0 = torch.from_numpy(point_ic[:, 1]).reshape(numpoint, 1).float().to(device)
        t_0.requires_grad = True
        x_ic = torch.from_numpy(point_ic[:, 0]).reshape(numpoint, 1).float().to(device)
        x_ic.requires_grad = True
        u_init = model_T(t_0, x_ic)
        u_exact = f_init(x_ic)
        return self.loss_function(u_init, u_exact)

    def loss_PDE(self, model_T, model_K, model_V, L, in_point, ft, model_vsm_beta, numpoint, time_scale, depth_scale):
        t = torch.from_numpy(in_point[:, 1]).reshape(numpoint, 1).float().to(device)
        t.requires_grad = True
        x = torch.from_numpy(in_point[:, 0]).reshape(numpoint, 1).float().to(device)
        x.requires_grad = True

        u = model_T(t, x)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

        K_pred = model_K(t, x, (time_scale - 1))  # coefficient of heat conduction
        heat_eq = K_pred * u_x
        u_xx = torch.autograd.grad(heat_eq, x, torch.ones_like(u), create_graph=True)[0]

        v_heat_cap = model_V(t, x, (time_scale - 1))

        freeze_mask, thaw_mask = get_FT_index(u)
        water_T = torch.zeros_like(u)

        if freeze_mask.any():  # Check for Frozen
            u_freeze = u[freeze_mask]
            vsm, beta = model_vsm_beta(t, x)
            vsm = vsm.reshape(numpoint, 1)
            beta = beta.reshape(numpoint, 1)
            s_frezz = smooth_theta_u(u_freeze, vsm[freeze_mask], beta[freeze_mask])
            # Compute the gradient of s_frezz with respect to u_freeze.
            ds_dutf = torch.autograd.grad(
                s_frezz,
                u_freeze,
                grad_outputs=torch.ones_like(s_frezz),
                create_graph=True
            )[0]

            water_T[freeze_mask] = ds_dutf

        f = L * water_T * u_t + v_heat_cap * u_t - u_xx
        loss_f = self.loss_function(f, ft)

        return loss_f
