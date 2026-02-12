import numpy as np
import torch
from scipy.stats import qmc
import matplotlib.pyplot as plt

class PINNSampler1D:
    def __init__(self, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0, base_seed=1234):
        self.x_min = x_min
        self.x_max = x_max
        self.t_min = t_min
        self.t_max = t_max
        self.base_seed = base_seed
        self.round_idx = 0

        self.init_points = None
        self.bc_points = None
        self.collocation_points = None

    def latin_hypercube_sampling(self, n_samples):
        """Latin Hypercube Sampling is implemented with an increment of the counter upon each invocation."""
        sampler = qmc.LatinHypercube(d=2, seed=self.base_seed + self.round_idx)
        self.round_idx += 1
        sample = sampler.random(n=n_samples)
        points = qmc.scale(sample, [self.x_min, self.t_min], [self.x_max, self.t_max])
        return points

    def initial_condition_sampling(self, n_points=100):
        x_points = np.linspace(self.x_min, self.x_max, n_points)
        t_points = np.zeros_like(x_points)
        points = np.vstack((x_points, t_points)).T
        return points

    def boundary_condition_sampling(self, n_points_per_boundary=50):
        t_left = np.linspace(self.t_min, self.t_max, n_points_per_boundary)
        x_left = np.zeros_like(t_left)
        left_points = np.vstack((x_left, t_left)).T

        t_right = np.linspace(self.t_min, self.t_max, n_points_per_boundary)
        x_right = np.ones_like(t_right)
        right_points = np.vstack((x_right, t_right)).T

        points = np.vstack((left_points, right_points))
        return points

    def initial_sampling(self, n_collocation=1000, n_init=100, n_bc=50):
        """An initial sampling is performed, with each invocation producing variable yet reproducible results."""
        self.init_points = self.initial_condition_sampling(n_init)
        self.bc_points = self.boundary_condition_sampling(n_bc)
        self.collocation_points = self.latin_hypercube_sampling(n_collocation)
        return self.init_points, self.bc_points, self.collocation_points


if __name__ == "__main__":
    np.random.seed(3407)
    torch.manual_seed(3407)

    sampler = PINNSampler1D(x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0, base_seed=1234)

    for i in range(1000):
        if i % 10 == 0:

            init_points, bc_points, collocation_points = sampler.initial_sampling(
                n_collocation=1000, n_init=100, n_bc=50
            )
