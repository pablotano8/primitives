import pymunk
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle


import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


class DMP1D:
    def __init__(self, start, goal, n_basis=3, alpha_x=3.0, alpha_z=10.0, beta_z=10.0/4.0, complexity = 1):
        self.start = start
        self.goal = goal
        self.n_basis = n_basis
        self.alpha_x = alpha_x
        self.alpha_z = alpha_z
        self.beta_z = beta_z
        self.complexity = complexity

        self.basis_centers = np.linspace(0, 1, n_basis)
        self.basis_widths = np.ones(n_basis) * 1.0 / (0.65 * (self.basis_centers[1] - self.basis_centers[0]) ** 2)
        self.weights = np.random.randn(n_basis)*self.complexity

    def basis(self, x):
        return np.exp(-self.basis_widths * (x - self.basis_centers)**2)
    
    def forcing_term(self, x):
        psi = self.basis(x)
        return (x * np.sum(psi * self.weights))

    def trajectory(self, T=1.0, dt=0.01):
        # Reset the system state
        self.y = self.start
        self.dy = 0
        self.x = 1
        self.z = 0

        # Record the state at each timestep
        y_track = []
        timesteps = int(T / dt)
        for _ in range(timesteps):
            self.y, self.z, self.x = self.step(self.y, self.z, self.x, dt)
            y_track.append(self.y)

        return np.array(y_track)

    
    def step(self, y, z, x, dt):
        f = self.forcing_term(x)
        dz = self.alpha_z * (self.beta_z * (self.goal - y) - z + f * (self.goal - self.start)) / self.alpha_x
        dx = -self.alpha_x * x
        z += dz * dt
        y += z * dt
        x += dx * dt
        return y, z, x
    
    def desired_velocity(self, y, z, x):
        f = self.forcing_term(x)
        dz = self.alpha_z * (self.beta_z * (self.goal - y) - z + f * (self.goal - self.start)) / self.alpha_x
        z += dz * 0.01
        return z


class Simulation:
    def __init__(self, world, dmp_x, dmp_y, start_position, T=1.0, dt=0.01, noise_std=0.00):
        self.world = world
        self.dmp_x = dmp_x
        self.dmp_y = dmp_y
        self.position = start_position
        self.velocity = np.array([0.0, 0.0])
        self.T = T
        self.dt = dt
        self.x = 1  # Initialize the canonical system state
        self.noise_std = noise_std * (np.abs(dmp_x.goal-dmp_x.start) + np.abs(dmp_y.goal-dmp_y.start))

    def run(self):
        position_track = []
        velocity_track = []
        min_distance_all = np.full(len(self.world.obstacles), float('inf'))  # Initialize as array of inf
        collision_happened = np.zeros(len(self.world.obstacles), dtype=bool)  # Initialize as array of False
        collision_position = [None] * len(self.world.obstacles)
        timesteps = int(self.T / self.dt)
        for _ in range(timesteps):
            desired_velocity_x = self.dmp_x.desired_velocity(self.position[0], self.velocity[0], self.x)
            desired_velocity_y = self.dmp_y.desired_velocity(self.position[1], self.velocity[1], self.x)
            desired_velocity = np.array([desired_velocity_x, desired_velocity_y])

            # Add control noise
            noise = np.random.normal(scale=self.noise_std, size=2)
            desired_velocity += noise

            # Get the actual velocity and new position from the world
            self.position, self.velocity, collision, _ , min_distance = self.world.actual_dynamics(self.position, desired_velocity, self.dt)
            if not any(collision_happened):
                min_distance_all = np.minimum(min_distance_all, min_distance)
            if not any(collision_happened):
                collision_happened = np.logical_or(collision_happened, collision)

            # Update the canonical system state
            self.x += -self.dmp_x.alpha_x * self.x * self.dt

            position_track.append(self.position)
            velocity_track.append(self.velocity)

        return np.array(position_track), np.array(velocity_track), collision_happened, collision_position, min_distance_all


    def plot(self, position_track, color='blue', plot_goal=True, alpha=0.5):
        # Add obstacles to the plot
        for body, obstacle in self.world.obstacles:
            vertices = np.array([body.local_to_world(v) for v in obstacle.get_vertices()])
            vertices = np.vstack((vertices, vertices[0]))  # Close the polygon by adding the first point at the end
            plt.plot(*vertices.T, color='black')
            plt.fill(*vertices.T, color=[0.8,0.8,0.8], alpha=0.5)  # Fill the polygon to make the obstacle solid

        # Check if position_track is a 2D array with at least one point before plotting
        if position_track.ndim == 2 and position_track.shape[1] == 2:
            plt.plot(position_track[:, 0], position_track[:, 1], '.', color=color, alpha=alpha, label='Goal = (1.0, 0.5)')
            
            # Check if position_track has at least one point to plot the last position
            if position_track.shape[0] > 0 and plot_goal:
                plt.plot(position_track[-1, 0], position_track[-1, 1], 'x', color=color)

        # if plot_goal:
        #     plt.scatter([self.dmp_x.start, self.dmp_x.goal], [self.dmp_y.start, self.dmp_y.goal], color='red')  # plot start and end points

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('2D DMP trajectories with obstacles')