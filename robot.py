import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import copy
from plot_trajectories import plot_example_trajectories, plot_predicted_vs_actual_trajectories
import matplotlib.pyplot as plt
from utils import sample_random_dmp_params, generate_trajectories_from_dmp_params, plot_value_function
from continuous_nav_envs import World, generate_random_positions
from dmps import DMP1D,Simulation
from plot_trajectories import generate_and_plot_trajectories_from_parameters

class GoalNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(GoalNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size,batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        return x, hidden
    
class PredNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(PredNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)
        return x, hidden

class RobotSimulation:
    def __init__(self, env, dmp_x, dmp_y,dmp_z, T=1.0, dt=0.01):
        self.env = env
        self.dmp_x = dmp_x
        self.dmp_y = dmp_y
        self.dmp_z = dmp_z
        self.velocity = np.array([0.0, 0.0])
        self.T = T
        self.dt = dt
        self.x = 1  # Initialize the canonical system state

    def run(self):
        position_track = []
        velocity_track = []

        timesteps = int(self.T / self.dt)
        for _ in range(timesteps):
            desired_velocity_x = self.dmp_x.desired_velocity(self.position[0], self.velocity[0], self.x)
            desired_velocity_y = self.dmp_y.desired_velocity(self.position[1], self.velocity[1], self.x)
            desired_velocity_z = self.dmp_z.desired_velocity(self.position[2], self.velocity[2], self.x)
            desired_velocity = np.array([desired_velocity_x, desired_velocity_y,desired_velocity_z])


            # Get the actual velocity and new position from the world
            self.position, self.velocity, collision, _ , min_distance = self.env.step(self.position, desired_velocity)

            # Update the canonical system state
            self.x += -self.dmp_x.alpha_x * self.x * self.dt

            position_track.append(self.position)
            velocity_track.append(self.velocity)

        return np.array(position_track), np.array(velocity_track)
    

def generate_data(
        world,
        world_bounds,
        number_of_trajectories=10000,
        complexity = 1,
        random_world = False):
    # Generate training data
    data = []
    for i in range(number_of_trajectories):  # Number of trajectories to generate
        if random_world:
            world = World(
                world_bounds=world_bounds,
                friction=0,
                obs_random_position= True,
                num_obstacles=2)

        if i%1000==0:
            print(f'Generated {i} of {number_of_trajectories} trajectories')
        # Generate random start and goal positions for first DMP
        start_position, _ = generate_random_positions(world, world_bounds=world_bounds,orientation = None)
        _, goal_position1 = generate_random_positions(world, world_bounds=world_bounds,orientation = None)

        # Initialize the first DMPs
        dmp_x1 = DMP1D(start=start_position[0], goal=goal_position1[0], n_basis=3, complexity=complexity)
        dmp_y1 = DMP1D(start=start_position[1], goal=goal_position1[1], n_basis=3, complexity=complexity)
        dmp_z1 = DMP1D(start=start_position[1], goal=goal_position1[1], n_basis=3, complexity=complexity)

        # Initialize the simulation with the world and the first DMPs
        simulation = Simulation(world, dmp_x1, dmp_y1, start_position, T=1.0, dt=0.01)

        # Run the first simulation and record the positions
        positions1, velocities1, collision1, _, _ = simulation.run()

        # Generate random goal position for second DMP
        _, goal_position2 = generate_random_positions(world, world_bounds=world_bounds,orientation = None)

        # Initialize the second DMPs with start position as the final position from the first DMP
        dmp_x2 = DMP1D(start=positions1[-1][0], goal=goal_position2[0], n_basis=3,complexity=complexity)
        dmp_y2 = DMP1D(start=positions1[-1][1], goal=goal_position2[1], n_basis=3,complexity=complexity)

        # Initialize the simulation with the world and the second DMPs
        simulation = Simulation(world, dmp_x2, dmp_y2, positions1[-1], T=1.0, dt=0.01)

        # Run the second simulation and record the positions
        positions2, velocities2, collision2,_ ,_= simulation.run()

        # Add the data to the training set
        dmp_params1 = [dmp_x1.start, dmp_y1.start, dmp_x1.goal, dmp_y1.goal, *dmp_x1.weights, *dmp_y1.weights]
        dmp_params2 = [dmp_x1.goal, dmp_y1.goal, dmp_x2.goal, dmp_y2.goal, *dmp_x2.weights, *dmp_y2.weights]
        distances_to_edges = np.array([np.array(point) for point in world.centers]).flatten()

        s_t = np.concatenate((positions1[0],distances_to_edges))
        data.append((s_t, True, positions1[-1], positions2[-1], dmp_params1, dmp_params2, any(collision1)*1, any(collision2)*1))

    # Split the data into training and validation sets
    split_idx = int(len(data) * 0.9)  # Use 80% of the data for training
    train_data = data[:split_idx]
    valid_data = data[split_idx:]

    # Shuffle training  data
    random.shuffle(train_data)

    train_data = [(torch.tensor(s_t, dtype=torch.float32),
                torch.tensor(s_t_plus_one, dtype=torch.float32),
                torch.tensor(final_position1, dtype=torch.float32),
                torch.tensor(final_position2, dtype=torch.float32),
                torch.tensor(dmp_params1, dtype=torch.float32),
                torch.tensor(dmp_params2, dtype=torch.float32),
                torch.tensor(collision1, dtype=torch.float32),
                torch.tensor(collision2, dtype=torch.float32)) for s_t, s_t_plus_one, final_position1,final_position2, dmp_params1,dmp_params2, collision1,collision2 in train_data]

    valid_data = [(torch.tensor(s_t, dtype=torch.float32),
                torch.tensor(s_t_plus_one, dtype=torch.float32),
                torch.tensor(final_position1, dtype=torch.float32),
                torch.tensor(final_position2, dtype=torch.float32),
                torch.tensor(dmp_params1, dtype=torch.float32),
                torch.tensor(dmp_params2, dtype=torch.float32),
                torch.tensor(collision1, dtype=torch.float32),
                torch.tensor(collision2, dtype=torch.float32)) for s_t, s_t_plus_one, final_position1,final_position2, dmp_params1,dmp_params2, collision1,collision2 in valid_data]
    return train_data, valid_data
