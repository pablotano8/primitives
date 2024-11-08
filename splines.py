import pymunk
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
from continuous_nav_envs import World, generate_random_positions


import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


class Spline:
    def __init__(self, start, goal, curvature, T=1.0, dt=0.01):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.curvature = curvature
        self.T = T  # Total time of the trajectory
        self.dt = dt  # Time step

        # Compute the control point for the quadratic Bezier curve
        direction = self.goal - self.start
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            raise ValueError("Start and goal positions cannot be the same")
        direction_unit = direction / direction_norm

        # Normal vector to the line from start to goal
        normal = np.array([-direction_unit[1], direction_unit[0]])

        # Control point (P1) adjusted by curvature parameter
        self.P1 = (self.start + self.goal) / 2 + self.curvature * normal

        # Precompute the trajectory
        self.precompute_trajectory()

    def bezier_curve(self, t):
        """Compute position on the Bezier curve at parameter t."""
        P0 = self.start
        P1 = self.P1
        P2 = self.goal
        return (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2

    def precompute_trajectory(self):
        """Precompute positions and velocities equally spaced along the curve in terms of arc length."""
        # Number of steps
        self.timesteps = int(self.T / self.dt)
        num_points = self.timesteps * 10  # Use more points for accurate arc length calculation

        # Parameter t along the curve
        ts = np.linspace(0, 1, num_points)
        positions = np.array([self.bezier_curve(t) for t in ts])

        # Compute cumulative arc length
        deltas = np.diff(positions, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        arc_lengths = np.hstack(([0], np.cumsum(distances)))
        total_length = arc_lengths[-1]

        # Equally spaced arc lengths
        equal_arc_lengths = np.linspace(0, total_length, self.timesteps)

        # Interpolate to find corresponding t values
        ts_equally_spaced = np.interp(equal_arc_lengths, arc_lengths, ts)

        # Compute positions at these t values
        self.positions = np.array([self.bezier_curve(t) for t in ts_equally_spaced])

        # Compute velocities
        self.velocities = np.diff(self.positions, axis=0) / self.dt
        # Add last velocity to maintain array size
        self.velocities = np.vstack((self.velocities, self.velocities[-1]))

        # For desired_velocity function, create time array
        self.times = np.linspace(0, self.T, self.timesteps)

    def desired_velocity(self, current_time):
        """Get desired velocity at the current time."""
        # Find the closest index in the precomputed trajectory
        index = int(np.clip(current_time / self.dt, 0, self.timesteps - 1))
        return self.velocities[index]

    def trajectory(self):
        """Return the precomputed trajectory positions."""
        return self.positions

class Spline1D:
    def __init__(self, spline, dim):
        self.spline = spline
        self.dim = dim  # 0 for x, 1 for y
        self.start = spline.start[dim]
        self.goal = spline.goal[dim]
        # Parameters to match DMP1D (not used but included for compatibility)
        self.alpha_x = 3.0
        self.alpha_z = 10.0
        self.beta_z = 10.0 / 4.0

    def basis(self, x):
        """Not used in spline but included for compatibility."""
        return None

    def forcing_term(self, x):
        """Not used in spline but included for compatibility."""
        return None

    def trajectory(self):
        """Return the trajectory in this dimension."""
        positions = self.spline.trajectory()
        return positions[:, self.dim]

    def desired_velocity(self, y, z, x, t):
        """Compute the desired velocity at current time t."""
        desired_vel = self.spline.desired_velocity(t)
        return desired_vel[self.dim]


class Simulation:
    def __init__(self, world, dmp_x, dmp_y, start_position, T=1.0, dt=0.01, noise_std=0.00):
        self.world = world
        self.dmp_x = dmp_x
        self.dmp_y = dmp_y
        self.position = np.array(start_position)
        self.velocity = np.array([0.0, 0.0])
        self.T = T
        self.dt = dt
        self.t = 0.0  # Current time
        self.noise_std = noise_std * (np.abs(dmp_x.goal - dmp_x.start) + np.abs(dmp_y.goal - dmp_y.start))

    def run(self):
        position_track = []
        velocity_track = []
        min_distance_all = np.full(len(self.world.obstacles), float('inf'))  # Initialize as array of inf
        collision_happened = np.zeros(len(self.world.obstacles), dtype=bool)  # Initialize as array of False
        collision_position = [None] * len(self.world.obstacles)
        timesteps = int(self.T / self.dt)
        for _ in range(timesteps):
            desired_velocity_x = self.dmp_x.desired_velocity(self.position[0], self.velocity[0], None, self.t)
            desired_velocity_y = self.dmp_y.desired_velocity(self.position[1], self.velocity[1], None, self.t)
            desired_velocity = np.array([desired_velocity_x, desired_velocity_y])

            # Add control noise
            noise = np.random.normal(scale=self.noise_std, size=2)
            desired_velocity += noise

            # Get the actual velocity and new position from the world
            self.position, self.velocity, collision, _, min_distance = self.world.actual_dynamics(
                self.position, desired_velocity, self.dt
            )
            if not any(collision_happened):
                min_distance_all = np.minimum(min_distance_all, min_distance)
            if not any(collision_happened):
                collision_happened = np.logical_or(collision_happened, collision)

            position_track.append(self.position.copy())
            velocity_track.append(self.velocity.copy())

            # Update time
            self.t += self.dt

        return np.array(position_track), np.array(velocity_track), collision_happened, collision_position, min_distance_all

    def plot(self, position_track, color='blue', plot_goal=True, alpha=0.5):
        # Add obstacles to the plot
        for body, obstacle in self.world.obstacles:
            vertices = np.array([body.local_to_world(v) for v in obstacle.get_vertices()])
            vertices = np.vstack((vertices, vertices[0]))  # Close the polygon by adding the first point at the end
            plt.plot(*vertices.T, color='black')
            plt.fill(*vertices.T, color=[0.8, 0.8, 0.8], alpha=0.5)  # Fill the polygon to make the obstacle solid

        # Check if position_track is a 2D array with at least one point before plotting
        if position_track.ndim == 2 and position_track.shape[1] == 2:
            plt.plot(position_track[:, 0], position_track[:, 1], color=color, alpha=alpha, label='Trajectory')

            # Check if position_track has at least one point to plot the last position
            if position_track.shape[0] > 0 and plot_goal:
                plt.plot(position_track[-1, 0], position_track[-1, 1], 'x', color=color, alpha=alpha)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('2D Trajectories with Obstacles')



def generate_trajectories_from_dmp_params(dmp_params1,dmp_params2,batch_size,batch_s_t,world):
        
        dmp_params1, dmp_params2 = dmp_params1.numpy(), dmp_params2.numpy()
        # Initialize and run DMP simulations for each item in the batch
        actual_final_position1,actual_final_position2, collision_info = [],[] , []
        for b in range(batch_size):
            spline = Spline(batch_s_t[b][:2], dmp_params1[b][2:4], dmp_params1[b][-1])
            # Create Spline1D instances for x and y dimensions
            dmp_x1 = Spline1D(spline, dim=0)
            dmp_y1 = Spline1D(spline, dim=1)
            # Now use these in your Simulation class
            simulation1 = Simulation(world, dmp_x1, dmp_y1, batch_s_t[b][:2])
            positions1, velocities1, collision1, collision_pos1, min_distance1 = simulation1.run()

            spline = Spline(positions1[-1][:2], dmp_params2[b][2:4], dmp_params2[b][-1])
            # Create Spline1D instances for x and y dimensions
            dmp_x2 = Spline1D(spline, dim=0)
            dmp_y2 = Spline1D(spline, dim=1)
            # Now use these in your Simulation class
            simulation2 = Simulation(world, dmp_x2, dmp_y2, positions1[-1][:2])
            positions2, velocities2, collision2, collision_pos2, min_distance2 = simulation2.run()


            actual_final_position1.append(np.array(positions1[-1]))
            actual_final_position2.append(np.array(positions2[-1]))
            collision_info.append({
                'collision1': collision1,
                'collision2': collision2,
                'collision_pos': collision_pos1,
                'collision_pos2': collision_pos2,
                'min_distance1': min_distance1,
                'min_distance2': min_distance2})
        return actual_final_position1,actual_final_position2, collision_info


def plot_example_trajectories(
        world, world_bounds, number_of_trajectories=1, random_position=True, orientation=None, complexity=1, circular=False,n_basis=3):
    
    plt.figure(figsize=(5, 4))
    ax = plt.gca()

    plt.xlim([world_bounds[0] - 0.04, world_bounds[1] + 0.05])
    plt.ylim([world_bounds[2] - 0.05, world_bounds[3] + 0.05])
    
    for _ in range(number_of_trajectories):
        start_position, goal_position = generate_random_positions(world, world_bounds=world_bounds, orientation=orientation)
        curvature1 = np.random.uniform(low=-complexity,high=complexity)

        spline1 = Spline(start_position, goal_position, curvature1)
        # Create Spline1D instances for x and y dimensions
        dmp_x1 = Spline1D(spline1, dim=0)
        dmp_y1 = Spline1D(spline1, dim=1)
        # Now use these in your Simulation class
        simulation = Simulation(world, dmp_x1, dmp_y1, start_position)

        positions1, velocities1, collision1, collision_pos1, min_distance1 = simulation.run()
        simulation.plot(positions1,color='blue',alpha=0.1)

        # Generate random goal position for second DMP
        _, goal_position2 = generate_random_positions(world, world_bounds=world_bounds,orientation = orientation)
        spline2 = Spline(positions1[-1], goal_position2, curvature1)
        # Create Spline1D instances for x and y dimensions
        dmp_x2 = Spline1D(spline2, dim=0)
        dmp_y2 = Spline1D(spline2, dim=1)
        # Now use these in your Simulation class
        simulation = Simulation(world, dmp_x2, dmp_y2, start_position)
        positions2, velocities1, collision1, collision_pos1, min_distance1 = simulation.run()
        simulation.plot(positions2,color='red',alpha=0.1)

    plt.show()


def generate_and_plot_trajectories_from_parameters(
        dmp_params1,
        dmp_params2,
        batch_size,
        batch_s_t,
        world,
        world_bounds,
        n_basis=3,
        circular=False,
        plot_only_first=False,
        gradual=False):
    
    dmp_params1, dmp_params2 = dmp_params1.detach().numpy(), dmp_params2.detach().numpy()
    batch_s_t = batch_s_t.detach().numpy()
    plt.figure(figsize=(3, 2.5))
    
    for b in range(batch_size):
        if gradual:
            alpha=b/batch_size * 0.9 + 0.1
        else:
            alpha=1

        spline = Spline(batch_s_t[b][:2], dmp_params1[b][2:4], dmp_params1[b][-1])
        # Create Spline1D instances for x and y dimensions
        dmp_x1 = Spline1D(spline, dim=0)
        dmp_y1 = Spline1D(spline, dim=1)
        # Now use these in your Simulation class
        simulation1 = Simulation(world, dmp_x1, dmp_y1, batch_s_t[b][:2])
        positions1, velocities1, collision1, collision_pos1, min_distance1 = simulation1.run()

        spline = Spline(positions1[-1][:2], dmp_params2[b][2:4], dmp_params2[b][-1])
        # Create Spline1D instances for x and y dimensions
        dmp_x2 = Spline1D(spline, dim=0)
        dmp_y2 = Spline1D(spline, dim=1)
        # Now use these in your Simulation class
        simulation2 = Simulation(world, dmp_x2, dmp_y2, positions1[-1][:2])
        positions2, velocities2, collision2, collision_pos2, min_distance2 = simulation2.run()

        # Plot the trajectories
        simulation1.plot(positions1, color='blue',plot_goal=False,alpha=alpha) 
        if not plot_only_first:
            simulation2.plot(positions2, color='red',plot_goal=False,alpha=alpha)
    
    plt.xlim([world_bounds[0]-0.1, world_bounds[1]+0.1])
    plt.ylim([world_bounds[2]-0.1, world_bounds[3]+0.1])
    plt.show()

def plot_predicted_vs_actual_trajectories(
        world,
        net,
        world_bounds,
        number_of_trajectories=1,
        orientation=None,
        complexity=1):

    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    

    plt.xlim([world_bounds[0] - 0.04, world_bounds[1] + 0.05])
    plt.ylim([world_bounds[2] - 0.05, world_bounds[3] + 0.05])
    
    for _ in range(number_of_trajectories):

        start_position, goal_position = generate_random_positions(world, world_bounds=world_bounds, orientation=orientation)
        curvature1 = np.random.uniform(low=-complexity,high=complexity)

        spline1 = Spline(start_position, goal_position, curvature1)
        # Create Spline1D instances for x and y dimensions
        dmp_x1 = Spline1D(spline1, dim=0)
        dmp_y1 = Spline1D(spline1, dim=1)
        # Now use these in your Simulation class
        simulation = Simulation(world, dmp_x1, dmp_y1, start_position)

        # Run the simulation and record the positions
        positions, velocities, collisions, collision_position, min_distance = simulation.run()

        # Compute the prediction of the model for the final position
        dmp_params =  [dmp_x1.start, dmp_y1.start,dmp_x1.goal, dmp_y1.goal, spline1.curvature]
        s_t = positions[0]
            
        inputs1 = torch.tensor([np.concatenate((s_t, dmp_params))], dtype=torch.float32).unsqueeze(1)
        
        # Forward pass for the first timestep
        outputs1, hidden = net(inputs1)
        
        # Prepare inputs for the second timestep
        padded_state = torch.zeros_like(torch.tensor(s_t))
        inputs2 = torch.tensor([np.concatenate((padded_state, dmp_params))], dtype=torch.float32).unsqueeze(1)
        
        # Forward pass for the second timestep
        outputs2, _ = net(inputs2, hidden)
        
        # Get the final predicted position
        predicted_final_position = outputs2.detach().numpy().squeeze()
        print("Predicted final position shape:", predicted_final_position.shape)

        # Plot the actual trajectory
        simulation.plot(positions, plot_goal=False,color='blue',alpha=0.5)
        
        # Plot the predicted final position
        plt.scatter(predicted_final_position[0], predicted_final_position[1], color='blue', marker='o', label='Predicted' if _ == 0 else "")
    
    plt.show()


if __name__ == "__main__":
    # Assume that the World class and the necessary imports are already provided

    # Initialize the world
    world_bounds = [0.1, 0.9, 0.1, 0.9]
    world = World(
        world_bounds=world_bounds,
        friction=0.5,
        num_obstacles=1,
        given_obstacles=[(0, 0.48), (0, 0.52), (0.6, 0.48), (0.6, 0.52)],
        
    )

    # Define desired parameters
    start_position = np.array([0.35, 0.2])
    false_start_position = np.array([0.5, 0.2])
    goal_position = np.array([0.2, 0.8])
    
    for curvature in np.linspace(-0.1, 0.1, 21):
        # Create a Spline instance
        spline = Spline(start_position, goal_position, curvature)

        # Create Spline1D instances for x and y dimensions
        dmp_x = Spline1D(spline, dim=0)
        dmp_y = Spline1D(spline, dim=1)

        # Now use these in your Simulation class
        simulation = Simulation(world, dmp_x, dmp_y, false_start_position)

        # Run the simulation
        position_track, velocity_track, collision_happened, collision_position, min_distance_all = simulation.run()

        # Plot the results
        simulation.plot(position_track)

    # Plot the goal point
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(goal_position[0], goal_position[1], 'ro', label='Goal')
    plt.show()