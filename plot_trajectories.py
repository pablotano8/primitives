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


from continuous_nav_envs import World,TrackWorld
from dmps import DMP1D, Simulation

from continuous_nav_envs import World,TrackWorld, generate_random_positions, is_inside_obstacle



def plot_example_trajectories_reach_goal(world, world_bounds, number_of_trajectories=10, use_two_prims = False,random_position=True, orientation=None, complexity=1, tolerance=0.05):
    # Generate random start and goal positions
    start_position, goal_position = [0.25,0.15],[0.5,0.5]

    plt.figure(figsize=(10, 8))
    for _ in range(number_of_trajectories):
        if use_two_prims:

            sub_x, sub_y = np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7)
            dmp_x1 = DMP1D(start=start_position[0], goal=sub_x, complexity=complexity)
            dmp_y1 = DMP1D(start=start_position[1], goal=sub_y, complexity=complexity) 

            # Simulate the trajectory in the world
            sim = Simulation(world, dmp_x1, dmp_y1, start_position, T=1.0, dt=0.01)
            position_track1, _, _, _, _ = sim.run()
            
            # Check if the trajectory got close to the goal
            final_position1 = position_track1[-1]

            dmp_x = DMP1D(start=final_position1[0], goal=goal_position[0], complexity=complexity)
            dmp_y = DMP1D(start=final_position1[1], goal=goal_position[1], complexity=complexity)
            
            # Simulate the trajectory in the world
            sim = Simulation(world, dmp_x, dmp_y, start_position=np.array([final_position1[0], final_position1[1]]), T=1.0, dt=0.01)
            position_track2, _, _, _, _ = sim.run()
            
            # Check if the trajectory got close to the goal
            final_position = position_track2[-1]
            distance_to_goal = np.linalg.norm(final_position - np.array([goal_position[0], goal_position[1]]))

            # Color-code the trajectory
            color = 'red' if distance_to_goal > tolerance else 'green'
            alpha=0.5

            # Plot the trajectories
            sim.plot(position_track1, color=color, plot_goal=False,alpha=alpha)
            sim.plot(position_track2, color=color, plot_goal=False,alpha=alpha)

        else:
            # Initialize the DMPs
            dmp_x = DMP1D(start=start_position[0], goal=goal_position[0], n_basis=2, complexity=complexity)
            dmp_y = DMP1D(start=start_position[1], goal=goal_position[1], n_basis=2, complexity=complexity)

            # Initialize the simulation with the world and the DMPs
            simulation = Simulation(world, dmp_x, dmp_y, start_position, T=1.0, dt=0.01)

            # Run the simulation and record the positions
            positions, _, _, _, _ = simulation.run()

            # Check if trajectory was close to the goal
            final_position = positions[-1]
            distance_to_goal = np.linalg.norm(final_position - goal_position)

            # Color-code the trajectory
            color = 'red' if distance_to_goal > tolerance else 'green'
            alpha = 0.5

            # Plot the trajectories
            simulation.plot(positions, color=color, plot_goal=False,alpha=alpha)

    plt.xlim([world_bounds[0] - 0.04, world_bounds[1] + 0.05])
    plt.ylim([world_bounds[2] - 0.05, world_bounds[3] + 0.05])
    plt.show()


def plot_predicted_vs_actual_trajectories(
        world,
        net,
        world_bounds,
        number_of_trajectories=1,
        random_position=True,
        orientation=None,
        complexity=1,
        circular=False,
        varying_wall=False,
        random_holes=False):
    def generate_positions():
        if circular:
            while True:
                start = np.random.uniform(-world.radius, world.radius, size=2)
                if np.linalg.norm(start) <= world.radius and not is_inside_obstacle(start, world):
                    break
            while True:
                goal = np.random.uniform(-world.radius, world.radius, size=2)
                if np.linalg.norm(goal) <= world.radius and not is_inside_obstacle(goal, world):
                    break
        else:
            start, goal = generate_random_positions(world, world_bounds=world_bounds, orientation=orientation)
        return start, goal

    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    
    if circular:
        theta = np.linspace(0, 2 * np.pi, 100)
        x = world.radius * np.cos(theta)
        y = world.radius * np.sin(theta)
        plt.plot(x, y, 'k--')  # Plot the circle with a dashed line
        plt.xlim([-world.radius - 0.1, world.radius + 0.1])
        plt.ylim([-world.radius - 0.1, world.radius + 0.1])
    else:
        plt.xlim([world_bounds[0] - 0.04, world_bounds[1] + 0.05])
        plt.ylim([world_bounds[2] - 0.05, world_bounds[3] + 0.05])
    
    for _ in range(number_of_trajectories):
        if circular:
            world.reset()
        if random_position:
            start_position, goal_position = generate_positions()
        else:
            start_position, goal_position = generate_random_positions(world, world_bounds=world_bounds, orientation=orientation)
        
        # Initialize the DMPs
        if circular:
            dmp_x = DMP1D(start=start_position[0], goal=goal_position[0], n_basis=3, complexity=complexity)
            dmp_y = DMP1D(start=start_position[1], goal=goal_position[1], n_basis=3, complexity=complexity)
        else:
            dmp_x = DMP1D(start=start_position[0], goal=goal_position[0], n_basis=3, complexity=complexity)
            dmp_y = DMP1D(start=start_position[1], goal=goal_position[1], n_basis=3, complexity=complexity)
        # Initialize the position and velocity of the agent
        start_velocity = np.array([0.0, 0.0])

        # Initialize the simulation with the world and the DMPs
        simulation = Simulation(world, dmp_x, dmp_y, start_position, T=1.0, dt=0.01)

        # Run the simulation and record the positions
        positions, velocities, collisions, collision_position, min_distance = simulation.run()

        # Compute the prediction of the model for the final position
        dmp_params = [dmp_x.start, dmp_y.start, dmp_x.goal, dmp_y.goal, *dmp_x.weights, *dmp_y.weights]
        
        # Prepare inputs for the first timestep
        if random_holes:
            vectors_to_obstacles = np.array([np.array(point) for point in world.centers]).flatten()
            s_t = np.concatenate((start_position,vectors_to_obstacles))
        elif varying_wall:
            wall_length = world.radius * 2 * world.wall_size # Wall length
            wall_shape = [(-wall_length / 2, -world.wall_thickness / 2), (wall_length / 2, -world.wall_thickness / 2),
                        (wall_length / 2, world.wall_thickness / 2), (-wall_length / 2, world.wall_thickness / 2)]
            if not world.wall_present:
                distances_to_edges = np.array([2,2,2,2])
            else:
                distances_to_edges = np.array([np.linalg.norm(start_position - np.array(point)) for point in wall_shape])
            s_t = np.concatenate(
                (start_position,
                np.array([world.wall_present*1.0]),
                distances_to_edges))
        else:
            distances_to_edges = np.array([np.linalg.norm(start_position - np.array(point)) for point in world.centers])
            s_t = np.concatenate((start_position,distances_to_edges))
            
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


def plot_example_trajectories(world, world_bounds, number_of_trajectories=1, random_position=True, orientation=None, complexity=1, circular=False):
    def generate_positions():
        if circular:
            while True:
                start, goal = generate_random_positions(world, world_bounds=world_bounds, orientation=orientation)
                if np.linalg.norm(start) <= world.radius:
                    break
        else:
            start, goal = generate_random_positions(world, world_bounds=world_bounds, orientation=orientation)
        return start, goal

    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    
    if circular:
        theta = np.linspace(0, 2 * np.pi, 100)
        x = world.radius * np.cos(theta)
        y = world.radius * np.sin(theta)
        plt.plot(x, y, 'k--')  # Plot the circle with a dashed line
        plt.xlim([-world.radius - 0.1, world.radius + 0.1])
        plt.ylim([-world.radius - 0.1, world.radius + 0.1])
    else:
        plt.xlim([world_bounds[0] - 0.04, world_bounds[1] + 0.05])
        plt.ylim([world_bounds[2] - 0.05, world_bounds[3] + 0.05])
    
    for _ in range(number_of_trajectories):
        if circular:
            world.reset()
        if random_position:
            start_position, goal_position = generate_positions()
        else:
            start_position, goal_position = generate_random_positions(world, world_bounds=world_bounds, orientation=orientation)
        
        # Initialize the DMPs
        dmp_x = DMP1D(start=start_position[0], goal=goal_position[0], n_basis=2, complexity=complexity)
        dmp_y = DMP1D(start=start_position[1], goal=goal_position[1], n_basis=2, complexity=complexity)
        # Initialize the position and velocity of the agent
        start_velocity = np.array([0.0, 0.0])

        # Initialize the simulation with the world and the DMPs
        simulation = Simulation(world, dmp_x, dmp_y, start_position, T=1.0, dt=0.01)

        # Run the simulation and record the positions
        positions, velocities, collisions, collision_position, min_distance = simulation.run()
        # Plot the trajectories
        simulation.plot(positions, plot_goal=False,color='blue',alpha=0.1)
    
        # Generate random goal position for second DMP
        _, goal_position2 = generate_random_positions(world, world_bounds=world_bounds,orientation = orientation)

        # Initialize the second DMPs with start position as the final position from the first DMP
        dmp_x2 = DMP1D(start=positions[-1][0], goal=goal_position2[0], n_basis=3,complexity=complexity)
        dmp_y2 = DMP1D(start=positions[-1][1], goal=goal_position2[1], n_basis=3,complexity=complexity)

        # Initialize the simulation with the world and the second DMPs
        simulation = Simulation(world, dmp_x2, dmp_y2, positions[-1], T=1.0, dt=0.01)

        # Run the second simulation and record the positions
        positions2, velocities2, collision2,_ ,_= simulation.run()
        simulation.plot(positions2, plot_goal=False,color='red',alpha=0.1)

    plt.show()


def generate_and_plot_trajectories_from_parameters(dmp_params1, dmp_params2, batch_size, batch_s_t, world,world_bounds,n_basis=3,circular=False,plot_only_first=False):
    dmp_params1, dmp_params2 = dmp_params1.detach().numpy(), dmp_params2.detach().numpy()
    batch_s_t = batch_s_t.detach().numpy()
    plt.figure(figsize=(3, 2.5))
    
    for b in range(batch_size):
        if circular:
            world.reset()
        dmp_x1 = DMP1D(start=dmp_params1[b][0], goal=dmp_params1[b][2], n_basis=n_basis)
        dmp_y1 = DMP1D(start=dmp_params1[b][1], goal=dmp_params1[b][3], n_basis=n_basis)
        dmp_x1.weights = dmp_params1[b][4:4+n_basis]
        dmp_y1.weights = dmp_params1[b][4+n_basis:4+n_basis+n_basis]
        simulation1 = Simulation(world, dmp_x1, dmp_y1, batch_s_t[b], T=1.0, dt=0.01)
        positions1, velocities1, collision1, collision_pos1, min_distance1 = simulation1.run()

        dmp_x2 = DMP1D(start=dmp_params2[b][0], goal=dmp_params2[b][2], n_basis=n_basis)
        dmp_y2 = DMP1D(start=dmp_params2[b][1], goal=dmp_params2[b][3], n_basis=n_basis)
        dmp_x2.weights = dmp_params2[b][4:4+n_basis]
        dmp_y2.weights = dmp_params2[b][4+n_basis:4+n_basis+n_basis]
        simulation2 = Simulation(world, dmp_x2, dmp_y2, positions1[-1], T=1.0, dt=0.01)
        positions2, velocities2, collision2, collision_pos2, min_distance2 = simulation2.run()

        # Plot the trajectories
        simulation1.plot(positions1, color='blue',plot_goal=False) 
        if not plot_only_first:
            simulation2.plot(positions2, color='red',plot_goal=False) 
    
    plt.xlim([world_bounds[0]-0.1, world_bounds[1]+0.1])
    plt.ylim([world_bounds[2]-0.1, world_bounds[3]+0.1])
    plt.show()


def generate_and_plot_trajectories_from_single_parameters(dmp_params, batch_size, batch_s_t, world,world_bounds):
    dmp_params = dmp_params.detach().numpy()
    batch_s_t = batch_s_t.detach().numpy()
    plt.figure(figsize=(3, 2.5))
    
    for b in range(batch_size):
        dmp_x1 = DMP1D(start=dmp_params[b][0], goal=dmp_params[b][2], n_basis=3)
        dmp_y1 = DMP1D(start=dmp_params[b][1], goal=dmp_params[b][3], n_basis=3)
        dmp_x1.weights = dmp_params[b][4:10]
        dmp_y1.weights = dmp_params[b][10:16]
        simulation1 = Simulation(world, dmp_x1, dmp_y1, batch_s_t[b], T=1.0, dt=0.01)
        positions1, velocities1, collision1, collision_pos1, min_distance1 = simulation1.run()


        # Plot the trajectories
        simulation1.plot(positions1, color='blue',plot_goal=False) 
    
    plt.xlim([world_bounds[0]-0.1, world_bounds[1]+0.1])
    plt.ylim([world_bounds[2]-0.1, world_bounds[3]+0.1])
    plt.show()


def plot_example_trajectories_level_2(world,world_bounds,plot_goal = True, num_trajectories=10, orientation = None):

    # Generate random start and goal positions
    start_position, goal_position1 = generate_random_positions(world, world_bounds=world_bounds)

    for _ in range(num_trajectories):
        start_position, goal_position = generate_random_positions(world, world_bounds=world_bounds,orientation = orientation)

        # Initialize the first DMPs
        dmp_x1 = DMP1D(start=start_position[0], goal=goal_position[0], n_basis=3,complexity = 1)
        dmp_y1 = DMP1D(start=start_position[1], goal=goal_position[1], n_basis=3,complexity = 1)

        # Initialize the position and velocity of the agent
        start_velocity = np.array([0.0, 0.0])

        # Initialize the simulation with the world and the first DMPs
        simulation = Simulation(world, dmp_x1, dmp_y1, start_position, T=1.0, dt=0.01)

        # Run the first simulation and record the positions
        positions1, velocities1, _ , _ = simulation.run()

        simulation.plot(positions1, color='blue', plot_goal=plot_goal)   # First DMP trajectory in blue

        # Generate random goal position for second DMP
        _, goal_position2 = generate_random_positions(world, world_bounds=world_bounds)

        # Initialize the second DMPs with start position as the final position from the first DMP
        dmp_x2 = DMP1D(start=positions1[-1][0], goal=goal_position2[0], n_basis=3,complexity = 1)
        dmp_y2 = DMP1D(start=positions1[-1][1], goal=goal_position2[1], n_basis=3,complexity = 1)

        # Initialize the simulation with the world and the second DMPs
        simulation = Simulation(world, dmp_x2, dmp_y2, positions1[-1], T=1.0, dt=0.01)

        # Run the second simulation and record the positions
        positions2, velocities2, collisions, _ = simulation.run()

        # Plot the combined trajectories
        simulation.plot(positions2, color='red',plot_goal=plot_goal)    # Second DMP trajectory in red
        plt.xlim([world_bounds[0]-0.04,world_bounds[1]+0.05])
        plt.ylim([world_bounds[2]-0.05,world_bounds[3]+0.05])

    plt.show()



def plot_2_consecutive_trajectories(world,world_bounds,net,net2):

    # Generate random start and goal positions
    start_position, goal_position1 = generate_random_positions(world, world_bounds=world_bounds)

    # Setting the number of trajectories
    num_trajectories = 1

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Color map to distinguish different trajectories
    colors = plt.cm.jet(np.linspace(0,1,num_trajectories))

        
    for i in range(num_trajectories):

        # Generate random goal position for first DMP
        _, goal_position1 = generate_random_positions(world, world_bounds=world_bounds)

        # Initialize the first DMPs
        dmp_x1 = DMP1D(start=start_position[0], goal=goal_position1[0], n_basis=3)
        dmp_y1 = DMP1D(start=start_position[1], goal=goal_position1[1], n_basis=3)

        # Initialize the position and velocity of the agent
        start_velocity = np.array([0.0, 0.0])

        # Initialize the simulation with the world and the first DMPs
        simulation = Simulation(world, dmp_x1, dmp_y1, start_position, T=1.0, dt=0.01)

        # Run the first simulation and record the positions
        positions1, velocities1, collisions = simulation.run()

        # Generate random goal position for second DMP
        _, goal_position2 = generate_random_positions(world, world_bounds=world_bounds)

        # Initialize the second DMPs with start position as the final position from the first DMP
        dmp_x2 = DMP1D(start=positions1[-1][0], goal=goal_position2[0], n_basis=3)
        dmp_y2 = DMP1D(start=positions1[-1][1], goal=goal_position2[1], n_basis=3)

        # Initialize the simulation with the world and the second DMPs
        simulation = Simulation(world, dmp_x2, dmp_y2, positions1[-1], T=1.0, dt=0.01)

        # Run the second simulation and record the positions
        positions2, velocities2 = simulation.run()

        # Compute the prediction of the model for the final position
        dmp_params1 = [dmp_x1.start, dmp_y1.start, dmp_x1.goal, dmp_y1.goal, *dmp_x1.weights, *dmp_y1.weights]
        dmp_params2 = [dmp_x2.start, dmp_y2.start, dmp_x2.goal, dmp_y2.goal, *dmp_x2.weights, *dmp_y2.weights]
        dmp_params = dmp_params1 + dmp_params2
        prediction1 = net(torch.tensor([np.concatenate((start_position, dmp_params1))], dtype=torch.float32))
        prediction2 = net2(torch.tensor([np.concatenate((start_position, dmp_params))], dtype=torch.float32))


        # Add obstacles to the plot
        for body, obstacle in world.obstacles:
            vertices = np.array([body.local_to_world(v) for v in obstacle.get_vertices()])
            vertices = np.vstack((vertices, vertices[0]))  # Close the polygon by adding the first point at the end
            plt.plot(*vertices.T, color='black')
            plt.fill(*vertices.T, color='grey', alpha=0.5)  # Fill the polygon to make the obstacle solid


        # Plot the combined trajectories
        # Plot the actual final position with a cross
        ax.scatter(positions1[0, 0], positions1[0, 1], color='tab:blue', marker='.')
        ax.scatter(positions1[-1, 0], positions1[-1, 1], color='tab:blue', marker='o')

        ax.scatter(prediction1[0][0].item(), prediction1[0][1].item(), color='tab:blue', marker='*')

        ax.plot(positions1[:, 0], positions1[:, 1], color=colors[i])   # First DMP trajectory
        ax.plot(positions2[:, 0], positions2[:, 1], color=colors[i])    # Second DMP trajectory

        # Plot the actual final position with a cross
        ax.scatter(positions2[-1, 0], positions2[-1, 1], color='tab:orange', marker='x')

        # Plot the model's predicted final position with a circle
        ax.scatter(prediction2[0][0].item(), prediction2[0][1].item(), color='tab:orange', marker='*')

        # Create the legend
        labels = ['Start Position', 'End Position of First DMP', 'Predicted End Position of First DMP', 
                'Actual End Position of Trajectory', 'Predicted End Position of Trajectory']
        handles = [plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='tab:blue', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=10),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='tab:blue', markersize=10),
                plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='tab:orange', markersize=10),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='tab:orange', markersize=10)]
        plt.legend(handles, labels, loc='upper left')


    ax.set_title('Predicted vs Actual Final Positions')
    plt.show()


def plot_reward_function(world,world_bounds, target_goal1, target_goal2):
    def reward(x,y):
        dist = np.mean(np.abs((np.array([x, y]) - target_goal1)))
        return -dist**0.3

    X, Y = np.meshgrid(np.linspace(0.1, 0.9, 100), np.linspace(0.1, 0.9, 100))
    Z = np.zeros_like(X)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[j, i] = reward(X[i, j], Y[i, j])  # Here we need to swap i and j for proper indexing

    plt.imshow(Z, extent=(0.1, 0.9, 0.1, 0.9), origin='lower', cmap='RdYlGn')
    plt.colorbar()
    start_position, goal_position = generate_random_positions(world, world_bounds=world_bounds)

    for _ in range(1):
        start_position, goal_position = generate_random_positions(world, world_bounds=world_bounds)
        # Initialize the DMPs
        dmp_x = DMP1D(start=start_position[0], goal=goal_position[0], n_basis=3)
        dmp_y = DMP1D(start=start_position[1], goal=goal_position[1], n_basis=3)
        # Initialize the position and velocity of the agent
        start_velocity = np.array([0.0, 0.0])

        # Initialize the simulation with the world and the DMPs
        simulation = Simulation(world, dmp_x, dmp_y, start_position, T=1.0, dt=0.01)

        # Run the simulation and record the positions
        positions, velocities, collisions = simulation.run()

        # Plot the trajectories
        simulation.plot(positions)
    plt.show()

# target_goal1 = np.array([0.6, 0.4])
# target_goal2 = np.array([0.4, 0.6])
# plot_reward_function(world, world_bounds, target_goal1, target_goal2)


