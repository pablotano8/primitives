import numpy as np
import matplotlib.pyplot as plt

import torch

from dmps import DMP1D, Simulation

from continuous_nav_envs import generate_random_positions, is_inside_obstacle
from mpl_toolkits.mplot3d import Axes3D


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
        random_holes=False,
        trip_wire=None):
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
    
    # Draw trip wires if enabled
    if trip_wire is not None:
        # For backward compatibility, convert boolean True to default trip wires
        if isinstance(trip_wire, bool) and trip_wire:
            trip_wire = [[(-1, -0.25), (-0.25, -0.25)], [(0.25, -0.25), (1, -0.25)]]
        
        # Draw each trip wire
        if not isinstance(trip_wire, bool):  # Skip if it's False
            for wire in trip_wire:
                p1, p2 = wire
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', alpha=0.7)
    
    for _ in range(number_of_trajectories):
        if circular:
            world.reset()
        if random_position:
            start_position, goal_position = generate_positions()
        else:
            start_position, goal_position = generate_random_positions(world, world_bounds=world_bounds, orientation=orientation)
        
        # Initialize the DMPs with complexity scaling
        # Calculate scaling factor based on distance between start and goal positions
        complexity_scaled = complexity * ((np.abs(start_position[0] - goal_position[0]) + np.abs(start_position[1] - goal_position[1])) / 2)
        dmp_x = DMP1D(start=start_position[0], goal=goal_position[0], n_basis=3, complexity=complexity_scaled)
        dmp_y = DMP1D(start=start_position[1], goal=goal_position[1], n_basis=3, complexity=complexity_scaled)
        
        # Initialize the position and velocity of the agent
        start_velocity = np.array([0.0, 0.0])

        # Initialize the simulation with the world and the DMPs
        simulation = Simulation(world, dmp_x, dmp_y, start_position, T=1.0, dt=0.01)

        # Run the simulation and record the positions
        positions, velocities, collisions, collision_position, min_distance = simulation.run()
        
        # Check for trip wire crossings if trip_wire is not None
        if trip_wire is not None:
            # For backward compatibility, convert boolean True to default trip wires
            if isinstance(trip_wire, bool) and trip_wire:
                trip_wire = [[(-1, -0.25), (-0.25, -0.25)], [(0.25, -0.25), (1, -0.25)]]
            
            # Process the trajectory
            if not isinstance(trip_wire, bool):  # Skip if it's False
                for wire in trip_wire:
                    # Each wire is defined by two points [(x1,y1), (x2,y2)]
                    p1, p2 = wire
                    
                    # Check each segment of the trajectory
                    for j in range(1, len(positions)):
                        current_x, current_y = positions[j]
                        prev_x, prev_y = positions[j-1]
                        
                        # Check if the trajectory segment intersects the trip wire
                        s1_x = current_x - prev_x
                        s1_y = current_y - prev_y
                        s2_x = p2[0] - p1[0]
                        s2_y = p2[1] - p1[1]
                        
                        denom = (-s2_x * s1_y + s1_x * s2_y)
                        if denom != 0:  # Non-parallel lines
                            s = (-s1_y * (prev_x - p1[0]) + s1_x * (prev_y - p1[1])) / denom
                            t = (s2_x * (prev_y - p1[1]) - s2_y * (prev_x - p1[0])) / denom
                            
                            if 0 <= s <= 1 and 0 <= t <= 1:  # Intersection found
                                # Calculate which side of the wire each point is on
                                wire_dir_x = p2[0] - p1[0]
                                wire_dir_y = p2[1] - p1[1]
                                
                                # Calculate normal vector to the wire (pointing "up")
                                normal_x = -wire_dir_y
                                normal_y = wire_dir_x
                                
                                # Make sure normal points "up" (positive y direction on average)
                                if normal_y < 0:
                                    normal_x = -normal_x
                                    normal_y = -normal_y
                                
                                # Check which side each point is on
                                prev_side = (prev_x - p1[0]) * normal_x + (prev_y - p1[1]) * normal_y
                                current_side = (current_x - p1[0]) * normal_x + (current_y - p1[1]) * normal_y
                                
                                # If crossing from below to above, terminate trajectory
                                if prev_side < 0 and current_side > 0:
                                    # Terminate trajectory at previous point
                                    positions = positions[:j]
                                    collisions = np.array([False])
                                    break
                
        
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


def plot_example_trajectories(
        world, world_bounds, number_of_trajectories=1, random_position=True, orientation=None, 
        complexity=1, circular=False, n_basis=3, trip_wire=None, start_position=np.array([0.2, -0.85]), goal_position=np.array([0.1, -0.0])):
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
    
    # Draw trip wires if enabled
    if trip_wire is not None:
        # For backward compatibility, convert boolean True to default trip wires
        if isinstance(trip_wire, bool) and trip_wire:
            trip_wire = [[(-1, -0.25), (-0.25, -0.25)], [(0.25, -0.25), (1, -0.25)]]
        
        # Draw each trip wire
        if not isinstance(trip_wire, bool):  # Skip if it's False
            for wire in trip_wire:
                p1, p2 = wire
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', alpha=0.7)
    
    for _ in range(number_of_trajectories):
        if circular:
            world.reset()
        if random_position:
            start_position, goal_position = generate_positions()
        else:
            start_position = start_position
            goal_position = goal_position

        # Initialize the DMPs
        complexity_scaled = complexity * ((np.abs(start_position[0] - goal_position[0]) + np.abs(start_position[1] - goal_position[1])) / 2)
        dmp_x = DMP1D(start=start_position[0], goal=goal_position[0], n_basis=n_basis, complexity=complexity_scaled)
        dmp_y = DMP1D(start=start_position[1], goal=goal_position[1], n_basis=n_basis, complexity=complexity_scaled)
        # Initialize the position and velocity of the agent
        start_velocity = np.array([0.0, 0.0])

        # Initialize the simulation with the world and the DMPs
        simulation = Simulation(world, dmp_x, dmp_y, start_position, T=1.0, dt=0.01)

        # Run the simulation and record the positions
        positions, velocities, collisions, collision_position, min_distance = simulation.run()
        
        # Track if first trajectory was cut
        first_trajectory_cut = False
        
        # Check for trip wire crossings if trip_wire is not None
        if trip_wire is not None:
            # For backward compatibility, convert boolean True to default trip wires
            if isinstance(trip_wire, bool) and trip_wire:
                trip_wire = [[(-1, -0.25), (-0.25, -0.25)], [(0.25, -0.25), (1, -0.25)]]
            
            # Process the trajectory
            if not isinstance(trip_wire, bool):  # Skip if it's False
                for wire in trip_wire:
                    # Each wire is defined by two points [(x1,y1), (x2,y2)]
                    p1, p2 = wire
                    
                    # Check each segment of the trajectory
                    for j in range(1, len(positions)):
                        current_x, current_y = positions[j]
                        prev_x, prev_y = positions[j-1]
                        
                        # Check if the trajectory segment intersects the trip wire
                        s1_x = current_x - prev_x
                        s1_y = current_y - prev_y
                        s2_x = p2[0] - p1[0]
                        s2_y = p2[1] - p1[1]
                        
                        denom = (-s2_x * s1_y + s1_x * s2_y)
                        if denom != 0:  # Non-parallel lines
                            s = (-s1_y * (prev_x - p1[0]) + s1_x * (prev_y - p1[1])) / denom
                            t = (s2_x * (prev_y - p1[1]) - s2_y * (prev_x - p1[0])) / denom
                            
                            if 0 <= s <= 1 and 0 <= t <= 1:  # Intersection found
                                # Calculate which side of the wire each point is on
                                wire_dir_x = p2[0] - p1[0]
                                wire_dir_y = p2[1] - p1[1]
                                
                                # Calculate normal vector to the wire (pointing "up")
                                normal_x = -wire_dir_y
                                normal_y = wire_dir_x
                                
                                # Make sure normal points "up" (positive y direction on average)
                                if normal_y < 0:
                                    normal_x = -normal_x
                                    normal_y = -normal_y
                                
                                # Check which side each point is on
                                prev_side = (prev_x - p1[0]) * normal_x + (prev_y - p1[1]) * normal_y
                                current_side = (current_x - p1[0]) * normal_x + (current_y - p1[1]) * normal_y
                                
                                # If crossing from below to above, terminate trajectory
                                if prev_side < 0 and current_side > 0:
                                    # Terminate trajectory at previous point
                                    positions = positions[:j]
                                    collisions = np.array([False])
                                    first_trajectory_cut = True
                                    break
                    if first_trajectory_cut:
                        break
                
        
        # Plot the trajectories
        simulation.plot(positions, plot_goal=False, color='blue', alpha=0.1)

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
        gradual=False,
        trip_wire=None):
    
    dmp_params1, dmp_params2 = dmp_params1.detach().numpy(), dmp_params2.detach().numpy()
    batch_s_t = batch_s_t.detach().numpy()
    plt.figure(figsize=(3, 2.5))
    
    # Draw trip wires if enabled
    if trip_wire is not None:
        # For backward compatibility, convert boolean True to default trip wires
        if isinstance(trip_wire, bool) and trip_wire:
            trip_wire = [[(-1, -0.25), (-0.25, -0.25)], [(0.25, -0.25), (1, -0.25)]]
        
        # Draw each trip wire
        if not isinstance(trip_wire, bool):  # Skip if it's False
            for wire in trip_wire:
                p1, p2 = wire
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', alpha=0.7)
    
    for b in range(batch_size):
        if gradual:
            alpha=b/batch_size * 0.7 + 0.3
        else:
            alpha=1
        if circular:
            world.reset()
            
        # Calculate scaling factor based on distance between start and goal positions
        start_x, start_y = dmp_params1[b][0], dmp_params1[b][1]
        goal_x, goal_y = dmp_params1[b][2], dmp_params1[b][3]
        complexity_scale = (np.abs(start_x - goal_x) + np.abs(start_y - goal_y)) / 2
            
        dmp_x1 = DMP1D(start=dmp_params1[b][0], goal=dmp_params1[b][2], n_basis=n_basis)
        dmp_y1 = DMP1D(start=dmp_params1[b][1], goal=dmp_params1[b][3], n_basis=n_basis)
        
        # Scale the weights by the complexity scale factor
        dmp_x1.weights = dmp_params1[b][4:4+n_basis] * complexity_scale
        dmp_y1.weights = dmp_params1[b][4+n_basis:4+n_basis+n_basis] * complexity_scale
        
        simulation1 = Simulation(world, dmp_x1, dmp_y1, batch_s_t[b], T=1.0, dt=0.01)
        positions1, velocities1, collision1, collision_pos1, min_distance1 = simulation1.run()
        
        # Track if first trajectory was cut
        first_trajectory_cut = False
        
        # Check for trip wire crossings if trip_wire is not None
        if trip_wire is not None:
            # For backward compatibility, convert boolean True to default trip wires
            if isinstance(trip_wire, bool) and trip_wire:
                trip_wire = [[(-1, -0.25), (-0.25, -0.25)], [(0.25, -0.25), (1, -0.25)]]
            
            # Process the first trajectory
            if not isinstance(trip_wire, bool):  # Skip if it's False
                for wire in trip_wire:
                    # Each wire is defined by two points [(x1,y1), (x2,y2)]
                    p1, p2 = wire
                    
                    # Check each segment of the trajectory
                    for j in range(1, len(positions1)):
                        current_x, current_y = positions1[j]
                        prev_x, prev_y = positions1[j-1]
                        
                        # Check if the trajectory segment intersects the trip wire
                        s1_x = current_x - prev_x
                        s1_y = current_y - prev_y
                        s2_x = p2[0] - p1[0]
                        s2_y = p2[1] - p1[1]
                        
                        denom = (-s2_x * s1_y + s1_x * s2_y)
                        if denom != 0:  # Non-parallel lines
                            s = (-s1_y * (prev_x - p1[0]) + s1_x * (prev_y - p1[1])) / denom
                            t = (s2_x * (prev_y - p1[1]) - s2_y * (prev_x - p1[0])) / denom
                            
                            if 0 <= s <= 1 and 0 <= t <= 1:  # Intersection found
                                # Calculate which side of the wire each point is on
                                wire_dir_x = p2[0] - p1[0]
                                wire_dir_y = p2[1] - p1[1]
                                
                                # Calculate normal vector to the wire (pointing "up")
                                normal_x = -wire_dir_y
                                normal_y = wire_dir_x
                                
                                # Make sure normal points "up" (positive y direction on average)
                                if normal_y < 0:
                                    normal_x = -normal_x
                                    normal_y = -normal_y
                                
                                # Check which side each point is on
                                prev_side = (prev_x - p1[0]) * normal_x + (prev_y - p1[1]) * normal_y
                                current_side = (current_x - p1[0]) * normal_x + (current_y - p1[1]) * normal_y
                                
                                # If crossing from below to above, terminate trajectory
                                if prev_side < 0 and current_side > 0:
                                    # Terminate trajectory at previous point
                                    positions1 = positions1[:j]
                                    collision1 = np.array([False])
                                    first_trajectory_cut = True
                                    break
                    if first_trajectory_cut:
                        break

        # Plot the first trajectory
        simulation1.plot(positions1, color='blue', plot_goal=False, alpha=alpha)
        
        if not plot_only_first:
            # Calculate scaling factor for second trajectory
            start_x, start_y = positions1[-1][0], positions1[-1][1]
            goal_x, goal_y = dmp_params2[b][2], dmp_params2[b][3]
            complexity_scale = (np.abs(start_x - goal_x) + np.abs(start_y - goal_y)) / 2
            
            dmp_x2 = DMP1D(start=dmp_params2[b][0], goal=dmp_params2[b][2], n_basis=n_basis)
            dmp_y2 = DMP1D(start=dmp_params2[b][1], goal=dmp_params2[b][3], n_basis=n_basis)
            
            # Scale the weights by the complexity scale factor
            dmp_x2.weights = dmp_params2[b][4:4+n_basis] * complexity_scale
            dmp_y2.weights = dmp_params2[b][4+n_basis:4+n_basis+n_basis] * complexity_scale
            
            simulation2 = Simulation(world, dmp_x2, dmp_y2, positions1[-1], T=1.0, dt=0.01)
            positions2, velocities2, collision2, collision_pos2, min_distance2 = simulation2.run()
            
            # Check for trip wire crossings for the second trajectory, but only if first wasn't cut
            if trip_wire is not None and not first_trajectory_cut:
                # For backward compatibility, convert boolean True to default trip wires
                if isinstance(trip_wire, bool) and trip_wire:
                    trip_wire = [[(-1, -0.25), (-0.25, -0.25)], [(0.25, -0.25), (1, -0.25)]]
                
                # Process the second trajectory
                if not isinstance(trip_wire, bool):  # Skip if it's False
                    for wire in trip_wire:
                        # Each wire is defined by two points [(x1,y1), (x2,y2)]
                        p1, p2 = wire
                        
                        # Check each segment of the trajectory
                        for j in range(1, len(positions2)):
                            current_x, current_y = positions2[j]
                            prev_x, prev_y = positions2[j-1]
                            
                            # Check if the trajectory segment intersects the trip wire
                            s1_x = current_x - prev_x
                            s1_y = current_y - prev_y
                            s2_x = p2[0] - p1[0]
                            s2_y = p2[1] - p1[1]
                            
                            denom = (-s2_x * s1_y + s1_x * s2_y)
                            if denom != 0:  # Non-parallel lines
                                s = (-s1_y * (prev_x - p1[0]) + s1_x * (prev_y - p1[1])) / denom
                                t = (s2_x * (prev_y - p1[1]) - s2_y * (prev_x - p1[0])) / denom
                                
                                if 0 <= s <= 1 and 0 <= t <= 1:  # Intersection found
                                    # Calculate which side of the wire each point is on
                                    wire_dir_x = p2[0] - p1[0]
                                    wire_dir_y = p2[1] - p1[1]
                                    
                                    # Calculate normal vector to the wire (pointing "up")
                                    normal_x = -wire_dir_y
                                    normal_y = wire_dir_x
                                    
                                    # Make sure normal points "up" (positive y direction on average)
                                    if normal_y < 0:
                                        normal_x = -normal_x
                                        normal_y = -normal_y
                                    
                                    # Check which side each point is on
                                    prev_side = (prev_x - p1[0]) * normal_x + (prev_y - p1[1]) * normal_y
                                    current_side = (current_x - p1[0]) * normal_x + (current_y - p1[1]) * normal_y
                                    
                                    # If crossing from below to above, terminate trajectory
                                    if prev_side < 0 and current_side > 0:
                                        # Terminate trajectory at previous point
                                        positions2 = positions2[:j]
                                        collision2 = np.array([False])
                                        break
            
            # Plot the second trajectory
            simulation2.plot(positions2, color='red', plot_goal=False, alpha=alpha)
    
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


def plot_example_trajectories_level_2(world,world_bounds,plot_goal = True, num_trajectories=10, orientation = None,n_basis=3):

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

def plot_dmp_weight_space(world, world_bounds, start_position, goal_position, complexity=1, n_basis=3, number_of_examples_per_weight=3, number_to_plot = 500):
    weight_range = (-complexity, complexity)
    plt.figure(figsize=(10, 8))
    
    # Warning about computational complexity
    total_trajectories = number_of_examples_per_weight**(2*n_basis)
    print(f"This will generate {total_trajectories} trajectories. Plotting {number_to_plot}")

    
    # Create weight sampling points for each basis function
    weights_values = np.linspace(weight_range[0], weight_range[1], number_of_examples_per_weight)
    
    # Add obstacles to the plot
    for body, obstacle in world.obstacles:
        vertices = np.array([body.local_to_world(v) for v in obstacle.get_vertices()])
        vertices = np.vstack((vertices, vertices[0]))  # Close the polygon
        plt.plot(*vertices.T, color='black')
        plt.fill(*vertices.T, color='grey', alpha=0.5)
    
    # Plot start and goal positions
    plt.scatter(start_position[0], start_position[1], color='black', s=50, marker='o', label='Start')
    plt.scatter(goal_position[0], goal_position[1], color='blue', s=100, marker='x', label='Goal',alpha=0.3)
    
    # For simple cases, we can use a full meshgrid to explore all combinations
    if n_basis == 1:
        # Simple case with only one basis function per dimension
        for wx in weights_values:
            for wy in weights_values:
                dmp_x = DMP1D(start=start_position[0], goal=goal_position[0], n_basis=n_basis)
                dmp_y = DMP1D(start=start_position[1], goal=goal_position[1], n_basis=n_basis)
                
                dmp_x.weights = np.array([wx])
                dmp_y.weights = np.array([wy])
                
                simulation = Simulation(world, dmp_x, dmp_y, start_position, T=1.0, dt=0.01)
                positions, _, _, _, _ = simulation.run()
                
                # Plot trajectory
                plt.plot(positions[:, 0], positions[:, 1], alpha=0.1, linewidth=1.5,color='blue')
    elif n_basis == 2:
        # For n_basis=2, we can still do a full enumeration with 4D parameter space
        for wx0 in weights_values:
            for wx1 in weights_values:
                for wy0 in weights_values:
                    for wy1 in weights_values:
                        dmp_x = DMP1D(start=start_position[0], goal=goal_position[0], n_basis=n_basis)
                        dmp_y = DMP1D(start=start_position[1], goal=goal_position[1], n_basis=n_basis)
                        
                        dmp_x.weights = np.array([wx0, wx1])
                        dmp_y.weights = np.array([wy0, wy1])
                        
                        simulation = Simulation(world, dmp_x, dmp_y, start_position, T=1.0, dt=0.01)
                        positions, _, _, _, _ = simulation.run()
                        
                        # Plot trajectory
                        plt.plot(positions[:, 0], positions[:, 1], alpha=0.1, linewidth=1.0,color='blue')
    else:
        # For higher dimensions, use random sampling
        num_samples = number_to_plot
        print(f"Using {num_samples} random samples to visualize the weight space")
        
        for _ in range(num_samples):
            dmp_x = DMP1D(start=start_position[0], goal=goal_position[0], n_basis=n_basis)
            dmp_y = DMP1D(start=start_position[1], goal=goal_position[1], n_basis=n_basis)
            
            # Randomly sample weights from the grid values
            dmp_x.weights = np.random.choice(weights_values, size=n_basis)
            dmp_y.weights = np.random.choice(weights_values, size=n_basis)
            
            simulation = Simulation(world, dmp_x, dmp_y, start_position, T=1.0, dt=0.01)
            positions, _, _, _, _ = simulation.run()
            
            # Plot trajectory with faint line
            plt.plot(positions[:, 0], positions[:, 1], alpha=0.1, linewidth=0.8,color='blue')
    
    plt.xlim([world_bounds[0] - 0.1, world_bounds[1] + 0.1])
    plt.ylim([world_bounds[2] - 0.1, world_bounds[3] + 0.1])
    
    if n_basis <= 2:
        plt.title(f'DMP Trajectory Space: {total_trajectories} trajectories with {n_basis} basis functions')
    else:
        plt.title(f'DMP Trajectory Space: {num_samples} sampled trajectories with {n_basis} basis functions')
    plt.show()
    plt.show()
