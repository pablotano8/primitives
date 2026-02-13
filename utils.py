import numpy as np
import matplotlib.pyplot as plt
import torch
from dmps import DMP1D,Simulation
import numpy as np
import torch

def sample_random_dmp_params(batch_s_t, N=3000):
    dmp_params = []
    for s_t in batch_s_t:
        for _ in range(N):
            dmp_params.append([
                s_t[0].detach().item(),
                s_t[1].detach().item(),
                0.1 + np.random.beta(0.5, 0.5) * (0.9 - 0.1),
                0.1 + np.random.beta(0.5, 0.5) * (0.9 - 0.1),
                *np.random.randn(6),
                *np.random.randn(6)])
    return torch.Tensor(dmp_params).view(len(batch_s_t), N, -1)


def generate_trajectories_from_dmp_params(dmp_params1, dmp_params2, batch_size, batch_s_t, world, circular=False, n_basis=3, trip_wire=None):
        
        if circular:
             world.reset()
        dmp_params1, dmp_params2 = dmp_params1.numpy(), dmp_params2.numpy()
        
        # Convert batch_s_t to numpy if it's a tensor
        if isinstance(batch_s_t, torch.Tensor):
            batch_s_t = batch_s_t.numpy()
            
        # Initialize and run DMP simulations for each item in the batch
        actual_final_position1, actual_final_position2, collision_info = [], [], []
        for b in range(batch_size):
            # Calculate scaling factor based on distance between start and goal positions
            # This is consistent with the complexity_scaled calculation in generate_data
            start_x, start_y = batch_s_t[b][0], batch_s_t[b][1]
            goal_x, goal_y = dmp_params1[b][2], dmp_params1[b][3]
            complexity_scale = (np.abs(start_x - goal_x) + np.abs(start_y - goal_y)) / 2
            
            dmp_x1 = DMP1D(start=batch_s_t[b][0], goal=dmp_params1[b][2], n_basis=n_basis)
            dmp_y1 = DMP1D(start=batch_s_t[b][1], goal=dmp_params1[b][3], n_basis=n_basis)
            
            # Scale the weights by the complexity scale factor
            dmp_x1.weights = dmp_params1[b][4:4+n_basis] * complexity_scale
            dmp_y1.weights = dmp_params1[b][4+n_basis:4+n_basis+n_basis] * complexity_scale
            
            simulation1 = Simulation(world, dmp_x1, dmp_y1, batch_s_t[b], T=1.0, dt=0.01)
            positions1, velocities1, collision1, collision_pos1, min_distance1 = simulation1.run()
            
            # Track if the first trajectory was cut by a trip wire
            first_trajectory_cut = False
            
            # Handle trip wire crossings if trip_wire is provided
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
                            # Using line segment intersection formula
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
                                    # We use the side with the higher y-value as "above"
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

            # Calculate scaling factor for second trajectory
            start_x, start_y = positions1[-1][0], positions1[-1][1]
            goal_x, goal_y = dmp_params2[b][2], dmp_params2[b][3]
            complexity_scale = (np.abs(start_x - goal_x) + np.abs(start_y - goal_y)) / 2
            
            dmp_x2 = DMP1D(start=positions1[-1][0], goal=dmp_params2[b][2], n_basis=n_basis)
            dmp_y2 = DMP1D(start=positions1[-1][1], goal=dmp_params2[b][3], n_basis=n_basis)
            
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

            actual_final_position1.append(np.array(positions1[-1]))
            actual_final_position2.append(np.array(positions2[-1]))
            collision_info.append({
                'collision1': collision1,
                'collision2': collision2,
                'collision_pos': collision_pos1,
                'collision_pos2': collision_pos2,
                'min_distance1': min_distance1,
                'min_distance2': min_distance2})
        return actual_final_position1, actual_final_position2, collision_info


def plot_value_function(net_value):
    net_value.eval()
    # Create a grid of points in the input space
    x = np.linspace(0.1, 0.9, 100)
    y = np.linspace(0.1, 0.9, 100)
    X, Y = np.meshgrid(x, y)
    grid = np.vstack([X.ravel(), Y.ravel()]).T

    # Convert grid to torch tensor
    grid_tensor = torch.tensor(grid, dtype=torch.float)

    # Evaluate the value function
    with torch.no_grad():
        values = net_value(grid_tensor).numpy()

    # Reshape the values for plotting
    Z = values.reshape(X.shape)

    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')

    # Labels and titles
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Value')
    ax.set_title('Value Function Surface')

    # Show the plot
    plt.show()
