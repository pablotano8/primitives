import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from plot_trajectories import plot_example_trajectories, plot_predicted_vs_actual_trajectories, generate_and_plot_trajectories_from_parameters
from utils import generate_trajectories_from_dmp_params
from continuous_nav_envs import CircularWorld
from continuous_nav_envs import generate_random_positions
from dmps import DMP1D,Simulation
import matplotlib.pyplot as plt


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
    
def split_list(data, n_chunks):
    """Split a list into n approximately equal chunks."""
    chunk_size = len(data) // n_chunks
    remainder = len(data) % n_chunks
    
    result = []
    start = 0
    for i in range(n_chunks):
        # Add an extra element to the first 'remainder' chunks
        chunk_end = start + chunk_size + (1 if i < remainder else 0)
        result.append(data[start:chunk_end])
        start = chunk_end
    
    return result


def create_valid_data_subset(valid_data, target_position=[0, -0.75], distance_threshold=0.05):
    target_position = torch.tensor(target_position, dtype=torch.float32)
    subset = []

    for data in valid_data:
        s_t = data[0][:2]
        distance = torch.norm(s_t - target_position)
        if distance <= distance_threshold:
            subset.append(data)
    
    return subset

def create_data_without_threat_zone(valid_data, target_position=[0, -0.75], distance_threshold=0.05):
    target_position = torch.tensor(target_position, dtype=torch.float32)
    subset = []

    for data in valid_data:
        s_t = data[0][:2]
        distance = torch.norm(s_t - target_position)
        if distance >= distance_threshold:
            subset.append(data)
    
    return subset

def create_data_with_wall_blocking_threat_zone(valid_data, height_threshold=-0.6):
    height_threshold = torch.tensor(height_threshold, dtype=torch.float32)
    subset = []

    for data in valid_data:
        height = data[0][1]
        if height >= height_threshold:
            subset.append(data)
    
    return subset

def create_data_without_edge_to_home(valid_data, target_position1=[0, -0.75], target_position2=[0, -0.75], distance_threshold=0.05):
    target_position1 = torch.tensor(target_position1, dtype=torch.float32)
    target_position2 = torch.tensor(target_position2, dtype=torch.float32)
    subset = []

    for data in valid_data:
        s_t = data[0][:2]
        distance1 = torch.norm(s_t - target_position1)
        distance2 = torch.norm(s_t - target_position2)
        if distance1 >= distance_threshold and distance2 >= distance_threshold:
            subset.append(data)
    return subset



def generate_initial_states(
        world,
        world_bounds,
        number_of_states=10000):
    # Generate training data
    data = []
    for i in range(number_of_states):  # Number of trajectories to generate
        
        world.reset()

        # Generate random start and goal positions for first DMP
        start_position, _ = generate_random_positions(world, world_bounds=world_bounds,orientation = None,circular=True)

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
        
        data.append((s_t, True))

    # Split the data into training and validation sets
    split_idx = int(len(data) * 0.9)  # Use 80% of the data for training
    train_data = data[:split_idx]
    valid_data = data[split_idx:]

    # Shuffle training  data
    random.shuffle(train_data)

    # Convert the data to tensors
    train_data = [(torch.tensor(s_t, dtype=torch.float32),
                torch.tensor(s_t_plus_one, dtype=torch.float32)) for s_t, s_t_plus_one in train_data]

    valid_data = [(torch.tensor(s_t, dtype=torch.float32),
                torch.tensor(s_t_plus_one, dtype=torch.float32)) for s_t, s_t_plus_one in valid_data]
    return train_data, valid_data




def visualize_initial_positions(train_data):
    # Extract initial positions from each data point
    positions = []
    for data_point in train_data:
        # The first element is s_t which contains the initial position
        s_t = data_point[0]
        # The first two elements of s_t are the x,y coordinates
        positions.append(s_t[:2].numpy())
    
    # Convert to numpy array
    positions = np.array(positions)
    
    # If there are more than 2000 points, randomly subsample
    if len(positions) > 10000:
        indices = np.random.choice(len(positions), 10000, replace=False)
        positions = positions[indices]
    
    # Create the scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(positions[:, 0], positions[:, 1], alpha=0.5, s=10)
    # Add a circle to represent the boundary of the world (radius=1)
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
    plt.gca().add_patch(circle)

    plt.show()


def generate_data(
        world,
        world_bounds,
        number_of_trajectories=10000,
        complexity = 1,
        orientation = None,
        varying_wall=False,
        full_state_space = False,
        trip_wire=None):
    # Generate training data
    data = []
    for i in range(number_of_trajectories):  # Number of trajectories to generate
        if varying_wall:
            world_bounds = [-1, 1, -1, 1]
            world = CircularWorld(
                num_obstacles=0,
                max_speed=100,
                radius=1,
                wall_present=np.random.uniform(0,1)<0.8,
                wall_size=np.random.uniform(0.3,1),
                wall_thickness=np.random.uniform(0,0.5))
        
        world.reset()
        if i%1000==0:
            print(f'Generated {i} of {number_of_trajectories} trajectories')
        # Generate random start and goal positions for first DMP
        start_position, _ = generate_random_positions(world, world_bounds=world_bounds,orientation = orientation,circular=True)
        _, goal_position1 = generate_random_positions(world, world_bounds=world_bounds,orientation = orientation)

        if full_state_space:
            wall_length = world.radius * 2 * world.wall_size # Wall length
            wall_shape = [(-wall_length / 2, -world.wall_thickness / 2), (wall_length / 2, -world.wall_thickness / 2),
                        (wall_length / 2, world.wall_thickness / 2), (-wall_length / 2, world.wall_thickness / 2)]
            if not world.wall_present:
                distances_to_edges = np.array([2,2,2,2])
            else:
                distances_to_edges = np.array([np.linalg.norm(start_position - np.array(point)) for point in wall_shape])
                
        # Initialize the first DMPs
        complexity_scaled = complexity * ((np.abs(start_position[0] - goal_position1[0]) + np.abs(start_position[1] - goal_position1[1])) / 2)
        dmp_x1 = DMP1D(start=start_position[0], goal=goal_position1[0], n_basis=3, complexity=complexity_scaled)
        dmp_y1 = DMP1D(start=start_position[1], goal=goal_position1[1], n_basis=3, complexity=complexity_scaled)

        # Initialize the position and velocity of the agent
        start_velocity = np.array([0.0, 0.0])

        # Initialize the simulation with the world and the first DMPs
        simulation = Simulation(world, dmp_x1, dmp_y1, start_position, T=1.0, dt=0.01)

        # Run the first simulation and record the positions
        positions1, velocities1, collision1, _, _ = simulation.run()

        # Generate random goal position for second DMP
        _, goal_position2 = generate_random_positions(world, world_bounds=world_bounds,orientation = orientation)

        # Initialize the second DMPs with start position as the final position from the first DMP
        complexity_scaled = complexity * ((np.abs(positions1[-1][0] - goal_position2[0]) + np.abs(positions1[-1][1] - goal_position2[1])) / 2)
        dmp_x2 = DMP1D(start=positions1[-1][0], goal=goal_position2[0], n_basis=3,complexity=complexity_scaled)
        dmp_y2 = DMP1D(start=positions1[-1][1], goal=goal_position2[1], n_basis=3,complexity=complexity_scaled)

        # Initialize the simulation with the world and the second DMPs
        simulation = Simulation(world, dmp_x2, dmp_y2, positions1[-1], T=1.0, dt=0.01)

        # Run the second simulation and record the positions
        positions2, velocities2, collision2,_ ,_= simulation.run()
        
        # Flag to track if any trajectory crossed a trip wire
        crossed_trip_wire = False
        
        # Check for trip wire crossings if trip_wire is provided
        if trip_wire is not None:
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
                                
                                # If crossing from below to above, mark as crossed
                                if prev_side < 0 and current_side > 0:
                                    crossed_trip_wire = True
                                    break
                    
                    if crossed_trip_wire:
                        break
            
            # Process the second trajectory with the same logic
            if not crossed_trip_wire and not isinstance(trip_wire, bool):  # Skip if it's False or already crossed
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
                                
                                # If crossing from below to above, mark as crossed
                                if prev_side < 0 and current_side > 0:
                                    crossed_trip_wire = True
                                    break
                    
                    if crossed_trip_wire:
                        break

        # Only add the data to the training set if the trajectory didn't cross any trip wire
        if not crossed_trip_wire:
            # Add the data to the training set
            dmp_params1 = [dmp_x1.start, dmp_y1.start, dmp_x1.goal, dmp_y1.goal, *dmp_x1.weights, *dmp_y1.weights]
            dmp_params2 = [dmp_x1.goal, dmp_y1.goal, dmp_x2.goal, dmp_y2.goal, *dmp_x2.weights, *dmp_y2.weights]
            if full_state_space:
                s_t = np.concatenate(
                    (positions1[0],
                    np.array([world.wall_present*1.0]),
                    distances_to_edges))
            else:
                s_t = positions1[0]
            if not world.wall_present:
                collision1, collision2 = np.array([0.0]), np.array([0.0])
            data.append((s_t, True, positions1[-1], positions2[-1], dmp_params1, dmp_params2, collision1*1, collision2*1))

    # Split the data into training and validation sets
    split_idx = int(len(data) * 0.9)  # Use 80% of the data for training
    train_data = data[:split_idx]
    valid_data = data[split_idx:]

    # Shuffle training  data
    random.shuffle(train_data)

    # Convert the data to tensors
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

def train_predictive_net(training_sets,
                         valid_data,
                         net,
                         num_training_sets=10,
                         learning_rate=0.0001,
                         num_epochs=50,
                         batch_size=32,
                         eval_freq=1,
                         weight_collisions=0.3):

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.MSELoss()
    criterion2 = nn.BCEWithLogitsLoss()
    
    average_validation_loss1, average_validation_loss2 = [] , []
    
    for epoch in range(num_epochs):
        # Select one of the training sets
        train_data_epoch = training_sets[epoch % num_training_sets]
        
        # Training
        net.train()
        train_losses = []
        for i in range(0, len(train_data_epoch), batch_size):
            batch = train_data_epoch[i:i + batch_size]
            batch_s_t = torch.stack([item[0] for item in batch])
            batch_final_position1 = torch.stack([item[2] for item in batch])
            batch_final_position2 = torch.stack([item[3] for item in batch])
            batch_dmp_params1 = torch.stack([item[4] for item in batch])
            batch_dmp_params2 = torch.stack([item[5] for item in batch])
            batch_coll1 = torch.stack([item[6] for item in batch]).view(-1)
            batch_coll2 = torch.stack([item[7] for item in batch]).view(-1)

            # Prepare inputs and targets for RNN
            inputs1 = torch.cat((batch_s_t, batch_dmp_params1), dim=-1).unsqueeze(1)  # (batch_size, seq_len=1, input_size)
            padded_state = torch.zeros_like(batch_s_t)
            inputs2 = torch.cat((padded_state, batch_dmp_params2), dim=-1).unsqueeze(1)  # (batch_size, seq_len=1, input_size)

            # Forward pass for both timesteps
            optimizer.zero_grad()
            outputs1, hidden = net(inputs1)
            outputs2, _ = net(inputs2, hidden)
            
            # Separate the outputs into position and length
            pred_pos1 = outputs1.squeeze(1)[:, :2]
            pred_coll1 = outputs1.squeeze(1)[:, 2]
            pred_pos2 = outputs2.squeeze(1)[:, :2]
            pred_coll2 = outputs2.squeeze(1)[:, 2]
            
            # Compute losses
            loss_pos1 = criterion(pred_pos1, batch_final_position1)
            loss_pos2 = criterion(pred_pos2, batch_final_position2)
            loss_coll1 = criterion2(pred_coll1, batch_coll1)
            loss_coll2 = criterion2(pred_coll2, batch_coll2)
            loss = loss_pos1 + loss_pos2 + weight_collisions*(loss_coll1 + loss_coll2)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        
        if epoch % eval_freq == 0:
            # Validation
            net.eval()
            valid_losses_pos1,valid_losses_pos2,valid_losses_coll,dumb_error1,dumb_error2 = [],[] , [] , [],[]
            with torch.no_grad():
                for i in range(0, len(valid_data), batch_size):
                    batch = valid_data[i:i + batch_size]
                    batch_s_t = torch.stack([item[0] for item in batch])
                    batch_final_position1 = torch.stack([item[2] for item in batch])
                    batch_final_position2 = torch.stack([item[3] for item in batch])
                    batch_dmp_params1 = torch.stack([item[4] for item in batch])
                    batch_dmp_params2 = torch.stack([item[5] for item in batch])
                    batch_coll1 = torch.stack([item[6] for item in batch]).view(-1)
                    batch_coll2 = torch.stack([item[7] for item in batch]).view(-1)

                    # Prepare inputs and targets for RNN
                    inputs1 = torch.cat((batch_s_t, batch_dmp_params1), dim=-1).unsqueeze(1)  # (batch_size, seq_len=1, input_size)
                    padded_state = torch.zeros_like(batch_s_t)
                    inputs2 = torch.cat((padded_state, batch_dmp_params2), dim=-1).unsqueeze(1)  # (batch_size, seq_len=1, input_size)

                    # Forward pass for both timesteps
                    outputs1, hidden = net(inputs1)
                    outputs2, _ = net(inputs2, hidden)
                    
                    # Separate the outputs into position and length
                    pred_pos1 = outputs1.squeeze(1)[:, :2]
                    pred_length1 = outputs1.squeeze(1)[:, 2]
                    pred_pos2 = outputs2.squeeze(1)[:, :2]
                    pred_length2 = outputs2.squeeze(1)[:, 2]
                    
                    # Compute losses
                    loss_pos1 = torch.mean(torch.abs(pred_pos1- batch_final_position1))
                    loss_pos2 = torch.mean(torch.abs(pred_pos2 - batch_final_position2))
                    loss_coll1 = criterion2(pred_length1, batch_coll1)
                    loss_coll2 = criterion2(pred_length2, batch_coll2)
                    loss_pos = (loss_pos1 + loss_pos2) / 2  # Average the losses for validation
                    loss_coll = (loss_coll1 + loss_coll2) / 2
                    dumb_error1.append(torch.mean(torch.abs(batch_final_position1- batch_dmp_params1[:,2:4])))
                    dumb_error2.append(torch.mean(torch.abs(batch_final_position2- batch_dmp_params2[:,2:4])))
                    valid_losses_pos1.append(loss_pos1.item())
                    valid_losses_pos2.append(loss_pos2.item())
                    valid_losses_coll.append(loss_coll.item())
            
            average_validation_loss1.append(np.mean(valid_losses_pos1))
            average_validation_loss2.append(np.mean(valid_losses_pos2))
            print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss = {np.mean(train_losses)}, Valid Loss Pos = {np.mean(valid_losses_pos2)}, Valid Loss Col = {np.mean(valid_losses_coll)}, Dumb Error1= {np.mean(dumb_error1)} Dumb Error2= {np.mean(dumb_error2)}')
    
    return average_validation_loss1,average_validation_loss2, net



def optimize_pred_nets_online(train_data_epoch,
                              batch_size,
                              net_goal,
                              net_preds,
                              optimizer_preds,
                              criterion,
                              bound_dmp_weights,
                              target_goal,
                              epsilon=0,
                              world=None):
    
    # Training Critics with Simulated Trajectories
    for net in net_preds:
        net.train()

    # We will only simulate 32 trajectories (it is expensive) from a random initial state 'i'
    i = random.randint(0, len(train_data_epoch) - batch_size)
    batch = train_data_epoch[i:i + batch_size]
    batch_s_t = torch.stack([item[0] for item in batch])
    
    # Prepare inputs for RNN
    inputs1 = batch_s_t.unsqueeze(1)  # (batch_size, seq_len=1, input_size)
    padded_state = torch.zeros_like(batch_s_t)
    inputs2 = padded_state.unsqueeze(1)  # (batch_size, seq_len=1, input_size)
    
    # Forward pass through net_goal to get DMP parameters
    with torch.no_grad():
        target_goal_batch = torch.stack([torch.tensor(target_goal) for item in batch])
        outputs1, hidden = net_goal(inputs1[:,:,:2]-target_goal_batch.unsqueeze(1))
        outputs2, _ = net_goal(inputs2[:,:,:2], hidden)
        
        dmp_params1 = outputs1.squeeze(1).clone()
        dmp_params2 = outputs2.squeeze(1).clone()

        # Force the start positions in dmp_params1 and dmp_params2, and center in home reference frame
        dmp_params1_positions = torch.cat([batch_s_t[:,:2], dmp_params1[:, 2:4] + target_goal_batch], dim=1)
        dmp_params2_positions = torch.cat([dmp_params1[:, 2:4] + target_goal_batch, dmp_params2[:, 2:4] + target_goal_batch], dim=1)

        # Separate the position and DMP weights for clamping
        dmp_params1_weights = dmp_params1[:, 4:].clamp(-bound_dmp_weights, bound_dmp_weights)
        dmp_params2_weights = dmp_params2[:, 4:].clamp(-bound_dmp_weights, bound_dmp_weights)

        # Concatenate the positions and clamped weights
        dmp_params1 = torch.cat([dmp_params1_positions, dmp_params1_weights], dim=1)
        dmp_params2 = torch.cat([dmp_params2_positions, dmp_params2_weights], dim=1)

    final_positions_1, final_positions_2, collision_info = generate_trajectories_from_dmp_params(
        dmp_params1=dmp_params1,
        dmp_params2=dmp_params2,
        batch_size=batch_size,
        batch_s_t=batch_s_t[:,:2],
        world=world,
        n_basis=3,
        circular=True)

    # Optimize Critics
    final_positions_1 = torch.tensor(final_positions_1)
    final_positions_2 = torch.tensor(final_positions_2)

    # Prepare inputs for both timesteps for net_preds
    inputs1 = torch.cat((batch_s_t, dmp_params1), dim=1).unsqueeze(1)
    inputs2 = torch.cat((padded_state, dmp_params2), dim=1).unsqueeze(1)
    
    # Forward pass for both timesteps
    optimizer_preds[0].zero_grad()
    outputs1, hidden = net_preds[0](inputs1)
    outputs2, _ = net_preds[0](inputs2, hidden)

    # Separate the outputs into position and length
    pred_pos1 = outputs1.squeeze(1)[:, :2]
    pred_pos2 = outputs2.squeeze(1)[:, :2]

    # Compute losses
    loss_pos1 = criterion(pred_pos1, final_positions_1.float())
    loss_pos2 = criterion(pred_pos2, final_positions_2.float())
    loss = loss_pos1 + loss_pos2

    # Backward pass and optimization
    loss.backward()
    optimizer_preds[0].step()

    return loss_pos1.item(), loss_pos2.item(), 0, net_preds


def train(train_data,
          valid_data,
          net_goal,
          net_preds,
          task,
          target_goal1=None,
          target_goal2=None,
          fine_tune_pred_nets=True,
          num_iterations_inverse_dynamics=5,
          num_samples_inverse_dynamics=200,
          num_training_sets=10,
          learning_rate=0.0001,
          num_epochs=100,
          batch_size=32,
          early_stopping_threshold=0,
          eval_freq=1,
          bound_dmp_weights=1,
          num_explo_episodes=-1,
          plot_trajectories=False,
          world=None,
          valid_batch_size=50,
          plot_only_first=False,
          testing_dumb_model = False,
          trip_wire = None):
    
    # Optimizer
    optimizer_goal = optim.Adam(net_goal.parameters(), lr=learning_rate, weight_decay=0)
    optimizer_preds = [optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0) for net in net_preds]

    stop_training, num_epochs_with_network = False, 0

    training_sets = split_list(train_data, 10)
    # Loss function
    criterion = nn.MSELoss()
    criterion2 = nn.BCEWithLogitsLoss()
    valid_losses, coll_perf1, coll_perf2 = [], [], []
    for epoch in range(num_epochs):
        # Select one of the training sets
        train_data_epoch = training_sets[epoch % num_training_sets]

        if not stop_training:
            # Training goal network
            net_goal.train()
            train_losses = []
            samples_inverse_dynamics = random.choices(train_data_epoch, k=num_samples_inverse_dynamics)
            for _ in range(num_iterations_inverse_dynamics):
                for i in range(0, len(samples_inverse_dynamics), batch_size):
                    batch = samples_inverse_dynamics[i:i+batch_size]
                    batch_s_t = torch.stack([item[0] for item in batch])
                    target_goal_batch = torch.stack([torch.tensor(target_goal2) for item in batch])

                    # Prepare inputs for RNN
                    inputs1 = batch_s_t.unsqueeze(1)  # (batch_size, seq_len=1, input_size)
                    padded_state = torch.zeros_like(batch_s_t)
                    inputs2 = padded_state.unsqueeze(1)  # (batch_size, seq_len=1, input_size)

                    # Forward pass through net_goal to get DMP parameters
                    optimizer_goal.zero_grad()
                    outputs1, hidden = net_goal(inputs1[:,:,:2]-target_goal_batch.unsqueeze(1))
                    outputs2, _ = net_goal(inputs2[:,:,:2], hidden)
                    
                    dmp_params1 = outputs1.squeeze(1)
                    dmp_params2 = outputs2.squeeze(1)

                    # Force the start positions in dmp_params1 and dmp_params2, and center in home reference frame
                    dmp_params1_positions = torch.cat([batch_s_t[:,:2], dmp_params1[:, 2:4] + target_goal_batch], dim=1)
                    dmp_params2_positions = torch.cat([dmp_params1[:, 2:4] + target_goal_batch, dmp_params2[:, 2:4] + target_goal_batch], dim=1)

                    # Separate the position and DMP weights for clamping
                    dmp_params1_weights = dmp_params1[:, 4:].clamp(-bound_dmp_weights, bound_dmp_weights)
                    dmp_params2_weights = dmp_params2[:, 4:].clamp(-bound_dmp_weights, bound_dmp_weights)
                    
                    # Calculate complexity scaling factors based on Manhattan distance
                    # For first trajectory: start -> goal1
                    start_pos1 = batch_s_t[:,:2]
                    goal_pos1 = dmp_params1[:, 2:4] + target_goal_batch
                    complexity_scale1 = ((torch.abs(start_pos1[:,0] - goal_pos1[:,0]) + 
                                        torch.abs(start_pos1[:,1] - goal_pos1[:,1])) / 2).unsqueeze(1)
                    
                    # For second trajectory: goal1 -> goal2
                    start_pos2 = dmp_params1[:, 2:4] + target_goal_batch
                    goal_pos2 = dmp_params2[:, 2:4] + target_goal_batch
                    complexity_scale2 = ((torch.abs(start_pos2[:,0] - goal_pos2[:,0]) + 
                                        torch.abs(start_pos2[:,1] - goal_pos2[:,1])) / 2).unsqueeze(1)
                    
                    # Apply scaling to weights
                    dmp_params1_weights = dmp_params1_weights * complexity_scale1.repeat(1, dmp_params1_weights.size(1))
                    dmp_params2_weights = dmp_params2_weights * complexity_scale2.repeat(1, dmp_params2_weights.size(1))

                    # Concatenate the positions and scaled weights
                    dmp_params1 = torch.cat([dmp_params1_positions, dmp_params1_weights], dim=1)
                    dmp_params2 = torch.cat([dmp_params2_positions, dmp_params2_weights], dim=1)

                    # Forward pass through net_preds to get final state predictions
                    inputs1 = torch.cat((batch_s_t, dmp_params1), dim=1).unsqueeze(1)
                    outputs1, hidden = net_preds[0](inputs1)
                    final_state_preds1 = outputs1.squeeze(1)

                    inputs2 = torch.cat((padded_state, dmp_params2), dim=1).unsqueeze(1)
                    outputs2, _ = net_preds[0](inputs2, hidden)
                    final_state_preds2 = outputs2.squeeze(1)

                    # Extract lengths from predictions
                    pred_coll1 = outputs1.squeeze(1)[:, 2]
                    pred_coll2 = outputs2.squeeze(1)[:, 2]

                    if testing_dumb_model:
                        final_state_preds1 = dmp_params1[:, 2:4]
                        final_state_preds2 = dmp_params2[:, 2:4]
                        pred_coll1 = False * pred_coll1
                        pred_coll2 = False * pred_coll2

                    # Calculate loss
                    target_goal_batch = torch.stack([torch.tensor(target_goal2) for item in batch])
                    loss_goal1 = criterion(final_state_preds1[:, :2], target_goal_batch)
                    loss_goal2 = criterion(final_state_preds2[:, :2], target_goal_batch)
                    loss_coll1 = criterion2(pred_coll1, torch.zeros_like(pred_coll1))
                    loss_coll2 = criterion2(pred_coll2, torch.zeros_like(pred_coll2))
                    outside_circle1 = torch.relu(torch.norm(final_state_preds1[:, :2], p=2, dim=1)- 1.0).mean()
                    outside_circle2 = torch.relu(torch.norm(final_state_preds2[:, :2], p=2, dim=1)- 1.0).mean()
                    distance1 = criterion(final_state_preds1[:, :2], batch_s_t[:,:2])
                    distance2 = criterion(final_state_preds2[:, :2], final_state_preds1[:, :2])
                    distance = criterion(final_state_preds2[:, :2], batch_s_t[:,:2])
                    target_subgoal = torch.stack([torch.tensor(target_goal1) for item in batch])
                    loss_subgoal = criterion(final_state_preds1[:, :2], target_subgoal)
                    return_distance = criterion(final_state_preds2[:, :2], batch_s_t[:,:2])

                    if task == "home_run_explo":
                        loss_goal = loss_goal2 +loss_goal1 +  0.25 * (
                            loss_coll1 + loss_coll2) + 1 * (outside_circle1 + outside_circle2)
                            
                        # loss_goal = loss_goal2 +loss_goal1 + 1 * (outside_circle1 + outside_circle2)
                        
                    elif task == "home_run_no_wall":
                        loss_goal = loss_goal1+loss_goal2

                    elif task == "home_run_one_primitive":
                        loss_goal = loss_goal1 + 0.25 * (
                            loss_coll1) + 1 * (
                                outside_circle1 )
                        # loss_goal = loss_goal1 
                        plot_only_first=True

                    elif task == "explore_obstacle":
                        loss_goal = 0.05* (-loss_coll1 -  loss_coll2) + 5 * distance2 + 10 * (outside_circle1 + outside_circle2)

                    elif task == "explore_and_return":
                        loss_goal = 0.5 * return_distance + loss_goal1 + 0.25 * (loss_coll1 + loss_coll2) + 1 * (outside_circle1 + outside_circle2)

                    elif task == "max_distance":
                        loss_goal = - distance + 0.25 * (loss_coll1 + loss_coll2) + 10 * (outside_circle1 + outside_circle2)
                    
                    elif task == "second_goal":
                        loss_goal = loss_goal2 + 0.25 * (
                            loss_coll1 + loss_coll2) + 1 * (outside_circle1 + outside_circle2)
                    else:
                        raise ValueError("Invalid task value. Expected 'home_run_explo' or 'home_run_escape'.")

                    if epoch > num_explo_episodes:
                        loss_goal.backward()
                        optimizer_goal.step()
            
            if fine_tune_pred_nets:
                epsilon = 1 if epoch < num_explo_episodes else 0
                for _ in range(1):
                    # Optimize critics with trajectories generated by the optimized Actor
                    loss_preds1, loss_preds2, loss_col, net_preds = optimize_pred_nets_online(train_data_epoch=train_data_epoch,
                                    batch_size=32,
                                    net_goal=net_goal,
                                    net_preds=net_preds,
                                    optimizer_preds=optimizer_preds,
                                    criterion=criterion,
                                    bound_dmp_weights=bound_dmp_weights,
                                    epsilon=epsilon,
                                    world=world,
                                    target_goal=target_goal2)
            
                train_losses.append((loss_preds1, loss_preds2, loss_col, loss_goal.item()))

            else:
                train_losses.append((loss_goal.item()))
                
        # Validation by Simulating in the environment
        net_goal.eval()
        if epoch % eval_freq == 0:
            with torch.no_grad():
                i = random.randint(0, len(valid_data)-valid_batch_size)
                batch = valid_data[i:i+valid_batch_size]
                batch_s_t = torch.stack([item[0] for item in batch])
                target_goal_batch = torch.stack([torch.tensor(target_goal2) for item in batch])

                # Prepare inputs for RNN
                inputs1 = batch_s_t.unsqueeze(1)  # (batch_size, seq_len=1, input_size)
                padded_state = torch.zeros_like(batch_s_t)
                inputs2 = padded_state.unsqueeze(1)  # (batch_size, seq_len=1, input_size)

                # Forward pass through net_goal to get DMP parameters
                outputs1, hidden = net_goal(inputs1[:,:,:2]-target_goal_batch.unsqueeze(1))
                outputs2, _ = net_goal(inputs2[:,:,:2], hidden)
                
                dmp_params1 = outputs1.squeeze(1)
                dmp_params2 = outputs2.squeeze(1)

                # Force the start positions in dmp_params1 and dmp_params2, and center in home reference frame
                dmp_params1_positions = torch.cat([batch_s_t[:,:2], dmp_params1[:, 2:4] + target_goal_batch], dim=1)
                dmp_params2_positions = torch.cat([dmp_params1[:, 2:4] + target_goal_batch, dmp_params2[:, 2:4] + target_goal_batch], dim=1)

                # Separate the position and DMP weights for clamping
                dmp_params1_weights = dmp_params1[:, 4:].clamp(-bound_dmp_weights, bound_dmp_weights)
                dmp_params2_weights = dmp_params2[:, 4:].clamp(-bound_dmp_weights, bound_dmp_weights)

                # Concatenate the positions and clamped weights
                dmp_params1 = torch.cat([dmp_params1_positions, dmp_params1_weights], dim=1)
                dmp_params2 = torch.cat([dmp_params2_positions, dmp_params2_weights], dim=1)

                # Generate trajectories with these parameters

                actual_final_position1, actual_final_position2, _ = generate_trajectories_from_dmp_params(
                    dmp_params1=dmp_params1,
                    dmp_params2=dmp_params2,
                    batch_size=valid_batch_size,
                    batch_s_t=batch_s_t[:,:2],
                    world=world,
                    circular=True,
                    n_basis=3)
                
                if plot_trajectories:
                    generate_and_plot_trajectories_from_parameters(
                        dmp_params1, dmp_params2, min(len(dmp_params1),10), batch_s_t[:,:2], world, world_bounds, n_basis=3,
                        circular=True,
                        plot_only_first=plot_only_first,
                        trip_wire=trip_wire)


                loss1 = np.mean(np.abs(np.array(actual_final_position1) - np.array(target_goal1)))
                loss2 = np.mean(np.abs(np.array(actual_final_position2) - np.array(target_goal2)))
                loss = np.max([1 - loss2,1-loss1])

                valid_losses.append(loss)

                if loss > early_stopping_threshold:
                    print(f'Stop training at epoch {epoch + 1}, valid loss: {loss}')
                    stop_training = True

                print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss = {np.mean(train_losses, axis=0)} , Valid Loss = {valid_losses[-1]:.2f}')

    return valid_losses, net_goal, net_preds


if __name__ == "__main__":

    # Initialize the world
    world_bounds = [-1, 1, -1, 1]
    world = CircularWorld(num_obstacles=0, max_speed=100, radius=1, wall_present=True,wall_size=0.6)

    # Check the environment
    plot_example_trajectories(
        world,world_bounds,
        number_of_trajectories=100,
        complexity=5.0,
        circular=True,
        random_position=False,
        n_basis=3,
        start_position=np.array([0.0, 0.85]),
        goal_position=np.array([0.0, -0.85]),
        # trip_wire = None)
        # trip_wire=[[(-0.30,-1.00),(-0.1,-0.4)],[(0.30,-1.00),(0.1,-0.4)]])
        trip_wire=[[(-0.7,-0.75),(0.2,0.0)],[(0.7,-0.75),(-0.2,0.0)]])
        # trip_wire=[[(-1.0,0.25),(0.0,0.0)],[(1.0,0.25),(0.0,0.0)]])
    

    # Collect Data for Predictive Net (Varying wall)
    train_data,valid_data= generate_data(
        world,
        world_bounds,
        number_of_trajectories=25000,
        orientation=None,
        complexity=5.0,
        varying_wall=True,
        full_state_space=True)
    
    training_sets = split_list(train_data, 10)

    # Initialize the network Level 1
    net = PredNet(input_size=2+2+2+6+5, hidden_size=64, output_size=3, dropout_rate=0.1)

    # Train Predictive Net
    loss1,loss2, net = train_predictive_net(
        training_sets,valid_data,net,
        batch_size = 32,
        num_epochs = 2000,
        learning_rate = 0.001,
        num_training_sets = 10,
        eval_freq = 20,
        weight_collisions = 0.05)


    plt.figure(figsize=(3.5,2))
    plt.plot(np.linspace(0,25000,len(loss1)),1-np.array(loss1)/2,color='blue',linewidth=3)
    plt.plot(np.linspace(0,25000,len(loss2)),1-np.array(loss2)/2,color='red',linewidth=3)
    plt.ylim(0.8,1)
    plt.show()


    plot_predicted_vs_actual_trajectories(
        world,
        net,
        world_bounds,
        number_of_trajectories=10,
        circular=True,
        complexity=5.0,
        varying_wall=True)

    # Collect Data during Exploration
    world = CircularWorld(num_obstacles=0, max_speed=100, radius=1, wall_present=True,wall_size=0.6)
    train_data_with_wall,valid_data_with_wall= generate_data(
        world,
        world_bounds,
        number_of_trajectories=100,
        orientation=None,
        complexity=1.5,
        varying_wall=False,
        full_state_space=True,
        # trip_wire=None)
        # trip_wire=[[(-0.30,-1.00),(-0.1,-0.4)],[(0.30,-1.00),(0.1,-0.4)]])
        trip_wire=[[(-0.7,-0.75),(0.2,0.0)],[(0.7,-0.75),(-0.2,0.0)]])
        # trip_wire=[[(-1.0,0.25),(0.0,0.0)],[(1.0,0.25),(0.0,0.0)]])
    visualize_initial_positions(train_data_with_wall)
        
    escape_valid_data_with_wall = create_valid_data_subset(valid_data_with_wall,target_position=torch.tensor([0, -0.85]), distance_threshold=0.3)

    # Collect Data during Escape
    new_world = CircularWorld(num_obstacles=0, max_speed=100, radius=1, wall_present=False,wall_size=0.6)
    train_data_without_wall,valid_data_without_wall= generate_data(
        new_world,
        world_bounds,
        number_of_trajectories=1000,
        orientation=None,
        complexity=1.5,
        varying_wall=False,
        full_state_space=True)
    escape_valid_data_without_wall = create_valid_data_subset(valid_data_without_wall,target_position=torch.tensor([0, -0.85]), distance_threshold=0.3)

    # Exploration with second wall
    world = CircularWorld(num_obstacles=0, max_speed=100, radius=1, wall_present=True,wall_size=0.5,second_wall=True)
    train_data_with_second_wall,valid_data_with_second_wall= generate_data(
        world,
        world_bounds,
        number_of_trajectories=10000,
        orientation=None,
        complexity=0.5,
        varying_wall=False,
        full_state_space=True)
    escape_valid_data_with_second_wall = create_valid_data_subset(valid_data_with_second_wall,target_position=torch.tensor([0, 0.85]), distance_threshold=0.3)

    # Create data with wall blocking threat zone
    data_with_wall_blocking_threat_zone = create_data_with_wall_blocking_threat_zone(
        train_data_with_wall,
        height_threshold=-0.4)

    # Create data without threat zone
    data_without_threat_zone = create_data_without_threat_zone(
        train_data_with_wall,
        target_position=torch.tensor([0, -0.85]),
        distance_threshold=0.3)

    # Create data without edge-to-home
    data_without_threat_zone = create_data_without_edge_to_home(
        train_data_with_wall,
        target_position1=torch.tensor([-0.6, -0.1]),
        target_position2=torch.tensor([0.6, -0.1]),
        distance_threshold=0.2)

    # Collect Data Square Obstacle
    world = CircularWorld(num_obstacles=0, max_speed=100, radius=1, wall_present=True,wall_size=0.4,wall_thickness=0.6)
    plot_example_trajectories(world,world_bounds,number_of_trajectories=10,complexity=2,circular=True)
    train_data_square_obs,valid_data_square_obs= generate_data(
        world,
        world_bounds,
        number_of_trajectories=10000,
        orientation=None,
        complexity=0.5,
        varying_wall=False,
        full_state_space=True)


    # Training Phase
    task = 'home_run_one_primitive'
    net_goal = GoalNet(input_size=2, hidden_size=64, output_size=2+2+6, dropout_rate=0.1)
    net_preds = [copy.deepcopy(net), copy.deepcopy(net)]
    valid_losses, net_goal, net_preds = train(
        train_data = train_data_with_wall,
        valid_data = valid_data_with_wall,
        net_goal = net_goal,
        net_preds = net_preds,
        target_goal1 = [0.,0.85],
        target_goal2 = [0.,0.85],
        task = task,
        fine_tune_pred_nets = False,
        num_samples_inverse_dynamics = 10000,
        num_iterations_inverse_dynamics = 1,
        bound_dmp_weights = 5.0,
        early_stopping_threshold = 1.0,
        learning_rate = 0.001,
        batch_size = 10,
        num_epochs = 10,
        plot_trajectories = True,
        world = world,
        valid_batch_size=5,
        trip_wire=None)
    

    # Testing Phase
    task = 'home_run_one_primitive'
    net_goal_test = copy.deepcopy(net_goal)
    valid_losses, net_goal_test, net_preds = train(
        train_data = train_data_without_wall,
        valid_data = escape_valid_data_without_wall,
        net_goal = net_goal_test,
        net_preds = net_preds,
        target_goal1 = [0,0.85],
        target_goal2 = [0,0.85],
        task = task,
        fine_tune_pred_nets = False,
        num_samples_inverse_dynamics = 100,
        num_iterations_inverse_dynamics = 1,
        bound_dmp_weights = 5.0,
        early_stopping_threshold = 1.00,
        learning_rate = 0.0001,
        batch_size = 15,
        num_epochs = 100,
        plot_trajectories = True,
        world=new_world,
        valid_batch_size=5,
        trip_wire=None,)