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
from utils import  generate_trajectories_from_dmp_params
from continuous_nav_envs import World, generate_random_positions
from dmps import DMP1D,Simulation
from plot_trajectories import generate_and_plot_trajectories_from_parameters, plot_dmp_weight_space

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
    
def generate_data(
        world,
        world_bounds,
        number_of_trajectories=10000,
        complexity = 1):
    # Generate training data
    data = []
    for i in range(number_of_trajectories):  # Number of trajectories to generate

        if i%1000==0:
            print(f'Generated {i} of {number_of_trajectories} trajectories')
        # Generate random start and goal positions for first DMP
        start_position, _ = generate_random_positions(world, world_bounds=world_bounds,orientation = None)
        _, goal_position1 = generate_random_positions(world, world_bounds=world_bounds,orientation = None)

        # Initialize the first DMPs
        dmp_x1 = DMP1D(start=start_position[0], goal=goal_position1[0], n_basis=3, complexity=complexity)
        dmp_y1 = DMP1D(start=start_position[1], goal=goal_position1[1], n_basis=3, complexity=complexity)

        # Initialize the position and velocity of the agent
        start_velocity = np.array([0.0, 0.0])

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

        s_t = positions1[0]
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


def train_predictive_net(train_data,
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
        random.shuffle(train_data)
        # Training
        net.train()
        train_losses = []
        for i in range(0, len(train_data)-batch_size, batch_size):
            batch = train_data[i:i + batch_size]
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
                    pred_coll1 = outputs1.squeeze(1)[:, 2]
                    pred_pos2 = outputs2.squeeze(1)[:, :2]
                    pred_coll2 = outputs2.squeeze(1)[:, 2]
                    
                    # Compute losses
                    loss_pos1 = torch.mean(torch.abs(pred_pos1- batch_final_position1))
                    loss_pos2 = torch.mean(torch.abs(pred_pos2 - batch_final_position2))
                    loss_coll1 = criterion2(pred_coll1, batch_coll1)
                    loss_coll2 = criterion2(pred_coll2, batch_coll2)
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


def train(train_data,
          valid_data,
          net_goal,
          net_preds,
          task,
          world,
          target_goal1=None,
          target_goal2=None,
          learning_rate=0.0001,
          num_epochs=100,
          batch_size=32,
          early_stopping_threshold=0,
          eval_freq=1,
          bound_dmp_weights = 1,
          plot_trajectories=False,
          number_trajectories_plot = 10,
          testing_dumb_model = False,
          valid_batch_size=100):
    
    # Optimizer
    optimizer_goal = optim.Adam(net_goal.parameters(), lr=learning_rate,weight_decay=0)

    stop_training = False

    iteration=0
    criterion = nn.MSELoss()
    criterion2 = nn.BCEWithLogitsLoss()
    criterion3 = nn.MSELoss(reduction='none')


    valid_losses, coll_perf1, coll_perf2 = [] , [] , [] 
    for epoch in range(num_epochs):
        if not stop_training:
            # Training goal network
            net_goal.train()
            train_losses = []
            for i in range(0, len(train_data)-batch_size,batch_size):
                batch = train_data[i:i+batch_size]
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

                # Concatenate the positions and clamped weights
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
                    
                # Calculate task-specific losses

                target_goal_batch = torch.stack([torch.tensor(target_goal2) for item in batch])
                loss_goal1 = criterion(final_state_preds1[:, :2], target_goal_batch)
                loss_goal2 = criterion(final_state_preds2[:, :2], target_goal_batch)
                loss_coll1 = criterion2(pred_coll1, torch.zeros_like(pred_coll1))
                loss_coll2 = criterion2(pred_coll2, torch.zeros_like(pred_coll2))
                loss_goal2 = criterion(final_state_preds2[:, :2], target_goal_batch)
                penalty_x1 = torch.relu(final_state_preds1[:, 0] - 0.9) + torch.relu(0.1 - final_state_preds1[:, 0])
                penalty_y1 = torch.relu(final_state_preds1[:, 1] - 0.9) + torch.relu(0.1 - final_state_preds1[:, 1])
                outside_square1 = (penalty_x1 + penalty_y1).mean()

                penalty_x2 = torch.relu(final_state_preds2[:, 0] - 0.9) + torch.relu(0.1 - final_state_preds2[:, 0])
                penalty_y2 = torch.relu(final_state_preds2[:, 1] - 0.9) + torch.relu(0.1 - final_state_preds2[:, 1])
                outside_square2 = (penalty_x2 + penalty_y2).mean()

                if task == "reach_final_goal":
                    loss_goal = 0.5*loss_goal1 + loss_goal2 + 0.25*  (loss_coll1 + loss_coll2) + outside_square1 + outside_square2
                    # loss_goal = loss_goal1 + loss_goal2
                elif task == "reach_two_goals":
                    target_goal_batch1 = torch.stack([torch.tensor(target_goal1) for _ in batch])
                    target_goal_batch2 = torch.stack([torch.tensor(target_goal2) for _ in batch])
                    # Compute per-sample losses for both goal assignments
                    loss_d1g1 = criterion3(final_state_preds1[:, :2], target_goal_batch1).sum(dim=1)
                    loss_d1g2 = criterion3(final_state_preds1[:, :2], target_goal_batch2).sum(dim=1)
                    loss_d2g1 = criterion3(final_state_preds2[:, :2], target_goal_batch1).sum(dim=1)
                    loss_d2g2 = criterion3(final_state_preds2[:, :2], target_goal_batch2).sum(dim=1)
                    # Total losses for both assignments
                    total_loss_A = loss_d1g1 + loss_d2g2  # Assign target_goal1 to final_state_preds1
                    total_loss_B = loss_d1g2 + loss_d2g1  # Assign target_goal2 to final_state_preds1
                    # Determine the assignment with the minimum total loss
                    mask = (total_loss_A < total_loss_B).float()
                    # Compute the final per-sample loss
                    loss_goal_per_sample = mask * total_loss_A + (1 - mask) * total_loss_B
                    # Compute the final loss as the mean over all samples
                    loss_goal = loss_goal_per_sample.mean() + 0.25 * (loss_coll1 + loss_coll2) + outside_square1 + outside_square2
                    
                elif task == "reach_final_goal_one_prim":
                    target_goal_batch2 = torch.stack([torch.tensor(target_goal2) for _ in batch])
                    loss_goal = criterion(final_state_preds1, target_goal_batch2)
                elif task == "maximize_total_distance":
                    loss_goal =-criterion(batch_s_t[:, :2], final_state_preds2[:, :2]) 
                    loss_goal +=  0.25*  (loss_coll1 + loss_coll2) + outside_square1 + outside_square2
                elif task == "goal_and_return":
                    target_goal_batch = torch.stack([torch.tensor(target_goal2) for item in batch])
                    loss_goal =  criterion(final_state_preds1[:, :2], target_goal_batch) + criterion(final_state_preds2[:, :2], batch_s_t[:, :2])
                    loss_goal +=  0.25*  (loss_coll1 + loss_coll2) + outside_square1 + outside_square2
                elif task == "explore_and_return":
                    loss_goal =  - criterion(final_state_preds1[:, :2], batch_s_t[:, :2]) + criterion(final_state_preds2[:, :2], batch_s_t[:, :2])
                    loss_goal +=  0.25*  (loss_coll1 + loss_coll2) + outside_square1 + outside_square2

                else:
                    raise ValueError("Invalid task value. Expected 'reach_final_goal', 'reach_two_goals', 'maximize_total_distance' or 'maximize_reward_function'.")
                
                optimizer_goal.zero_grad()
                loss_goal.backward()
                optimizer_goal.step()
                iteration+=1

            train_losses.append(( loss_goal.item()))

        # Validation by Simulating in the environment
        net_goal.eval()
        if epoch%eval_freq == 0:
            print(iteration)
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
                actual_final_position1,actual_final_position2, collision_info = generate_trajectories_from_dmp_params(
                    dmp_params1=dmp_params1,
                    dmp_params2=dmp_params2,
                    batch_size=valid_batch_size,
                    batch_s_t=np.array(batch_s_t[:,:2]),
                    world=world)
                
                if plot_trajectories:
                    which = np.random.choice(valid_batch_size-number_trajectories_plot-1)
                    generate_and_plot_trajectories_from_parameters(
                        dmp_params1[which:which+number_trajectories_plot,:],
                        dmp_params2[which:which+number_trajectories_plot,:],
                        number_trajectories_plot,
                        batch_s_t[which:which+number_trajectories_plot,:2],
                        world,world_bounds)
                    
                if task == "reach_final_goal" or task=='use_rl':
                    loss = np.mean(np.abs(np.array(actual_final_position2)- np.array(target_goal2)))
                    loss = 1 - loss
                elif task == "reach_two_goals":
                    # Compute per-sample distances for both assignments
                    # Option 1: assign target_goal1 to actual_final_position1
                    dist1_option1 = np.abs(np.array(actual_final_position1) - np.array(target_goal1))
                    dist2_option1 =  np.abs(np.array(actual_final_position2) - np.array(target_goal2))
                    total_loss_option1 = dist1_option1 + dist2_option1

                    # Option 2: assign target_goal1 to actual_final_position2
                    dist1_option2 = np.abs(np.array(actual_final_position1) - np.array(target_goal2))
                    dist2_option2 = np.abs(np.array(actual_final_position2) - np.array(target_goal1))
                    total_loss_option2 = dist1_option2 + dist2_option2

                    # For each sample, pick the minimal total loss
                    total_loss = np.minimum(total_loss_option1, total_loss_option2)

                    # Compute the average loss over the batch
                    loss = 1 - np.mean(total_loss)/2
                    
                elif task == 'goal_and_return':
                    loss1 = np.mean(np.abs(np.array(actual_final_position1)- np.array(target_goal2)))
                    loss2 = np.mean(np.abs(np.array(actual_final_position2)- np.array(batch_s_t)))
                    loss = 1 - (loss1 + loss2) / 2
                elif task == "reach_final_goal_one_prim":
                    loss = np.mean(np.abs(np.array(actual_final_position1)- np.array(target_goal2)))
                    loss = 1 - loss
                elif task == "maximize_total_distance":
                    loss = np.mean(np.abs(np.array(actual_final_position2)- np.array(batch_s_t)))
                    loss = loss / 1.66
                elif task == "explore_and_return":
                    loss = - np.mean(np.abs(np.array(actual_final_position1)- np.array(batch_s_t)))/0.5 + np.mean(np.abs(np.array(actual_final_position2)- np.array(batch_s_t)))
                valid_losses.append(loss)

                if loss>early_stopping_threshold:
                    print(f'Stop training at epoch {epoch+1}, valid loss: {loss}')
                    stop_training = True

                print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {np.mean(train_losses, axis=0)}, Coll Perf = [{np.mean(coll_perf1):.2f},{np.mean(coll_perf2):.2f}], Valid Loss = {valid_losses[-1]:.2f}')

    return valid_losses, net_goal, net_preds





if __name__ == "__main__":

    # Initialize the world
    world_bounds = [0.1, 0.9, 0.1, 0.9]
    world = World(
        world_bounds=world_bounds,
        friction=1,
        num_obstacles=1,
        given_obstacles = [(0, 0.48), (0, 0.52), (0.6, 0.48), (0.6, 0.52)])

    # Check the environment
    plot_example_trajectories(world,world_bounds,number_of_trajectories=5,complexity=1.0)

    # Collect Data for Predictive Net
    train_data,valid_data= generate_data(
        world,
        world_bounds,
        number_of_trajectories=5_000,
        complexity=1.0)

    # Initialize the network Level 1
    net = PredNet(input_size=2+2+2+6, hidden_size=64, output_size=3, dropout_rate=0.1)

    # Train Predictive Net
    loss1,loss2, net = train_predictive_net(
        train_data,valid_data,net,
        batch_size = 32,
        num_epochs = 1000,
        learning_rate = 0.001,
        num_training_sets = 10,
        eval_freq = 50,
        weight_collisions = 0.05)
    net.eval()

    plt.figure(figsize=(3.5,2))
    plt.plot(np.linspace(0,1500,len(loss1)),1-np.array(loss1),color='blue',linewidth=3)
    plt.plot(np.linspace(0,1500,len(loss2)),1-np.array(loss2),color='red',linewidth=3)
    plt.ylim(0.8,1.0)
    plt.show()

    
    plot_predicted_vs_actual_trajectories(
        world,
        net,
        world_bounds,
        number_of_trajectories=4,
        circular=False,
        complexity=0.5)
    

    valid_loss_task = {'reach_final_goal':[]  }
    
    # Train New Tasks
    all_valid_loss = []
    for task in valid_loss_task:
        net_goal = GoalNet(input_size=2, hidden_size=64, output_size=2+2+6, dropout_rate=0.1)
        net_preds = [copy.deepcopy(net), copy.deepcopy(net)]
        target_goal1 = [0.3,0.85]
        target_goal2 = [0.3,0.85]
        valid_losses, net_goal, net_preds = train(
            train_data,
            valid_data,
            net_goal,
            net_preds,
            world=world,
            target_goal1 = target_goal1,
            target_goal2 = target_goal2,
            task = task,
            bound_dmp_weights=0.9,
            early_stopping_threshold = 0.99,
            eval_freq=5,
            learning_rate=0.001,
            batch_size=10,
            num_epochs=100,
            plot_trajectories=True,
            number_trajectories_plot = 6,
            testing_dumb_model=False)
        valid_loss_task[task] = valid_losses
        all_valid_loss += valid_losses