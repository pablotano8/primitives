
import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import os
import gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os
import sys


# from mujoco_plots import visualize_predictions

DIRECTIONS = [
    (-1e3, 0),
    (1e3, 0),
    (0, 1e3),
    (0, -1e3),
    (-1e3, 1e3),
    (1e3, -1e3),
    (1e3, 1e3),
    (-1e3, -1e3),
    (0,0)
]


class RNNMujoco(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_rate=0.1, symm=False,bidirectional=False):
        super(RNNMujoco, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout_rate if num_layers > 1 else 0,
                           bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.symm = symm
        self.bidirectional = bidirectional

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)

        # If bidirectional, sum the outputs from both directions BEFORE the FC layer
        if self.bidirectional:
            out_forward = out[:, :, :self.rnn.hidden_size]
            out_backward = out[:, :, self.rnn.hidden_size:]
            out = out_forward + out_backward
        
        # Apply dropout to the outputs of the RNN layer
        out = self.dropout(out)
        
        # Apply the fc layer on every timestep's output
        out = self.fc(out)

        if self.symm:
            out = out * torch.tensor([1, 0.7, 1, 0.7, 1, 0.7, 1, 0.7])
    
        return out, hidden

    def init_hidden(self, batch_size):
        # If the LSTM is bidirectional, the number of directions is 2, else it's 1
        num_directions = 2 if self.rnn.bidirectional else 1
        
        # Initialize hidden and cell states
        # Shape: (num_layers * num_directions, batch_size, hidden_size)
        return (torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size),
                torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size))



def train_pred_net_rnn(train_data,
                       valid_data,
                       pred_net,
                       learning_rate=0.0001,
                       num_epochs=50,
                       batch_size=32,
                       eval_freq=1,
                       clip_value=1.0,
                       plan_length=10,
                       verbose = True,
                       random_padding=True):

    # Filter train_data up to plan_length
    train_data = [element[:plan_length] for element in train_data if len(element) >= plan_length]
    valid_data = [element[:plan_length] for element in valid_data if len(element) >= plan_length]

    # Optimizer
    optimizer = optim.Adam(pred_net.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Loss function
    criterion = nn.MSELoss()
    average_validation_loss = []

    processed_batches = -1
    for epoch in range(num_epochs):
        
        # Training
        pred_net.train()
        train_losses = []

        for i in range(0, len(train_data), batch_size):
            sequences = train_data[i:i+batch_size]
            t_star = torch.randint(0, plan_length, (len(sequences),))  # Assuming T=10
            # Extract data from sequences and pad after t*
            s_t_seq_list, final_state_seq_list, dmp_params_seq_list = [], [], []
            for idx, seq in enumerate(sequences):
                s_t = [t[0] if j <= random_padding*t_star[idx] else torch.zeros_like(t[0]) for j, t in enumerate(seq)]
                # s_t = [t[0] for t in seq]
                final_state = [t[1] for t in seq]
                dmp_params = [t[2] for t in seq]

                s_t_seq_list.append(torch.stack(s_t))
                final_state_seq_list.append(torch.stack(final_state))
                dmp_params_seq_list.append(torch.stack(dmp_params))

            s_t_seq = torch.stack(s_t_seq_list)
            final_state_seq = torch.stack(final_state_seq_list)
            dmp_params_seq = torch.stack(dmp_params_seq_list)
            
            s_t_extended_seq = torch.cat((s_t_seq, dmp_params_seq), dim=-1)
            target_seq = final_state_seq

            optimizer.zero_grad()

            hidden = pred_net.init_hidden(len(sequences))

            output, _ = pred_net(s_t_extended_seq, hidden)
            # Calculate the loss for the entire batch
            batch_loss = criterion(output, target_seq)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(pred_net.parameters(), clip_value)

            optimizer.step()

            processed_batches +=1
            train_losses.append(batch_loss.item())

            if processed_batches % eval_freq == 0:
                # Validation
                pred_net.eval()
                valid_losses = []
                future_step_losses = [0.0] * plan_length  # Assuming T=10
                count_steps = [0] * plan_length

                with torch.no_grad():
                    for i in range(0, len(valid_data), batch_size):
                        sequences = valid_data[i:i+batch_size]
                        
                        # Extract data from sequences and pad after t*
                        s_t_seq_list, final_state_seq_list, dmp_params_seq_list = [], [], []
                        for idx, seq in enumerate(sequences):
                            s_t = [t[0] if j <= 0 else torch.zeros_like(t[0]) for j, t in enumerate(seq)]
                            # s_t = [t[0] for t in seq]
                            final_state = [t[1] for t in seq]
                            dmp_params = [t[2] for t in seq]

                            s_t_seq_list.append(torch.stack(s_t))
                            final_state_seq_list.append(torch.stack(final_state))
                            dmp_params_seq_list.append(torch.stack(dmp_params))

                        s_t_seq = torch.stack(s_t_seq_list)
                        final_state_seq = torch.stack(final_state_seq_list)
                        dmp_params_seq = torch.stack(dmp_params_seq_list)
                        
                        s_t_extended_seq = torch.cat((s_t_seq, dmp_params_seq), dim=-1)
                        target_seq = final_state_seq

                        hidden = pred_net.init_hidden(len(sequences))
                        output, _ = pred_net(s_t_extended_seq, hidden)

                        for j in range(output.size(1)):
                            loss_per_feature = torch.abs(output[:, j, :] - target_seq[:, j, :])
                            loss = torch.mean(loss_per_feature)  # Taking the mean across features
                            future_step_losses[j] += loss.item()
                            count_steps[j] += 1
                                                
                        batch_loss = torch.mean(torch.abs(output - target_seq))
                        valid_losses.append(batch_loss.item())

                future_step_avg_losses = [total_loss / count if count != 0 else 0 for total_loss, count in zip(future_step_losses, count_steps)]
                average_validation_loss.append(future_step_avg_losses)
                if verbose:
                    print(f'Epoch {processed_batches*batch_size  / (len(train_data)):.4f}: Train = {np.mean(train_losses):.4f}, Valid = {np.mean(valid_losses):.4f}')
                    print([f"{loss:.4f}" for loss in future_step_avg_losses])

    return average_validation_loss, pred_net


class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout





def map_to_direction(output):
    with torch.no_grad():
        direction = output*1

        direction[np.abs(direction) < 0.25] = 0

        number_of_steps = np.max(np.abs(output)).item() * 333

        direction[direction>0]=1
        direction[direction<0]=-1
        
        idx = np.where([d==(direction[0].item()*1e3,direction[1].item()*1e3) for d in DIRECTIONS])[0][0]

        return idx, int(number_of_steps)


class RNNPlanner(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.1, tau=1.0):
        super(RNNPlanner, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout_rate if num_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout_rate)
        self.direction_fc = nn.Linear(hidden_size, 2)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        
        # Apply dropout to the outputs of the RNN layer
        out = self.dropout(out)
        
        # Get the logits (before softmax)
        out = -1 + torch.sigmoid(self.direction_fc(out)) * 2

        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size),
                torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))





def generate_figure_preds(plan_length=3,T_range = [200,500]):

    env = gym.make('AntBulletEnv-v0')
    # Reset environment
    obs = env.reset()
    with SuppressPrints():
        models = [PPO.load(os.path.join('trained_nets', f"ppo_ant_dir_{i}")) for i in range(9)]

    done = False

    visited_positions = []
    dumb_preds = []
    
    x_dumb, y_dumb = 0,0
    for i in range(plan_length):
        T = np.random.randint(T_range[0], T_range[1])
        chosen_idx = np.random.randint(0, 9)
        chosen_model = models[chosen_idx]
        env.robot.walk_target_x, env.robot.walk_target_y = DIRECTIONS[chosen_idx]
        env.walk_target_x, env.walk_target_y = DIRECTIONS[chosen_idx]
        print((chosen_idx,T))
        
        for t in range(T):
            action, _ = chosen_model.predict(obs)
            obs, _, done, _ = env.step(action)
            # Render the environment and save the frame to the video
            if t % 20 == 0:
                x, y, _ = env.robot.body_xyz
                visited_positions.append((x, y))
                x_dumb += DIRECTIONS[chosen_idx][0]*0.0005
                y_dumb  += DIRECTIONS[chosen_idx][1]*0.0005
                print((x,y))
                print((x_dumb, y_dumb))
                print(np.abs(x_dumb-x)+np.abs(y_dumb-y)/2)
                dumb_preds.append((x_dumb,y_dumb) )

            if done:
                break

    # Release the video writer object
    x_positions, y_positions = zip(*visited_positions)
    x_dumb, y_dumb = zip(*dumb_preds)

    plt.scatter(x_positions, y_positions,color='#4B0082', s=11,alpha=0.8) 
    plt.scatter(x_dumb, y_dumb,color='grey', s=5,alpha=0.8) 

    # Plotting the reference lines
    # plt.axhline(0, color='grey', alpha=0.3)  # y = 0
    # plt.axvline(0, color='grey', alpha=0.3)  # x = 0
    # plt.plot([-13, 13], [-13, 13], color='grey', alpha=0.3)  # x = y
    # plt.plot([-13, 13], [13, -13], color='grey', alpha=0.3)  # x = -y
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)

    plt.show()




def compute_dumb_pred_error(num_trials = 1000, plan_length=3,T_range = [200,500]):


    env = gym.make('AntBulletEnv-v0')
    dumb_errors = []
    for trial in range(num_trials):
        print(trial)
        # Reset environment
        obs = env.reset()
        with SuppressPrints():
            models = [PPO.load(os.path.join('trained_nets', f"ppo_ant_dir_{i}")) for i in range(9)]

        done = False

        current_state = np.concatenate([env.robot.body_xyz[:-1]])
        current_state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        hidden_planner = planner.init_hidden(1)
        padded_zeros = torch.zeros(current_state_tensor.shape[0], plan_length-1, current_state_tensor.shape[2])
        planner_input = torch.cat((current_state_tensor, padded_zeros), dim=1)
        goals_pred, _ = planner(planner_input, hidden_planner)

        visited_positions = []
        dumb_preds = []
        
        x_dumb, y_dumb = 0,0
        for i in range(plan_length):
            T = np.random.randint(T_range[0], T_range[1])
            chosen_idx = np.random.randint(0, 9)
            chosen_model = models[chosen_idx]
            env.robot.walk_target_x, env.robot.walk_target_y = DIRECTIONS[chosen_idx]
            env.walk_target_x, env.walk_target_y = DIRECTIONS[chosen_idx]
            print((chosen_idx,T))
            
            for t in range(T):
                action, _ = chosen_model.predict(obs)
                obs, _, done, _ = env.step(action)
                # Render the environment and save the frame to the video
                if t % 20 == 0:
                    x, y, _ = env.robot.body_xyz
                    visited_positions.append((x, y))
                    x_dumb += DIRECTIONS[chosen_idx][0]*0.0005
                    y_dumb  += DIRECTIONS[chosen_idx][1]*0.0005

                if done:
                    break
            dumb_errors.append((i,(np.abs(x_dumb-x)+np.abs(y_dumb-y))/2) )

    error1,error2,error3 = [] , [] , [] 
    for e in np.array(dumb_errors):
        if e[0]==0:
            error1.append(e[1])
        elif e[0]==1:
            error2.append(e[1])
        elif e[0]==2:
            error3.append(e[1])

    return error1,error2,error3 


def generate_data_walking_policies(num_sequences=20, 
                                   path="trained_nets", 
                                   validation_size=0.05, 
                                   T_range = [50,333]):

    env = gym.make('AntBulletEnv-v0')

    # Load all models
    with SuppressPrints():
        models = [PPO.load(os.path.join(path, f"ppo_ant_dir_{i}")) for i in range(9)]

    # Data collection
    data, sequence = [] , [] 
    obs = env.reset()

    for i in range(num_sequences):
        if i%100==0:
            print(f'Generated {i} episodes of {num_sequences}')
        chosen_idx = np.random.randint(0, 9)
        chosen_model = models[chosen_idx]
        initial_state = np.concatenate([obs, env.robot.body_xyz])
        T = np.random.randint(T_range[0], T_range[1])
        env.robot.walk_target_x, env.robot.walk_target_y = DIRECTIONS[chosen_idx]
        env.walk_target_x, env.walk_target_y = DIRECTIONS[chosen_idx]

        for t in range(T):
            action, _ = chosen_model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            
            # If T timesteps are reached or episode terminates
            if t == T - 1 or done:
                final_state = np.concatenate([obs, env.robot.body_xyz])
                prim_params = (DIRECTIONS[chosen_idx][0]*(t+1)*0.001*0.003,DIRECTIONS[chosen_idx][1]*(t+1)*0.001*0.003)  # t+1 because t is 0-indexed
                sequence.append((initial_state, final_state, prim_params))
                
                # If the episode ends prematurely or we have enough data for this sequence
                if done:
                    data.append(sequence)
                    obs = env.reset()
                    sequence = []
                    break

    # Split the data into training and validation sets
    split_idx = int(len(data) * (1-validation_size))
    train_data_sequences = data[:split_idx]
    valid_data_sequences = data[split_idx:]

    # Convert sequences to tensors
    train_data = []
    for seq in train_data_sequences:
        tensor_seq = [(torch.tensor(initial_state, dtype=torch.float32),
                       torch.tensor(final_state, dtype=torch.float32),
                       torch.tensor(prim_params, dtype=torch.float32)) for initial_state, final_state, prim_params in seq]
        train_data.append(tensor_seq)

    valid_data = []
    for seq in valid_data_sequences:
        tensor_seq = [(torch.tensor(initial_state, dtype=torch.float32),
                       torch.tensor(final_state, dtype=torch.float32),
                       torch.tensor(prim_params, dtype=torch.float32)) for initial_state, final_state, prim_params in seq]
        valid_data.append(tensor_seq)

    return train_data, valid_data



def extract_position_from_data(train_data, valid_data):

    # Modify train_data
    for sequence_idx, sequence in enumerate(train_data):
        modified_sequence = []
        for data_point in sequence:
            modified_sequence.append((data_point[0][-3:-1].clone(), data_point[1][-3:-1].clone(), data_point[2].clone()))
        train_data[sequence_idx] = modified_sequence

    # Modify valid_data
    for sequence_idx, sequence in enumerate(valid_data):
        modified_sequence = []
        for data_point in sequence:
            modified_sequence.append((data_point[0][-3:-1].clone(), data_point[1][-3:-1].clone(), data_point[2].clone()))
        valid_data[sequence_idx] = modified_sequence
        
    return train_data, valid_data

def train_ppo_walking():

    path = 'trained_nets'
    # Training each model
    i=0
    for x_dir, y_dir in DIRECTIONS:
        env = gym.make('AntBulletEnv-v0')
        env.walk_target_x, env.robot.walk_target_x = x_dir, x_dir
        env.walk_target_y, env.robot.walk_target_y = y_dir, y_dir
        
        model = PPO("MlpPolicy", env, verbose=1)

        model.learn(total_timesteps=500000)
        
        # Save the model
        model_name = f"{path}/ppo_ant_dir_{i}"
        model.save(model_name)
        env.close()
        i+=1




def generate_episode_with_planner(planner,env,plan_length=2):

    env = gym.make('AntBulletEnv-v0')

    # Load all models
    models = [PPO.load(os.path.join("trained_nets", f"ppo_ant_dir_{i}")) for i in range(9)]

    # Data collection
    data, sequence = [] , []
    # Data collection
    obs,done = env.reset(),False
    current_state = np.concatenate([env.robot.body_xyz[:-1]])
    current_state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    hidden_planner = planner.init_hidden(1)
    padded_zeros = torch.zeros(current_state_tensor.shape[0], plan_length-1, current_state_tensor.shape[2])
    planner_input = torch.cat((current_state_tensor, padded_zeros), dim=1)

    # Cumpute goals with Planner
    goals_pred, _ = planner(planner_input, hidden_planner)
    
    initial_state = np.concatenate([env.robot.body_xyz[:-1]])

    for i in range(plan_length):

        prim_array = goals_pred[0][i].clone().detach().numpy()
        chosen_idx,T = map_to_direction(prim_array)
        chosen_model = models[chosen_idx]
        env.robot.walk_target_x, env.robot.walk_target_y = DIRECTIONS[chosen_idx]
        env.walk_target_x, env.walk_target_y = DIRECTIONS[chosen_idx]

        for t in range(T):
            action, _ = chosen_model.predict(obs)
            obs, _, done, _ = env.step(action)
            
            if t == T - 1 or done:
                final_state = np.concatenate([env.robot.body_xyz[:-1]])
                final_state = np.concatenate([env.robot.body_xyz[:-1]])
                prim_params = (DIRECTIONS[chosen_idx][0]*(t+1)*0.001*0.003,DIRECTIONS[chosen_idx][1]*(t+1)*0.001*0.003)  # t+1 because t is 0-indexed
                sequence.append((initial_state, final_state, prim_params))
                
            if done:
                break
        current_state = np.concatenate([env.robot.body_xyz[:-1]])
        current_state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

    data.append(sequence)
    # Split the data into training and validation sets
    train_data_sequences = data

    # Convert sequences to tensors
    train_data = []
    for seq in train_data_sequences:
        tensor_seq = [(torch.tensor(initial_state, dtype=torch.float32),
                       torch.tensor(final_state, dtype=torch.float32),
                       torch.tensor(new_dmp_params, dtype=torch.float32)) for initial_state, final_state, new_dmp_params in seq]
        train_data.append(tensor_seq)

    return train_data



def evaluate_planner(planner, env, task='one_goal', num_eval_episodes=1, plan_length=2):
    planner.eval()
    eval_rewards = []
    with torch.no_grad():
        env = gym.make('AntBulletEnv-v0')
        with SuppressPrints():
            models = [PPO.load(os.path.join("trained_nets", f"ppo_ant_dir_{i}"),env) for i in range(9)]

        for episode in range(num_eval_episodes):
            episode_reward = []

            # Data collection
            obs,done = env.reset(),False
            current_state = np.concatenate([env.robot.body_xyz[:-1]])
            current_state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
            hidden_planner = planner.init_hidden(1)
            padded_zeros = torch.zeros(current_state_tensor.shape[0], plan_length-1, current_state_tensor.shape[2])
            planner_input = torch.cat((current_state_tensor, padded_zeros), dim=1)

            # Cumpute goals with Planner
            goals_pred, _ = planner(planner_input, hidden_planner)

            for i in range(plan_length):
                prim_array = goals_pred[0][i].clone().detach().numpy()
                chosen_idx,T = map_to_direction(prim_array)
                chosen_model = models[chosen_idx]
                env.robot.walk_target_x, env.robot.walk_target_y = DIRECTIONS[chosen_idx]
                env.walk_target_x, env.walk_target_y = DIRECTIONS[chosen_idx]
                print((chosen_idx,T))
                for t in range(T):
                    action, _ = chosen_model.predict(obs)
                    obs, _, done, _ = env.step(action)
                    if done:
                        break
                print(np.concatenate([env.robot.body_xyz[:-1]]))
                episode_reward.append(env.robot.body_xyz[0])

            total_episode_reward = episode_reward[-1]
            eval_rewards.append(total_episode_reward)

    avg_reward = np.mean(eval_rewards)

    return avg_reward



def train_planner_walk(train_data, planner, pred_net,env,
                      finetune_loops=0,
                      plan_length=7,
                      eval_freq = 1,
                      num_eval_episodes=10,
                      learning_rate=0.0001,
                      num_epochs=50,
                      batch_size=32,
                      clip_value=1.0,
                      early_stopping_threshold=1,
                      ):
    

    train_data = [seq[:plan_length] for seq in train_data if len(seq) >= plan_length]
    # Optimizer
    optimizer = optim.Adam(planner.parameters(), lr=learning_rate, weight_decay=1e-5)

    all_avg_rewards, finetune_count, processed_batches, vae_loss, pred_loss = [] , 0 , -1, 0 , 0
    epoch=0
    while True:
        
        # Training
        planner.train()
        pred_net.train()
        train_rewards = []

        for i in range(0, len(train_data), batch_size):
            processed_batches+=1
            optimizer.zero_grad() 
            
            sequences = train_data[i:i+batch_size]

            # Extract data from sequences
            state_seq_list = [torch.stack([t[0] for t in seq]) for seq in sequences]
            state_seq = torch.stack(state_seq_list)
            
            # Initialize hidden states for both the planner and pred_net at the beginning of each sequence
            hidden_pred_net = pred_net.init_hidden(len(state_seq))
            hidden_planner = planner.init_hidden(len(state_seq))

            # Build padded input to Planner
            current_state = state_seq[:, 0:1]  # Add sequence dimension
            padded_zeros = torch.zeros_like(state_seq[:, 1:])
            planner_input = torch.cat((current_state, padded_zeros), dim=1)

            # Cumpute goals with Planner
            goals_pred, _ = planner(planner_input, hidden_planner)

            # Build padded input to Pred Net
            first_elem = torch.cat((state_seq[:, 0], goals_pred[:, 0]), dim=1)
            padded_states = torch.zeros_like(state_seq[:, 1:])
            subsequent_elems = torch.cat((padded_states, goals_pred[:, 1:]), dim=2)
            s_t_extended_seq = torch.cat((first_elem.unsqueeze(1), subsequent_elems), dim=1)
            predicted_states, _ = pred_net(s_t_extended_seq, hidden_pred_net)


            # Distance Losses

            # One goal
            distance_loss = (torch.abs(predicted_states[:, 1, 0]-3) + torch.abs(predicted_states[:, 1, 1]-4.5) + 
                             torch.abs(predicted_states[:, 2, 0]-3) + torch.abs(predicted_states[:, 2, 1]-4.5)).mean()       

            # Two goals
            # distance_loss = (torch.abs(predicted_states[:, 0, 0]-0) + torch.abs(predicted_states[:, 0, 1]-2) + 
            #                  torch.abs(predicted_states[:, 1, 0]-3) + torch.abs(predicted_states[:, 1, 1]-2) + 
            #                  torch.abs(predicted_states[:, 2, 0]-3) + torch.abs(predicted_states[:, 2, 1]-2)).mean()       

            # Explore and Return
            # distance_loss = (predicted_states[:, 0, 0] + predicted_states[:, 0, 1] + 
            #                   torch.abs(predicted_states[:, 1, 0]-current_state[:,0,0]) + 
            #                   torch.abs(predicted_states[:, 1, 1]-current_state[:,0,1]) + 
            #                   torch.abs(predicted_states[:, 2, 0]-current_state[:,0,0]) + 
            #                   torch.abs(predicted_states[:, 2, 1]-current_state[:,0,1])).mean()       

            
            # One goal + Obstaclle
            # distance_loss = (torch.abs(predicted_states[:, 2, 0]-3) + torch.abs(predicted_states[:, 2, 1]-0.5) + 
            #                  - torch.abs(predicted_states[:, 0, 1])- torch.abs(predicted_states[:, 1, 1])).mean()      

            # Safe Area
            # distance_loss = (5*torch.abs(predicted_states[:, :, 0]-2.5) -torch.clamp(torch.var(predicted_states[:, :, 1]),0,2)).mean() 


            # Total Loss
            loss = distance_loss 
                
            # Optimize
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(planner.parameters(), clip_value)
            optimizer.step()
                
            # Append
            train_rewards.append(loss.item())

            if processed_batches%eval_freq==0:
                epoch+=1
                # Finetune Pred Net
                for _ in range(finetune_loops):
                    train_data_pred = generate_episode_with_planner(planner,env,plan_length=plan_length)
                    if len(train_data_pred)>0:
                        finetune_count +=1
                        pred_loss, pred_net = train_pred_net_rnn(
                            train_data=train_data_pred,
                            valid_data=train_data_pred,
                            pred_net=pred_net,
                            plan_length=plan_length,
                            learning_rate=0.001,
                            num_epochs=2,
                            batch_size=1,
                            eval_freq=1,
                            verbose=False,
                            random_padding=True
                        )

                reward = evaluate_planner(planner, env,num_eval_episodes=num_eval_episodes, plan_length=plan_length)
                all_avg_rewards.append(reward)

                print(f'Epoch {processed_batches*batch_size  / (len(train_data)):.4f}: Train = {loss.item():.4f}  Dist: {distance_loss:.4f}   Rec:{vae_loss:.4f}  Pred:{np.mean(pred_loss):.4f}   Perf: {reward:.4f}  Tuned: {finetune_count}')
                finetune_count = 0
                # Check for early stopping condition
                if reward > early_stopping_threshold:
                    print("Stopping early due to surpassing the performance threshold.")
                    break
            if epoch>num_epochs:
                return planner, all_avg_rewards
            



if __name__ == '__main__':

    # train_ppo_walking() # Train PPO policies for all walking directions (they are already saved)

    # Generate datasets with the trained models
    train_data_rnn, valid_data_rnn = generate_data_walking_policies(num_sequences=500,T_range=[50,333])

    # with open('datasets/train_data_walk_T50to333.pkl', 'wb') as f:
    #     pickle.dump(train_data_rnn, f)
    # with open('datasets/valid_data_walk_T50to333.pkl', 'wb') as f:
    #     pickle.dump(valid_data_rnn, f)

    with open('datasets/train_data_walk_T50to333.pkl', 'rb') as f:
        train_data_rnn = pickle.load(f)
    with open('datasets/valid_data_walk_T50to333.pkl', 'rb') as f:
        valid_data_rnn = pickle.load(f)

    train_data,valid_data = extract_position_from_data(train_data_rnn,valid_data_rnn)
    # train_data,valid_data = train_data_rnn,valid_data_rnn

    # Instantiate the RNN
    rnn_pred_net = RNNMujoco(
        input_size=2 + 2,
        hidden_size=64,
        output_size=2,
        num_layers=2,
        dropout_rate=0.1,
        bidirectional=False
    )

    average_validation_loss, rnn_pred_net = train_pred_net_rnn(
        train_data=train_data,
        valid_data=valid_data,
        pred_net=rnn_pred_net,
        plan_length=3,
        learning_rate=0.001,
        num_epochs=10,
        batch_size=32,
        eval_freq=100,
        random_padding=False
    )

    # error1,error2,error3 = compute_dumb_pred_error(1000)
    dumb_errors = 1 - np.array([1.41 ,2.96 ,3.75])/20  # (20 is the approximate size of the arena)
    plt.figure(figsize=(4,2))
    perf = np.array(average_validation_loss)
    plt.plot(np.linspace(0,1000,len(perf[:1000,0])),1-np.array(perf[:1000,0])/20,color='#4B0082',linewidth=3)
    plt.plot(np.linspace(0,1000,len(perf[:1000,1])),1-np.array(perf[:1000,1])/20,color='#4B0082',linewidth=3,alpha=0.7)
    plt.plot(np.linspace(0,1000,len(perf[:1000,2])),1-np.array(perf[:1000,2])/20,color='#4B0082',linewidth=3,alpha=0.3)
    plt.show()
    # torch.save(rnn_pred_net.state_dict(), 'trained_nets/rnn_Ant_walk_T50to300_planlength4.pth')

    rnn_pred_net.load_state_dict(torch.load('trained_nets/rnn_Ant_walk_T50to300.pth'))

    planner = RNNPlanner(
        input_size=2,
        hidden_size=64,
        num_layers=2,
        dropout_rate=0
    )

    env = gym.make('AntBulletEnv-v0')

    planner,all_avg_rewards = train_planner_walk(
        train_data=train_data,
        planner=planner,
        pred_net=rnn_pred_net,
        env=env,
        eval_freq=2000,
        num_eval_episodes = 1,
        learning_rate=0.001,
        num_epochs=10,
        batch_size=5,
        early_stopping_threshold=100,
        plan_length=3,
        finetune_loops=0,
    )