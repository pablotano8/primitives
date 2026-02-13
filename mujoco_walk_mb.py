"""
Model-Based RL Baseline for Mujoco Ant Walking Task

This implements a Dyna-style model-based RL approach where:
1. We use the same world model (RNNMujoco) as in the original agent
2. Instead of differentiable reward optimization, we use PPO to train the planning network
3. Training uses simulated trajectories from the world model (same number of samples/iterations)
4. Evaluation uses the same zero-shot metrics as the original model
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import os
import sys
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import gymnasium as gym
import pybullet_envs_gymnasium  # <-- this registers AntBulletEnv-v0 etc.



DIRECTIONS = [
    (-1e3, 0),
    (1e3, 0),
    (0, 1e3),
    (0, -1e3),
    (-1e3, 1e3),
    (1e3, -1e3),
    (1e3, 1e3),
    (-1e3, -1e3),
    (0, 0)
]


class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class RNNMujoco(nn.Module):
    """World Model: Predicts next state given current state and primitive parameters"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_rate=0.1, symm=False, bidirectional=False):
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


class RNNPlanner(nn.Module):
    """Planning Network: Outputs primitive parameters given state"""
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
        
        # Get the output (direction parameters)
        out = -1 + torch.sigmoid(self.direction_fc(out)) * 2

        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size),
                torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))


def map_to_direction(output):
    """Map continuous output to discrete direction and number of steps"""
    with torch.no_grad():
        direction = output * 1

        direction[np.abs(direction) < 0.25] = 0

        number_of_steps = np.max(np.abs(output)).item() * 333

        direction[direction > 0] = 1
        direction[direction < 0] = -1
        
        idx = np.where([d == (direction[0].item() * 1e3, direction[1].item() * 1e3) for d in DIRECTIONS])[0][0]

        return idx, int(number_of_steps)


def generate_data_walking_policies(num_sequences=20, 
                                   path=None, 
                                   validation_size=0.05, 
                                   T_range=[50, 333]):
    """Generate training data from walking policies"""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_nets')

    env = gym.make('AntBulletEnv-v0')

    # Load all models
    with SuppressPrints():
        models = [PPO.load(os.path.join(path, f"ppo_ant_dir_{i}")) for i in range(9)]

    # Data collection
    data, sequence = [], []
    obs, _ = env.reset()

    for i in range(num_sequences):
        if i % 100 == 0:
            print(f'Generated {i} episodes of {num_sequences}')
        chosen_idx = np.random.randint(0, 9)
        chosen_model = models[chosen_idx]
        initial_state = np.concatenate([obs, env.unwrapped.robot.body_xyz])
        T = np.random.randint(T_range[0], T_range[1])
        env.unwrapped.robot.walk_target_x, env.unwrapped.robot.walk_target_y = DIRECTIONS[chosen_idx]
        env.unwrapped.walk_target_x, env.unwrapped.walk_target_y = DIRECTIONS[chosen_idx]

        for t in range(T):
            action, _ = chosen_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # If T timesteps are reached or episode terminates
            if t == T - 1 or done:
                final_state = np.concatenate([obs, env.unwrapped.robot.body_xyz])
                prim_params = (DIRECTIONS[chosen_idx][0] * (t + 1) * 0.001 * 0.003, 
                              DIRECTIONS[chosen_idx][1] * (t + 1) * 0.001 * 0.003)
                sequence.append((initial_state, final_state, prim_params))
                
                # If the episode ends prematurely or we have enough data for this sequence
                if done:
                    data.append(sequence)
                    obs, _ = env.reset()
                    sequence = []
                    break

    # Split the data into training and validation sets
    split_idx = int(len(data) * (1 - validation_size))
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
    """Extract only position information from the full state"""
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


def train_pred_net_rnn(train_data,
                       valid_data,
                       pred_net,
                       learning_rate=0.0001,
                       num_epochs=50,
                       batch_size=32,
                       eval_freq=1,
                       clip_value=1.0,
                       plan_length=10,
                       verbose=True,
                       random_padding=True):
    """Train the world model"""
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
            sequences = train_data[i:i + batch_size]
            t_star = torch.randint(0, plan_length, (len(sequences),))
            # Extract data from sequences and pad after t*
            s_t_seq_list, final_state_seq_list, dmp_params_seq_list = [], [], []
            for idx, seq in enumerate(sequences):
                s_t = [t[0] if j <= random_padding * t_star[idx] else torch.zeros_like(t[0]) for j, t in enumerate(seq)]
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

            processed_batches += 1
            train_losses.append(batch_loss.item())

            if processed_batches % eval_freq == 0:
                # Validation
                pred_net.eval()
                valid_losses = []
                future_step_losses = [0.0] * plan_length
                count_steps = [0] * plan_length

                with torch.no_grad():
                    for i in range(0, len(valid_data), batch_size):
                        sequences = valid_data[i:i + batch_size]
                        
                        # Extract data from sequences and pad after t*
                        s_t_seq_list, final_state_seq_list, dmp_params_seq_list = [], [], []
                        for idx, seq in enumerate(sequences):
                            s_t = [t[0] if j <= 0 else torch.zeros_like(t[0]) for j, t in enumerate(seq)]
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
                            loss = torch.mean(loss_per_feature)
                            future_step_losses[j] += loss.item()
                            count_steps[j] += 1
                                                
                        batch_loss = torch.mean(torch.abs(output - target_seq))
                        valid_losses.append(batch_loss.item())

                future_step_avg_losses = [total_loss / count if count != 0 else 0 for total_loss, count in zip(future_step_losses, count_steps)]
                average_validation_loss.append(future_step_avg_losses)
                if verbose:
                    print(f'Epoch {processed_batches * batch_size / (len(train_data)):.4f}: Train = {np.mean(train_losses):.4f}, Valid = {np.mean(valid_losses):.4f}')
                    print([f"{loss:.4f}" for loss in future_step_avg_losses])

    return average_validation_loss, pred_net


class ModelBasedEnv(gym.Env):
    """
    Gym environment that uses the world model (RNNMujoco) for transition dynamics.
    This enables Dyna-style planning where PPO trains on simulated trajectories.
    """
    def __init__(self, 
                 world_model, 
                 target_goal,
                 task='one_goal',
                 plan_length=3):
        super(ModelBasedEnv, self).__init__()
        
        self.world_model = world_model
        self.world_model.eval()
        self.target_goal = torch.tensor(target_goal, dtype=torch.float32)
        self.task = task
        self.plan_length = plan_length
        
        # State dimension: position (2)
        self.state_dim = 2
        
        # Action space: primitive parameters for plan_length primitives
        # Each primitive: 2 dimensions (x_dir, y_dir)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2 * plan_length,), dtype=np.float32)
        
        # Observation space
        self.observation_space = gym.spaces.Box(low=-20., high=20., shape=(self.state_dim,), dtype=np.float32)
        
        self.s_t = None
        self.start_position = None

    def step(self, action):
        """Execute action using the world model for state prediction."""
        action = torch.tensor(action, dtype=torch.float32)
        
        # Parse action into primitive parameters
        prim_params = action.reshape(self.plan_length, 2)
        
        # Get state tensor for world model input
        current_state = torch.tensor(self.start_position, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Use world model to predict sequence of positions
        predicted_positions = []
        with torch.no_grad():
            hidden = self.world_model.init_hidden(1)
            
            for i in range(self.plan_length):
                # Build input: state + primitive parameters
                if i == 0:
                    state_input = current_state.squeeze(1)
                else:
                    state_input = torch.zeros_like(current_state.squeeze(1))
                
                prim_input = prim_params[i:i+1]
                full_input = torch.cat([state_input, prim_input], dim=-1).unsqueeze(1)
                
                # Predict next position
                output, hidden = self.world_model(full_input, hidden)
                pred_pos = output.squeeze()
                predicted_positions.append(pred_pos)
        
        # Calculate reward based on task
        if self.task == 'one_goal':
            # Task: reach goal at timestep 1 and 2
            distance_loss = 0.3*(torch.abs(predicted_positions[0][0] - 3) + 
                           torch.abs(predicted_positions[0][1] - 2) + 
                           torch.abs(predicted_positions[1][0] - 3) + 
                           torch.abs(predicted_positions[1][1] - 2) + 
                           torch.abs(predicted_positions[2][0] - 3) + 
                           torch.abs(predicted_positions[2][1] - 2)).item()
            reward = -distance_loss
            
        elif self.task == 'two_goals':
            # Task: reach two goals
            target_goal2 = torch.tensor([3.0, 2.0], dtype=torch.float32)
            distance_loss = (torch.abs(predicted_positions[0][0] - 0) + 
                           torch.abs(predicted_positions[0][1] - 2) + 
                           torch.abs(predicted_positions[1][0] - 3) + 
                           torch.abs(predicted_positions[1][1] - 2) + 
                           torch.abs(predicted_positions[2][0] - 3) + 
                           torch.abs(predicted_positions[2][1] - 2)).item()
            reward = -distance_loss
            
        elif self.task == 'explore_and_return':
            # Task: explore and return to start
            current_state_tensor = torch.tensor(self.start_position, dtype=torch.float32)
            distance_loss = (predicted_positions[0][0] + predicted_positions[0][1] + 
                          torch.abs(predicted_positions[1][0] - current_state_tensor[0]) + 
                          torch.abs(predicted_positions[1][1] - current_state_tensor[1]) + 
                          torch.abs(predicted_positions[2][0] - current_state_tensor[0]) + 
                          torch.abs(predicted_positions[2][1] - current_state_tensor[1])).item()
            reward = -distance_loss
            
        elif self.task == 'goal_and_obstacle':
            # Task: reach goal while avoiding obstacle
            distance_loss = (torch.abs(predicted_positions[2][0] - 3) + 
                           torch.abs(predicted_positions[2][1] - 2) - 
                           torch.abs(predicted_positions[0][1]) - 
                           torch.abs(predicted_positions[1][1])).item()
            reward = -distance_loss
            
        elif self.task == 'safe_area':
            # Task: stay in safe area
            distance_loss = (5 * torch.abs(predicted_positions[-1][0] - 2.5) - 
                           torch.clamp(torch.var(torch.stack([p[1] for p in predicted_positions])), 0, 2)).item()
            reward = -distance_loss
        
        # Store predictions
        self.predicted_positions = [p.numpy() for p in predicted_positions]
        self.prim_params = prim_params
        
        # Update state to final position
        self.s_t = predicted_positions[-1].numpy()
        
        terminated = True
        truncated = False
        return self.s_t, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset environment with starting position at origin"""
        if seed is not None:
            np.random.seed(seed)
        
        self.start_position = np.array([0.0, 0.0])
        self.s_t = self.start_position.copy()
        return self.s_t, {}


class RealWorldEnv(gym.Env):
    """
    Gym environment that uses real physics simulation for evaluation.
    """
    def __init__(self, 
                 target_goal,
                 task='one_goal',
                 plan_length=3):
        super(RealWorldEnv, self).__init__()
        
        self.target_goal = torch.tensor(target_goal, dtype=torch.float32)
        self.task = task
        self.plan_length = plan_length
        
        self.state_dim = 2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2 * plan_length,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-20., high=20., shape=(self.state_dim,), dtype=np.float32)
        
        self.s_t = None
        self.start_position = None
        self.env = None
        self.models = None

    def step(self, action):
        """Execute action using real physics simulation"""
        action = torch.tensor(action, dtype=torch.float32)
        prim_params = action.reshape(self.plan_length, 2)
        
        # Initialize environment and models if needed
        if self.env is None:
            self.env = gym.make('AntBulletEnv-v0')
            _dir = os.path.dirname(os.path.abspath(__file__))
            with SuppressPrints():
                self.models = [PPO.load(os.path.join(_dir, "trained_nets", f"ppo_ant_dir_{i}")) for i in range(9)]
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Execute primitives in sequence
        positions = [np.array(self.start_position)]
        for i in range(self.plan_length):
            prim_array = prim_params[i].clone().detach().numpy()
            chosen_idx, T = map_to_direction(prim_array)
            chosen_model = self.models[chosen_idx]
            self.env.unwrapped.robot.walk_target_x, self.env.unwrapped.robot.walk_target_y = DIRECTIONS[chosen_idx]
            self.env.unwrapped.walk_target_x, self.env.unwrapped.walk_target_y = DIRECTIONS[chosen_idx]
            
            for t in range(T):
                action_low, _ = chosen_model.predict(obs)
                obs, _, terminated, truncated, _ = self.env.step(action_low)
                done = terminated or truncated
                
                # Sample positions every 10 steps for better visualization
                if t % 10 == 0:
                    position = np.array([self.env.unwrapped.robot.body_xyz[0], self.env.unwrapped.robot.body_xyz[1]])
                    positions.append(position)
                
                if done:
                    break
            
            final_position = np.array([self.env.unwrapped.robot.body_xyz[0], self.env.unwrapped.robot.body_xyz[1]])
            positions.append(final_position)
        
        self.positions = positions
        self.prim_params = prim_params
        
        # Calculate reward based on task
        if self.task == 'one_goal':
            loss = 0.3*np.mean(np.abs(positions[-1] - np.array(self.target_goal)) + np.abs(positions[-2] - np.array(self.target_goal)) + np.abs(positions[-3] - np.array(self.target_goal)))
            reward = -loss
        elif self.task == 'two_goals':
            # Implement other tasks as needed
            loss = np.mean(np.abs(positions[-1] - np.array(self.target_goal)))
            reward = -loss
        elif self.task == 'goal_and_obstacle':
            loss = (np.abs(positions[-1][0] - 3) + np.abs(positions[-1][1] - 2) - 
                    0.1*np.abs(positions[1][1]) - 0.1*np.abs(positions[2][1]))
            reward = -loss
        
        self.s_t = positions[-1]
        terminated = True
        truncated = False
        return self.s_t, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.start_position = np.array([0.0, 0.0])
        self.s_t = self.start_position.copy()
        return self.s_t, {}


def evaluate_model(model, target_goal, task='one_goal', num_episodes=50, plan_length=3):
    """Evaluate the trained policy using real physics simulation."""
    env = RealWorldEnv(
        target_goal=target_goal,
        task=task,
        plan_length=plan_length
    )
    
    performances = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        initial_state = env.start_position.copy()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, _ = env.step(action)
        
        # Get positions from the environment
        positions = env.positions
        
        # Compute loss based on task type
        if task == 'one_goal':
            # Single goal at the end
            loss = 0.3*(np.abs(positions[1][0] - 3) + np.abs(positions[1][1] - 2) +
                    np.abs(positions[2][0] - 3) + np.abs(positions[2][1] - 2) +
                    np.abs(positions[3][0] - 3) + np.abs(positions[3][1] - 2))
        
        elif task == 'two_goals':
            # Two goals: waypoints at each step
            # Matches: (0,2), (3,2), (3,2)
            loss = (np.abs(positions[1][0] - 0) + np.abs(positions[1][1] - 2) +
                    np.abs(positions[2][0] - 3) + np.abs(positions[2][1] - 2) +
                    np.abs(positions[3][0] - 3) + np.abs(positions[3][1] - 2))
        
        elif task == 'explore_return':
            # Explore away then return to start
            loss = (positions[1][0] + positions[1][1] +  # Maximize first position
                    np.abs(positions[2][0] - initial_state[0]) +  # Return close to start
                    np.abs(positions[2][1] - initial_state[1]) +
                    np.abs(positions[3][0] - initial_state[0]) +
                    np.abs(positions[3][1] - initial_state[1]))
        
        elif task == 'goal_and_obstacle':
            # Goal at end, avoid obstacles in middle steps
            loss = (np.abs(positions[3][0] - 3) + np.abs(positions[3][1] - 2) -
                    0.1*np.abs(positions[1][1]) - 0.1*np.abs(positions[2][1]))
        
        elif task == 'safe_area':
            # Stay near x=2.5, maximize variety in y
            positions_array = np.array(positions[1:])  # Skip initial
            loss = 5 * np.mean(np.abs(positions_array[:, 0] - 2.5)) - np.clip(np.var(positions_array[:, 1]), 0, 2)
        
        else:
            # Default to single goal
            loss = np.mean(np.abs(positions[-1] - np.array(target_goal)))
        
        # Normalize by approximate arena size (20)
        performance = 1 - loss / 20
        performances.append(performance)
    
    return np.mean(performances)


def plot_trajectories_mb(model, target_goal, task='one_goal', plan_length=3, num_plots=1):
    """Plot example trajectories from the model-based policy"""
    env = RealWorldEnv(
        target_goal=target_goal,
        task=task,
        plan_length=plan_length
    )
    
    for _ in range(num_plots):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, _ = env.step(action)
        
        positions = np.array(env.positions)
        plt.figure(figsize=(6, 6))
        plt.plot(positions[:, 0], positions[:, 1], 'o-', color="#555555", linewidth=1, markersize=5)
        plt.scatter([target_goal[0]], [target_goal[1]], color='red', s=200, marker='*', zorder=5)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Model-Based RL Trajectory ({task})')
        plt.show()


class CustomEvalCallback:
    """Callback for evaluating the model during training"""
    def __init__(self, eval_freq, target_goal, task='one_goal', plan_length=3, verbose=0, plot=False):
        self.eval_freq = eval_freq
        self.target_goal = target_goal
        self.task = task
        self.plan_length = plan_length
        self.verbose = verbose
        self.eval_log = []
        self.plot = plot
        self.n_calls = 0

    def __call__(self, locals_dict, globals_dict):
        self.n_calls += 1
        
        if self.n_calls % self.eval_freq == 0:
            model = locals_dict.get('self')
            performance = evaluate_model(
                model,
                self.target_goal,
                self.task,
                num_episodes=50,
                plan_length=self.plan_length
            )
            self.eval_log.append(performance)
            if self.verbose > 0:
                print(f"N_Calls: {self.n_calls}   Performance: {performance:.4f}")
            
            if self.plot:
                plot_trajectories_mb(
                    model,
                    self.target_goal,
                    self.task,
                    self.plan_length
                )
        return True


def train_model_based_rl(world_model, 
                         target_goal,
                         task='one_goal',
                         plan_length=3,
                         total_timesteps=10000,
                         eval_freq=1000,
                         learning_rate=0.001,
                         batch_size=10,
                         n_steps=200,
                         finetune_loops=0,
                         plot=False):
    """
    Train a planning network using PPO with the world model for simulation.
    """
    env = ModelBasedEnv(
        world_model=world_model,
        target_goal=target_goal,
        task=task,
        plan_length=plan_length
    )
    
    policy_kwargs = dict(
        net_arch=dict(pi=[64], vf=[64])
    )
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size
    )
    
    # Manual evaluation loop since stable_baselines3 callbacks work differently
    eval_log = []
    steps_per_iteration = n_steps
    num_iterations = total_timesteps // steps_per_iteration
    finetune_count = 0
    pred_loss = 0
    
    for iteration in range(num_iterations):
        model.learn(total_timesteps=steps_per_iteration, reset_num_timesteps=False)
        
        if (iteration + 1) * steps_per_iteration % eval_freq == 0:
            # Finetune world model
            for _ in range(finetune_loops):
                train_data_pred = generate_episode_with_model(model, plan_length=plan_length)
                if len(train_data_pred) > 0:
                    finetune_count += 1
                    pred_loss, world_model = train_pred_net_rnn(
                        train_data=train_data_pred,
                        valid_data=train_data_pred,
                        pred_net=world_model,
                        plan_length=plan_length,
                        learning_rate=0.001,
                        num_epochs=2,
                        batch_size=1,
                        eval_freq=1,
                        verbose=False,
                        random_padding=True
                    )
            
            performance = evaluate_model(
                model,
                target_goal,
                task=task,
                num_episodes=50,
                plan_length=plan_length
            )
            eval_log.append(performance)
            print(f"Iteration {iteration + 1}/{num_iterations}: Performance = {performance:.4f}  Tuned: {finetune_count}")
            finetune_count = 0
            
            if plot:
                plot_trajectories_mb(model, target_goal, task, plan_length)
    
    return model, eval_log


def generate_episode_with_model(model, plan_length=3):
    """Generate an episode using the trained model-based policy for world model finetuning"""
    env = gym.make('AntBulletEnv-v0')
    
    # Load all primitive models
    _dir = os.path.dirname(os.path.abspath(__file__))
    with SuppressPrints():
        models = [PPO.load(os.path.join(_dir, "trained_nets", f"ppo_ant_dir_{i}")) for i in range(9)]
    
    # Data collection
    data, sequence = [], []
    obs, _ = env.reset()
    initial_state = np.concatenate([env.unwrapped.robot.body_xyz[:-1]])
    
    # Get action from model
    action, _ = model.predict(np.array([0.0, 0.0]), deterministic=True)
    prim_params = action.reshape(plan_length, 2)
    
    for i in range(plan_length):
        prim_array = prim_params[i]
        chosen_idx, T = map_to_direction(prim_array)
        chosen_model = models[chosen_idx]
        env.unwrapped.robot.walk_target_x, env.unwrapped.robot.walk_target_y = DIRECTIONS[chosen_idx]
        env.unwrapped.walk_target_x, env.unwrapped.walk_target_y = DIRECTIONS[chosen_idx]
        
        for t in range(T):
            action_low, _ = chosen_model.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action_low)
            done = terminated or truncated
            
            if t == T - 1 or done:
                final_state = np.concatenate([env.unwrapped.robot.body_xyz[:-1]])
                prim_params_scaled = (DIRECTIONS[chosen_idx][0]*(t+1)*0.001*0.003, 
                                     DIRECTIONS[chosen_idx][1]*(t+1)*0.001*0.003)
                sequence.append((initial_state, final_state, prim_params_scaled))
                
            if done:
                break
        
        if not done:
            initial_state = np.concatenate([env.unwrapped.robot.body_xyz[:-1]])
    
    if len(sequence) > 0:
        data.append(sequence)
    
    # Convert to tensors
    train_data = []
    for seq in data:
        tensor_seq = [(torch.tensor(initial_state, dtype=torch.float32),
                       torch.tensor(final_state, dtype=torch.float32),
                       torch.tensor(prim_params, dtype=torch.float32)) 
                      for initial_state, final_state, prim_params in seq]
        train_data.append(tensor_seq)
    
    return train_data


if __name__ == "__main__":
    
    # ============ Phase 1: Train World Model ============
    print("=" * 50)
    print("Phase 1: Training World Model (Mujoco Ant)")
    print("=" * 50)
    
    # Load or generate data
    try:
        with open('datasets/train_data_walk_T50to333.pkl', 'rb') as f:
            train_data_rnn = pickle.load(f)
        with open('datasets/valid_data_walk_T50to333.pkl', 'rb') as f:
            valid_data_rnn = pickle.load(f)
        print("Loaded existing training data")
    except FileNotFoundError:
        print("Generating new training data...")
        train_data_rnn, valid_data_rnn = generate_data_walking_policies(num_sequences=500, T_range=[50, 333])
        
        # Save the data
        os.makedirs('datasets', exist_ok=True)
        with open('datasets/train_data_walk_T50to333.pkl', 'wb') as f:
            pickle.dump(train_data_rnn, f)
        with open('datasets/valid_data_walk_T50to333.pkl', 'wb') as f:
            pickle.dump(valid_data_rnn, f)
    
    # Extract position data
    train_data, valid_data = extract_position_from_data(train_data_rnn, valid_data_rnn)
    
    # Instantiate the RNN world model
    rnn_pred_net = RNNMujoco(
        input_size=2 + 2,  # position (2) + primitive params (2)
        hidden_size=64,
        output_size=2,  # output position (2)
        num_layers=2,
        dropout_rate=0.1,
        bidirectional=False
    )
    
    # Try to load existing model
    try:
        rnn_pred_net.load_state_dict(torch.load('trained_nets/rnn_Ant_walk_T50to300.pth'))
        print("Loaded existing world model")
    except FileNotFoundError:
        print("Training new world model...")
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
        
        # Save the model
        os.makedirs('trained_nets', exist_ok=True)
        torch.save(rnn_pred_net.state_dict(), 'trained_nets/rnn_Ant_walk_T50to300.pth')
        
        # Plot learning curves
        plt.figure(figsize=(4, 2))
        perf = np.array(average_validation_loss)
        plt.plot(np.linspace(0, 1000, len(perf[:1000, 0])), 1 - np.array(perf[:1000, 0]) / 20, 
                color='#4B0082', linewidth=3)
        plt.plot(np.linspace(0, 1000, len(perf[:1000, 1])), 1 - np.array(perf[:1000, 1]) / 20, 
                color='#4B0082', linewidth=3, alpha=0.7)
        plt.plot(np.linspace(0, 1000, len(perf[:1000, 2])), 1 - np.array(perf[:1000, 2]) / 20, 
                color='#4B0082', linewidth=3, alpha=0.3)
        plt.xlabel("Training Steps")
        plt.ylabel("World Model Performance")
        plt.title("World Model Training")
        plt.show()
    
    rnn_pred_net.eval()
    
    # ============ Phase 2: Train Planning Network with PPO ============
    print("=" * 50)
    print("Phase 2: Training Planning Network (Model-Based PPO)")
    print("=" * 50)
    
    # Task: Reach One Goal
    target_goal = [3.0, 2.0]
    
    model, performance_log = train_model_based_rl(
        world_model=rnn_pred_net,
        target_goal=target_goal,
        task='two_goals',
        plan_length=3,
        total_timesteps=5000,
        eval_freq=2000,
        learning_rate=0.001,
        batch_size=5,
        n_steps=200,
        finetune_loops=2,
        plot=False
    )
    
    # ============ Phase 3: Evaluation ============
    print("=" * 50)
    print("Phase 3: Zero-Shot Evaluation")
    print("=" * 50)
    
    final_performance = evaluate_model(
        model, 
        target_goal, 
        task='two_goals',
        num_episodes=100,
        plan_length=3
    )
    print(f"Final Zero-Shot Performance: {final_performance:.4f}")

    
    plot_trajectories_mb(model, target_goal=[3,2], plan_length=3)
    