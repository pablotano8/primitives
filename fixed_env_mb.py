"""
Model-Based RL Baseline for Fixed Environment Task

This implements a Dyna-style model-based RL approach where:
1. We use the same world model (PredNet) as in the original agent
2. Instead of differentiable reward optimization, we use PPO to train the planning network
3. Training uses simulated trajectories from the world model (same number of samples/iterations)
4. Evaluation uses the same zero-shot metrics as the original model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from plot_trajectories import plot_example_trajectories, plot_predicted_vs_actual_trajectories
from utils import generate_trajectories_from_dmp_params
from continuous_nav_envs import World, generate_random_positions
from plot_trajectories import generate_and_plot_trajectories_from_parameters
from dmps import DMP1D, Simulation

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class PredNet(nn.Module):
    """World Model: Predicts next state given current state and DMP parameters"""
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
    """Planning Network: Outputs DMP parameters given state"""
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(GoalNet, self).__init__()
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
        complexity=1):
    """Generate training data for the world model"""
    data = []
    for i in range(number_of_trajectories):
        if i % 1000 == 0:
            print(f'Generated {i} of {number_of_trajectories} trajectories')

        start_position, _ = generate_random_positions(world, world_bounds=world_bounds, orientation=None)
        _, goal_position1 = generate_random_positions(world, world_bounds=world_bounds, orientation=None)

        dmp_x1 = DMP1D(start=start_position[0], goal=goal_position1[0], n_basis=3, complexity=complexity)
        dmp_y1 = DMP1D(start=start_position[1], goal=goal_position1[1], n_basis=3, complexity=complexity)

        simulation = Simulation(world, dmp_x1, dmp_y1, start_position, T=1.0, dt=0.01)
        positions1, velocities1, collision1, _, _ = simulation.run()

        _, goal_position2 = generate_random_positions(world, world_bounds=world_bounds, orientation=None)

        dmp_x2 = DMP1D(start=positions1[-1][0], goal=goal_position2[0], n_basis=3, complexity=complexity)
        dmp_y2 = DMP1D(start=positions1[-1][1], goal=goal_position2[1], n_basis=3, complexity=complexity)

        simulation = Simulation(world, dmp_x2, dmp_y2, positions1[-1], T=1.0, dt=0.01)
        positions2, velocities2, collision2, _, _ = simulation.run()

        dmp_params1 = [dmp_x1.start, dmp_y1.start, dmp_x1.goal, dmp_y1.goal, *dmp_x1.weights, *dmp_y1.weights]
        dmp_params2 = [dmp_x1.goal, dmp_y1.goal, dmp_x2.goal, dmp_y2.goal, *dmp_x2.weights, *dmp_y2.weights]

        s_t = positions1[0]
        data.append((s_t, True, positions1[-1], positions2[-1], dmp_params1, dmp_params2, any(collision1) * 1, any(collision2) * 1))

    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]
    random.shuffle(train_data)

    train_data = [(torch.tensor(s_t, dtype=torch.float32),
                   torch.tensor(s_t_plus_one, dtype=torch.float32),
                   torch.tensor(final_position1, dtype=torch.float32),
                   torch.tensor(final_position2, dtype=torch.float32),
                   torch.tensor(dmp_params1, dtype=torch.float32),
                   torch.tensor(dmp_params2, dtype=torch.float32),
                   torch.tensor(collision1, dtype=torch.float32),
                   torch.tensor(collision2, dtype=torch.float32)) for s_t, s_t_plus_one, final_position1, final_position2, dmp_params1, dmp_params2, collision1, collision2 in train_data]

    valid_data = [(torch.tensor(s_t, dtype=torch.float32),
                   torch.tensor(s_t_plus_one, dtype=torch.float32),
                   torch.tensor(final_position1, dtype=torch.float32),
                   torch.tensor(final_position2, dtype=torch.float32),
                   torch.tensor(dmp_params1, dtype=torch.float32),
                   torch.tensor(dmp_params2, dtype=torch.float32),
                   torch.tensor(collision1, dtype=torch.float32),
                   torch.tensor(collision2, dtype=torch.float32)) for s_t, s_t_plus_one, final_position1, final_position2, dmp_params1, dmp_params2, collision1, collision2 in valid_data]
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
    """Train the world model (PredNet)"""
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    criterion2 = nn.BCEWithLogitsLoss()

    average_validation_loss1, average_validation_loss2 = [], []

    for epoch in range(num_epochs):
        random.shuffle(train_data)

        net.train()
        train_losses = []
        for i in range(0, len(train_data) - batch_size, batch_size):
            batch = train_data[i:i + batch_size]
            batch_s_t = torch.stack([item[0] for item in batch])
            batch_final_position1 = torch.stack([item[2] for item in batch])
            batch_final_position2 = torch.stack([item[3] for item in batch])
            batch_dmp_params1 = torch.stack([item[4] for item in batch])
            batch_dmp_params2 = torch.stack([item[5] for item in batch])
            batch_coll1 = torch.stack([item[6] for item in batch]).view(-1)
            batch_coll2 = torch.stack([item[7] for item in batch]).view(-1)

            inputs1 = torch.cat((batch_s_t, batch_dmp_params1), dim=-1).unsqueeze(1)
            padded_state = torch.zeros_like(batch_s_t)
            inputs2 = torch.cat((padded_state, batch_dmp_params2), dim=-1).unsqueeze(1)

            optimizer.zero_grad()
            outputs1, hidden = net(inputs1)
            outputs2, _ = net(inputs2, hidden)

            pred_pos1 = outputs1.squeeze(1)[:, :2]
            pred_coll1 = outputs1.squeeze(1)[:, 2]
            pred_pos2 = outputs2.squeeze(1)[:, :2]
            pred_coll2 = outputs2.squeeze(1)[:, 2]

            loss_pos1 = criterion(pred_pos1, batch_final_position1)
            loss_pos2 = criterion(pred_pos2, batch_final_position2)
            loss_coll1 = criterion2(pred_coll1, batch_coll1)
            loss_coll2 = criterion2(pred_coll2, batch_coll2)
            loss = loss_pos1 + loss_pos2 + weight_collisions * (loss_coll1 + loss_coll2)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if epoch % eval_freq == 0:
            net.eval()
            valid_losses_pos1, valid_losses_pos2 = [], []
            with torch.no_grad():
                for i in range(0, len(valid_data), batch_size):
                    batch = valid_data[i:i + batch_size]
                    batch_s_t = torch.stack([item[0] for item in batch])
                    batch_final_position1 = torch.stack([item[2] for item in batch])
                    batch_final_position2 = torch.stack([item[3] for item in batch])
                    batch_dmp_params1 = torch.stack([item[4] for item in batch])
                    batch_dmp_params2 = torch.stack([item[5] for item in batch])

                    inputs1 = torch.cat((batch_s_t, batch_dmp_params1), dim=-1).unsqueeze(1)
                    padded_state = torch.zeros_like(batch_s_t)
                    inputs2 = torch.cat((padded_state, batch_dmp_params2), dim=-1).unsqueeze(1)

                    outputs1, hidden = net(inputs1)
                    outputs2, _ = net(inputs2, hidden)

                    pred_pos1 = outputs1.squeeze(1)[:, :2]
                    pred_pos2 = outputs2.squeeze(1)[:, :2]

                    loss_pos1 = torch.mean(torch.abs(pred_pos1 - batch_final_position1))
                    loss_pos2 = torch.mean(torch.abs(pred_pos2 - batch_final_position2))
                    valid_losses_pos1.append(loss_pos1.item())
                    valid_losses_pos2.append(loss_pos2.item())

            average_validation_loss1.append(np.mean(valid_losses_pos1))
            average_validation_loss2.append(np.mean(valid_losses_pos2))
            print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss = {np.mean(train_losses):.4f}, Valid Loss Pos1 = {np.mean(valid_losses_pos1):.4f}, Valid Loss Pos2 = {np.mean(valid_losses_pos2):.4f}')

    return average_validation_loss1, average_validation_loss2, net


class ModelBasedEnv(gym.Env):
    """
    Gym environment that uses the world model (PredNet) for transition dynamics.
    This enables Dyna-style planning where PPO trains on simulated trajectories.
    """
    def __init__(self, 
                 world_model, 
                 target_goal,
                 task='reach_final_goal',
                 target_goal2=None,
                 bound_dmp_weights=1.0,
                 world=None,
                 world_bounds=None):
        super(ModelBasedEnv, self).__init__()
        
        self.world_model = world_model
        self.world_model.eval()
        self.target_goal = torch.tensor(target_goal, dtype=torch.float32)
        self.target_goal2 = torch.tensor(target_goal2 if target_goal2 else target_goal, dtype=torch.float32)
        self.task = task
        self.bound_dmp_weights = bound_dmp_weights
        self.world = world
        self.world_bounds = world_bounds if world_bounds else [0.1, 0.9, 0.1, 0.9]
        
        # Action space: DMP parameters for 2 primitives
        # Each DMP: goal (2) + weights (6) = 8
        # Total: 16 params
        self.action_space = spaces.Box(low=-3, high=3, shape=(16,), dtype=np.float32)
        
        # Observation space: position (2)
        self.observation_space = spaces.Box(low=0., high=1., shape=(2,), dtype=np.float32)
        
        self.s_t = None
        self.start_position = None

    def step(self, action):
        """
        Execute action using the world model for state prediction.
        """
        action = torch.tensor(action, dtype=torch.float32)
        
        # Parse action into DMP parameters
        goal1 = action[:2]
        weights1 = action[2:8].clamp(-self.bound_dmp_weights, self.bound_dmp_weights)
        goal2 = action[8:10]
        weights2 = action[10:16].clamp(-self.bound_dmp_weights, self.bound_dmp_weights)
        
        # Construct full DMP params
        start_pos = torch.tensor(self.start_position, dtype=torch.float32)
        dmp_params1 = torch.cat([start_pos, goal1, weights1])
        dmp_params2 = torch.cat([goal1, goal2, weights2])
        
        # Get state tensor for world model input
        s_t = torch.tensor(self.s_t, dtype=torch.float32)
        
        # Use world model to predict final positions
        with torch.no_grad():
            inputs1 = torch.cat((s_t, dmp_params1)).unsqueeze(0).unsqueeze(0)
            padded_state = torch.zeros_like(s_t)
            inputs2 = torch.cat((padded_state, dmp_params2)).unsqueeze(0).unsqueeze(0)
            
            outputs1, hidden = self.world_model(inputs1)
            outputs2, _ = self.world_model(inputs2, hidden)
            
            pred_pos1 = outputs1.squeeze()[:2]
            pred_coll1 = torch.sigmoid(outputs1.squeeze()[2])
            pred_pos2 = outputs2.squeeze()[:2]
            pred_coll2 = torch.sigmoid(outputs2.squeeze()[2])
        
        # Calculate reward based on task
        criterion = nn.MSELoss()
        
        # Boundary penalties
        penalty_x1 = torch.relu(pred_pos1[0] - 0.9) + torch.relu(0.1 - pred_pos1[0])
        penalty_y1 = torch.relu(pred_pos1[1] - 0.9) + torch.relu(0.1 - pred_pos1[1])
        outside_square1 = (penalty_x1 + penalty_y1).item()
        
        penalty_x2 = torch.relu(pred_pos2[0] - 0.9) + torch.relu(0.1 - pred_pos2[0])
        penalty_y2 = torch.relu(pred_pos2[1] - 0.9) + torch.relu(0.1 - pred_pos2[1])
        outside_square2 = (penalty_x2 + penalty_y2).item()
        
        if self.task == 'reach_final_goal':
            loss_goal1 = criterion(pred_pos1, self.target_goal).item()
            loss_goal2 = criterion(pred_pos2, self.target_goal).item()
            reward = -(0.5 * loss_goal1 + loss_goal2) - 0.25 * (pred_coll1.item() + pred_coll2.item()) - outside_square1 - outside_square2
            
        elif self.task == 'reach_two_goals':
            loss_goal1 = criterion(pred_pos1, self.target_goal).item()
            loss_goal2 = criterion(pred_pos2, self.target_goal2).item()
            # Also try reverse assignment
            loss_goal1_rev = criterion(pred_pos1, self.target_goal2).item()
            loss_goal2_rev = criterion(pred_pos2, self.target_goal).item()
            # Use better assignment
            total_loss = min(loss_goal1 + loss_goal2, loss_goal1_rev + loss_goal2_rev)
            reward = -total_loss - 0.25 * (pred_coll1.item() + pred_coll2.item()) - outside_square1 - outside_square2
            
        elif self.task == 'goal_and_return':
            initial_pos = torch.tensor(self.start_position, dtype=torch.float32)
            loss_goal1 = criterion(pred_pos1, self.target_goal).item()
            loss_return = criterion(pred_pos2, initial_pos).item()
            reward = -(loss_goal1 + loss_return) - 0.25 * (pred_coll1.item() + pred_coll2.item()) - outside_square1 - outside_square2
            
        elif self.task == 'maximize_total_distance':
            initial_pos = torch.tensor(self.start_position, dtype=torch.float32)
            distance = criterion(pred_pos2, initial_pos).item()
            reward = distance - 0.25 * (pred_coll1.item() + pred_coll2.item()) - outside_square1 - outside_square2
        
        # Store predictions for evaluation
        self.pred_pos1 = pred_pos1.numpy()
        self.pred_pos2 = pred_pos2.numpy()
        self.pred_coll1 = pred_coll1.item() > 0.5
        self.pred_coll2 = pred_coll2.item() > 0.5
        self.dmp_params1 = dmp_params1
        self.dmp_params2 = dmp_params2
        
        # Update state
        self.s_t = pred_pos2.numpy()
        
        terminated = True
        truncated = False
        return self.s_t, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset environment with random initial state"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.start_position = generate_random_positions(self.world, world_bounds=self.world_bounds)[0]
        self.s_t = self.start_position
        return self.s_t, {}


class RealWorldEnv(gym.Env):
    """
    Gym environment that uses real physics simulation for evaluation.
    """
    def __init__(self, 
                 target_goal,
                 task='reach_final_goal',
                 target_goal2=None,
                 bound_dmp_weights=1.0,
                 world=None,
                 world_bounds=None):
        super(RealWorldEnv, self).__init__()
        
        self.target_goal = torch.tensor(target_goal, dtype=torch.float32)
        self.target_goal2 = torch.tensor(target_goal2 if target_goal2 else target_goal, dtype=torch.float32)
        self.task = task
        self.bound_dmp_weights = bound_dmp_weights
        self.world = world
        self.world_bounds = world_bounds if world_bounds else [0.1, 0.9, 0.1, 0.9]
        
        self.action_space = spaces.Box(low=-3, high=3, shape=(16,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0., high=1., shape=(2,), dtype=np.float32)
        
        self.s_t = None
        self.start_position = None

    def step(self, action):
        """Execute action using real physics simulation"""
        action = torch.tensor(action, dtype=torch.float32)
        
        goal1 = action[:2]
        weights1 = action[2:8].clamp(-self.bound_dmp_weights, self.bound_dmp_weights)
        goal2 = action[8:10]
        weights2 = action[10:16].clamp(-self.bound_dmp_weights, self.bound_dmp_weights)
        
        start_pos = torch.tensor(self.start_position, dtype=torch.float32)
        dmp_params1 = torch.cat([start_pos, goal1, weights1]).unsqueeze(0)
        dmp_params2 = torch.cat([goal1, goal2, weights2]).unsqueeze(0)
        
        final_position1, final_position2, collision_info = generate_trajectories_from_dmp_params(
            dmp_params1=dmp_params1,
            dmp_params2=dmp_params2,
            batch_size=1,
            batch_s_t=torch.tensor([self.start_position]),
            world=self.world)
        
        self.final_position1 = np.array(final_position1[0])
        self.final_position2 = np.array(final_position2[0])
        self.collision1 = collision_info[0]['collision1'].max() > 0
        self.collision2 = collision_info[0]['collision2'].max() > 0
        self.dmp_params1 = dmp_params1
        self.dmp_params2 = dmp_params2
        
        # Calculate reward
        criterion = nn.MSELoss()
        
        if self.task == 'reach_final_goal':
            loss = np.mean(np.abs(self.final_position2 - np.array(self.target_goal)))
            reward = -loss
        elif self.task == 'reach_two_goals':
            loss1 = np.mean(np.abs(self.final_position1 - np.array(self.target_goal)))
            loss2 = np.mean(np.abs(self.final_position2 - np.array(self.target_goal2)))
            reward = -(loss1 + loss2) / 2
        elif self.task == 'goal_and_return':
            loss1 = np.mean(np.abs(self.final_position1 - np.array(self.target_goal)))
            loss2 = np.mean(np.abs(self.final_position2 - np.array(self.start_position)))
            reward = -(loss1 + loss2) / 2
        elif self.task == 'maximize_total_distance':
            distance = np.mean(np.abs(self.final_position2 - np.array(self.start_position)))
            reward = distance
        
        self.s_t = self.final_position2
        terminated = True
        truncated = False
        return self.s_t, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.start_position = generate_random_positions(self.world, world_bounds=self.world_bounds)[0]
        self.s_t = self.start_position
        return self.s_t, {}


def evaluate_model(model, world, target_goal, task='reach_final_goal', target_goal2=None, 
                   num_episodes=50, world_bounds=None):
    """
    Evaluate the trained policy using real physics simulation.
    """
    env = RealWorldEnv(
        target_goal=target_goal,
        task=task,
        target_goal2=target_goal2,
        world=world,
        world_bounds=world_bounds
    )
    
    performances = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, _ = env.step(action)
        
        if task == 'reach_final_goal':
            loss = np.mean(np.abs(env.final_position2 - np.array(target_goal)))
            performance = 1 - loss
        elif task == 'reach_two_goals':
            # Try both assignments
            loss_opt1 = np.mean(np.abs(env.final_position1 - np.array(target_goal))) + np.mean(np.abs(env.final_position2 - np.array(target_goal2)))
            loss_opt2 = np.mean(np.abs(env.final_position1 - np.array(target_goal2))) + np.mean(np.abs(env.final_position2 - np.array(target_goal)))
            loss = min(loss_opt1, loss_opt2)
            performance = 1 - loss / 2
        elif task == 'goal_and_return':
            loss1 = np.mean(np.abs(env.final_position1 - np.array(target_goal)))
            loss2 = np.mean(np.abs(env.final_position2 - np.array(env.start_position)))
            performance = 1 - (loss1 + loss2) / 2
        elif task == 'maximize_total_distance':
            distance = np.mean(np.abs(env.final_position2 - np.array(env.start_position)))
            performance = distance / 1.66
        
        performances.append(performance)
    
    return np.mean(performances)


def plot_trajectories_mb(model, world, target_goal, task='reach_final_goal', target_goal2=None, world_bounds=None):
    """Plot example trajectories from the model-based policy"""
    env = RealWorldEnv(
        target_goal=target_goal,
        task=task,
        target_goal2=target_goal2,
        world=world,
        world_bounds=world_bounds
    )
    
    obs, _ = env.reset()
    initial_position = torch.tensor([env.start_position])
    action, _ = model.predict(obs, deterministic=True)
    _, _, _, _, _ = env.step(action)
    
    generate_and_plot_trajectories_from_parameters(
        env.dmp_params1, 
        env.dmp_params2, 
        1, 
        initial_position, 
        world, 
        world_bounds, 
        n_basis=3, 
        circular=False
    )


class CustomEvalCallback(BaseCallback):
    """Callback for evaluating the model during training"""
    def __init__(self, eval_freq, world, target_goal, task='reach_final_goal', target_goal2=None,
                 world_bounds=None, verbose=0, plot=False):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.world = world
        self.target_goal = target_goal
        self.target_goal2 = target_goal2
        self.task = task
        self.world_bounds = world_bounds
        self.eval_log = []
        self.plot = plot

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            performance = evaluate_model(
                self.model,
                self.world,
                self.target_goal,
                self.task,
                self.target_goal2,
                num_episodes=50,
                world_bounds=self.world_bounds
            )
            self.eval_log.append(performance)
            if self.verbose > 0:
                print(f"N_Calls: {self.n_calls}   Performance: {performance:.4f}")
            
            if self.plot:
                plot_trajectories_mb(
                    self.model,
                    self.world,
                    self.target_goal,
                    self.task,
                    self.target_goal2,
                    self.world_bounds
                )
        return True


def train_model_based_rl(world_model, 
                         train_data,
                         target_goal,
                         task='reach_final_goal',
                         target_goal2=None,
                         world=None,
                         world_bounds=None,
                         total_timesteps=10000,
                         eval_freq=1000,
                         learning_rate=0.001,
                         batch_size=10,
                         n_steps=200,
                         plot=False):
    """
    Train a planning network using PPO with the world model for simulation.
    """
    env = ModelBasedEnv(
        world_model=world_model,
        target_goal=target_goal,
        task=task,
        target_goal2=target_goal2,
        world=world,
        world_bounds=world_bounds
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
    
    eval_callback = CustomEvalCallback(
        eval_freq=eval_freq,
        world=world,
        target_goal=target_goal,
        task=task,
        target_goal2=target_goal2,
        world_bounds=world_bounds,
        verbose=1,
        plot=plot
    )
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    return model, eval_callback.eval_log


if __name__ == "__main__":
    
    # Initialize the world
    world_bounds = [0.1, 0.9, 0.1, 0.9]
    world = World(
        world_bounds=world_bounds,
        friction=1,
        num_obstacles=1,
        given_obstacles=[[(0, 0.48), (0, 0.52), (0.6, 0.48), (0.6, 0.52)]]
    )

    # Check the environment
    plot_example_trajectories(world, world_bounds, number_of_trajectories=5, complexity=1.0)

    # ============ Phase 1: Train World Model ============
    print("=" * 50)
    print("Phase 1: Training World Model")
    print("=" * 50)
    
    train_data, valid_data = generate_data(
        world,
        world_bounds,
        number_of_trajectories=5000,
        complexity=1.0
    )

    # Initialize the world model
    net = PredNet(input_size=2 + 2 + 2 + 6, hidden_size=64, output_size=3, dropout_rate=0.1)

    # Train World Model
    loss1, loss2, net = train_predictive_net(
        train_data, valid_data, net,
        batch_size=32,
        num_epochs=1000,
        learning_rate=0.001,
        num_training_sets=10,
        eval_freq=50,
        weight_collisions=0.05
    )
    net.eval()

    plt.figure(figsize=(3.5, 2))
    plt.plot(np.linspace(0, 1500, len(loss1)), 1 - np.array(loss1), color='blue', linewidth=3)
    plt.plot(np.linspace(0, 1500, len(loss2)), 1 - np.array(loss2), color='red', linewidth=3)
    plt.ylim(0.8, 1.0)
    plt.title("World Model Training")
    plt.show()

    plot_predicted_vs_actual_trajectories(
        world, net, world_bounds,
        number_of_trajectories=4,
        circular=False,
        complexity=0.5
    )

    # ============ Phase 2: Train Planning Network with PPO ============
    print("=" * 50)
    print("Phase 2: Training Planning Network (Model-Based PPO)")
    print("=" * 50)
    
    # Collect states for training
    train_data_mp, valid_data_mp = generate_data(world, world_bounds, number_of_trajectories=100, complexity=1.0)

    # Task: Reach Final Goal
    target_goal = [0.3, 0.85]
    
    model, performance_log = train_model_based_rl(
        world_model=net,
        train_data=train_data_mp,
        target_goal=target_goal,
        task='reach_final_goal',
        world=world,
        world_bounds=world_bounds,
        total_timesteps=10000,  # Match original model's sample efficiency
        eval_freq=1000,
        learning_rate=0.005,
        batch_size=10,
        n_steps=200,
        plot=True
    )

    # ============ Phase 3: Evaluation ============
    print("=" * 50)
    print("Phase 3: Zero-Shot Evaluation")
    print("=" * 50)
    
    final_performance = evaluate_model(
        model, world, target_goal, 
        task='reach_final_goal', 
        num_episodes=100,
        world_bounds=world_bounds
    )
    print(f"Final Zero-Shot Performance (reach_final_goal): {final_performance:.4f}")

    # Plot learning curve
    plt.figure(figsize=(5, 3))
    plt.plot(np.linspace(0, 50000, len(performance_log)), performance_log, color=[0.5, 0.5, 0.5], alpha=0.3)
    if len(performance_log) > 5:
        window = min(11, len(performance_log) if len(performance_log) % 2 == 1 else len(performance_log) - 1)
        plt.plot(np.linspace(0, 50000, len(performance_log)), savgol_filter(performance_log, window, 1), color=[0.3, 0.3, 0.3], linewidth=3)
    plt.xlabel("Training Steps")
    plt.ylabel("Performance")
    plt.title("Model-Based RL (Fixed Env - Reach Final Goal)")
    plt.show()

    # Save results
    with open("performances/fixed_env_mb_reach_final_goal.pkl", "wb") as f:
        pickle.dump(performance_log, f)

    # ============ Additional Task: Two Goals ============
    print("=" * 50)
    print("Training for Two Goals Task")
    print("=" * 50)
    
    target_goal1 = [0.6, 0.35]
    target_goal2 = [0.85, 0.85]
    
    model_two_goals, performance_log_two_goals = train_model_based_rl(
        world_model=net,
        train_data=train_data_mp,
        target_goal=target_goal1,
        task='reach_two_goals',
        target_goal2=target_goal2,
        world=world,
        world_bounds=world_bounds,
        total_timesteps=10000,
        eval_freq=1000,
        learning_rate=0.005,
        batch_size=10,
        n_steps=200,
        plot=True
    )

    final_performance_two_goals = evaluate_model(
        model_two_goals, world, target_goal1, 
        task='reach_two_goals',
        target_goal2=target_goal2,
        num_episodes=100,
        world_bounds=world_bounds
    )
    print(f"Final Zero-Shot Performance (reach_two_goals): {final_performance_two_goals:.4f}")

    # Save results
    with open("performances/fixed_env_mb_two_goals.pkl", "wb") as f:
        pickle.dump(performance_log_two_goals, f)

    # Plot final trajectories
    for _ in range(3):
        plot_trajectories_mb(model, world, target_goal, task='reach_final_goal', world_bounds=world_bounds)
