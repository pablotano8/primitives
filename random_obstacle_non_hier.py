import gym
from gym import spaces
import numpy as np
from continuous_nav_envs import ContinuousActionWorld, RandomRectangleWorld, World
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from plot_trajectories import plot_example_trajectories
from continuous_nav_envs import generate_random_positions

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

    def init_hidden(self, batch_size):
        # If the LSTM is bidirectional, the number of directions is 2, else it's 1
        num_directions = 1
        
        # Initialize hidden and cell states
        # Shape: (num_layers * num_directions, batch_size, hidden_size)
        return (torch.zeros(1 * num_directions, batch_size, self.hidden_size),
                torch.zeros(1 * num_directions, batch_size, self.hidden_size))


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

    def init_hidden(self, batch_size):
        # If the LSTM is bidirectional, the number of directions is 2, else it's 1
        num_directions = 1
        
        # Initialize hidden and cell states
        # Shape: (num_layers * num_directions, batch_size, hidden_size)
        return (torch.zeros(1 * num_directions, batch_size, self.hidden_size),
                torch.zeros(1 * num_directions, batch_size, self.hidden_size))



class CustomEnv(gym.Env):
    def __init__(self,world_eval = None, max_steps=200,task='one_goal',step_size=0.1):
        super(CustomEnv, self).__init__()

        self.world_eval = world_eval
        # Define action and observation space
        self.step_size = step_size
        # Actions are angles (0 to 2π)
        self.action_space = spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32)

        # Observations are the current coordinates in the world
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)

        # Define environment specifics
        self.goal = np.array([0.2, 0.8])
        self.max_steps = max_steps
        self.current_step = 0
        self.current_position = None

        self.task = task
        if self.task == 'one_goal':
            self.goals = [np.array([0.4, 0.6])]
        elif self.task == 'two_goals':
            self.goals = [np.array([0.4, 0.6]), np.array([0.6, 0.4])]
        self.active_goal_idx = 0


    def reset(self):
        self.world_bounds = [-1, 1, -1, 1]
        if self.world_eval is not None:
            self.world = self.world_eval
        else:
            self.world = RandomRectangleWorld(
                num_obstacles=0,
                max_speed=100,
                radius=1,
                wall_present=np.random.uniform(0,1)<0.8,
                wall_size=np.random.uniform(0.3,1),
                wall_thickness=np.random.uniform(0,0.5))
        
        self.start_position, _ = generate_random_positions(self.world, world_bounds=self.world_bounds,orientation = None,circular=True)

        wall_length = self.world.radius * 2 * self.world.wall_size # Wall length
        wall_shape = [(-wall_length / 2, -self.world.wall_thickness / 2), (wall_length / 2, -self.world.wall_thickness / 2),
                    (wall_length / 2, self.world.wall_thickness / 2), (-wall_length / 2, self.world.wall_thickness / 2)]
        if not self.world.wall_present:
            distances_to_edges = np.array([2,2,2,2])
        else:
            distances_to_edges = np.array([np.linalg.norm(self.start_position - np.array(point)) for point in wall_shape])
            
        self.current_position = np.concatenate(
            (self.start_position,
            np.array([self.world.wall_present*1.0]),
            distances_to_edges))
        
        self.current_step = 0
        self.done = False
        self.all_distances = []
        return self.current_position

    def step(self, action):
        desired_velocity = self.action_to_velocity(action[0])
        step_size = min(max(0,action[1]),self.step_size)
        # Compute the actual dynamics using the desired velocity
        new_position, _, _, _, _ = self.world.actual_dynamics(
            np.array([self.current_position[0],self.current_position[1]]),
            np.array([desired_velocity[0],desired_velocity[1]]),
            step_size)
        self.current_position[0] = new_position[0]
        self.current_position[1] = new_position[1]
        self.current_step += 1
        # Calculate reward

        self.done = self.current_step >= self.max_steps

        reward = 0

        info = {}

        return self.current_position, reward, self.done, info

    def action_to_velocity(self, action):
        # Convert the action (angle) to a unit vector
        u = np.array([np.cos(action), np.sin(action)])
        # Multiply the unit vector with the step size to get the desired velocity
        desired_velocity = u
        return desired_velocity
    
    def render(self, mode='human'):
        pass  # Rendering can be implemented if desired

    def close(self):
        pass



def generate_data(env, number_of_episodes=20, min_steps=10, rnn=True, validation_size=0.1):

    data = []

    for i in range(number_of_episodes):
        if i % 1000 ==0:
            print(f'Generated {i} of {number_of_episodes} episodes')

        obs = env.reset()
        done=False
        previous_state = obs*1  # Using obs as the initial state
        step_count=0
        sequence = []

        while True:
            # Produce random action (a random angle in range 0 to 2*pi)
            action_dir = np.random.uniform(0, 2*np.pi)
            action_step = np.random.uniform(0, env.step_size)
            action = np.array([action_dir,action_step])
            obs, _, done, _ = env.step(action)
            step_count += 1
            
            sequence.append((previous_state*1, obs*1, action*1))
            previous_state = obs*1
            
            # If the episode ended, reset the environment
            if done:
                data.append(sequence)
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
                       torch.tensor(action, dtype=torch.float32)) for initial_state, final_state, action in seq]
        train_data.append(tensor_seq)

    valid_data = []
    for seq in valid_data_sequences:
        tensor_seq = [(torch.tensor(initial_state, dtype=torch.float32),
                       torch.tensor(final_state, dtype=torch.float32),
                       torch.tensor(action, dtype=torch.float32)) for initial_state, final_state, action in seq]
        valid_data.append(tensor_seq)

    if not rnn:
        train_data = [item for sublist in train_data for item in sublist]
        valid_data = [item for sublist in valid_data for item in sublist]

    return train_data, valid_data



def generate_episode_with_planner(env, planner, validation_size=0):

    data = []

    obs = env.reset()
    initial_state = obs  # Using obs as the initial state
    step_count=0
    sequence = []

    while True:
        # Produce random action (a random angle in range 0 to 2*pi)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        action = np.array([planner(obs_tensor)[0].item()])
    
        obs, _, done, _ = env.step(action)
        step_count += 1

        current_state = obs  # Using obs as the current state
        sequence.append((initial_state, current_state, action))
        initial_state = current_state
        
        # If the episode ended, reset the environment
        if done:
            data.append(sequence)
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
                       torch.tensor(action, dtype=torch.float32)) for initial_state, final_state, action in seq]
        train_data.append(tensor_seq)

    valid_data = []
    for seq in valid_data_sequences:
        tensor_seq = [(torch.tensor(initial_state, dtype=torch.float32),
                       torch.tensor(final_state, dtype=torch.float32),
                       torch.tensor(action, dtype=torch.float32)) for initial_state, final_state, action in seq]
        valid_data.append(tensor_seq)

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
                       verbose = True,
                       random_padding=True):

    # Filter train_data up to plan_length
    train_data = [element[:plan_length] for element in train_data if len(element) >= plan_length]
    valid_data = [element[:plan_length] for element in valid_data if len(element) >= plan_length]

    iteration=0
    # Optimizer
    optimizer = optim.Adam(pred_net.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Loss function
    criterion = nn.MSELoss()
    average_validation_loss = []

    processed_batches = -1
    training_sets = np.array_split(train_data, 10)
    for epoch in range(num_epochs):
        
        train_data_epoch = training_sets[epoch % 10]
        # Training
        pred_net.train()
        train_losses = []

        for i in range(0, len(train_data_epoch), batch_size):
            sequences = train_data_epoch[i:i+batch_size]
            t_star = torch.randint(0, plan_length, (len(sequences),))  # Assuming T=10
            # Extract data from sequences and pad after t*
            s_t_seq_list, final_state_seq_list, action_seq_list = [], [], []
            for idx, seq in enumerate(sequences):
                s_t = [t[0] if j <= random_padding*t_star[idx] else torch.zeros_like(t[0]) for j, t in enumerate(seq)]
                final_state = [t[1][:2] for t in seq]
                action = [t[2] for t in seq]

                s_t_seq_list.append(torch.stack(s_t))
                final_state_seq_list.append(torch.stack(final_state))
                action_seq_list.append(torch.stack(action))

            s_t_seq = torch.stack(s_t_seq_list)
            final_state_seq = torch.stack(final_state_seq_list)
            action_seq = torch.stack(action_seq_list)
            
            s_t_extended_seq = torch.cat((s_t_seq, action_seq), dim=-1)
            target_seq = final_state_seq

            optimizer.zero_grad()

            hidden = pred_net.init_hidden(len(sequences))

            output, _ = pred_net(s_t_extended_seq, hidden)
            # Calculate the loss for the entire batch
            batch_loss = criterion(output, target_seq)
            batch_loss.backward()
            iteration+=1
            torch.nn.utils.clip_grad_norm_(pred_net.parameters(), clip_value)

            optimizer.step()

            processed_batches +=1
            train_losses.append(batch_loss.item())

            if processed_batches % eval_freq == 0:
                print(iteration)
                # Validation
                pred_net.eval()
                valid_losses = []
                future_step_losses = [0.0] * plan_length  # Assuming T=10
                count_steps = [0] * plan_length

                with torch.no_grad():
                    for i in range(0, len(valid_data), batch_size):
                        sequences = valid_data[i:i+batch_size]
                        
                        # Extract data from sequences and pad after t*
                        s_t_seq_list, final_state_seq_list, action_seq_list = [], [], []
                        for idx, seq in enumerate(sequences):
                            s_t = [t[0] if j <= 0 else torch.zeros_like(t[0]) for j, t in enumerate(seq)]
                            # s_t = [t[0] for t in seq]
                            final_state = [t[1][:2] for t in seq]
                            action = [t[2] for t in seq]

                            s_t_seq_list.append(torch.stack(s_t))
                            final_state_seq_list.append(torch.stack(final_state))
                            action_seq_list.append(torch.stack(action))

                        s_t_seq = torch.stack(s_t_seq_list)
                        final_state_seq = torch.stack(final_state_seq_list)
                        action_seq = torch.stack(action_seq_list)
                        
                        s_t_extended_seq = torch.cat((s_t_seq, action_seq), dim=-1)
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

                average_validation_loss.append(np.mean(valid_losses))
                future_step_avg_losses = [total_loss / count if count != 0 else 0 for total_loss, count in zip(future_step_losses, count_steps)]
                if verbose:
                    print(f'Epoch {epoch + 1}/{num_epochs}: Train = {np.mean(train_losses):.4f}, Valid = {np.mean(valid_losses):.4f}')
                    print([f"{loss:.4f}" for loss in future_step_avg_losses])

    return average_validation_loss, pred_net




def evaluate_planner(env,planner, target_goal,number_of_episodes=20,plot_trajectory = False):

    performances = []
    sequences = []
    for i in range(number_of_episodes):

        obs = env.reset()
        initial_state = obs  # Using obs as the initial state
        step_count=0
        sequence = []

        while True:
            obs_tensor = torch.tensor(obs[0:2], dtype=torch.float32).unsqueeze(0).unsqueeze(1)
            action = planner(obs_tensor)[0][0][0].detach().numpy() + np.random.normal(0,0.01,2)
            obs, _, done, _ = env.step(action)
            step_count += 1

            sequence.append(np.array([obs_tensor.numpy().flatten(),obs.flatten()]))
            
            # If the episode ended, reset the environment
            if done:
                distance = np.mean(np.abs(obs[:2] - target_goal))
                performances.append(1-distance)
                obs = env.reset()
                break
        sequences.append(sequence)

    if plot_trajectory:
        for sequence in sequences[-4:]:
            # Plot a trajectory
            position_track = sequence
            for i in range(1,len(position_track) - 2):
                plt.plot([position_track[i][0][0], position_track[i][1][0]],[position_track[i][0][1], position_track[i][1][1]] ,color='blue', alpha=0.6,)
                plt.plot(position_track[i][0][0],position_track[i][0][1] ,'.',color='blue', alpha=0.1,)

            for body, obstacle in env.world.obstacles:
                vertices = np.array([body.local_to_world(v) for v in obstacle.get_vertices()])
                vertices = np.vstack((vertices, vertices[0]))  # Close the polygon by adding the first point at the end
                plt.plot(*vertices.T, color='black')
                plt.fill(*vertices.T, color='grey', alpha=0.5)  # Fill the polygon to make the obstacle solid
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.show()
    return np.mean(performances)


def train_planner_rnn(train_data, planner, pred_net,env,
                      plan_length=7,
                      eval_freq = 1,
                      learning_rate=0.0001,
                      num_epochs=50,
                      batch_size=32,
                      clip_value=1.0,
                      target_goal = None,
                      finetune_loops = 2,
                      plot_trajectory = False
                      ):
    

    iteration=0
    train_data = [seq[:plan_length] for seq in train_data if len(seq) >= plan_length]

    # Optimizer
    optimizer = optim.Adam(planner.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    all_avg_rewards, finetune_count, processed_batches, vae_loss, pred_loss = [] , 0 , -1, 0 , 0
    training_sets = np.array_split(train_data, 10)
    for epoch in range(num_epochs):
        
        train_data_epoch = training_sets[epoch % 10]
        # Training
        planner.train()
        pred_net.eval()
        train_rewards = []

        for i in range(0, len(train_data_epoch), batch_size):
            processed_batches+=1
            optimizer.zero_grad() 
            
            sequences = train_data_epoch[i:i+batch_size]

            # Extract data from sequences
            state_seq_list = [torch.stack([t[0] for t in seq]) for seq in sequences]
            state_seq = torch.stack(state_seq_list)
            
            # Initialize hidden states for both the planner and pred_net at the beginning of each sequence
            hidden_pred_net = pred_net.init_hidden(len(state_seq))
            hidden_planner = planner.init_hidden(len(state_seq))

            # Build padded input to Planner
            current_state = state_seq[:, 0:1,:2]  # Add sequence dimension
            padded_zeros = torch.zeros_like(state_seq[:, 1:,:2])
            planner_input = torch.cat((current_state, padded_zeros), dim=1)

            # Cumpute goals with Planner
            goals_pred, _ = planner(planner_input, hidden_planner)

            proposed_direction = goals_pred[:,:, 0].clamp(0,2*np.pi)
            proposed_step_size = goals_pred[:,:, 1].clamp(0,env.step_size)
            goals_pred = torch.stack([proposed_direction, proposed_step_size], dim=-1)

            # Padded input # Iterative input without Simulaition
            first_elem = torch.cat((state_seq[:, 0], goals_pred[:, 0]), dim=1)
            padded_states = torch.zeros_like(state_seq[:, 1:])
            subsequent_elems = torch.cat((padded_states, goals_pred[:, 1:]), dim=2)
            s_t_extended_seq = torch.cat((first_elem.unsqueeze(1), subsequent_elems), dim=1)
            predicted_states, _ = pred_net(s_t_extended_seq, hidden_pred_net)

            # Distance Losses
            target_goal_batch = torch.stack([torch.tensor(target_goal) for item in sequences])
            loss = criterion(predicted_states[:,-1,:],target_goal_batch)

            # Optimize
            loss.backward()
            optimizer.step()
            iteration+=1
                
            # Append
            train_rewards.append(loss.item())

        if epoch%eval_freq==0:
            print(iteration)

            reward = evaluate_planner(env,planner,target_goal,plot_trajectory=plot_trajectory, number_of_episodes=100)
            all_avg_rewards.append(reward)

            print(f'Epoch {epoch + 1}/{num_epochs}: Train = {loss.item():.4f}  Pred:{np.mean(pred_loss):.4f}   Perf: {reward:.4f}  Tuned: {finetune_count}')
            finetune_count = 0
    return planner, all_avg_rewards






if __name__=='__main__':

    # Instantiate the world
    # Initialize the world
    world_bounds = [-1, 1, -1, 1]
    world = RandomRectangleWorld(num_obstacles=0, max_speed=100, radius=1, wall_present=True,wall_size=0.6,friction=0)

    # Check the environment
    plot_example_trajectories(world,world_bounds,number_of_trajectories=10,complexity=2.5,circular=True)

    env = CustomEnv(world_eval=None, step_size=0.1)

    # Generate the data
    train_data, valid_data = generate_data(env,number_of_episodes=25000)

    # Plot a trajectory
    world = RandomRectangleWorld(num_obstacles=0, max_speed=100, radius=1, wall_present=True,wall_size=0.6,friction=0)
    env_eval = CustomEnv(world_eval=world, step_size=0.1)
    env_eval.reset()
    data, _ = generate_data(env_eval,number_of_episodes=10)
    position_track = data[0]
    for i in range(len(position_track) - 2):
        plt.plot([position_track[i][0][0], position_track[i][1][0]],[position_track[i][0][1], position_track[i][1][1]] ,color='black', alpha=0.6,)

    for body, obstacle in env_eval.world.obstacles:
        vertices = np.array([body.local_to_world(v) for v in obstacle.get_vertices()])
        vertices = np.vstack((vertices, vertices[0]))  # Close the polygon by adding the first point at the end
        plt.plot(*vertices.T, color='black')
        plt.fill(*vertices.T, color='grey', alpha=0.5)  # Fill the polygon to make the obstacle solid
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.show()


    pred_losses,perfs= [] , []
    for PLAN_LENGTH in [2,5,10,15,20,25,37,50,75,100,150,200]:
        print(PLAN_LENGTH)
        # Instantiate the RNN
        pred_net = PredNet(
            input_size=2 + 4 + 3,
            hidden_size=64,
            output_size=2,
            dropout_rate=0.1,
        )

        average_validation_loss, pred_net = train_pred_net_rnn(
            train_data = train_data,
            valid_data = valid_data,
            pred_net = pred_net,
            plan_length = PLAN_LENGTH,
            learning_rate = 0.001,
            num_epochs = 1000,
            batch_size = 100,
            eval_freq = 2000,
            random_padding = False
        )
        pred_losses.append(average_validation_loss)

        # Instantiate the RNN
        planner = GoalNet(
            input_size=2,
            hidden_size=64,
            output_size=2,
            dropout_rate=0,
        )

        world = RandomRectangleWorld(num_obstacles=0, max_speed=100, radius=1, wall_present=True,wall_size=0.6,friction=0)
        env_eval = CustomEnv(world_eval=world, step_size=0.1)

        planner, all_avg_rewards = train_planner_rnn(train_data, planner, pred_net,env_eval,
                        plan_length=PLAN_LENGTH,
                        eval_freq = 1,
                        learning_rate=0.001,
                        num_epochs=202,
                        batch_size=10,
                        target_goal = [0,0.85],
                        finetune_loops=0,
                        plot_trajectory=True,
                        )
        
        perfs.append(all_avg_rewards[-1])
    


    plan_lengths = [2,5,10,15,20,25,37,50,75,100,150,200]

    preds = [0.0217,0.0301,0.0369,0.0550,0.0576,0.0619,0.0640,0.0980,0.1012,0.1521,0.1720,0.2374]
    perfs = [0.5506,0.7487,0.7102,0.4962,0.4263,0.4071,0.2049,0.2095,0.1581,0.0762,0.3515,0.251]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(3,2.5))
    plt.plot(plan_lengths,1-np.array(preds),linewidth=5, color='black')
    plt.show()

    plt.figure(figsize=(3,2.5))
    plt.plot(plan_lengths,perfs,linewidth=5, color='black')
    plt.ylim(0.5,1)
    plt.show()

    # MP Search Space (i.e. number of grid points assuming 1 grid point every 0.1 for the every parameter)
    mp_space = []
    grid_points = 0.1
    for plan_length in [2,5,10,15,20,25,37,50,75,100,150,200]:
        #assume 1 grid point 
        angle_points =  (2*np.pi)/grid_points
        # spline_points =  (np.round(comp/grid_points))**6 #6 splines (3 for each dimension)
        goal_points = (2/grid_points)**2

        mp_space.append(plan_length*angle_points)

    comp_space1 = (np.round(0.5/grid_points))**2 + (1/grid_points)**2

    import matplotlib.pyplot as plt
    plt.figure(figsize=(3,2.5))
    plt.plot(plan_lengths,mp_space,linewidth=5, color='black')
    plt.plot(100,125,'.')
    plt.plot(200,250,'.')
    plt.show()
