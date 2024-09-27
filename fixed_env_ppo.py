import pymunk
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pickle

from continuous_nav_envs import World, generate_random_positions
from dmps import DMP1D,Simulation

import gym
from gym import spaces
import numpy as np
import torch

from stable_baselines3 import PPO
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from utils import sample_random_dmp_params, generate_trajectories_from_dmp_params
from plot_trajectories import plot_example_trajectories
from plot_trajectories import generate_and_plot_trajectories_from_parameters


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, reward_threshold: float, check_freq: int):
        super(EarlyStoppingCallback, self).__init__()
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq
        self.episode_rewards = []
        self.evaluate_only = False  # Flag to indicate if we should only evaluate

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-self.check_freq:])
            if mean_reward > self.reward_threshold:
                print(f'Switching to evaluation only because mean reward {mean_reward} is above threshold {self.reward_threshold}')
                self.evaluate_only = True  # Set flag to only evaluate

        return not self.evaluate_only  # Return False if we should only evaluate, which stops training

    def _on_rollout_end(self) -> None:
        self.episode_rewards.append(self.locals["rewards"])

        
class CustomEnv(gym.Env):
    def __init__(self, criterion, target_goal, task = 'final_goal', target_goal2 = [0.5,0.5]):
        super(CustomEnv, self).__init__()

        self.criterion = criterion
        self.task = task
        self.target_goal = torch.tensor(target_goal)
        self.target_goal2 = torch.tensor(target_goal2)
        self.action_space = spaces.Box(low=-3, high=3, shape=(20,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0., high=1., shape=(2,))

        self.s_t = None

    def step(self, action):
        action = torch.tensor(action).unsqueeze(0)  # add batch dimension

        dmp_params1, dmp_params2 = torch.split(action, 10,dim=1)
        dmp_params1[:,:4] = dmp_params1[:,:4]
        dmp_params1[:,4:] = dmp_params1[:,4:].clamp(-1.0, 1.0)

        dmp_params2[:,:4] = dmp_params2[:,:4]
        dmp_params2[:,4:] = dmp_params2[:,4:].clamp(-1.0, 1.0)

        final_position1, final_position2, collision_info = generate_trajectories_from_dmp_params(
            dmp_params1=dmp_params1,
            dmp_params2=dmp_params2,
            batch_size=1,
            batch_s_t=torch.tensor([self.start_position]),
            world=world)  # add batch dimension
        
        self.initial_pos = torch.Tensor(self.start_position).squeeze().numpy()
        self.final_position2 = torch.Tensor(final_position2).squeeze().numpy()
        self.final_position1 = torch.Tensor(final_position1).squeeze().numpy()
        self.collision1 = (torch.Tensor([collision_info[0]['collision1'].max()*1]) > 0).squeeze().item()
        self.collision2 = (torch.Tensor([collision_info[0]['collision2'].max()*1]) > 0).squeeze().item()


        if self.task == 'final_goal':
            reward = -torch.mean(torch.abs(torch.tensor(self.final_position2) - self.target_goal)).item()-torch.mean(torch.abs(torch.tensor(self.final_position1) - self.target_goal)).item()
        elif self.task == 'goal_and_return':
            reward = -((torch.mean(
                torch.abs(torch.tensor(self.final_position1) - self.target_goal) ).item()) + (torch.mean(
                torch.abs(torch.tensor(self.final_position2) - self.start_position) ).item()))/2
        elif self.task == 'max_reward':
            reward = -torch.abs(torch.Tensor([self.final_position1[0]])).item() - torch.abs(torch.Tensor([self.final_position2[0]])).item()
        elif self.task == 'max_travel_dist':
            reward = torch.mean(torch.abs(torch.tensor(self.final_position2) - torch.tensor(self.final_position1)) ).item() + torch.mean(torch.abs(torch.tensor(self.final_position1) - torch.tensor(self.initial_pos)) ).item()
        elif self.task == "reach_two_goals_no_collisions":
            reward = 1-((torch.mean(
                torch.abs(torch.tensor(self.final_position2) - self.target_goal2) ).item()) + (torch.mean(
                torch.abs(torch.tensor(self.final_position1) - self.target_goal) ).item()))
            reward-= (self.collision1 or self.collision2)/5
                
            reward = reward

        self.s_t = self.final_position2

        done = True  # every episode ends after one step
        return self.s_t, reward, done, {}

    def reset(self):
        self.start_position = generate_random_positions(world, world_bounds=world_bounds)[0]
        self.s_t = self.start_position
        return self.s_t


def evaluate_model(model,
                   task = 'final_goal',
                   target_goal=[0.5,0.5],
                   target_goal2=[0.5,0.5],
                   pure_ppo = True,
                   net_inverse=None,
                   use_importance_sampling=False,
                   train_data=None):
    env = CustomEnv(criterion=criterion, target_goal=target_goal, target_goal2= target_goal2, task = task)

    num_episodes = 50
    distances, net_inverse_error = [] , []
    for _ in range(num_episodes):
        obs = env.reset()
        action, _ = model.predict(obs)
        _, _, _, _ = env.step(action)
        if task == 'final_goal':
            distance = np.mean(np.abs(env.final_position2 - np.array(env.target_goal)))
        elif task == 'goal_and_return':
            distance = 1- (np.mean(np.abs(env.final_position1 - np.array(env.target_goal))) + np.mean(np.abs(env.final_position2 - np.array(env.start_position))))/2
        elif task == 'max_reward':
            distance =  ((np.abs(env.final_position1[0] ) + np.abs(env.final_position2[0] )))/ 1.96
        elif task == 'max_travel_dist':
            distance =  (np.mean(np.abs(env.final_position2 - np.array(env.final_position1))) + np.mean(np.abs(env.final_position1 - np.array(env.initial_pos))))/1.1
        elif task == "reach_two_goals_no_collisions":
            distance = 1 - (np.mean(np.abs(env.final_position2 - np.array(env.target_goal2))) + np.mean(np.abs(env.final_position1 - np.array(env.target_goal))))
            distance -= (env.collision1 or env.collision2)/5
        if not pure_ppo:
            net_inverse_error.append((np.mean(np.abs(env.final_position2 - np.array(action[2:]))) + np.mean(np.abs(env.final_position1 - np.array(action[:2]))))/2)
        else:
            net_inverse_error.append(np.array(0))
        distances.append(distance)
    return np.mean(distances), np.mean(net_inverse_error)

def plot_trajectories(model,
                   task = 'final_goal',
                   target_goal=[0.5,0.5],
                   target_goal2=[0.5,0.5]):
    
    env = CustomEnv(criterion=criterion, target_goal=target_goal, target_goal2= target_goal2, task = task)
    
    obs = env.reset()
    initial_position = torch.Tensor([env.start_position])
    action, _ = model.predict(obs)
    _, _, _, _ = env.step(action)
    
    action = torch.tensor(action).unsqueeze(0)  # add batch dimension

    dmp_params1, dmp_params2 = torch.split(action, 10,dim=1)

    # Force the start positions in dmp_params1 and dmp_params2, and center in home reference frame
    dmp_params1_positions = torch.cat([torch.Tensor(env.start_position), dmp_params1[0][2:4] ])
    dmp_params2_positions = torch.cat([dmp_params1[0][2:4] , dmp_params2[0][2:4]])

    # Separate the position and DMP weights for clamping
    dmp_params1_weights = dmp_params1[0][4:].clamp(-1, 1)
    dmp_params2_weights = dmp_params2[0][4:].clamp(-1, 1)

    # Concatenate the positions and clamped weights
    dmp_params1 = torch.cat([dmp_params1_positions, dmp_params1_weights]).unsqueeze(dim=0)
    dmp_params2 = torch.cat([dmp_params2_positions, dmp_params2_weights]).unsqueeze(dim=0)

    generate_and_plot_trajectories_from_parameters(dmp_params1, dmp_params2, 1, initial_position, world, world_bounds, n_basis=3, circular=False)


class CustomEvalCallback(BaseCallback):
    def __init__(
            self,
            eval_freq,
            task = 'final_goal',
            verbose=0,
            target_goal=[0.5,0.5],
            target_goal2=[0.5,0.5],
            pure_ppo = True,
            net_inverse = None,
            use_importance_sampling= False,
            train_data = False,
            plot = False):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_log = []
        self.task = task
        self.target_goal = target_goal
        self.target_goal2 = target_goal2
        self.pure_ppo = pure_ppo
        self.net_inverse = net_inverse
        self.use_importance_sampling = use_importance_sampling
        self.train_data = train_data
        self.plot = plot
        self.world = world

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_distance, net_inverse_error = evaluate_model(
                self.model,
                task = self.task,
                target_goal=self.target_goal, 
                target_goal2=self.target_goal2, 
                pure_ppo = self.pure_ppo, 
                net_inverse=self.net_inverse,
                use_importance_sampling=self.use_importance_sampling,
                train_data=self.train_data)
            self.eval_log.append(mean_distance)
            if self.verbose > 0:
                print(f"N_Calls: {self.n_calls}   Performance: {mean_distance}")
        if self.plot and self.n_calls % self.eval_freq == 0:
            plot_trajectories(
                self.model,
                task = self.task,
                target_goal=self.target_goal, 
                target_goal2=self.target_goal2 )
        return True
            

if __name__ == "__main__":

    # Initialize the world
    world_bounds = [0.1, 0.9, 0.1, 0.9]
    world = World(
        world_bounds=world_bounds,
        friction=1,
        num_obstacles=1,
        given_obstacles= [(0, 0.48), (0, 0.52), (0.6, 0.48), (0.6, 0.52)])
    plot_example_trajectories(world,world_bounds,number_of_trajectories=5,complexity=2.0)


    valid_losses = None
    all_valid_loss = None
    ### PPO Control #####

    ##### Final Goal PPO ####

    CONT_LEARN = True
    criterion = nn.MSELoss()
    target_goal = [0.3,0.85]
    policy_kwargs = dict(
        net_arch=[dict(pi=[64], vf=[64])]
        )
    env = CustomEnv(criterion=criterion, target_goal=target_goal, task = 'final_goal')
    model = PPO('MlpPolicy', 
                env, 
                verbose=0, 
                policy_kwargs=policy_kwargs,
                learning_rate=0.001,
                n_steps=200, 
                batch_size=10)

    eval_callback = CustomEvalCallback(eval_freq=1000, verbose=1, task='final_goal', target_goal=target_goal,plot=False)
    total_timesteps = 50000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    final_goal_control = eval_callback.eval_log