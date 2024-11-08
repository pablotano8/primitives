import pymunk
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pickle

from continuous_nav_envs import World, generate_random_positions, CircularWorld
from dmps import DMP1D,Simulation
from scipy.signal import savgol_filter
import gym
from gym import spaces
import numpy as np
import torch

from stable_baselines3 import PPO, TD3
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from utils import sample_random_dmp_params, generate_trajectories_from_dmp_params
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


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
    def __init__(self, criterion, target_goal, world=None,task = 'final_goal', target_goal2 = [0.5,0.5]):
        super(CustomEnv, self).__init__()

        self.criterion = criterion
        self.task = task
        self.target_goal = torch.tensor(target_goal)
        self.target_goal2 = torch.tensor(target_goal2)
        self.action_space = spaces.Box(low=-3, high=3, shape=(20,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(7,))
        self.world = world

        self.s_t = None

    def step(self, action):
        action = torch.tensor(action).unsqueeze(0)  # add batch dimension

        dmp_params1, dmp_params2 = torch.split(action, 10,dim=1)

        # Force the start positions in dmp_params1 and dmp_params2, and center in home reference frame
        dmp_params1_positions = torch.cat([torch.Tensor(self.s_t[:2]), dmp_params1[0][2:4] ])
        dmp_params2_positions = torch.cat([dmp_params1[0][2:4] , dmp_params2[0][2:4]])

        # Separate the position and DMP weights for clamping
        dmp_params1_weights = dmp_params1[0][4:].clamp(-1, 1)
        dmp_params2_weights = dmp_params2[0][4:].clamp(-1, 1)

        # Concatenate the positions and clamped weights
        dmp_params1 = torch.cat([dmp_params1_positions, dmp_params1_weights]).unsqueeze(dim=0)
        dmp_params2 = torch.cat([dmp_params2_positions, dmp_params2_weights]).unsqueeze(dim=0)

        final_position1, final_position2, collision_info = generate_trajectories_from_dmp_params(
            dmp_params1=dmp_params1,
            dmp_params2=dmp_params2,
            batch_size=1,
            batch_s_t=torch.tensor([self.s_t[:2]]),
            world=self.world,
            circular=True,
            n_basis=3)  # add batch dimension
        
        self.initial_pos = torch.Tensor(self.s_t[:2]).squeeze()
        self.final_position2 = torch.Tensor(final_position2).squeeze()
        self.final_position1 = torch.Tensor(final_position1).squeeze()
        self.collision1 = (torch.Tensor([collision_info[0]['collision1'].max()*1]) > 0).squeeze().item()
        self.collision2 = (torch.Tensor([collision_info[0]['collision2'].max()*1]) > 0).squeeze().item()


        criterion = nn.MSELoss()
        loss_goal1 = criterion(self.final_position1, self.target_goal).item()
        loss_goal2 = criterion(self.final_position2, self.target_goal).item()
        outside_circle1 = torch.relu(torch.norm(self.final_position1, p=2)- 1.0).mean().item()
        outside_circle2 = torch.relu(torch.norm(self.final_position1, p=2)- 1.0).mean().item()
        if self.task == 'final_goal':
            # reward = -loss_goal1 - loss_goal2  - 0.25 * (self.collision1 + self.collision2) - outside_circle1 - outside_circle2
            reward = -loss_goal1 - loss_goal2 

        wall_length = self.world.radius * 2 * self.world.wall_size # Wall length
        wall_shape = [(-wall_length / 2, -self.world.wall_thickness / 2), (wall_length / 2, -self.world.wall_thickness / 2),
                    (wall_length / 2, self.world.wall_thickness / 2), (-wall_length / 2, self.world.wall_thickness / 2)]
        if not self.world.wall_present:
            distances_to_edges = np.array([2,2,2,2])
        else:
            distances_to_edges = np.array([np.linalg.norm(final_position2 - np.array(point)) for point in wall_shape])
        self.s_t =  np.concatenate((self.final_position2.numpy(),np.array([self.world.wall_present*1.0]),distances_to_edges))
        self.world.reset()
        done = True  # assuming the episode ends after one step
        return self.s_t, reward, done, {}

    def reset(self):
        if self.world is None:
            self.world = CircularWorld(
                num_obstacles=0,
                max_speed=100,
                radius=1,
                wall_present=np.random.uniform(0,1)<0.8,
                wall_size=np.random.uniform(0.3,1),
                wall_thickness=np.random.uniform(0,0.5))
        self.world.reset()
        start_position = generate_random_positions(self.world, world_bounds=world_bounds,circular = True)[0]
        wall_length = self.world.radius * 2 * self.world.wall_size # Wall length
        wall_shape = [(-wall_length / 2, -self.world.wall_thickness / 2), (wall_length / 2, -self.world.wall_thickness / 2),
                    (wall_length / 2, self.world.wall_thickness / 2), (-wall_length / 2, self.world.wall_thickness / 2)]
        if not self.world.wall_present:
            distances_to_edges = np.array([2,2,2,2])
        else:
            distances_to_edges = np.array([np.linalg.norm(start_position - np.array(point)) for point in wall_shape])
        self.s_t =  np.concatenate((start_position,np.array([self.world.wall_present*1.0]),distances_to_edges))
            
        return self.s_t


def evaluate_model(model,
                   task = 'final_goal',
                   target_goal=[0.5,0.5],
                   target_goal2=[0.5,0.5],
                   pure_ppo = True,
                   net_inverse=None,
                   use_importance_sampling=False,
                   train_data=None,
                   world=None):
    env = CustomEnv(criterion=criterion, target_goal=target_goal, target_goal2= target_goal2, task = task,world=world)

    num_episodes = 50
    perfs, net_inverse_error = [] , []
    for _ in range(num_episodes):
        world.reset()
        obs = env.reset()
        action, _ = model.predict(obs)
        _, _, _, _ = env.step(action)
        if task == 'final_goal':
            distance1 = np.mean(np.abs(env.final_position1.numpy() - np.array(env.target_goal.numpy())))
            distance2 = np.mean(np.abs(env.final_position2.numpy() - np.array(env.target_goal.numpy())))
            performance = np.max(np.array([1 - distance2,1-distance1]))
        perfs.append(performance)
    return np.mean(perfs)

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
            world = None):
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
        self.world = world

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_distance = evaluate_model(
                self.model,
                task = self.task,
                target_goal=self.target_goal, 
                target_goal2=self.target_goal2, 
                pure_ppo = self.pure_ppo, 
                net_inverse=self.net_inverse,
                use_importance_sampling=self.use_importance_sampling,
                train_data=self.train_data,
                world=self.world)
            self.eval_log.append(mean_distance)
            print(f"N_Calls: {self.n_calls}   Performance: {mean_distance}")
        return True
            

if __name__ == "__main__":

    # Initialize the world
    world_bounds = [-1, 1, -1, 1]
    world = World(num_obstacles=9, max_speed=100, world_bounds=world_bounds)


    valid_losses = None
    all_valid_loss = None
    ### PPO Control #####

##### Escaping Rat ####

    CONT_LEARN = False
    criterion = nn.MSELoss()
    target_goal = [0.,0.85]
    policy_kwargs = dict(
        net_arch=[dict(pi=[64], vf=[64])]
        )
    world_bounds = [-1, 1, -1, 1]

    env = CustomEnv(criterion=criterion, target_goal=target_goal, task = 'final_goal',world=None)

    model = TD3("MlpPolicy", env, verbose=0)
    # model = PPO('MlpPolicy', 
    #             env, 
    #             verbose=0, 
    #             policy_kwargs=policy_kwargs,
    #             learning_rate=0.001,
    #             n_steps=200, 
    #             batch_size=10)
    # model = PPO('MlpPolicy', 
    #             env, 
    #             verbose=0)

    world = CircularWorld(num_obstacles=0, max_speed=100, radius=1, wall_present=True,wall_size=0.6)
    eval_callback = CustomEvalCallback(eval_freq=1000, verbose=2, task='final_goal', target_goal=target_goal, world=world)
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    performance = eval_callback.eval_log

    with open("performances/rat_td3_multi_env.pkl", "wb") as f:
        pickle.dump(performance, f)


    plt.figure(figsize=(5,3))
    plt.plot(np.linspace(0,500000,len(performance)),performance,color=[0.1,0.1,0.1],alpha=0.3)
    plt.plot(np.linspace(0,500000,len(performance)),savgol_filter(performance,31,1),color=[0.1,0.1,0.1],linewidth=3)
    plt.ylim(0.2,1.0)
    plt.show()