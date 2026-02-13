import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from dmps import DMP1D
import gymnasium as gym

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

class RobotSimulation:
    def __init__(self, env, dmp_x, dmp_y,dmp_z, start_position, T=1.0, dt=0.01):
        self.env = env
        self.dmp_x = dmp_x
        self.dmp_y = dmp_y
        self.dmp_z = dmp_z
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.desired_velocity = np.array([0.0, 0.0, 0.0])
        self.position = start_position*1
        self.perfect_position = start_position*1
        self.T = T
        self.dt = dt
        self.x = 1  # Initialize the canonical system state

    def run(self):
        position_track = []
        object_track = []

        timesteps = int(self.T / self.dt)
        for _ in range(timesteps):
            desired_velocity_x = self.dmp_x.desired_velocity(self.position[0], self.velocity[0], self.x)
            desired_velocity_y = self.dmp_y.desired_velocity(self.position[1], self.velocity[1], self.x)
            desired_velocity_z = self.dmp_z.desired_velocity(self.position[2], self.velocity[2], self.x)
            self.desired_velocity = np.array([desired_velocity_x, desired_velocity_y,desired_velocity_z,0])

            # Get the actual velocity and new position from the world
            prev_position = self.position * 1
            obs = self.env.step(self.desired_velocity*10)
            self.position = obs[0]['observation'][:3]
            self.velocity = (self.position - prev_position)
            
            # Update the canonical system state
            self.x += -self.dmp_x.alpha_x * self.x * self.dt

            position_track.append(self.position)
            object_track.append(obs[0]['observation'][3:6])


        return np.array(position_track), np.array(object_track)
    

def plot_example_trajectories_robot(
        env,
        bounds,
        number_of_trajectories=10,
        complexity = 1,
        aim_for_block = False,
        plot_block=True):
    # Generate training data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_zlim(0.4, 0.45)

    for i in range(number_of_trajectories):  # Number of trajectories to generate
        obs = env.reset()
        start_position = obs[0]['observation'][:3]
        start_position_object = obs[0]['observation'][3:6]
        if not aim_for_block:
            goal_position1 = np.random.uniform(low=np.array([bounds[0],bounds[2],bounds[4]]),high=np.array([bounds[1],bounds[3],bounds[5]]))
        else:
            goal_position1 = np.random.uniform(
                low=np.array([
                    start_position_object[0]-0.1,
                    start_position_object[1]-0.1,
                    start_position_object[2]-0.005
                    ]),high=np.array([
                        start_position_object[0]+0.1,
                        start_position_object[1]+0.1,
                        start_position_object[2]+0.005
                        ]))
            
        # Initialize the first DMPs
        dmp_x1 = DMP1D(start=start_position[0], goal=goal_position1[0], n_basis=1, complexity=complexity)
        dmp_y1 = DMP1D(start=start_position[1], goal=goal_position1[1], n_basis=1, complexity=complexity)
        dmp_z1 = DMP1D(start=start_position[2], goal=goal_position1[2], n_basis=1, complexity=complexity)

        # Initialize the simulation with the world and the first DMPs
        simulation = RobotSimulation(env, dmp_x1, dmp_y1,dmp_z1, start_position)

        # Run the first simulation and record the positions
        positions1, object1 = simulation.run()
        
        goal_position2 = np.random.uniform(low=np.array([bounds[0],bounds[2],bounds[4]]),high=np.array([bounds[1],bounds[3],bounds[5]]))

        # Initialize the first DMPs
        dmp_x2 = DMP1D(start=positions1[-1][0], goal=goal_position2[0], n_basis=1, complexity=complexity)
        dmp_y2 = DMP1D(start=positions1[-1][1], goal=goal_position2[1], n_basis=1, complexity=complexity)
        dmp_z2 = DMP1D(start=positions1[-1][2], goal=goal_position2[2], n_basis=1, complexity=complexity)

        # Initialize the simulation with the world and the first DMPs
        simulation = RobotSimulation(env, dmp_x2, dmp_y2,dmp_z2, positions1[-1])

        # Run the first simulation and record the positions
        positions2, object2 = simulation.run()

        ax.scatter(start_position[0], start_position[1], start_position[2], color='b', label='Start', marker='o')
        ax.scatter(goal_position1[0], goal_position1[1], goal_position1[2], color='b', label='Goal', marker='x')
        ax.plot(positions1[:, 0], positions1[:, 1], positions1[:, 2], '.',label='Trajectory', color='b')
        ax.scatter(goal_position2[0], goal_position2[1], goal_position2[2], color='r', label='Goal', marker='x')
        ax.plot(positions2[:, 0], positions2[:, 1], positions2[:, 2], '.',label='Trajectory', color='r')
        if plot_block:
            ax.plot(object1[:, 0], object1[:, 1], object1[:, 2], '.',label='Trajectory', color='green')
            ax.plot(object2[:, 0], object2[:, 1], object2[:, 2], '.',label='Trajectory', color='orange')
    plt.show()

def plot_predictions_robot(
        env,
        bounds,
        number_of_trajectories=10,
        complexity = 1,
        aim_for_block = False):
    # Generate training data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(number_of_trajectories):  # Number of trajectories to generate
        obs = env.reset()
        start_position = obs[0]['observation'][:3]
        start_position_object = obs[0]['observation'][3:6]
        if not aim_for_block:
            goal_position1 = np.random.uniform(low=np.array([bounds[0],bounds[2],bounds[4]]),high=np.array([bounds[1],bounds[3],bounds[5]]))
        else:
            goal_position1 = np.random.uniform(
                low=np.array([
                    start_position_object[0]-0.1,
                    start_position_object[1]-0.1,
                    start_position_object[2]-0.005
                    ]),high=np.array([
                        start_position_object[0]+0.1,
                        start_position_object[1]+0.1,
                        start_position_object[2]+0.005
                        ]))

        # Initialize the first DMPs
        dmp_x1 = DMP1D(start=start_position[0], goal=goal_position1[0], n_basis=1, complexity=complexity)
        dmp_y1 = DMP1D(start=start_position[1], goal=goal_position1[1], n_basis=1, complexity=complexity)
        dmp_z1 = DMP1D(start=start_position[2], goal=goal_position1[2], n_basis=1, complexity=complexity)

        # Initialize the simulation with the world and the first DMPs
        simulation = RobotSimulation(env, dmp_x1, dmp_y1,dmp_z1, start_position)

        # Run the first simulation and record the positions
        positions1, object1 = simulation.run()
        
        goal_position2 = np.random.uniform(low=np.array([bounds[0],bounds[2],bounds[4]]),high=np.array([bounds[1],bounds[3],bounds[5]]))

        # Initialize the first DMPs
        dmp_x2 = DMP1D(start=positions1[-1][0], goal=goal_position2[0], n_basis=1, complexity=complexity)
        dmp_y2 = DMP1D(start=positions1[-1][1], goal=goal_position2[1], n_basis=1, complexity=complexity)
        dmp_z2 = DMP1D(start=positions1[-1][2], goal=goal_position2[2], n_basis=1, complexity=complexity)

        # Initialize the simulation with the world and the first DMPs
        simulation = RobotSimulation(env, dmp_x2, dmp_y2,dmp_z2, positions1[-1])

        # Run the first simulation and record the positions
        positions2, object2 = simulation.run()

        ax.scatter(start_position[0], start_position[1], start_position[2], color='b', label='Start', marker='o')
        # ax.scatter(goal_position1[0], goal_position1[1], goal_position1[2], color='b', label='Goal', marker='x')
        ax.plot(positions1[:, 0], positions1[:, 1], positions1[:, 2], '.',label='Trajectory', color='b')
        # ax.scatter(goal_position2[0], goal_position2[1], goal_position2[2], color='r', label='Goal', marker='x')
        ax.plot(positions2[:, 0], positions2[:, 1], positions2[:, 2], '.',label='Trajectory', color='r')
        ax.plot(object1[:, 0], object1[:, 1], object1[:, 2], '.',label='Trajectory', color='green')
        ax.plot(object2[:, 0], object2[:, 1], object2[:, 2], '.',label='Trajectory', color='orange')


        dmp_params1 = [dmp_x1.goal, dmp_y1.goal,dmp_z1.goal, *dmp_x1.weights, *dmp_y1.weights, *dmp_z1.weights]
        dmp_params2 = [dmp_x2.goal, dmp_y2.goal,dmp_z2.goal, *dmp_x2.weights, *dmp_y2.weights,  *dmp_z2.weights]

        s_t = start_position_object[:2]

        s_t = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0)
        dmp_params1 = torch.tensor(dmp_params1, dtype=torch.float32).unsqueeze(0)
        dmp_params2 = torch.tensor(dmp_params2, dtype=torch.float32).unsqueeze(0)

        # Prepare inputs and targets for RNN
        inputs1 = torch.cat((s_t, dmp_params1), dim=-1).unsqueeze(1)  # (batch_size, seq_len=1, input_size)
        padded_state = torch.zeros_like(s_t)
        inputs2 = torch.cat((padded_state, dmp_params2), dim=-1).unsqueeze(1)  # (batch_size, seq_len=1, input_size)

        # Forward pass for both timesteps
        outputs1, hidden = net(inputs1)
        outputs2, _ = net(inputs2, hidden)
        
        # Separate the outputs into position and length
        pred_pos1 = outputs1.squeeze(1)[:, :3].squeeze().detach().numpy()
        pred_obj1 = outputs1.squeeze(1)[:, 3:5].squeeze().detach().numpy()
        pred_pos2 = outputs2.squeeze(1)[:, :3].squeeze().detach().numpy()
        pred_obj2 = outputs2.squeeze(1)[:, 3:5].squeeze().detach().numpy()
        
        pred_obj1 += start_position_object
        pred_obj2 += pred_obj1

        ax.plot(pred_pos1[0], pred_pos1[1], pred_pos1[2], 'x',label='Trajectory', color='blue')
        ax.plot(pred_pos2[0], pred_pos2[1], pred_pos2[2], 'x',label='Trajectory', color='red')
        ax.plot(pred_obj1[0], pred_obj1[1], start_position_object[2], 'x',label='Trajectory', color='green')
        ax.plot(pred_obj2[0], pred_obj2[1], start_position_object[2], 'x',label='Trajectory', color='orange')
        ax.plot(start_position_object[0], start_position_object[1], start_position_object[2], 'o',label='Trajectory', color='black')

    plt.show()


def generate_data(
        env,
        bounds,
        number_of_trajectories=10000,
        complexity = 1,
        aim_for_block = False):
    # Generate training data
    data = []
    for i in range(number_of_trajectories):  # Number of trajectories to generate

        if i%1000==0:
            print(f'Generated {i} of {number_of_trajectories} trajectories')

        obs = env.reset()
        start_position = obs[0]['observation'][:3]
        start_position_object = obs[0]['observation'][3:6]
        if not aim_for_block:
            goal_position1 = np.random.uniform(low=np.array([bounds[0],bounds[2],bounds[4]]),high=np.array([bounds[1],bounds[3],bounds[5]]))
        else:
            goal_position1 = np.random.uniform(
                low=np.array([
                    start_position_object[0]-0.1,
                    start_position_object[1]-0.1,
                    start_position_object[2]-0.005
                    ]),high=np.array([
                        start_position_object[0]+0.1,
                        start_position_object[1]+0.1,
                        start_position_object[2]+0.005
                        ]))
            
        # Initialize the first DMPs
        dmp_x1 = DMP1D(start=start_position[0], goal=goal_position1[0], n_basis=1, complexity=complexity)
        dmp_y1 = DMP1D(start=start_position[1], goal=goal_position1[1], n_basis=1, complexity=complexity)
        dmp_z1 = DMP1D(start=start_position[2], goal=goal_position1[2], n_basis=1, complexity=complexity)

        # Initialize the simulation with the world and the first DMPs
        simulation = RobotSimulation(env, dmp_x1, dmp_y1,dmp_z1, start_position)

        # Run the first simulation and record the positions
        positions1, object1 = simulation.run()
        
        goal_position2 = np.random.uniform(low=np.array([bounds[0],bounds[2],bounds[4]]),high=np.array([bounds[1],bounds[3],bounds[5]]))

        # Initialize the first DMPs
        dmp_x2 = DMP1D(start=positions1[-1][0], goal=goal_position2[0], n_basis=1, complexity=complexity)
        dmp_y2 = DMP1D(start=positions1[-1][1], goal=goal_position2[1], n_basis=1, complexity=complexity)
        dmp_z2 = DMP1D(start=positions1[-1][2], goal=goal_position2[2], n_basis=1, complexity=complexity)

        # Initialize the simulation with the world and the first DMPs
        simulation = RobotSimulation(env, dmp_x2, dmp_y2,dmp_z2, positions1[-1])

        # Run the first simulation and record the positions
        positions2, object2 = simulation.run()

        # Add the data to the training set
        dmp_params1 = [dmp_x1.goal, dmp_y1.goal,dmp_z1.goal, *dmp_x1.weights, *dmp_y1.weights, *dmp_z1.weights]
        dmp_params2 = [dmp_x2.goal, dmp_y2.goal,dmp_z2.goal, *dmp_x2.weights, *dmp_y2.weights,  *dmp_z2.weights]

        s_t = start_position_object
        data.append((s_t, True, positions1[-1], positions2[-1], dmp_params1, dmp_params2, object1[-1],  object2[-1]))

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
                torch.tensor(object1, dtype=torch.float32),
                torch.tensor(object2, dtype=torch.float32)) for s_t, s_t_plus_one, final_position1,final_position2, dmp_params1,dmp_params2, object1,object2 in train_data]

    valid_data = [(torch.tensor(s_t, dtype=torch.float32),
                torch.tensor(s_t_plus_one, dtype=torch.float32),
                torch.tensor(final_position1, dtype=torch.float32),
                torch.tensor(final_position2, dtype=torch.float32),
                torch.tensor(dmp_params1, dtype=torch.float32),
                torch.tensor(dmp_params2, dtype=torch.float32),
                torch.tensor(object1, dtype=torch.float32),
                torch.tensor(object2, dtype=torch.float32)) for s_t, s_t_plus_one, final_position1,final_position2, dmp_params1,dmp_params2, object1,object2 in valid_data]
    return train_data, valid_data

def train_predictive_net(train_data,
                         valid_data,
                         net,
                         learning_rate=0.0001,
                         num_epochs=50,
                         batch_size=32,
                         eval_freq=1,
                         weight_object=1,
                         weight_pos = 1):
    
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # Loss function
    criterion = nn.MSELoss()
    huber_criterion = nn.SmoothL1Loss()

    average_validation_loss1, average_validation_loss2 = [] , []
    
    for epoch in range(num_epochs):

        random.shuffle(train_data)
        # Training
        net.train()
        train_losses = []
        for i in range(0, len(train_data)-batch_size, batch_size):
            batch = train_data[i:i + batch_size]
            batch_s_t = torch.stack([item[0][:2] for item in batch])
            batch_final_position1 = torch.stack([item[2] for item in batch])
            batch_final_position2 = torch.stack([item[3] for item in batch])
            batch_dmp_params1 = torch.stack([item[4] for item in batch])
            batch_dmp_params2 = torch.stack([item[5] for item in batch])
            batch_obj1 = torch.stack([item[6][:2] for item in batch])
            batch_obj2 = torch.stack([item[7][:2] for item in batch])

            # Prepare inputs and targets for RNN
            inputs1 = torch.cat((batch_s_t, batch_dmp_params1), dim=-1).unsqueeze(1)
            padded_state = torch.zeros_like(batch_s_t)
            inputs2 = torch.cat((padded_state, batch_dmp_params2), dim=-1).unsqueeze(1)

            # Forward pass for both timesteps
            optimizer.zero_grad()
            outputs1, hidden = net(inputs1)
            outputs2, _ = net(inputs2, hidden)
            
            # Separate the outputs into position and length
            pred_pos1 = outputs1.squeeze(1)[:, :3]
            pred_obj1 = outputs1.squeeze(1)[:, 3:5]
            pred_pos2 = outputs2.squeeze(1)[:, :3] 
            pred_obj2 = outputs2.squeeze(1)[:, 3:5]

            # Compute losses
            loss_pos1 = criterion(pred_pos1, batch_final_position1)
            loss_pos2 = criterion(pred_pos2, batch_final_position2)
            loss_obj1 = huber_criterion(pred_obj1, batch_obj1 - batch_s_t)
            loss_obj2 = huber_criterion(pred_obj2, batch_obj2 - batch_obj1)
            loss = weight_pos * (loss_pos1 + loss_pos2) + weight_object * (loss_obj1 + loss_obj2)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        
        if epoch % eval_freq == 0:
            # Validation
            net.eval()
            valid_losses_pos1,valid_losses_pos2,valid_losses_obj,dumb_error1,dumb_error2, dumb_error3, dumb_error4= [],[] , [] , [],[] , [], []
            with torch.no_grad():
                for i in range(0, len(valid_data), batch_size):
                    batch = valid_data[i:i + batch_size]
                    batch_s_t = torch.stack([item[0][:2] for item in batch])
                    batch_final_position1 = torch.stack([item[2] for item in batch])
                    batch_final_position2 = torch.stack([item[3] for item in batch])
                    batch_dmp_params1 = torch.stack([item[4] for item in batch])
                    batch_dmp_params2 = torch.stack([item[5] for item in batch])
                    batch_obj1 = torch.stack([item[6][:2] for item in batch])
                    batch_obj2 = torch.stack([item[7][:2] for item in batch])

                    # Prepare inputs and targets for RNN
                    inputs1 = torch.cat((batch_s_t, batch_dmp_params1), dim=-1).unsqueeze(1)  # (batch_size, seq_len=1, input_size)
                    padded_state = torch.zeros_like(batch_s_t)
                    inputs2 = torch.cat((padded_state, batch_dmp_params2), dim=-1).unsqueeze(1)  # (batch_size, seq_len=1, input_size)

                    # Forward pass for both timesteps
                    outputs1, hidden = net(inputs1)
                    outputs2, _ = net(inputs2, hidden)
                    
                    # Separate the outputs into position and length
                    pred_pos1 = outputs1.squeeze(1)[:, :3]
                    pred_obj1 = outputs1.squeeze(1)[:, 3:5]
                    pred_pos2 = outputs2.squeeze(1)[:, :3]
                    pred_obj2 = outputs2.squeeze(1)[:, 3:5]

                    # Compute losses
                    loss_pos1 = torch.mean(torch.abs(pred_pos1- batch_final_position1))
                    loss_pos2 = torch.mean(torch.abs(pred_pos2 - batch_final_position2))
                    loss_obj1 = torch.mean(torch.abs(pred_obj1 + batch_s_t  - batch_obj1  ))
                    loss_obj2 = torch.mean(torch.abs(pred_obj2 + batch_obj1 - batch_obj2 )) 
                    loss_pos = (loss_pos1 + loss_pos2) / 2 
                    loss_obj = (loss_obj1 + loss_obj2) / 2
                    dumb_error1.append(torch.mean(torch.abs(batch_final_position1- batch_dmp_params1[:,:3])))
                    dumb_error2.append(torch.mean(torch.abs(batch_final_position2- batch_dmp_params2[:,:3])))
                    dumb_error3.append(torch.mean(torch.abs(batch_obj1-batch_s_t)))
                    dumb_error4.append(torch.mean(torch.abs(batch_obj2-batch_obj1)))
                    valid_losses_pos1.append(loss_pos1.item())
                    valid_losses_pos2.append(loss_pos2.item())
                    valid_losses_obj.append(loss_obj.item())

            average_validation_loss1.append(np.mean(valid_losses_pos1))
            average_validation_loss2.append(np.mean(valid_losses_pos2))
            print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss = {np.mean(train_losses):.4f}, Pos = {np.mean(valid_losses_pos2):.4f}, Obj = {np.mean(valid_losses_obj):.4f}, DumbPos1= {np.mean(dumb_error1):.4f} DumbPos2= {np.mean(dumb_error2):.4f}  DumbObj1= {np.mean(dumb_error3):.4f}  DumbObj2= {np.mean(dumb_error4):.4f}')
    
    return average_validation_loss1,average_validation_loss2, net


if __name__ == "__main__":

    env = gym.make('FetchPush-v2', max_episode_steps=200)
    bounds = [1.15,1.55,0.55,0.95,0.42,0.43] #checked by plotting tons of desired goals
    plot_example_trajectories_robot(env,bounds,1,complexity=1,aim_for_block=False,plot_block=True)

    # Collect Data for Predictive Net
    train_data,valid_data= generate_data(
        env,
        bounds,
        number_of_trajectories=100_000,
        complexity=0.5,
        aim_for_block=False)

    # Train Predictive Net
    net = PredNet(input_size=6+2, hidden_size=64, output_size=5, dropout_rate=0.1)

    # Train Predictive Net
    loss1,loss2, net = train_predictive_net(
        train_data,valid_data,net,
        batch_size = 32,
        num_epochs = 100,
        learning_rate = 0.001,
        eval_freq = 10,
        weight_object = 1)
    net.eval()

    plot_predictions_robot(env,bounds,1,complexity=0.5,aim_for_block=True)