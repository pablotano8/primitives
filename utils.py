
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


def generate_trajectories_from_dmp_params(dmp_params1,dmp_params2,batch_size,batch_s_t,world,circular=False,n_basis=3):
        
        if circular:
             world.reset()
        dmp_params1, dmp_params2 = dmp_params1.numpy(), dmp_params2.numpy()
        # Initialize and run DMP simulations for each item in the batch
        actual_final_position1,actual_final_position2, collision_info = [],[] , []
        for b in range(batch_size):
            dmp_x1 = DMP1D(start=batch_s_t[b][0], goal=dmp_params1[b][2], n_basis=n_basis)
            dmp_y1 = DMP1D(start=batch_s_t[b][1], goal=dmp_params1[b][3], n_basis=n_basis)
            dmp_x1.weights = dmp_params1[b][4:4+n_basis]
            dmp_y1.weights = dmp_params1[b][4+n_basis:4+n_basis+4+n_basis]
            simulation1 = Simulation(world, dmp_x1, dmp_y1, batch_s_t[b], T=1.0, dt=0.01)
            positions1, velocities1, collision1, collision_pos1, min_distance1 = simulation1.run()

            dmp_x2 = DMP1D(start=positions1[-1][0], goal=dmp_params2[b][2], n_basis=n_basis)
            dmp_y2 = DMP1D(start=positions1[-1][1], goal=dmp_params2[b][3], n_basis=n_basis)
            dmp_x2.weights = dmp_params2[b][4:4+n_basis]
            dmp_y2.weights = dmp_params2[b][4+n_basis:4+n_basis+4+n_basis]
            simulation2 = Simulation(world, dmp_x2, dmp_y2, positions1[-1], T=1.0, dt=0.01)
            positions2, velocities2, collision2, collision_pos2, min_distance2 = simulation2.run()

            actual_final_position1.append(np.array(positions1[-1]))
            actual_final_position2.append(np.array(positions2[-1]))
            collision_info.append({
                'collision1': collision1,
                'collision2': collision2,
                'collision_pos': collision_pos1,
                'collision_pos2': collision_pos2,
                'min_distance1': min_distance1,
                'min_distance2': min_distance2})
        return actual_final_position1,actual_final_position2, collision_info


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
