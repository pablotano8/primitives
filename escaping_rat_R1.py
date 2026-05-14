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


class ParticleFilter:
    """Particle filter over the 4 wall corner positions (8 scalars).

    Corners are stored in canonical order [BL, BR, TR, TL] to match the
    wall_shape construction at line ~294. The filter assumes a stationary
    wall (no process noise on the dynamics), with a small Gaussian jitter
    added on resampling to avoid particle degeneracy.
    """

    def __init__(self, n_particles=500, x_range=(-1.0, 1.0),
                 y_range=(-0.25, 0.25), resample_jitter=0.01, seed=None):
        self.n = n_particles
        self.x_range = x_range
        self.y_range = y_range
        self.resample_jitter = resample_jitter
        self.rng = np.random.default_rng(seed)

        # Independent uniform prior per corner coordinate.
        bl = np.stack([self.rng.uniform(x_range[0], 0.0, size=n_particles),
                       self.rng.uniform(y_range[0], 0.0, size=n_particles)], axis=-1)
        br = np.stack([self.rng.uniform(0.0, x_range[1], size=n_particles),
                       self.rng.uniform(y_range[0], 0.0, size=n_particles)], axis=-1)
        tr = np.stack([self.rng.uniform(0.0, x_range[1], size=n_particles),
                       self.rng.uniform(0.0, y_range[1], size=n_particles)], axis=-1)
        tl = np.stack([self.rng.uniform(x_range[0], 0.0, size=n_particles),
                       self.rng.uniform(0.0, y_range[1], size=n_particles)], axis=-1)
        # shape (n_particles, 4, 2) corners in BL, BR, TR, TL order
        self.particles = np.stack([bl, br, tr, tl], axis=1)
        self.log_weights = np.full(n_particles, -np.log(n_particles))

    def sample(self, n):
        """Draw n particles proportional to current weights. Returns (n, 4, 2)."""
        w = np.exp(self.log_weights - np.max(self.log_weights))
        w = w / w.sum()
        idx = self.rng.choice(self.n, size=n, replace=True, p=w)
        return self.particles[idx].copy()

    def update(self, positions, observations, sigmas, max_useful_sigma=1.0):
        """Update weights with a batch of (position, distance-observation, sigma) triples.

        positions:    (B, 2) numpy float
        observations: (B, 4) numpy float -- noisy distances to corners (BL, BR, TR, TL)
        sigmas:       (B,)   numpy float -- per-observation noise std (one sigma per row)
        max_useful_sigma: rows whose sigma exceeds this are dropped (likelihood would
                          be flat across particles anyway -- skip the work).
        """
        if positions.shape[0] == 0:
            return
        keep = sigmas <= max_useful_sigma
        if not keep.any():
            return
        positions = positions[keep]
        observations = observations[keep]
        sigmas = sigmas[keep]

        # predicted distances per (batch, particle, corner): (B, N, 4)
        diff = self.particles[None, :, :, :] - positions[:, None, None, :]
        pred = np.linalg.norm(diff, axis=-1)
        # log-likelihood per (batch, particle): sum over 4 corners of N(obs; pred, sigma^2)
        sig2 = (sigmas ** 2)[:, None, None] + 1e-12
        log_lik = -0.5 * ((observations[:, None, :] - pred) ** 2) / sig2
        log_lik = log_lik.sum(axis=-1) - 0.5 * 4 * np.log(2 * np.pi * sig2[:, :, 0])
        # accumulate across the batch (assume independence)
        self.log_weights = self.log_weights + log_lik.sum(axis=0)
        # numerical stability: keep max log-weight at 0
        self.log_weights -= np.max(self.log_weights)

        # ESS-based resampling
        w = np.exp(self.log_weights)
        w = w / w.sum()
        ess = 1.0 / np.sum(w ** 2)
        if ess < self.n / 2:
            idx = self.rng.choice(self.n, size=self.n, replace=True, p=w)
            self.particles = self.particles[idx]
            # regularized resampling jitter
            self.particles = self.particles + self.rng.normal(
                0.0, self.resample_jitter, size=self.particles.shape)
            self.log_weights = np.full(self.n, -np.log(self.n))

    def summary(self, true_corners=None):
        """Return weighted mean (4, 2), per-coord std (4, 2), ESS, and L2 to truth (or None)."""
        w = np.exp(self.log_weights - np.max(self.log_weights))
        w = w / w.sum()
        mean = (w[:, None, None] * self.particles).sum(axis=0)
        var = (w[:, None, None] * (self.particles - mean[None, :, :]) ** 2).sum(axis=0)
        std = np.sqrt(var)
        ess = 1.0 / np.sum(w ** 2)
        l2 = None if true_corners is None else float(np.linalg.norm(mean - true_corners))
        return {"mean": mean, "std": std, "ess": float(ess), "l2_to_truth": l2}


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



def _trajectory_near_edges(positions, edge_points, edge_radius):
    for pt in positions:
        for ep in edge_points:
            if (pt[0] - ep[0]) ** 2 + (pt[1] - ep[1]) ** 2 <= edge_radius ** 2:
                return True
    return False


def distance_to_wall(positions, true_corners):
    """Euclidean distance from each position to the closest point on the wall AABB.

    positions:    (B, 2) numpy float
    true_corners: (4, 2) numpy float, in BL/BR/TR/TL order
    Returns:      (B,)   numpy float
    """
    positions = np.atleast_2d(positions)
    x_min, x_max = true_corners[:, 0].min(), true_corners[:, 0].max()
    y_min, y_max = true_corners[:, 1].min(), true_corners[:, 1].max()
    dx = np.maximum(0.0, np.maximum(x_min - positions[:, 0], positions[:, 0] - x_max))
    dy = np.maximum(0.0, np.maximum(y_min - positions[:, 1], positions[:, 1] - y_max))
    return np.sqrt(dx * dx + dy * dy)


def observation_noise_std(positions, true_corners, sigma_min=0.002, beta=12.0,
                          cutoff_distance=0.4):
    """Position-dependent observation noise: tight near the wall, exploding far from it.

    With the defaults (sigma_min=0.002, beta=12), useful observations are confined
    to a narrow band around the wall: sigma is ~0.002 on the wall, ~0.022 at d=0.2,
    ~0.81 at d=0.5, and effectively infinite past d=1. Combined with the cutoff,
    observations from positions farther than `cutoff_distance` are returned with
    a sentinel std so large that the likelihood is flat across particles -- the
    filter ignores them. This reproduces the user's intent: the posterior should
    only tighten when the agent has actually been close to the wall.
    """
    d = distance_to_wall(positions, true_corners)
    sigma = sigma_min * np.exp(beta * d)
    if cutoff_distance is not None:
        sigma = np.where(d > cutoff_distance, 1e6, sigma)
    return sigma


def _pf_unit_check(verbose=True):
    """Sanity check: feed observations from positions sweeping around the wall
    (near, for triangulation) vs from a single far position (vanishing SNR).
    Confirms noise schedule and filter behavior."""
    true_corners = np.array([(-0.3, -0.05), (0.3, -0.05),
                             (0.3, 0.05), (-0.3, 0.05)])

    pf_near = ParticleFilter(n_particles=500, seed=0)
    initial_std = pf_near.summary(true_corners)["std"].mean()
    # Sweep positions around (and just outside) the wall; multiple vantage points
    # are required to triangulate corners (a single position only constrains each
    # corner to a circle).
    rng = np.random.default_rng(0)
    near_positions = np.stack([
        rng.uniform(-0.5, 0.5, size=200),
        rng.choice([-1, 1], size=200) * rng.uniform(0.06, 0.15, size=200),
    ], axis=-1)
    for p in near_positions:
        p = p.reshape(1, 2)
        sig = observation_noise_std(p, true_corners)
        obs = noisy_distances_to_corners(p, true_corners, sig)
        pf_near.update(p, obs, sig)
    near_summary = pf_near.summary(true_corners)
    if verbose:
        print(f'[PF unit] near sweep: initial std={initial_std:.4g}, '
              f'final std={near_summary["std"].mean():.4g}, '
              f'L2={near_summary["l2_to_truth"]:.4g}')
    assert near_summary["std"].mean() < initial_std / 3.0, (
        f'PF should tighten with near-wall sweep: {near_summary["std"].mean()} vs initial {initial_std}')
    assert near_summary["l2_to_truth"] < 0.15, (
        f'PF mean should converge near truth with triangulation: L2={near_summary["l2_to_truth"]}')

    pf_far = ParticleFilter(n_particles=500, seed=1)
    far_pos = np.array([[0.0, 0.95]])  # near top of arena, very high sigma
    for _ in range(200):
        sig = observation_noise_std(far_pos, true_corners)
        obs = noisy_distances_to_corners(far_pos, true_corners, sig)
        pf_far.update(far_pos, obs, sig)
    far_summary = pf_far.summary(true_corners)
    if verbose:
        print(f'[PF unit] far  pos sigma={float(observation_noise_std(far_pos, true_corners)):.4g}'
              f', final std={far_summary["std"].mean():.4g}, L2={far_summary["l2_to_truth"]:.4g}')
    # Far should barely tighten the posterior compared to near sweep.
    assert far_summary["std"].mean() > 3.0 * near_summary["std"].mean(), (
        'Far-from-wall observations should leave the posterior much broader than near')


def plot_particle_posterior(pf, true_corners=None, title=None):
    """Scatter all particle corner positions, overlaying the true corners."""
    labels = ['BL', 'BR', 'TR', 'TL']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    plt.figure(figsize=(6, 6))
    for k in range(4):
        pts = pf.particles[:, k, :]
        plt.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.3,
                    color=colors[k], label=f'particles {labels[k]}')
        if true_corners is not None:
            plt.scatter(true_corners[k, 0], true_corners[k, 1],
                        s=120, marker='x', color=colors[k],
                        label=f'true {labels[k]}')
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    plt.gca().add_patch(circle)
    plt.gca().set_aspect('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    if title:
        plt.title(title)
    plt.legend(fontsize=7, loc='upper right', ncol=2)
    plt.show()


def noisy_distances_to_corners(positions, true_corners, sigmas, rng=None):
    """Return noisy distances from each position to each of the 4 corners.

    positions:    (B, 2)
    true_corners: (4, 2)
    sigmas:       (B,)
    Returns:      (B, 4) clipped to >= 0
    """
    if rng is None:
        rng = np.random
    positions = np.atleast_2d(positions)
    true_distances = np.linalg.norm(
        positions[:, None, :] - true_corners[None, :, :], axis=-1)  # (B, 4)
    noise = rng.normal(0.0, 1.0, size=true_distances.shape) * sigmas[:, None]
    return np.maximum(0.0, true_distances + noise)


def generate_edge_biased_data(
        world,
        world_bounds,
        number_of_trajectories=10000,
        complexity=1.5,
        orientation=None,
        full_state_space=True,
        bias='high',
        edge_points=((-0.6, 0.0), (0.6, 0.0)),
        edge_radius=0.15,
        max_attempts_multiplier=20,
        return_trajectories=False):
    """Generate two-primitive exploration data biased by trajectory proximity to the
    wall edges at `edge_points`.

    bias='high' -> keep only trajectories that pass within `edge_radius` of an edge.
    bias='low'  -> keep only trajectories that never pass within `edge_radius` of an edge.
    """
    if bias not in ('high', 'low'):
        raise ValueError("bias must be 'high' or 'low'")

    data = []
    trajectories = []
    attempts = 0
    max_attempts = number_of_trajectories * max_attempts_multiplier

    while len(data) < number_of_trajectories and attempts < max_attempts:
        attempts += 1
        world.reset()

        if attempts % 1000 == 0:
            print(f'Edge-biased ({bias}): kept {len(data)}/{number_of_trajectories} after {attempts} attempts')

        start_position, _ = generate_random_positions(world, world_bounds=world_bounds, orientation=orientation, circular=True)
        _, goal_position1 = generate_random_positions(world, world_bounds=world_bounds, orientation=orientation)

        if full_state_space:
            wall_length = world.radius * 2 * world.wall_size
            wall_shape = [(-wall_length / 2, -world.wall_thickness / 2), (wall_length / 2, -world.wall_thickness / 2),
                          (wall_length / 2, world.wall_thickness / 2), (-wall_length / 2, world.wall_thickness / 2)]
            if not world.wall_present:
                distances_to_edges = np.array([2, 2, 2, 2])
            else:
                distances_to_edges = np.array([np.linalg.norm(start_position - np.array(point)) for point in wall_shape])

        complexity_scaled = complexity * ((np.abs(start_position[0] - goal_position1[0]) + np.abs(start_position[1] - goal_position1[1])) / 2)
        dmp_x1 = DMP1D(start=start_position[0], goal=goal_position1[0], n_basis=3, complexity=complexity_scaled)
        dmp_y1 = DMP1D(start=start_position[1], goal=goal_position1[1], n_basis=3, complexity=complexity_scaled)

        simulation = Simulation(world, dmp_x1, dmp_y1, start_position, T=1.0, dt=0.01)
        positions1, velocities1, collision1, _, _ = simulation.run()

        _, goal_position2 = generate_random_positions(world, world_bounds=world_bounds, orientation=orientation)
        complexity_scaled = complexity * ((np.abs(positions1[-1][0] - goal_position2[0]) + np.abs(positions1[-1][1] - goal_position2[1])) / 2)
        dmp_x2 = DMP1D(start=positions1[-1][0], goal=goal_position2[0], n_basis=3, complexity=complexity_scaled)
        dmp_y2 = DMP1D(start=positions1[-1][1], goal=goal_position2[1], n_basis=3, complexity=complexity_scaled)

        simulation = Simulation(world, dmp_x2, dmp_y2, positions1[-1], T=1.0, dt=0.01)
        positions2, velocities2, collision2, _, _ = simulation.run()

        near = (_trajectory_near_edges(positions1, edge_points, edge_radius)
                or _trajectory_near_edges(positions2, edge_points, edge_radius))

        if bias == 'high' and not near:
            continue
        if bias == 'low' and near:
            continue

        dmp_params1 = [dmp_x1.start, dmp_y1.start, dmp_x1.goal, dmp_y1.goal, *dmp_x1.weights, *dmp_y1.weights]
        dmp_params2 = [dmp_x1.goal, dmp_y1.goal, dmp_x2.goal, dmp_y2.goal, *dmp_x2.weights, *dmp_y2.weights]
        if full_state_space:
            s_t = np.concatenate((positions1[0], np.array([world.wall_present * 1.0]), distances_to_edges))
        else:
            s_t = positions1[0]
        if not world.wall_present:
            collision1, collision2 = np.array([0.0]), np.array([0.0])

        data.append((s_t, True, positions1[-1], positions2[-1], dmp_params1, dmp_params2, collision1 * 1, collision2 * 1))
        if return_trajectories:
            trajectories.append((np.array(positions1), np.array(positions2)))

    if len(data) < number_of_trajectories:
        print(f'Warning: only collected {len(data)}/{number_of_trajectories} edge-biased ({bias}) trajectories in {attempts} attempts.')

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
                   torch.tensor(collision2, dtype=torch.float32))
                  for s_t, s_t_plus_one, final_position1, final_position2, dmp_params1, dmp_params2, collision1, collision2 in train_data]

    valid_data = [(torch.tensor(s_t, dtype=torch.float32),
                   torch.tensor(s_t_plus_one, dtype=torch.float32),
                   torch.tensor(final_position1, dtype=torch.float32),
                   torch.tensor(final_position2, dtype=torch.float32),
                   torch.tensor(dmp_params1, dtype=torch.float32),
                   torch.tensor(dmp_params2, dtype=torch.float32),
                   torch.tensor(collision1, dtype=torch.float32),
                   torch.tensor(collision2, dtype=torch.float32))
                  for s_t, s_t_plus_one, final_position1, final_position2, dmp_params1, dmp_params2, collision1, collision2 in valid_data]

    if return_trajectories:
        return train_data, valid_data, trajectories
    return train_data, valid_data


def plot_edge_biased_trajectories(
        world,
        trajectories,
        edge_points=((-0.6, 0.0), (0.6, 0.0)),
        edge_radius=0.15,
        circular=True,
        title=None,
        max_trajectories=200):
    """Plot exploratory trajectories collected by `generate_edge_biased_data`, in the
    style of `plot_example_trajectories`. Each entry of `trajectories` is a tuple
    (positions1, positions2) for the two-primitive rollout."""
    plt.figure(figsize=(5, 4))
    ax = plt.gca()

    if circular:
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(world.radius * np.cos(theta), world.radius * np.sin(theta), 'k--')
        plt.xlim([-world.radius - 0.1, world.radius + 0.1])
        plt.ylim([-world.radius - 0.1, world.radius + 0.1])

    if getattr(world, 'wall_present', False):
        wall_length = world.radius * 2 * world.wall_size
        wt = world.wall_thickness
        rect = plt.Rectangle((-wall_length / 2, -wt / 2), wall_length, wt,
                             facecolor='gray', edgecolor='black', alpha=0.6)
        ax.add_patch(rect)

    for ep in edge_points:
        ax.add_patch(plt.Circle(ep, edge_radius, fill=False, color='red', linestyle=':', alpha=0.8))
        ax.plot(ep[0], ep[1], 'rx', markersize=8)

    n = min(len(trajectories), max_trajectories)
    for positions1, positions2 in trajectories[:n]:
        plt.plot(positions1[:, 0], positions1[:, 1], color='blue', alpha=0.1)
        plt.plot(positions2[:, 0], positions2[:, 1], color='green', alpha=0.1)
        plt.plot(positions1[0, 0], positions1[0, 1], 'o', color='black', markersize=2, alpha=0.4)

    ax.set_aspect('equal')
    if title is not None:
        plt.title(title)
    plt.show()


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
          trip_wire = None,
          pf=None,
          true_corners=None,
          pf_log_freq=1):

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

                    # Bayesian wall-state estimator: sample first (using stale posterior),
                    # then later update the filter with this batch's noisy observations.
                    # Only the distance-to-corner features (cols 3..6) of batch_s_t are
                    # replaced with samples drawn from the posterior; positions (cols 0..1)
                    # and the wall_present flag (col 2) stay ground-truth.
                    batch_s_t_pm = batch_s_t
                    if pf is not None and true_corners is not None:
                        batch_positions_np = batch_s_t[:, :2].detach().cpu().numpy().astype(np.float64)
                        sampled_corners = pf.sample(batch_s_t.shape[0])  # (B, 4, 2)
                        posterior_distances = np.linalg.norm(
                            batch_positions_np[:, None, :] - sampled_corners, axis=-1)  # (B, 4)
                        batch_s_t_pm = batch_s_t.clone()
                        batch_s_t_pm[:, 3:7] = torch.tensor(
                            posterior_distances, dtype=torch.float32)

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

                    # Forward pass through net_preds to get final state predictions.
                    # batch_s_t_pm has corner distances drawn from the posterior over
                    # wall corners (matches batch_s_t when pf is None).
                    inputs1 = torch.cat((batch_s_t_pm, dmp_params1), dim=1).unsqueeze(1)
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

                    # Update the particle filter with this batch's noisy observations
                    # AFTER PM has used the (stale) posterior for its forward pass.
                    if pf is not None and true_corners is not None:
                        valid_mask = (batch_s_t[:, 2] > 0.5).detach().cpu().numpy()
                        if valid_mask.any():
                            obs_positions = batch_positions_np[valid_mask]
                            sigmas = observation_noise_std(obs_positions, true_corners)
                            obs = noisy_distances_to_corners(
                                obs_positions, true_corners, sigmas)
                            pf.update(obs_positions, obs, sigmas)

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

                if pf is not None and (epoch % pf_log_freq == 0):
                    pf_summary = pf.summary(true_corners=true_corners)
                    mean_std = float(pf_summary["std"].mean())
                    l2 = pf_summary["l2_to_truth"]
                    print(f'  [PF] ESS={pf_summary["ess"]:.1f}, mean per-coord std={mean_std:.4f}'
                          + (f', L2(mean,truth)={l2:.4f}' if l2 is not None else ''))

    return valid_losses, net_goal, net_preds


if __name__ == "__main__":

    # Sanity-check the Bayesian wall estimator before any training.
    _pf_unit_check()

    # Initialize the world
    world_bounds = [-1, 1, -1, 1]
    world = CircularWorld(num_obstacles=0, max_speed=100, radius=1, wall_present=True,wall_size=0.6)

    # Check the environment
    plot_example_trajectories(
        world,world_bounds,
        number_of_trajectories=100,
        complexity=2.0,
        circular=True,
        random_position=True,
        n_basis=3,
        start_position=np.array([0.0, 0.85]),
        goal_position=np.array([0.0, -0.85]),
        trip_wire = None)
        # trip_wire=[[(-0.30,-1.00),(-0.1,-0.4)],[(0.30,-1.00),(0.1,-0.4)]])
        # trip_wire=[[(-0.7,-0.75),(0.2,0.0)],[(0.7,-0.75),(-0.2,0.0)]])
        # trip_wire=[[(-1.0,0.25),(0.0,0.0)],[(1.0,0.25),(0.0,0.0)]])
    

    # Collect Data for Predictive Net (Varying wall)
    train_data,valid_data= generate_data(
        world,
        world_bounds,
        number_of_trajectories=25000,
        orientation=None,
        complexity=1.5,
        varying_wall=True,
        full_state_space=True)
    
    training_sets = split_list(train_data, 10)

    # Initialize the network
    net = PredNet(input_size=2+2+2+6+5, hidden_size=64, output_size=3, dropout_rate=0.1)

    # Train Predictive Net (World Model)
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
        complexity=2.0,
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
        trip_wire=None)
        # trip_wire=[[(-0.30,-1.00),(-0.1,-0.4)],[(0.30,-1.00),(0.1,-0.4)]])
        # trip_wire=[[(-0.7,-0.75),(0.2,0.0)],[(0.7,-0.75),(-0.2,0.0)]])
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

    # Collect Data during Exploration -- Edge-biased pipelines (R1 extra analysis)
    # Test whether exploratory policies with high density near the wall edges at
    # (-0.6, 0) and (0.6, 0) yield more "edge-directed" two-primitive escape policies
    # than exploratory policies with low density near those edges.
    edge_points = ((-0.6, 0.0), (0.6, 0.0))
    edge_radius = 0.3

    world_edge = CircularWorld(num_obstacles=0, max_speed=100, radius=1, wall_present=True, wall_size=0.6)
    train_data_high_edge, valid_data_high_edge, traj_high_edge = generate_edge_biased_data(
        world_edge,
        world_bounds,
        number_of_trajectories=1000,
        complexity=1.5,
        orientation=None,
        full_state_space=True,
        bias='high',
        edge_points=edge_points,
        edge_radius=edge_radius,
        return_trajectories=True)

    train_data_low_edge, valid_data_low_edge, traj_low_edge = generate_edge_biased_data(
        world_edge,
        world_bounds,
        number_of_trajectories=1000,
        complexity=1.5,
        orientation=None,
        full_state_space=True,
        bias='low',
        edge_points=edge_points,
        edge_radius=edge_radius,
        return_trajectories=True,)

    plot_edge_biased_trajectories(
        world_edge, traj_high_edge, edge_points=edge_points, edge_radius=edge_radius,
        circular=True, title='Exploration: high density near edges')
    plot_edge_biased_trajectories(
        world_edge, traj_low_edge, edge_points=edge_points, edge_radius=edge_radius,
        circular=True, title='Exploration: low density near edges')

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
    task = 'home_run_explo'
    net_goal = GoalNet(input_size=2, hidden_size=64, output_size=2+2+6, dropout_rate=0.1)
    net_preds = [copy.deepcopy(net), copy.deepcopy(net)]

    # The exploration data train_data_high_edge was generated on world_edge,
    # so the agent's noisy observations should be consistent with that wall.
    pf_world = world_edge
    pf_wall_length = pf_world.radius * 2 * pf_world.wall_size
    true_corners_train = np.array([
        (-pf_wall_length / 2, -pf_world.wall_thickness / 2),
        ( pf_wall_length / 2, -pf_world.wall_thickness / 2),
        ( pf_wall_length / 2,  pf_world.wall_thickness / 2),
        (-pf_wall_length / 2,  pf_world.wall_thickness / 2),
    ])
    pf = ParticleFilter(n_particles=500, x_range=(-1.0, 1.0),
                        y_range=(-0.25, 0.25), seed=0)

    valid_losses, net_goal, net_preds = train(
        train_data = train_data_low_edge,
        valid_data = valid_data_low_edge,
        net_goal = net_goal,
        net_preds = net_preds,
        target_goal1 = [0.,0.85],
        target_goal2 = [0.,0.85],
        task = task,
        fine_tune_pred_nets = False,
        num_samples_inverse_dynamics = 10000,
        num_iterations_inverse_dynamics = 1,
        bound_dmp_weights = 1.5,
        early_stopping_threshold = 1.0,
        learning_rate = 0.001,
        batch_size = 10,
        num_epochs = 10,
        plot_trajectories = True,
        world = world,
        valid_batch_size=5,
        trip_wire=None,
        pf=pf,
        true_corners=true_corners_train)

    # Quick visualization of the posterior over wall corners after training.
    plot_particle_posterior(pf, true_corners_train,
                            title='Posterior over wall corners after Training Phase')
    

    # Testing Phase
    task = 'home_run_explo'
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
        bound_dmp_weights = 1.5,
        early_stopping_threshold = 1.00,
        learning_rate = 0.0001,
        batch_size = 15,
        num_epochs = 100,
        plot_trajectories = True,
        world=new_world,
        valid_batch_size=5,
        trip_wire=None,)