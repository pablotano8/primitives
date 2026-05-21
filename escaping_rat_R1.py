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
    """Closed-form Gaussian posterior over (x_left, x_right).

    Despite the legacy name, this is not a particle filter -- it's a pair of
    independent univariate Gaussians updated by the standard conjugate
    Gaussian formula. Each observation is a noisy direct reading of x_left
    (or x_right) with noise std scaled by the agent's distance to that edge
    segment. The posterior is exact, unbiased, and shrinks monotonically as
    1/sqrt(N) with evidence -- no resampling, no degeneracy, no bias from
    non-linear distance observations.

    Wall thickness is known; the 4 corners are derived from (xL, xR) as
        BL=(xL,-t/2)  BR=(xR,-t/2)  TR=(xR,+t/2)  TL=(xL,+t/2)
    """

    def __init__(self, wall_thickness=0.2,
                 xL_prior_mean=-0.5, xL_prior_std=0.3,
                 xR_prior_mean=+0.5, xR_prior_std=0.3,
                 sigma_min=0.05, slope=1.0,
                 n_particles=500, seed=None,
                 # legacy kwargs accepted for backwards compat
                 x_left_range=None, x_right_range=None, resample_jitter=None):
        if x_left_range is not None:
            xL_prior_mean = 0.5 * (x_left_range[0] + x_left_range[1])
            xL_prior_std = (x_left_range[1] - x_left_range[0]) / np.sqrt(12)
        if x_right_range is not None:
            xR_prior_mean = 0.5 * (x_right_range[0] + x_right_range[1])
            xR_prior_std = (x_right_range[1] - x_right_range[0]) / np.sqrt(12)

        self.wall_thickness = float(wall_thickness)
        self.xL_mean = float(xL_prior_mean)
        self.xL_var = float(xL_prior_std) ** 2
        self.xR_mean = float(xR_prior_mean)
        self.xR_var = float(xR_prior_std) ** 2
        self.sigma_min = sigma_min
        self.slope = slope
        self.n_particles = n_particles   # only used for sample() / plotting
        self.rng = np.random.default_rng(seed)
        self.n_obs_applied = 0

    def _sigma_to_edge(self, positions, x_edge):
        """Per-position noise std for an observation of x_edge.
        Distance is to the closest point on the vertical edge segment at x=x_edge."""
        positions = np.atleast_2d(positions)
        t2 = self.wall_thickness / 2
        dx = positions[:, 0] - x_edge
        dy = np.maximum(0.0, np.abs(positions[:, 1]) - t2)
        d = np.sqrt(dx * dx + dy * dy)
        return self.sigma_min + self.slope * d

    def _corners_from_params(self, params):
        """(..., 2) [xL, xR] -> (..., 4, 2) corners in BL, BR, TR, TL order."""
        t2 = self.wall_thickness / 2
        xL = params[..., 0]
        xR = params[..., 1]
        return np.stack([
            np.stack([xL, np.full_like(xL, -t2)], axis=-1),
            np.stack([xR, np.full_like(xR, -t2)], axis=-1),
            np.stack([xR, np.full_like(xR,  t2)], axis=-1),
            np.stack([xL, np.full_like(xL,  t2)], axis=-1),
        ], axis=-2)

    @property
    def particles(self):
        """N i.i.d. draws from the posterior, shape (N, 2). For plotting."""
        return np.stack([
            self.rng.normal(self.xL_mean, np.sqrt(self.xL_var), size=self.n_particles),
            self.rng.normal(self.xR_mean, np.sqrt(self.xR_var), size=self.n_particles),
        ], axis=-1)

    @property
    def log_weights(self):
        """All draws equally weighted (closed-form posterior)."""
        return np.zeros(self.n_particles)

    def sample(self, n):
        """Draw n joint posterior samples and convert to 4-corner positions."""
        params = np.stack([
            self.rng.normal(self.xL_mean, np.sqrt(self.xL_var), size=n),
            self.rng.normal(self.xR_mean, np.sqrt(self.xR_var), size=n),
        ], axis=-1)
        return self._corners_from_params(params)

    def update(self, positions, true_corners):
        """Fold in a batch of noisy xL/xR observations from these positions.

        positions:    (B, 2) numpy float, agent positions
        true_corners: (4, 2) numpy float, ground-truth corners -- only the
                      min/max of column 0 (xL, xR) are used.
        """
        positions = np.atleast_2d(positions)
        if positions.shape[0] == 0:
            return
        true_xL = float(true_corners[:, 0].min())
        true_xR = float(true_corners[:, 0].max())

        # generate noisy direct readings of xL and xR
        sig_L = self._sigma_to_edge(positions, true_xL)
        sig_R = self._sigma_to_edge(positions, true_xR)
        obs_L = true_xL + self.rng.normal(0.0, sig_L)
        obs_R = true_xR + self.rng.normal(0.0, sig_R)

        # conjugate Gaussian update for xL (vectorized over the batch)
        prec_obs_L = 1.0 / (sig_L ** 2)
        new_prec_L = 1.0 / self.xL_var + prec_obs_L.sum()
        self.xL_mean = (self.xL_mean / self.xL_var + (obs_L * prec_obs_L).sum()) / new_prec_L
        self.xL_var = 1.0 / new_prec_L

        prec_obs_R = 1.0 / (sig_R ** 2)
        new_prec_R = 1.0 / self.xR_var + prec_obs_R.sum()
        self.xR_mean = (self.xR_mean / self.xR_var + (obs_R * prec_obs_R).sum()) / new_prec_R
        self.xR_var = 1.0 / new_prec_R

        self.n_obs_applied += positions.shape[0]

    def summary(self, true_corners=None):
        mean_params = np.array([self.xL_mean, self.xR_mean])
        std_params = np.sqrt(np.array([self.xL_var, self.xR_var]))
        t2 = self.wall_thickness / 2
        mean_corners = np.array([
            [self.xL_mean, -t2], [self.xR_mean, -t2],
            [self.xR_mean,  t2], [self.xL_mean,  t2],
        ])
        l2 = None if true_corners is None else float(np.linalg.norm(mean_corners - true_corners))
        return {
            "mean_params": mean_params,
            "std_params": std_params,
            "mean_corners": mean_corners,
            "ess": float(self.n_particles),
            "l2_to_truth": l2,
        }


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


def observation_noise_std(positions, true_corners, sigma_min=0.03, slope=0.8):
    """Per-corner observation noise: smooth, no hard cutoff.

    Returns shape (B, 4): one sigma per (observation, corner). Sigma for corner
    k grows LINEARLY with the agent's distance to corner k:

        sigma_k = sigma_min + slope * d_to_corner_k

    Defaults give sigma ~ 0.03 right at the corner, ~ 0.19 at d=0.2, ~ 0.43 at
    d=0.5, ~ 0.83 at d=1.0. Smooth degradation of information with distance,
    no binary cutoff -- distant observations contribute weakly but still
    contribute, so posterior shrinkage is gradual as evidence accumulates.
    """
    positions = np.atleast_2d(positions)
    d = np.linalg.norm(positions[:, None, :] - true_corners[None, :, :], axis=-1)
    return sigma_min + slope * d


def _pf_unit_check(verbose=True):
    """Sanity check the per-corner observation model.

    HIGH-edge proxy: positions cycle near each of the 4 corners -> all corners
        should converge to truth.
    LOW-edge proxy: positions stay over the wall middle (close to wall surface
        but >cutoff_distance from any corner) -> posterior stays near prior.
    """
    true_corners = np.array([(-0.6, -0.1), (0.6, -0.1),
                             (0.6, 0.1), (-0.6, 0.1)])
    rng = np.random.default_rng(0)

    pf_near = ParticleFilter(n_particles=500, wall_thickness=0.2, seed=0)
    initial_std = pf_near.summary(true_corners)["std_params"].mean()
    # Cycle near each corner with a small random offset (within cutoff).
    high_edge_positions = []
    for _ in range(50):
        for c in true_corners:
            high_edge_positions.append(c + rng.uniform(-0.08, 0.08, size=2))
    high_edge_positions = np.array(high_edge_positions)
    pf_near.update(high_edge_positions, true_corners)
    near_summary = pf_near.summary(true_corners)
    if verbose:
        print(f'[PF unit] high-edge sweep (near each corner): '
              f'initial std_params={initial_std:.4g}, '
              f'final mean_params={near_summary["mean_params"].round(3).tolist()}, '
              f'std_params={near_summary["std_params"].round(4).tolist()}, '
              f'L2(corners,truth)={near_summary["l2_to_truth"]:.4g}, '
              f'obs applied={pf_near.n_obs_applied}')
    assert near_summary["std_params"].mean() < initial_std / 3.0, (
        f'PF should tighten when agent visits each corner: {near_summary["std_params"]} vs initial {initial_std}')
    assert near_summary["l2_to_truth"] < 0.1, (
        f'PF mean corners should converge to truth: L2={near_summary["l2_to_truth"]}')

    # LOW-edge proxy: positions over the wall middle (close to wall surface,
    # far from any lateral corner). Under the smooth noise model they still
    # contribute, just much more weakly per observation.
    n_low = len(high_edge_positions)  # match observation count
    pf_low = ParticleFilter(n_particles=500, wall_thickness=0.2, seed=2)
    low_edge_positions = np.stack([
        rng.uniform(-0.3, 0.3, size=n_low),       # central x -- far from corners at x=+-0.6
        rng.choice([-1, 1], size=n_low) * rng.uniform(0.0, 0.15, size=n_low),  # near wall surface
    ], axis=-1)
    pf_low.update(low_edge_positions, true_corners)
    low_summary = pf_low.summary(true_corners)
    if verbose:
        print(f'[PF unit] low-edge sweep (over wall middle): '
              f'final std_params={low_summary["std_params"].round(4).tolist()}, '
              f'L2(corners,truth)={low_summary["l2_to_truth"]:.4g}, '
              f'obs applied={pf_low.n_obs_applied}')
    # With the smooth noise model, low-edge observations DO contribute weakly
    # (no hard cutoff). Posterior should be noticeably broader than high-edge
    # for an equal number of observations -- per-obs information is much lower.
    assert low_summary["std_params"].mean() > 1.5 * near_summary["std_params"].mean(), (
        f'Low-edge regime should keep posterior broader than high-edge: '
        f'low={low_summary["std_params"]} vs high={near_summary["std_params"]}')


def plot_particle_posterior(pf, true_corners=None, title=None):
    """Visualize the posterior over (x_left, x_right).

    Two panels:
      - left: scatter of particles in (x_left, x_right) space, with truth marked
      - right: each particle drawn as a candidate wall rectangle (transparent),
               with the true wall outlined in black.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    w = np.exp(pf.log_weights - np.max(pf.log_weights))
    w = w / w.sum()
    pts = pf.particles
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.4, c=w, cmap='viridis')
    if true_corners is not None:
        xL_true = true_corners[:, 0].min()
        xR_true = true_corners[:, 0].max()
        ax.scatter([xL_true], [xR_true], s=200, marker='x', color='red',
                   linewidths=2.5, label=f'truth ({xL_true:.2f}, {xR_true:.2f})')
        ax.legend(fontsize=8)
    ax.set_xlabel('x_left'); ax.set_ylabel('x_right')
    ax.set_xlim(-1.05, 0.05); ax.set_ylim(-0.05, 1.05)
    ax.set_title('posterior over (x_left, x_right)')
    ax.grid(alpha=0.3)

    ax = axes[1]
    t2 = pf.wall_thickness / 2
    # subsample particles for plotting
    idx = np.argsort(w)[-200:]
    for i in idx:
        xL, xR = pts[i]
        ax.add_patch(plt.Rectangle((xL, -t2), xR - xL, pf.wall_thickness,
                                   fill=True, alpha=0.02 + 0.3 * w[i] / w[idx].max(),
                                   color='tab:blue', edgecolor=None))
    if true_corners is not None:
        xL_true = true_corners[:, 0].min()
        xR_true = true_corners[:, 0].max()
        ax.add_patch(plt.Rectangle((xL_true, -t2), xR_true - xL_true, pf.wall_thickness,
                                   fill=False, edgecolor='red', linewidth=2,
                                   label='true wall'))
        ax.legend(fontsize=8)
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    ax.set_aspect('equal'); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_title('candidate walls (top 200 particles)')

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def noisy_distances_to_corners(positions, true_corners, sigmas, rng=None):
    """Return noisy distances from each position to each of the 4 corners.

    positions:    (B, 2)
    true_corners: (4, 2)
    sigmas:       (B, 4) per-corner noise std (or (B,) broadcast across corners)
    Returns:      (B, 4) clipped to >= 0
    """
    if rng is None:
        rng = np.random
    positions = np.atleast_2d(positions)
    sigmas = np.asarray(sigmas)
    if sigmas.ndim == 1:
        sigmas = np.repeat(sigmas[:, None], 4, axis=1)
    true_distances = np.linalg.norm(
        positions[:, None, :] - true_corners[None, :, :], axis=-1)  # (B, 4)
    # don't draw on a sentinel sigma -- it can hit overflow and waste cycles
    noise = rng.normal(0.0, 1.0, size=true_distances.shape) * np.minimum(sigmas, 1e3)
    return np.maximum(0.0, true_distances + noise)


def random_exploration_waypoints(world, world_bounds, n_trajectories=1000,
                                 complexity=1.5, waypoints_per_trajectory=5,
                                 verbose=True):
    """Simulate N random two-DMP trajectories and return subsampled waypoints.

    Used to seed the wall-corner posterior BEFORE escape-policy training. Each
    rollout produces ~200 raw timesteps, but adjacent timesteps are highly
    correlated -- treating them as independent observations would massively
    overcount evidence. We evenly subsample `waypoints_per_trajectory` points
    from each of the two DMPs, so each rollout contributes ~2*K observations.
    """
    all_positions = []
    for i in range(n_trajectories):
        if verbose and i and i % 200 == 0:
            print(f'random exploration: {i}/{n_trajectories}')
        world.reset()
        start_position, _ = generate_random_positions(
            world, world_bounds=world_bounds, orientation=None, circular=True)
        _, goal1 = generate_random_positions(world, world_bounds=world_bounds, orientation=None)
        c1 = complexity * ((abs(start_position[0] - goal1[0]) + abs(start_position[1] - goal1[1])) / 2)
        sim = Simulation(world,
                         DMP1D(start=start_position[0], goal=goal1[0], n_basis=3, complexity=c1),
                         DMP1D(start=start_position[1], goal=goal1[1], n_basis=3, complexity=c1),
                         start_position, T=1.0, dt=0.01)
        positions1, _, _, _, _ = sim.run()

        _, goal2 = generate_random_positions(world, world_bounds=world_bounds, orientation=None)
        c2 = complexity * ((abs(positions1[-1][0] - goal2[0]) + abs(positions1[-1][1] - goal2[1])) / 2)
        sim = Simulation(world,
                         DMP1D(start=positions1[-1][0], goal=goal2[0], n_basis=3, complexity=c2),
                         DMP1D(start=positions1[-1][1], goal=goal2[1], n_basis=3, complexity=c2),
                         positions1[-1], T=1.0, dt=0.01)
        positions2, _, _, _, _ = sim.run()

        for positions in (positions1, positions2):
            arr = np.array(positions)
            if waypoints_per_trajectory and len(arr) > waypoints_per_trajectory:
                idx = np.linspace(0, len(arr) - 1, waypoints_per_trajectory).astype(int)
                arr = arr[idx]
            all_positions.append(arr)
    return np.concatenate(all_positions, axis=0)


def update_pf_from_exploration(pf, true_corners, waypoints, batch_size=10, shuffle=True):
    """Stream `waypoints` through the PF in mini-batches, mutating its posterior."""
    if shuffle:
        order = np.random.permutation(len(waypoints))
        waypoints = waypoints[order]
    for i in range(0, len(waypoints), batch_size):
        pf.update(waypoints[i:i + batch_size], true_corners)


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

                    # The PF posterior is FROZEN inside train(): wall-state
                    # estimation happens entirely in a pre-training random
                    # exploration phase. This isolates PM-learning performance
                    # from state-estimation progress so we can study how PM
                    # quality varies with the agent's prior wall-knowledge.

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
                    mp = pf_summary["mean_params"]; sp = pf_summary["std_params"]
                    l2 = pf_summary["l2_to_truth"]
                    print(f'  [PF] xL={mp[0]:+.3f}+-{sp[0]:.3f}, xR={mp[1]:+.3f}+-{sp[1]:.3f}, '
                          f'ESS={pf_summary["ess"]:.1f}'
                          + (f', L2(corners,truth)={l2:.4f}' if l2 is not None else '')
                          + f', obs applied={pf.n_obs_applied}')

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
    # The PF infers (x_left, x_right) only; wall thickness is given.
    pf = ParticleFilter(n_particles=25, wall_thickness=pf_world.wall_thickness,
                        x_left_range=(-1.0, 0.0), x_right_range=(0.0, 1.0), seed=0)

    # Pre-training: random exploration to seed the wall-edge posterior. All
    # state estimation happens here -- train() then runs with the PF frozen,
    # so PM-learning performance can be studied as a function of how much
    # prior wall-knowledge the agent had before escape-policy training. Vary
    # n_trajectories to sweep over "degrees of pre-training learning".
    print('Running random exploration to build the wall-edge posterior...')
    explo_waypoints = random_exploration_waypoints(
        pf_world, world_bounds, n_trajectories=5, complexity=1.5)
    update_pf_from_exploration(pf, true_corners_train, explo_waypoints, batch_size=10)
    pf_summary = pf.summary(true_corners=true_corners_train)
    print(f'Posterior after random exploration: '
          f'xL={pf_summary["mean_params"][0]:+.3f}+-{pf_summary["std_params"][0]:.4f}, '
          f'xR={pf_summary["mean_params"][1]:+.3f}+-{pf_summary["std_params"][1]:.4f}, '
          f'obs applied={pf.n_obs_applied}, '
          f'L2(corners,truth)={pf_summary["l2_to_truth"]:.4f}')
    plot_particle_posterior(pf, true_corners_train,
                            title='Posterior after random exploration (pre-train)')

    valid_losses, net_goal, net_preds = train(
        train_data = train_data_low_edge,
        valid_data = valid_data_low_edge,
        net_goal = net_goal,
        net_preds = net_preds,
        target_goal1 = [0.,0.85],
        target_goal2 = [0.,0.85],
        task = task,
        fine_tune_pred_nets = False,
        num_samples_inverse_dynamics = 1000,
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
    # Should look near-prior (broad blobs) -- low-edge trajectories provide
    # very few observations within the cutoff distance of the wall.
    plot_particle_posterior(pf, true_corners_train,
                            title='Posterior over wall corners after Training Phase (low-edge)')
    

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