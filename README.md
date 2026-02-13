# Composing Predictable Primitives for Zero-Shot Learning

Code for the paper: **"Composing predictable primitives for zero-shot learning"** by Pablo Tano, Jacob Bakermans, Charles Findling, Tiago Branco, and Alexandre Pouget.

## Overview

This repository implements a framework for **zero-shot behavioral adaptability** — solving new tasks on the first attempt without trial and error. The approach combines:

1. **Dynamic Movement Primitives (DMPs)** as simple, parametric behavioral building blocks.
2. **A World Model (WM)** — an LSTM that learns to predict the outcomes of sequences of primitives via self-supervised learning.
3. **A Planning Model (PM)** — an LSTM that proposes primitive sequences, optimized offline by backpropagating through the frozen world model.

The key insight is that composing simple, highly predictable primitives keeps gradient-based open-loop planning tractable and produces extreme sample efficiency.

---

## File Descriptions and Mapping to Paper Figures

### Core Infrastructure

| File | Description |
|------|-------------|
| `dmps.py` | Implements 1D Dynamic Movement Primitives (`DMP1D`) and a `Simulation` class that executes 2D DMP pairs in a physics world with collision detection. This is the foundation primitive used by all experiments. |
| `continuous_nav_envs.py` | Defines all 2D navigation environments using pymunk physics: `RandomRectangleWorld` (circular boundary with optional wall), `RandomHolesWorld` (square world with circular hole obstacles), and `ContinuousActionWorld` (step-based variant for non-hierarchical baselines). |
| `utils.py` | Shared utility functions for sampling DMP parameters, executing sequential DMPs in a world, and visualization helpers. |
| `plot_trajectories.py` | Visualization utilities for plotting trajectories and comparing world model predictions against actual DMP simulations. |

### Random Obstacle Environment (Figure 2a–b, 3, 4, 5 and 6)

A point agent navigates around random rectangular obstacles in a circular arena. The world model receives position + wall-edge distances as input and predicts outcomes of 2-primitive sequences.

| File | Description | Figures |
|------|-------------|---------|
| `random_obstacle.py` | **Main approach.** Trains the LSTM world model on randomly generated DMP trajectories, then trains the LSTM planning model by differentiable optimization through the frozen world model. Supports tasks: reach goal, two goals, move away, explore and return. | **Fig. 2a–b** (WM prediction accuracy + zero-shot performance); **Fig. 6** (PM optimization landscape and collision penalties) |
| `random_obstacle_ppo.py` | **Model-free RL baseline.** Trains PPO directly on the real physics environment (no world model). Hierarchical action space outputs DMP parameters. | **Fig. 2b** (PPO comparison bars at 50K and 1M episodes); **Fig. 3c–d** (No World Model ablation) |
| `random_obstacle_mb.py` | **Model-based RL baseline.** Dyna-style approach: trains the same world model, then trains a PPO policy inside a simulated environment built from the world model. | **Fig. 2b** (model-based RL comparison) |
| `random_obstacle_non_hier.py` | **Non-hierarchical baseline (No Primitives ablation).** Replaces DMPs with sequences of elementary step actions (angle + fixed step size). Uses an LSTM world model and planner operating over raw actions. Tests multiple planning horizons. | **Fig. 3a–b** (No Primitives: prediction accuracy degrades with horizon; zero-shot performance vs. planning horizon) |

### Random Holes Environment (Figure 2c–d)

A square world with two large circular holes that terminate trajectories on contact. The agent must reach goals while avoiding the holes.

| File | Description | Figures |
|------|-------------|---------|
| `random_holes.py` | **Main approach.** Same architecture as `random_obstacle.py` adapted for the holes environment. Trains world model on 200K trajectories across random worlds, then trains the planner on a specific world configuration. | **Fig. 2c–d** (WM prediction + zero-shot performance on reach-one-goal and reach-two-goals with holes) |
| `random_holes_ppo.py` | **Model-free RL baseline.** PPO trained directly on the holes environment. | **Fig. 2d** (PPO comparison bars) |
| `random_holes_mb.py` | **Model-based RL baseline.** Dyna-style PPO using the learned world model. | **Fig. 2d** (model-based RL comparison) |

### MuJoCo Ant Environment (Figures 2e–f, 7)

A 28-dimensional MuJoCo Ant quadruped. Primitives are PPO-trained walking policies parameterized by direction and duration, producing naturalistic locomotion.

| File | Description | Figures |
|------|-------------|---------|
| `mujoco_walk_pol.py` | **Main approach.** Full pipeline: (1) trains 9 PPO walking policies (8 cardinal directions + stationary), (2) collects trajectory data, (3) trains LSTM world model on (x,y) position sequences, (4) trains LSTM planner through the world model. Supports tasks: single goal, sequential goals, obstacle avoidance, explore-and-return, safe-area exploration. | **Fig. 2e–f** (WM accuracy, zero-shot Ant trajectories); **Fig. 7b–f** (zero-shot trajectories on 5 tasks); **Fig. 7g–k** (walking vs. uncorrelated primitives comparison) |
| `mujoco_walk_mb.py` | **Model-based RL baseline.** Dyna-style PPO for the MuJoCo Ant. | **Fig. 2f** (model-based RL comparison) |

### Escaping Rat Environment (Figures 4, 5, 9)

A circular platform with a shelter and removable wall, modeling the mouse escape experiments of Shamash et al. (2021, 2023). Includes a trip-wire mechanism to filter trajectories, simulating the constraint that mice only experience certain paths during exploration.

| File | Description | Figures |
|------|-------------|---------|
| `escaping_rat.py` | **Main approach.** Implements the full escape model: world model trained with experience constrained by trip-wire filtering (mimicking exploration patterns), then planner trained offline. Reproduces edge-vector vs. home-vector escape strategies across all experimental conditions. Includes data subset functions for analyzing how exploration constraints shape escape trajectories. | **Fig. 4** (all 9 experimental conditions: exploration, escape with/without wall, model predictions); **Fig. 5a** (comparison showing necessity of all components); **Fig. 9** (wall-length manipulation + shelter relocation) |
| `escaping_rat_ppo.py` | **RL baseline for escape.** Trains TD3 (despite filename) on the escape environment. | **Fig. 5a** (No World Model ablation — model-free RL fails without escape rewards during exploration) |

### Ablation & Analysis (Figures 3e–i)

| File | Description | Figures |
|------|-------------|---------|
| `complexity_tradeoff.py` | **Complexity vs. compositionality analysis.** Systematically varies DMP complexity and measures its effect on world model accuracy and task performance. Compares compositional (2-primitive sequences) vs. non-compositional (single complex primitive) agents. Shows that compositional agents bypass the complexity–controllability tradeoff. | **Fig. 3e–h** (predictability and controllability vs. complexity; compositional vs. non-compositional performance) |
| `direct_optimization.py` | **Direct parameter optimization baseline.** Instead of an amortized planner network, directly optimizes DMP parameters through the world model via gradient descent for each initial state. Demonstrates the optimization landscape and convergence properties. | **Fig. 1h** (example PM optimization iterations); **Fig. 6** (optimization landscape visualization) |

### Fixed Environment (Additional baselines)

A square world with a single fixed horizontal wall. Used for controlled comparisons where the obstacle configuration does not vary.

| File | Description |
|------|-------------|
| `fixed_env.py` | **Main approach** for the fixed-wall environment. Same WM+PM architecture. |
| `fixed_env_ppo.py` | **PPO baseline** for the fixed-wall environment. |
| `fixed_env_mb.py` | **Model-based PPO baseline** for the fixed-wall environment. |

---

## Quick Start

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- pymunk (2D physics)
- MuJoCo + mujoco-py (for Ant experiments)
- Stable-Baselines3 (for PPO/TD3 baselines)
- Matplotlib

### Running Experiments

Each experiment file can be run directly. For example:

```bash
# Train world model + planner on random obstacle environment
python random_obstacle.py

# Train world model + planner on MuJoCo Ant
python mujoco_walk_pol.py

# Run the escaping rat experiment
python escaping_rat.py

# Run the complexity tradeoff analysis
python complexity_tradeoff.py
```

## Citation
