import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import eigsh, splu, spsolve
from scipy.sparse import identity
from scipy.sparse import random as sparse_random

# Hybrid Shift-and-Invert Power Method + RL
class ShiftInvertPowerRL:
    def __init__(self, A, sigma=0.0, shift_tol=1e-6):
        self.A = A
        self.sigma = sigma  # Shift value
        self.shift_tol = shift_tol
        self.dim = A.shape[0]
        self._precompute_shifted_matrix()

    def _precompute_shifted_matrix(self):
        """Precompute (A - σI)^-1 for sparse systems"""
        if issparse(self.A):
            I = identity(self.dim, format='csr')
            self.M = self.A - self.sigma * I
            self.M_solver = splu(self.M.tocsc())
        else:
            self.M = self.A - self.sigma * np.eye(self.dim)
            self.M_inv = np.linalg.inv(self.M)

    def solve_shifted_system(self, b):
        """Solve (A - σI)x = b"""
        if issparse(self.A):
            return self.M_solver.solve(b)
        else:
            return np.linalg.solve(self.M, b)

    def hybrid_power_iteration(self, num_iter=10):
        """Warm start using shift-and-invert power method"""
        x = np.random.randn(self.dim)
        x /= np.linalg.norm(x)

        for _ in range(num_iter):
            x = self.solve_shifted_system(x)
            x /= np.linalg.norm(x) + 1e-8
        return x

# RL Environment for Eigenvector Refinement
class ShiftInvertEnv(gym.Env):
    def __init__(self, A, sigma, target_eigenvalue):
        super().__init__()
        self.solver = ShiftInvertPowerRL(A, sigma)
        self.target_eigenvalue = target_eigenvalue
        self.dim = A.shape[0]

        # Observation: current vector + residual history
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                           shape=(self.dim + 3,), dtype=np.float32)

        # Action: vector adjustment directions
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.dim,), dtype=np.float32)

        # Initialize with hybrid power method
        self.state = self.solver.hybrid_power_iteration(num_iter=20)
        self.residual_history = []

    def reset(self):
        self.state = self.solver.hybrid_power_iteration(num_iter=20)
        self.residual_history = []
        return self._get_obs()

    def step(self, action):
        # RL-guided adjustment
        adjusted_vec = self.state + 0.1 * action
        adjusted_vec /= np.linalg.norm(adjusted_vec) + 1e-8

        # Power method step for stability
        refined_vec = self.solver.solve_shifted_system(adjusted_vec)
        refined_vec /= np.linalg.norm(refined_vec) + 1e-8

        # Calculate metrics
        residual = self.solver.A @ refined_vec - self.target_eigenvalue * refined_vec
        residual_norm = np.linalg.norm(residual)
        self.residual_history.append(residual_norm)

        # Reward shaping
        reward = -residual_norm - 0.1 * np.log(residual_norm + 1e-8)

        self.state = refined_vec
        done = residual_norm < self.solver.shift_tol

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """Augment state with residual statistics"""
        residual_stats = [
            np.mean(self.residual_history[-10:]),
            np.min(self.residual_history[-10:]),
            np.std(self.residual_history[-10:])
        ] if self.residual_history else [0, 0, 0]
        return np.concatenate([self.state, residual_stats])


def train_hybrid_eigen_solver(A, sigma=0.0, total_timesteps=50000):

    eigvals, eigvecs = eigsh(A, k=1, sigma=sigma)
    target_eigenvalue = eigvals[0]
    actual_eigenvector = eigvecs[:, 0]


    env = ShiftInvertEnv(A, sigma, target_eigenvalue)


    policy_kwargs = dict(net_arch=[256, 256, 128])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        verbose=1
    )


    model.learn(total_timesteps=total_timesteps)


    obs = env.reset()
    best_vec = env.state
    best_residual = np.inf

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        current_residual = np.linalg.norm(A @ env.state - target_eigenvalue * env.state)

        if current_residual < best_residual:
            best_residual = current_residual
            best_vec = env.state

    v1 = best_vec / (np.linalg.norm(best_vec) + 1e-8)
    v2 = actual_eigenvector / (np.linalg.norm(actual_eigenvector) + 1e-8)
    cos_sim = np.dot(v1, v2)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)  # Handle numerical errors
    cos_dist = 2 * np.arccos(abs(cos_sim))/ np.pi

    print(f"Final residual norm: {best_residual:.4e}")
    print(f"Eigenvector norm check: {np.linalg.norm(best_vec):.4f}")
    print(f"Cosine distance to actual eigenvector: {cos_dist:.6f}")

    return best_vec, best_residual, cos_dist


import numpy as np
from scipy.sparse import random as sparse_random
import pandas as pd

# Settings
sizes = [5, 10, 50, 100, 500, 1000]
runs_per_size = 4
sigma = 0.5

# Storage for results
results = []

for size in sizes:
    for run in range(1, runs_per_size + 1):
        print(f"\nRunning size {size}, iteration {run}")
        np.random.seed(42 + run)  # Different seed for variability

        A = sparse_random(size, size, density=0.01, format='csr')
        A = (A + A.T) * 0.5  # Make symmetric

        _, residual, cos_dist = train_hybrid_eigen_solver(A, sigma=sigma)

        results.append({
            "Matrix Size": size,
            "Run": run,
            "Final Residual": residual,
            "Cosine Distance": cos_dist
        })

# Convert to DataFrame for display
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))


