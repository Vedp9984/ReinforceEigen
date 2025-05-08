
# Hybrid Shift-and-Invert Power Method with RL: Extended Documentation

## Overview
This code implements a **hybrid numerical-RL approach** to compute eigenvectors of sparse/dense matrices. It combines:
1. **Shift-and-Invert Power Method**: Traditional iterative linear algebra for eigenvalue problems.
2. **Reinforcement Learning (RL)**: Policy-guided refinement of eigenvectors using the PPO algorithm.

**Key Innovation**: Uses RL to escape local minima and accelerate convergence where traditional methods might stagnate.



## Component Deep Dive

### 1. Core Numerical Engine (`ShiftInvertPowerRL` Class)
**Mathematical Foundation**: Solves eigenvalue problems using shifted systems:
```math
(A - σI)x = λx → (A - σI)^{-1}x = \frac{1}{λ}x
```
Where σ is a shift near the target eigenvalue.

#### Critical Methods:
| Method | Purpose | Technical Details |
|--------|---------|-------------------|
| `_precompute_shifted_matrix` | Factorizes `(A - σI)` | - Sparse: LU decomposition via `splu`<br>- Dense: Explicit matrix inverse |
| `solve_shifted_system` | Solves `(A - σI)x = b` | - Sparse: Uses precomputed LU factors<br>- Dense: Uses `np.linalg.solve` |
| `hybrid_power_iteration` | Warm-start eigenvector | Combines power iteration with shift-and-invert acceleration |

**Sparse Optimization**: Achieves O(n) complexity for sparse matrices via:
- Compressed Sparse Row (CSR) format
- SuperLU decomposition (`splu`)

---

### 2. RL Environment (`ShiftInvertEnv` Class)
**Observation Space** (`dim + 3`):
- **Primary State**: Current eigenvector estimate (`dim` elements)
- **Residual Context**: 
  - Mean of last 10 residuals
  - Minimum residual in window
  - Standard deviation of residuals

**Action Space** (`dim`-dimensional):
- Continuous adjustments ∈ [-1, 1] to eigenvector components
- Scaled by 0.1 before application:  
  `adjusted_vec = state + 0.1 * action`

#### Reward Engineering:
```python
reward = -residual_norm - 0.1 * log(residual_norm + 1e-8)
```
- **Primary Term**: `-residual_norm` directly penalizes eigenvector error
- **Logarithmic Term**: Encourages faster convergence in early stages
- **Tolerance**: Episode terminates when `residual_norm < shift_tol` (1e-6)

---

### 3. RL Training Pipeline (`train_hybrid_eigen_solver`)
#### Phase 1: Ground Truth Computation
Uses ARPACK (`eigsh`) to find:
- Target eigenvalue (closest to σ)
- Reference eigenvector for validation

#### Phase 2: PPO Configuration
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    gamma=0.99,
    policy_kwargs=dict(net_arch=[256, 256, 128])
)
```
- **Network Architecture**: 3-layer MLP (256 → 256 → 128 neurons)
- **Discount Factor**: γ=0.99 prioritizes long-term convergence
- **Learning Rate**: 1e-4 balances stability and speed

#### Phase 3: Hybrid Training
1. RL agent interacts with environment for 50,000 timesteps
2. Combines policy updates with power iteration stabilization

#### Phase 4: Evaluation Metrics
- **Residual Norm**: $||Av - λv||$
- **Cosine Distance**:  
  ```math
  \text{cos\_dist} = \frac{2}{\pi} \arccos(|\text{cos\_sim}|)
  ```
  Measures angular deviation from true eigenvector (0=perfect alignment, 1=orthogonal).

---

## RL Algorithm Classification
**PPO (Proximal Policy Optimization)** is used here:
```
RL Algorithms
└── Model-Free RL
    └── Policy Optimization
        └── PPO
```

**Why PPO?**
- Handles continuous action spaces (required for vector adjustments)
- Avoids destructive policy updates via clipped objective
- Sample-efficient for medium-dimensional problems (≈1,000D actions here)


---

## Performance Considerations
### Strengths
- **Sparse Efficiency**: Handles 1000x1000 matrices via LU decomposition
- **Hybrid Convergence**: RL reduces iterations needed after warm-start
- **Adaptability**: Learns matrix-specific adjustment strategies

### Limitations
- **Training Cost**: ~50k timesteps required per matrix
- **Eigenvalue Clustering**: Struggles when σ is near multiple eigenvalues
- **Symmetric Assumption**: Requires $A = A^\in

## Extension Opportunities
1. **Multi-Eigenvalue Support**: Modify environment to track multiple eigenvectors
2. **Adaptive σ**: Let RL agent adjust shift during training
3. **Distributed Training**: Scale to 10k+ matrices via MPI
4. **Alternative RL Algorithms**: Test SAC/TD3 for comparison

This hybrid paradigm demonstrates how traditional numerics and modern RL can synergize for challenging linear algebra problems.
```