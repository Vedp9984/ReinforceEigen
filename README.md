# ReinforceEigen
## RL Algorithm
***Objective***: For any $ A \in \mathbb{R}^{n \times n} $ Find eigenvector for given eignvalue $\lambda$  
***State***: The currnt guess $v$ for eigenvector  
***Action*** Gadient descent:
$v_{n+1} = v_n - \mu (A-\lambda 𝟙)^\intercal (A-\lambda 𝟙)v_n$  
***Reward***: $R=-||Ax-\lambda x||^2$
