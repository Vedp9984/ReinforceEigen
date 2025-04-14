# ReinforceEigen
## Not really RL Algorithm Dedeepya396
***Objective***: For any $A \in \mathbb{R}^{n \times n}$ Find eigenvector for given eignvalue $\lambda$  
***State***: The currnt guess $v$ for eigenvector  
***Action*** Gadient descent:
$v_{n+1} = v_n - \mu (A-\lambda 𝟙)^\intercal (A-\lambda 𝟙)v_n$  
***Reward***: $r=-||Ax-\lambda x||^2$  
***Hyperparametres***:  Learning Rate $\mu$

## Not really RL Algorithm Vedp9984
***Objective***: For any $A \in \mathbb{R}^{n \times n}$ Find eigenvector for given eignvalue $\lambda$  
***State***: The currnt guess $v$ for eigenvector  
***Action*** Gadient descent:
$v_{n+1} = v_n - 2\mu \frac{Av-rv}{v^\intercal v}$  
***Reward***: $r=\frac{v^\intercal A v}{v^\intercal v}$  
***Hyperparametres***:  Learning Rate $\mu$



