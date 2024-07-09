# Notes during the experiment
- Install [matplotlib 3.3.3](https://github.com/openai/spinningup/issues/332) to plot 
- [Walkerv2 env](https://mgoulao.github.io/gym-docs/environments/mujoco/walker2d/)
- act_dim: 6 (continuous action space)
- obs_dim: 17 (continuous observation space)


## Replay Buffer notes
- The replay buffer is **not a dynamic Python list**. 
- `obs_buff` is state buffer. The size is (#time_steps x #states)
- `act_buff` is action buffer. The size is (#time_steps x #actions)
- `val` is the estimation V (value function) using the critic. The size is (#time_steps)
- `logp` is the log pi (a|s). The size is (#time_steps)
- `adv` is the advantage function. The size is (#time_steps)
- `ret` is the rewards to go. The size is (#time_steps)
- `gamma` is discounting factor, while `lam` is the lamda in GAE
- `adv_buf` is [r0 + gamma * vhat1 - vhat0 + gamma * lam * (r1 + gamma * vhat2 - vhat1) + ... , r1 + gamma * vhat2 - vhat1 + ..., ...]
- `ret_buf` is [r0 + gamma r1 + gamma^2 r2 + ..., r1 + gamma r2 + ..., ...]

## General Method notes
- The policy loss function is techniquely not a loss! 
- The policy loss function equals to (log pi(.|s) @ adv(.,s) ).mean(). The adv is there due to GAE.
- The critic loss function equals to (NN(obs) - return(obs))^2
- The update of actor and critic is per trajectory.

## Vanilla PG notes
- The current implementation supports parallelization.
- The actor is update once per trajectory, while the critic is update multiple times per trajectory.
- On-policy algorithm.
- Support both discrete and continuous action space.
- Actor:
  - Continuous action space:
    - `log_std`: [-0.5, ..., -0.5] with the shape is (#actions)
    - `mu`: Neural network with input as (#time_steps x #states) and output as (#time_steps x #actions)
    - Both `log_std` and `mu` are parameters to be optimized.
    - `_distribution(obs)`: Normal(mu(obs), exp(log_std))
    - `_log_prob_from_distribution(pi, act)`: _distribution.log_prob(Normal.sample())
  - Discrete action space:
    - Similar to above, but only using `mu` instead of `log_std` 
    - Categorical(Neural network(obs))
- Critic: Neural network with input as (#time_steps x #states) and output as (#time_steps x 1)

## Soft Actor-critic notes

## PPO notes

## Configuration notes
- Epoch is trajectorie or episode or rollout.