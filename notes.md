# Notes during the experiment
- Install [matplotlib 3.3.3](https://github.com/openai/spinningup/issues/332) to plot 
- [Walkerv2 env](https://mgoulao.github.io/gym-docs/environments/mujoco/walker2d/)
- act_dim: 6 (continuous action space)
- obs_dim: 17 (continuous observation space)


## Replay Buffer notes
- The replay buffer is not a regular dynamic Python list. 
- `obs_buff` is state buffer. The size is (#time_steps x #states)
- val is the estimation V (value function) using the critic. The size is (#time_steps)
- logp is the log pi (a|s) 
- adv is the advantage function
- ret is the rewards to go
- gamma is discounting factor, while lam is the lamda in GAE
- adv_buf is [r0 + gamma * vhat1 - vhat0 + gamma * lam * (r1 + gamma * vhat2 - vhat1) + ... , r1 + gamma * vhat2 - vhat1, ]
- ret_buf is [r0 + gamma r1 + gamma^2 r2 + ..., r1 + gamma r2 + ..., ...]

## General Method notes
- The policy loss function is techniquely not a loss! 
- The policy loss function equals (log pi(.|s) @ adv(.,s) ).mean(). The adv is there due to GAE.
- The update is per trajectory
- 

## Vanilla PG notes
- Only update the actor once per trajectory

## Configuration notes
- Epoch is # trajectories or episodes or rollouts