import numpy as np
from gymnasium.wrappers import TimeLimit
from my_cartpole_env import CartPoleEnv
from plotting import plot_returns
from collections import deque
from hyperparameters import *
from common import *


def run_q_learning(env,train_timesteps_M):
    
    returns = deque(maxlen=1000)
    returns.append(0)
    saved_returns = []
    saved_timesteps = []
    episode = 0
    state, _ = env.reset()
    episode_reward = 0

    # Learning rate and exploration rate schedules flags
    alpha_schedule = False
    epsilon_schedule = False

    for timestep in range(int(train_timesteps_M * 1e6)):
        
        if epsilon_schedule:
            ### FILL IN HERE ###
            raise NotImplementedError("Epsilon schedule not implemented")
        else:
            current_epsilon = EPSILON

        if alpha_schedule:
            ### FILL IN HERE ###
            raise NotImplementedError("Alpha schedule not implemented")
        else:
            current_alpha = ALPHA

        indices = state_to_indices(state, [x_vals, x_dot_vals, theta_vals, theta_dot_vals])
        
        ### FILL IN HERE ###
        raise NotImplementedError("Action selection not implemented")

        next_state, _, terminated, truncated, _ = env.step(int(action))
        next_indices = state_to_indices(next_state, [x_vals, x_dot_vals, theta_vals, theta_dot_vals])
        reward = reward_function(next_state, terminated) * -1
        episode_reward += reward

        # Q-value update
        best_next_q = np.max(Q[next_indices])
        ### FILL IN HERE ###
        raise NotImplementedError("Q-value update not implemented")

        state = next_state
        if terminated or truncated:
            returns.append(episode_reward)
            episode += 1
            state, _ = env.reset()
            episode_reward = 0
      
        if timestep % 20000 == 0 and timestep > 0:
            print(f"Timestep {timestep/1e6:.2f}M | Return: {np.mean(returns):.4f} | Eps: {current_epsilon:.5f} | Alpha: {current_alpha:.5f}")
            saved_returns.append(np.mean(returns))
            saved_timesteps.append(timestep/1e6)
            plot_returns(saved_returns, saved_timesteps)

    return Q


if __name__ == "__main__":

    # train agent
    train_env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT)
    train_env = TimeLimit(train_env, max_episode_steps=MAX_EPISODE_STEPS)
    Q = run_q_learning(train_env, train_timesteps_M=TRAIN_TIMESTEPS_M)

    # evaluate agent
    returns = evaluate_agent(train_env, type="Q", Q=Q)
    print_statistics(returns)
    train_env.close()

    # play agent
    play_env = CartPoleEnv(render_mode="human", x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT)
    play_env = TimeLimit(play_env, max_episode_steps=MAX_EPISODE_STEPS)
    play_agent(play_env, type="Q", Q=Q)
    play_env.close()

