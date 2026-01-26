import numpy as np
from gymnasium.wrappers import TimeLimit
from my_cartpole_env import CartPoleEnv
from plotting import plot_policy_value
from common import *
from hyperparameters import *


def value_iteration():
    global V

    # Pre-calculate the current state costs for efficiency
    current_state_costs = quadratic_cost(state_tensor)

    for it in range(N_ITERATIONS):
        V_old = V.copy()
        V_new = []

        for action in ACTION_VALS:
            next_states = dynamics(state_tensor, action)
            next_indices = state_to_indices(next_states, [x_vals, x_dot_vals, theta_vals, theta_dot_vals])

            ### FILL IN HERE ###
            raise NotImplementedError("Bellman update not implemented")
            
            # Handle termination
            new_estimate = np.where(is_terminal[next_indices], TERMINAL_COST, future_val)

            V_new.append(new_estimate)

        ### FILL IN HERE ### 
        raise NotImplementedError("Value update not implemented")

        ### FILL IN HERE ###
        raise NotImplementedError("Check convergence not implemented")
        print(f"[{it}] diff = {diff:.5f}", "value stats: ", np.min(V), np.max(V), np.mean(V), end='\r')
        if diff < DELTA: print(f"VI completed in {it} iterations with diff {diff:.5f}"); break
    return V

def compute_policy(V):
    ### FILL IN HERE ### hint: Compute policy
    raise NotImplementedError("Compute policy not implemented")
    return policy


if __name__ == "__main__":

    # value iteration
    V = value_iteration()
    policy = compute_policy(V)
    plot_policy_value(policy, V, N_BINS, x_vals, theta_vals)

    # evaluate agent
    env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    returns = evaluate_agent(env, type="DP", policy=policy)
    print_statistics(returns)
    env.close()

    # play agent
    play_env = CartPoleEnv(render_mode="human", x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT)
    play_env = TimeLimit(play_env, max_episode_steps=MAX_EPISODE_STEPS)
    play_agent(play_env, type="DP", policy=policy)
    play_env.close()


