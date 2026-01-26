import numpy as np
from gymnasium.wrappers import TimeLimit
from my_cartpole_env import CartPoleEnv
from collections import deque

from hyperparameters import *

# STATE SPACE
x_vals = np.linspace(-X_LIMIT, X_LIMIT, N_BINS)
x_dot_vals = np.linspace(-X_VEL_LIMIT, X_VEL_LIMIT, N_BINS)
theta_vals = np.linspace(-THETA_LIMIT, THETA_LIMIT, N_BINS)
theta_dot_vals = np.linspace(-THETA_VEL_LIMIT, THETA_VEL_LIMIT, N_BINS)
mesh = np.meshgrid(x_vals, x_dot_vals, theta_vals, theta_dot_vals, indexing='ij')
state_tensor = np.array(mesh) # Shape: (4, N, N, N, N)

# EMPTY ARRAYS
ACTION_VALS = np.array([0.0, 1.0]) 
V = np.zeros((N_BINS, N_BINS, N_BINS, N_BINS))
Q = np.zeros((N_BINS, N_BINS, N_BINS, N_BINS, 2))

# Pre-calculate which states in the grid are already terminal for efficient computation
is_terminal = (state_tensor[2] <= theta_vals[0]) | (state_tensor[2] >= theta_vals[-1]) | \
              (state_tensor[0] <= x_vals[0]) | (state_tensor[0] >= x_vals[-1])


def dynamics(state_tensor, action_val, continuous_action=False):
    # PHYSICS PARAMETERS
    GRAVITY = 9.8
    MASSCART = 1.0
    MASSPOLE = 0.1
    TOTAL_MASS = MASSPOLE + MASSCART
    LENGTH = 0.5
    TAU = 0.02
    FORCE_MAG = 10.0

    x, x_dot, theta, theta_dot = state_tensor
    if continuous_action:
        force = action_val[0]
    else:
        force = FORCE_MAG if action_val == 1 else -FORCE_MAG

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    temp = (force + MASSPOLE * LENGTH * theta_dot**2 * sin_t) / TOTAL_MASS
    thetaacc = (GRAVITY * sin_t - cos_t * temp) / (LENGTH * (4.0/3.0 - MASSPOLE * cos_t**2 / TOTAL_MASS))
    xacc = temp - MASSPOLE * LENGTH * thetaacc * cos_t / TOTAL_MASS

    # Euler Integration
    return np.array([
        x + TAU * x_dot,
        np.clip(x_dot + TAU * xacc, -X_VEL_LIMIT, X_VEL_LIMIT),
        theta + TAU * theta_dot,
        np.clip(theta_dot + TAU * thetaacc, -THETA_VEL_LIMIT, THETA_VEL_LIMIT)
    ])


# Indexing function 
def state_to_indices(state, grids):
    indices = []
    for i in range(4):
        # Using clip to prevent out-of-bounds errors
        idx = np.round((state[i] - grids[i][0]) / (grids[i][1] - grids[i][0])).astype(int)
        indices.append(np.clip(idx, 0, len(grids[i]) - 1))
    return tuple(indices)

# Reward function
def reward_function(state, terminated):
    if terminated:
        return TERMINAL_COST
    return quadratic_cost(state)

# Quadratic cost function
def quadratic_cost(state):
    ### FILL IN HERE ### hint: CART_COST_WEIGHT, POLE_ANGLE_COST_WEIGHT
    raise NotImplementedError("Quadratic cost function not implemented")

def evaluate_agent(env, type, policy=None, Q=None):
    if type == "Q":
        assert Q is not None
    elif type == "DP":
        assert policy is not None
    elif type == "MPC":
        assert policy is not None
    else:
        raise ValueError(f"Invalid type: {type}")

    state, _ = env.reset()
    episode_reward = 0

    returns = deque(maxlen=NUM_EPISODES_EVAL)

    for i in range(NUM_EPISODES_EVAL):
        state, _ = env.reset()
        # Reset MPC warm start buffer for new episode
        if type == "MPC":
            policy.reset()
        episode_reward = 0

        while True:
            indices = state_to_indices(state, [x_vals, x_dot_vals, theta_vals, theta_dot_vals])
            
            if type == "Q":
                action = np.argmax(Q[indices])
            elif type == "DP":
                action = policy[indices]
            elif type == "MPC":
                action = policy.control(state)
            state, _, terminated, truncated, _ = env.step(action)
            reward = reward_function(state, terminated) * (-1 if type == "Q" else 1)
            episode_reward += reward

            if terminated or truncated:
                returns.append(episode_reward)
                print(f"Episode {i} Mean Return: {np.mean(returns)}", end="\r")
                break

    env.close()
    return returns

    
def play_agent(env, type, Q=None, policy=None):
    if type == "Q":
        assert Q is not None
    elif type == "DP":
        assert policy is not None
    elif type == "MPC":
        assert policy is not None
    else:
        raise ValueError(f"Invalid type: {type}")

    n_episodes = 100
    for episode in range(n_episodes):
        state, _ = env.reset()
        # Reset MPC warm start buffer for new episode
        if type == "MPC":
            policy.reset()
        done = False
        total_cost = 0
        t = 0
        
        print(f"--- Starting Episode {episode+1} ---")
        
        while not done:
            indices = state_to_indices(state, [x_vals, x_dot_vals, theta_vals, theta_dot_vals])
            
            if type == "Q":
                action = np.argmax(Q[indices])
            elif type == "DP":
                action = policy[indices]
            elif type == "MPC":
                action = policy.control(state)

            state, _, terminated, truncated, info = env.step(action)
            cost = reward_function(state, terminated) * (-1 if type == "Q" else 1)

            done = terminated or truncated
            total_cost += cost
            t += 1
            
            if done:
                reason = "Terminated (Fell/Out of Bounds)" if terminated else "Truncated (Time Limit)"
                print(f"Finished at t={t} | Reason: {reason} | Episode cost: {total_cost}")

    env.close()


def print_statistics(returns):
    print("************ RETURN STATISTICS ************")
    print(f"Number of episodes: {len(returns)}")
    print(f"Average return: {np.mean(returns)}")
    print(f"Standard deviation: {np.std(returns)}")
    print(f"Min return: {np.min(returns)}")
    print(f"Max return: {np.max(returns)}")
    print("*******************************************")