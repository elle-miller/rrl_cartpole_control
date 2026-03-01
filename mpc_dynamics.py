
import numpy as np
from my_cartpole_env import CartPoleEnv
from gymnasium.wrappers import TimeLimit
from hyperparameters import *
from common import *
from plotting import plot_loss_and_returns
import torch
import torch.nn as nn


class DynamicsMPC:
    def __init__(self, dynamics_model, H=10, max_iters=5):

        # MPC/iLQR Parameters
        self.H = H          # Horizon length
        self.max_iters = max_iters    # iLQR iterations per time step
        self.Q = np.diag([1.0, 0.1, 10.0, 0.1]) # State cost (x, x_dot, theta, theta_dot)
        self.R = np.array([[0.01]])             # Control cost
        
        # Warm start buffer
        self.U_guess = np.zeros((self.H, 1))

        # Learned dynamics model (nn.Module)
        self.dynamics_model = dynamics_model

    def _predict(self, x, u):
        """Run dynamics NN: numpy in -> numpy out."""
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            u_t = torch.tensor(u, dtype=torch.float32).reshape(1, -1)
            next_x = self.dynamics_model(x_t, u_t)
            return next_x.squeeze(0).numpy()

    def get_jacobians(self, x, u):
        """Finite-difference Jacobians, batched for speed."""
        eps = 1e-6
        nx, nu = 4, 1

        with torch.no_grad():
            # Batch A: 2*nx forward passes in one batch
            x_batch = np.tile(x, (2 * nx, 1))
            for i in range(nx):
                x_batch[2 * i, i] += eps
                x_batch[2 * i + 1, i] -= eps
            x_t = torch.tensor(x_batch, dtype=torch.float32)
            u_t = torch.tensor(u, dtype=torch.float32).reshape(1, -1).expand(2 * nx, -1)
            out = self.dynamics_model(x_t, u_t).numpy()
            A = ((out[::2] - out[1::2]) / (2 * eps)).T

            # Batch B: 2*nu forward passes in one batch
            u_batch = np.tile(u, (2 * nu, 1))
            for i in range(nu):
                u_batch[2 * i, i] += eps
                u_batch[2 * i + 1, i] -= eps
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).expand(2 * nu, -1)
            u_t = torch.tensor(u_batch, dtype=torch.float32)
            out = self.dynamics_model(x_t, u_t).numpy()
            B = ((out[::2] - out[1::2]) / (2 * eps)).T

        return A, B

    def solve_ilqr(self, x0, U_init):
        """The iLQR Solver"""
        self.dynamics_model.eval()
        U = U_init.copy()
        X = np.zeros((self.H + 1, 4))
        X[0] = x0
        
        # Initial Rollout
        for t in range(self.H):
            X[t+1] = self._predict(X[t], U[t])
            
        for _ in range(self.max_iters):
            # Backward Pass
            ks = [np.zeros((1, 1))] * self.H
            Ks = [np.zeros((1, 4))] * self.H
            
            # Terminal Value Function derivatives
            Vx = self.Q @ X[-1]
            Vxx = self.Q
            
            for t in reversed(range(self.H)):
                A, B = self.get_jacobians(X[t], U[t])
                
                # Gradients of the cost
                lx = self.Q @ X[t]
                lu = self.R @ U[t]
                
                # Q-function derivatives
                Qx = lx + A.T @ Vx
                Qu = lu + B.T @ Vx
                Qxx = self.Q + A.T @ Vxx @ A
                Quu = self.R + B.T @ Vxx @ B
                Qux = B.T @ Vxx @ A
                
                # Control gains
                k = -np.linalg.inv(Quu) @ Qu
                K = -np.linalg.inv(Quu) @ Qux
                
                ks[t] = k
                Ks[t] = K
                
                # Update Value Function for next step
                Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
                Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            
            # Forward Pass (Line search simplified for brevity)
            X_new = np.zeros_like(X)
            X_new[0] = x0
            U_new = np.zeros_like(U)
            
            for t in range(self.H):
                U_new[t] = U[t] + ks[t] + Ks[t] @ (X_new[t] - X[t])
                X_new[t+1] = self._predict(X_new[t], U_new[t])
            
            X, U = X_new, U_new
            
        ks = np.array(ks)
        Ks = np.array(Ks)
        return U, X, ks, Ks

    def reset(self):
        """Reset the warm start buffer for a new episode"""
        self.U_guess = np.zeros((self.H, 1))
    
    def control(self, state):
        """MPC interface: solve and shift"""
        U_opt, _, _, _ = self.solve_ilqr(state, self.U_guess)
        # Extract first action and ensure it's a scalar
        action = U_opt[0, 0] if U_opt.ndim == 2 else U_opt[0]
        
        # Clip action to valid range [-1, 1]
        action = float(np.clip(action, -1.0, 1.0))
        
        # Warm start shift
        self.U_guess[:-1] = U_opt[1:]
        self.U_guess[-1] = 0
        
        return action

class DynamicsModel(nn.Module):
    """Learned dynamics with optional input normalization for better training."""
    # CartPole state/action scales for normalization (from hyperparameters)
    STATE_SCALE = np.array([X_LIMIT, X_VEL_LIMIT, THETA_LIMIT, THETA_VEL_LIMIT], dtype=np.float32)
    ACTION_SCALE = np.array([1.0], dtype=np.float32)

    def __init__(self, normalize_inputs=True):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.model = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def _normalize(self, states, actions):
        if not self.normalize_inputs:
            return states, actions
        s_scale = torch.tensor(self.STATE_SCALE, device=states.device)
        a_scale = torch.tensor(self.ACTION_SCALE, device=actions.device)
        return states / s_scale, actions / a_scale

    def forward(self, states, actions):
        s_norm, a_norm = self._normalize(states, actions)
        x = torch.cat([s_norm, a_norm], dim=1)
        x_delta = self.model(x)
        return states + x_delta


def collect_data(env, agent, num_samples, random_policy=False, exploration_noise=0.0):
    print("Collecting data for dynamics model...")
    data_states = []
    data_actions = []
    data_next_states = []

    state, _ = env.reset()
    while len(data_states) < num_samples:
        remaining = num_samples - len(data_states)
        print(f"\rCollecting data for dynamics model... {remaining} samples to go", end="", flush=True)
        if random_policy:
            action = np.random.uniform(low=-1, high=1, size=(1,))[0]
        else:
            action = agent.control(state)
            if exploration_noise > 0:
                action = np.clip(action + np.random.normal(0, exploration_noise), -1, 1)
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()
            continue
        data_states.append(state.copy())
        data_actions.append(np.array([action]))
        data_next_states.append(next_state.copy())
        state = next_state

    print()  # newline after progress
    data_states = np.array(data_states)
    data_actions = np.array(data_actions)
    data_next_states = np.array(data_next_states)
    states = torch.tensor(data_states, dtype=torch.float32)
    actions = torch.tensor(data_actions, dtype=torch.float32)
    next_states = torch.tensor(data_next_states, dtype=torch.float32)
    return states, actions, next_states

if __name__ == "__main__":

    env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT, continuous_action=True, disturbance=0)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    play_env = CartPoleEnv(render_mode="human", x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT, continuous_action=True, disturbance=0)
    play_env = TimeLimit(play_env, max_episode_steps=MAX_EPISODE_STEPS)

    dynamics_model = DynamicsModel(normalize_inputs=True)
    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=0.001)

    NUM_ROLLOUTS = 15
    INIT_SAMPLES = 100000   # Random policy for broad coverage
    ROLLOUT_SAMPLES = 1000  # Per rollout with MPC (more than before)
    REPLAY_CAP = 100000    # Max transitions in replay buffer
    NUM_EPOCHS = 20        # More training as dataset grows
    BATCH_SIZE = 512
    H_MPC_LEARNED = 10     # Shorter horizon reduces compounding error
    EXPLORATION_NOISE = 0.1  # When using MPC, add noise for coverage

    agent = DynamicsMPC(dynamics_model, H=H_MPC_LEARNED, max_iters=5)

    # Replay buffer (dataset aggregation)
    replay_states = []
    replay_actions = []
    replay_next_states = []

    all_losses = []
    all_returns = []

    for i in range(NUM_ROLLOUTS):
        # Collect data: random for first rollout, MPC with exploration thereafter
        if i == 0:
            random_policy = True
            num_collect = INIT_SAMPLES

            exploration = 0.0
        else:
            random_policy = False
            num_collect = ROLLOUT_SAMPLES
            exploration = EXPLORATION_NOISE

        states, actions, next_states = collect_data(env, agent, num_collect, random_policy, exploration)

        # Add to replay buffer (FIFO if over cap)
        replay_states.append(states.numpy())
        replay_actions.append(actions.numpy())
        replay_next_states.append(next_states.numpy())
        all_s = np.concatenate(replay_states)
        all_a = np.concatenate(replay_actions)
        all_n = np.concatenate(replay_next_states)
        if len(all_s) > REPLAY_CAP:
            idx = np.random.choice(len(all_s), REPLAY_CAP, replace=False)
            all_s, all_a, all_n = all_s[idx], all_a[idx], all_n[idx]
            replay_states = [all_s]
            replay_actions = [all_a]
            replay_next_states = [all_n]
        else:
            replay_states = [all_s]
            replay_actions = [all_a]
            replay_next_states = [all_n]

        dataset_size = len(all_s)
        num_batches = max(1, dataset_size // BATCH_SIZE)

        # Train on full replay buffer
        cumulative_loss = 0.0
        for epoch in range(NUM_EPOCHS):
            idx = np.random.permutation(dataset_size)
            states_shuffled = torch.tensor(all_s[idx], dtype=torch.float32)
            actions_shuffled = torch.tensor(all_a[idx], dtype=torch.float32)
            next_states_shuffled = torch.tensor(all_n[idx], dtype=torch.float32)

            epoch_loss = 0.0
            for batch in range(num_batches):
                start = batch * BATCH_SIZE
                end = min(start + BATCH_SIZE, dataset_size)
                batch_states = states_shuffled[start:end]
                batch_actions = actions_shuffled[start:end]
                batch_next_states = next_states_shuffled[start:end]

                optimizer.zero_grad()
                predicted_next_states = dynamics_model(batch_states, batch_actions)
                loss = nn.MSELoss()(predicted_next_states, batch_next_states)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            cumulative_loss += epoch_loss / num_batches
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Rollout {i+1}, Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss / num_batches:.9f} (dataset size: {dataset_size})")

        avg_loss = cumulative_loss / NUM_EPOCHS
        all_losses.append(avg_loss)
        returns = evaluate_agent(env, type="MPC", policy=agent)
        avg_return = np.mean(returns)
        all_returns.append(avg_return)
        print(f"Rollout {i+1}: Avg loss: {avg_loss:.6f}, Avg return: {avg_return:.2f}")

        plot_loss_and_returns(all_losses, all_returns)

    play_agent(play_env, type="MPC", policy=agent)


