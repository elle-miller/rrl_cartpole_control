from matplotlib.projections import HammerAxes
import numpy as np
from my_cartpole_env import CartPoleEnv
from gymnasium.wrappers import TimeLimit
from hyperparameters import *
from common import *
from plotting import plot_returns


class CartpoleMPC:
    def __init__(self, H=10, max_iters=5):

        # MPC/iLQR Parameters
        self.H = H          # Horizon length
        self.max_iters = max_iters    # iLQR iterations per time step
        ### FILL IN HERE ### hint: Q, R from provided cost function parameters
        # self.Q = np.diag([...])
        # self.R = np.array([[...]])
        raise NotImplementedError("MPC/iLQR Parameters not implemented")

        # Warm start buffer
        self.U_guess = np.zeros((self.H, 1))

    def get_jacobians(self, x, u):
        """Numerical Jacobians (Finite Difference) for A and B"""
        eps = 1e-6
        nx = len(x)
        nu = len(u)
        
        A = np.zeros((nx, nx))
        B = np.zeros((nx, nu))
        
        for i in range(nx):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            A[:, i] = (dynamics(x_plus, u, continuous_action=True) - dynamics(x_minus, u, continuous_action=True)) / (2 * eps)
            
        for i in range(nu):
            u_plus = u.copy()
            u_plus[i] += eps
            u_minus = u.copy()
            u_minus[i] -= eps
            B[:, i] = (dynamics(x, u_plus, continuous_action=True) - dynamics(x, u_minus, continuous_action=True)) / (2 * eps)
            
        return A, B

    def solve_ilqr(self, x0, U_init):
        """The iLQR Solver"""
        U = U_init.copy()
        X = np.zeros((self.H + 1, 4))
        X[0] = x0
        
        # Initial Rollout
        for t in range(self.H):
            X[t+1] = dynamics(X[t], U[t], continuous_action=True)
            
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

                ### FILL IN HERE ### hint: Q-function derivatives, control gains, value function update
                raise NotImplementedError("iLQR not implemented")
                
            # Forward Pass (Line search simplified for brevity)
            X_new = np.zeros_like(X)
            X_new[0] = x0
            U_new = np.zeros_like(U)
            
            for t in range(self.H):
                ### FILL IN HERE ### hint: compute U_new[t] and X_new[t+1]
                raise NotImplementedError("iLQR not implemented")
            
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


if __name__ == "__main__":

    # ----------------------
    USE_MPC = True  # Set to False for Open-Loop iLQR
    DISTURBANCE = 0
    H_MPC = 5
    H_ILQR = 100
    # ----------------------

    env = CartPoleEnv(x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT, continuous_action=True, disturbance=DISTURBANCE)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    play_env = CartPoleEnv(render_mode="human", x_threshold=X_LIMIT, theta_threshold_radians=THETA_LIMIT, continuous_action=True, disturbance=DISTURBANCE)
    play_env = TimeLimit(play_env, max_episode_steps=MAX_EPISODE_STEPS)

    if USE_MPC:
        print("Running in Closed-Loop MPC mode...")

        mpc = CartpoleMPC(H_MPC, max_iters=5)
        returns = evaluate_agent(env, type="MPC", policy=mpc)
        print_statistics(returns)
        play_agent(play_env, type="MPC", policy=mpc)
    else:
        print("Running in Open-Loop iLQR mode...")

        mpc = CartpoleMPC(H_ILQR, max_iters=5)
        state, _ = play_env.reset()
            
        while True:
            # Solve for the entire horizon
            u_plan, x_plan, ks, Ks = mpc.solve_ilqr(state, np.zeros((H_ILQR, 1)))
            
            for t in range(H_ILQR):
                action = u_plan[t] + ks[t] + Ks[t] @ (state - x_plan[t]) 
                state, _, terminated, truncated, info = play_env.step(action[0])
            
                if terminated or truncated:
                    state, _ = play_env.reset()
                    reason = "Terminated (Fell/Out of Bounds)" if terminated else "Truncated (Time Limit)"
                    print(f"Finished at t={t} | Reason: {reason}")
                    break
    
