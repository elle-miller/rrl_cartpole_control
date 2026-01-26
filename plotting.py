import numpy as np
import matplotlib.pyplot as plt


def plot_policy_value(policy, V, N_bins, x_vals, theta_vals):
    mid = N_bins // 2
    v_slice = V[:, mid, :, mid].T
    p_slice = policy[:, mid, :, mid].T
    dx = x_vals[1] - x_vals[0]
    dt = np.rad2deg(theta_vals[1] - theta_vals[0])

    extent = [
        x_vals[0] - dx/2, x_vals[-1] + dx/2, 
        np.rad2deg(theta_vals[0]) - dt/2, np.rad2deg(theta_vals[-1]) + dt/2
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(v_slice, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    im2 = ax2.imshow(p_slice, origin='lower', extent=extent, aspect='auto', cmap='managua')
    fig.colorbar(im1, ax=ax1,)
    fig.colorbar(im2, ax=ax2, ticks=[0, 1])
    ax1.set_title(r"Value Function $V(x, \dot{x}=0, \theta, \dot{\theta}=0)$")
    ax2.set_title(r"Policy $\pi(s)$")    

    x_edges = np.linspace(x_vals[0] - dx/2, x_vals[-1] + dx/2, N_bins + 1)
    t_edges = np.linspace(np.rad2deg(theta_vals[0]) - dt/2, np.rad2deg(theta_vals[-1]) + dt/2, N_bins + 1)

    for ax in [ax1, ax2]:
        ax.set_xticks(x_edges, minor=True)
        ax.set_yticks(t_edges, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.6)
        ax.axhline(0, color='white', linewidth=1.5, alpha=0.7) # Horizontal zero (Theta=0)
        ax.axvline(0, color='white', linewidth=1.5, alpha=0.7) # Vertical zero (X=0)
        ax.set_axisbelow(False)
        ax.set_xlabel('Cart Position (m)')
        ax.set_ylabel('Pole Angle (deg)')

    plt.tight_layout()
    plt.show()

def plot_returns(returns, timesteps):
    plt.plot(timesteps, returns)
    plt.xlabel('Timesteps (M)')
    plt.ylabel('Return')
    plt.title('Returns')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('returns.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
