import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# === Your Functions ===

def modal_shape(L, x, n):
    return np.sin(n * np.pi * x / L)

def point_load(L, x, x_in):
    a = x_in
    b = x_in + L / 1000
    pload = np.where((x >= a) & (x <= b), 1, 0)
    return pload

def modal_excitation(L, n, p_x, x_grid):
    phi_n = modal_shape(L, x_grid, n)
    return np.trapz(p_x * phi_n, x_grid)

def modal_response(F_n, mu_n, omega_n, xi_n, omega):
    denominator = mu_n * (omega_n**2 - omega**2 + 2j * xi_n * omega_n * omega)
    z_n = F_n / denominator
    return z_n

def physical_response(z_n, L, x_out, n):
    return z_n * modal_shape(L, x_out, n)

def natural_pulsation(E, rho, L, I, A, n):
    return (n * np.pi / L)**2 * np.sqrt(E * I / (rho * A))

# === Parameters ===
E = 70e9
rho = 2700
L = 1.0
b = 1e-2
h = 1e-3
I = b * h**3 / 12
A = b * h

omega = np.linspace(1, 1000, 1000) * 2 * np.pi  # rad/s
x_out = L / 2
x_in = L / 4
xi = 0.00  # damping
x_grid = np.linspace(0, L, 1000)
p_x = point_load(L, x_grid, x_in)


# load u_exact from file u_exact.npy 
u_exact = np.load('iplot/u_exact.npy')
u_approx_all = np.load('iplot/u_approx_all.npy')  # Add this line at the top


# === Plot Setup ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(bottom=0.25, hspace=0.4)

# Main frequency response plot
line_exact, = ax1.plot(omega / (2 * np.pi), np.abs(u_exact), color='black', label='Exact', lw=2)
line_approx, = ax1.plot([], [], color='blue', linestyle='--', label='Approx (N modes)')

# Vertical lines at natural frequencies
for n in range(1, 20):
    freq_n = natural_pulsation(E, rho, L, I, A, n) / (2 * np.pi)
    ax1.axvline(freq_n, color='red', linestyle='--', lw=0.5)

ax1.set_ylabel('|u(x_out)|')
ax1.set_yscale('log')
ax1.grid(which='both', linestyle=':')
ax1.legend()

# Error plot
error_line, = ax2.plot([], [], color='purple', label='Absolute Error')
error_threshold = 1e-8
ax2.axhline(error_threshold, color='gray', linestyle='--', label='Error Threshold')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Error')
ax2.set_yscale('log')
ax2.grid(which='both', linestyle=':')
ax2.set_ylim([1e-11, 1e-2])
ax2.legend()

# === Slider ===
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Number of modes', 1, 50, valinit=6, valstep=1)

# Track minimum modes required
minimum_modes_required = []

def update(val):
    N = int(slider.val)
    u_approx = u_approx_all[N - 1]
    abs_error = np.abs(u_exact - u_approx)

    # Update frequency response and error plot
    line_approx.set_data(omega / (2 * np.pi), np.abs(u_approx))
    line_approx.set_label(f'Approx ({N} modes)')
    error_line.set_data(omega / (2 * np.pi), abs_error)

    # Track minimum modes required to stay under threshold
    if np.max(abs_error) < error_threshold and N not in minimum_modes_required:
        minimum_modes_required.append(N)

    ax1.legend()
    ax2.relim()
    ax2.autoscale_view()
    fig.canvas.draw_idle()


# Initialize plot
update(6)
slider.on_changed(update)

plt.show()

