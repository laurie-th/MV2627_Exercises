import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider

# === Parameters ===
E = 70e9
rho = 2700
L = 1.0
b = 1e-2
h = 1e-3
I = b * h**3 / 12
A = b * h
N_modes = 6
xi = 0.00
x_grid = np.linspace(0, L, 500)
measurement_points = [0.25 * L, 0.5 * L, 0.75 * L]
colors = ['blue', 'orange', 'green']

# Frequency and Time Setup
N_freqs = 1024
omega = np.linspace(0, 100 * 2 * np.pi, N_freqs)
freq = omega / (2 * np.pi)
dt = 1 / (2 * freq[-1])
t_irfft = np.linspace(0, 2 * N_freqs * dt, N_freqs)

# === Helper Functions ===
def modal_shape(L, x, n):
    return np.sin(n * np.pi * x / L)

def dynamic_load(x, t):
    return np.maximum(np.arctan(2 * x - 2 + t), 0)

def modal_excitation(p_x, phi_n, x):
    return np.trapz(p_x * phi_n, x)

def compute_modal_mass(rho, A, L, n):
    return rho * A * L / 2

def natural_pulsation(E, rho, L, I, A, n):
    return (n * np.pi / L)**2 * np.sqrt(E * I / (rho * A))

def modal_response(Fn, mu_n, omega_n, xi_n, omega):
    denom = mu_n * (omega_n**2 - omega**2 + 2j * xi_n * omega_n * omega)
    return Fn / denom

def physical_response(z_n, phi_n):
    return z_n * phi_n

# === Modal Data ===
modal_data_all = [
    (n, compute_modal_mass(rho, A, L, n), natural_pulsation(E, rho, L, I, A, n))
    for n in range(1, N_modes + 1)
]

selected_modes = [1]  # Default
t_current = 2.0       # Default

# === Figure and Subplots ===
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8, hspace=0.4)

# --- Load Profile Plot ---
line_load, = ax1.plot([], [], label='Load profile', color='red')
for x_out in measurement_points:
    ax1.axvline(x_out, color=colors[measurement_points.index(x_out)], linestyle='--', lw=0.5, label=f'x = {round(x_out, 2)} m')
ax1.set_ylabel('Load Magnitude p(x, t)')
ax1.set_xlabel('Beam Position x (m)')
ax1.set_title('Dynamic Load Profile Along Beam')
ax1.set_ylim(0, 2)
ax1.set_xlim(0, L)
ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

# --- Frequency Domain Plot ---
lines_response = [ax2.plot([], [], label=f'x = {round(x_out, 2)} m', color=colors[i])[0]
                  for i, x_out in enumerate(measurement_points)]
ax2.set_ylabel('|u(x)|')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_yscale('log')
ax2.set_ylim(1e-8, 1e2)
ax2.set_xlim(0, 100)
ax2.set_title('Frequency Domain Response')
ax2.grid(which='both', linestyle=':')
for n in range(1, N_modes + 1):
    freq_n = natural_pulsation(E, rho, L, I, A, n) / (2 * np.pi)
    ax2.axvline(freq_n, color='red', linestyle='--', lw=0.5)
ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

# --- Time Domain Plot ---
lines_time = []
for i, x in enumerate(measurement_points):
    line, = ax3.plot([], [], label=f"x = {x:.2f} m", color=colors[i])
    lines_time.append(line)
ax3.set_xlim(0, 2)
ax3.set_ylim(-0.03, 0.03)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Displacement u(x, t)')
ax3.set_title('Time-Domain Response')
ax3.grid(True)

# === Widgets ===
# CheckButtons for mode selection
check_ax = plt.axes([0.82, 0.1, 0.1, 0.2])
check_labels = [f'Mode {n}' for n in range(1, N_modes + 1)]
check_states = [n in selected_modes for n in range(1, N_modes + 1)]
check = CheckButtons(check_ax, check_labels, check_states)

# Slider for time
slider_ax = plt.axes([0.1, 0.01, 0.7, 0.03])
time_slider = Slider(slider_ax, 't(s)', 0.0, 10.0, valinit=t_current, valstep=0.01)

# === Update Function ===
def update_plot(val=None):
    t_load = time_slider.val
    p_xt = dynamic_load(x_grid, t_load) 
    line_load.set_data(x_grid, p_xt)

    selected = [i + 1 for i, state in enumerate(check.get_status()) if state]

    for i, x_out in enumerate(measurement_points):
        u_total = np.zeros_like(omega, dtype=complex)
        for n, mu_n, omega_n in modal_data_all:
            if n in selected:
                phi_n_x = modal_shape(L, x_grid, n)
                Fn = modal_excitation(p_xt, phi_n_x, x_grid)
                z_n = modal_response(Fn, mu_n, omega_n, xi, omega)
                phi_n_out = modal_shape(L, x_out, n)
                u_total += physical_response(z_n, phi_n_out)

        # Update frequency domain
        u_mag = np.abs(u_total)
        lines_response[i].set_data(freq, u_mag)

        # Update time domain
        u_t = np.fft.irfft(u_total, n=N_freqs)
        lines_time[i].set_data(t_irfft, u_t)

    fig.canvas.draw_idle()

# === Connect Widgets ===
check.on_clicked(update_plot)
time_slider.on_changed(update_plot)

# === Initial Plot ===
update_plot()
plt.show()

