import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrow
from matplotlib.widgets import Slider

# --- Define modal shape function ---
def modal_shape(L, x, n):
    x = np.asarray(x)
    return np.sin(n * np.pi * x / L)

# --- Define modal excitation ---
def modal_excitation(p_func, L, n):
    x_vals = np.linspace(0, L, 1000)
    phi_n = modal_shape(L, x_vals, n)
    return np.trapz(p_func(x_vals) * phi_n, x_vals)

# --- Define FRF ---
def FRF_with_Fn(L, w, x_out, w_nat, mu_nat, xi_nat, F_n_list):
    H = np.zeros_like(w, dtype=complex)
    for i, (w_n, mu, xi, F_n) in enumerate(zip(w_nat, mu_nat, xi_nat, F_n_list)):
        phi_out = modal_shape(L, x_out, i + 1)
        num = F_n * phi_out
        den = mu * (w_n**2 - w**2 + 2j * xi * w_n * w)
        H += num / den
    return H

# --- Beam properties ---
E = 70e9
rho = 2700
L = 1
b = 1e-2
h = 1e-3
I = b * h**3 / 12
A = b * h

def compute_modal_mass(rho, A, L, n):
    return rho * A * L / 2

def natural_pulsation(E, rho, L, I, A, n):
    return (n * np.pi / L)**2 * np.sqrt(E * I / (rho * A))

def p_func(x, x0, sigma=0.01):
    return np.exp(-((x - x0)**2) / (2 * sigma**2))

n_list = np.arange(1, 6)
mu_n = compute_modal_mass(rho, A, L, n=n_list) * np.ones((len(n_list), 1))
w_n = natural_pulsation(E, rho, L, I, A, n=n_list)
xi_n = 0.01 * np.ones((len(n_list), 1))
w_array = np.linspace(0, 100 * 2 * np.pi, 1024)

# Measurement points
measurement_points = [0.25 * L, 0.5 * L, 0.75 * L]

# Plot setup
fig, axs = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1, 1]})

# Beam visualization
axs[0].set_xlim(-0.05, L + 0.05)
axs[0].set_ylim(-0.1, 0.2)
axs[0].axis('off')
axs[0].plot([0, L], [0, 0], 'k-', lw=4)

# Measurement points
colors = ['blue', 'orange', 'green']
for i, x in enumerate(measurement_points):
    axs[0].plot([x, x], [-0.1, 0.01], color=colors[i], linestyle='--')
    axs[0].text(x, -0.12, f"x_out={x:.2f} L", ha='center', fontsize=10, color=colors[i])

# Pin supports
axs[0].add_patch(Polygon([[0, 0], [-0.02, -0.05], [0.02, -0.05]], closed=True, color='gray'))
axs[0].add_patch(Polygon([[L, 0], [L - 0.02, -0.05], [L + 0.02, -0.05]], closed=True, color='gray'))

# Arrow placeholder
arrow_y = 0.085
arrow = axs[0].add_patch(FancyArrow(0, arrow_y, 0, -0.05, width=0.005,
                                    head_width=0.02, head_length=0.02, color='red'))

# FRF plot
frf_lines = [axs[1].plot([], [], label=f"x_out={x:.2f} L")[0] for x in measurement_points]
axs[1].set_xlim(0, 100)
axs[1].set_yscale('log')
axs[1].set_ylim(1e-5, 1e2)
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("|u(x)|")
axs[1].legend()
axs[1].grid(True, which='both', linestyle=':')

# # Phase plot
# phase_lines = [axs[2].plot([], [], label=f"x_out={x:.2f} L")[0] for x in measurement_points]
# axs[2].set_xlim(0, 100)
# axs[2].set_ylim(-180, 180)
# axs[2].set_title("FRF Phase (degrees)")
# axs[2].set_xlabel("Frequency (Hz)")
# axs[2].set_ylabel("Phase (°)")
# axs[2].legend()
# axs[2].grid(True)


# Modal excitation bar plot
modal_bar = axs[2].bar(n_list, np.zeros_like(n_list), color='salmon', width=0.6)
axs[2].set_ylim(0, 1)
axs[2].set_xlabel("Mode number")
axs[2].set_ylabel("|F_n|")
axs[2].grid(True)

# Slider setup
slider_ax = plt.axes([0.16, 0.8, 0.7, 0.03])  # [left, bottom, width, height]
x_slider = Slider(slider_ax, 'Load (x_in)', 0, 1, valinit=0.0, valstep=0.01)


# --- Update function ---
def update(x_in):
    global arrow
    p = lambda x: p_func(x, x_in) * 30
    F_n_list = [modal_excitation(p, L, n_i) for n_i in n_list]

    for j, x_out in enumerate(measurement_points):
        H = FRF_with_Fn(L=L, w=w_array, x_out=x_out, w_nat=w_n, mu_nat=mu_n, xi_nat=xi_n, F_n_list=F_n_list)
        frf_lines[j].set_data(w_array / (2 * np.pi), np.abs(H))
        # phase_lines[j].set_data(w_array / (2 * np.pi), np.angle(H, deg=True))

    for bar, height in zip(modal_bar, np.abs(F_n_list)):
        bar.set_height(height)

    # Remove old arrow if it exists
    if arrow is not None:
        arrow.remove()

    # Add a new arrow
    arrow_y = 0.085
    arrow = axs[0].add_patch(
        FancyArrow(x_in, arrow_y, 0, -0.05, width=0.005, head_width=0.02, head_length=0.02, color='red')
    )

    fig.canvas.draw_idle()


x_slider.on_changed(update)

# Initial update
update(0.0)

plt.show()
