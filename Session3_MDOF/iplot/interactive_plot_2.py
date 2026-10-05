import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from scipy import linalg as la

# System parameters
m = 1
k = 16
k1 = 16
c = 0.1
c1 = 0.1

def build_3DOF_matrices(m, c, c1, k, k1):
    M = np.array([[m, 0, 0], [0, m, 0], [0, 0, m]])
    K = np.array([[k1+k, -k, 0], [-k, 2*k, -k], [0, -k, k]])
    C = np.array([[c1+c, -c, 0], [-c, 2*c, -c], [0, -c, c]])
    return M, C, K

def calc_eigenvalues_eigenvectors(M, K):
    lamb, psi = la.eig(K, M)
    idx = lamb.argsort()
    lamb = lamb[idx]
    psi = psi[:, idx]
    return lamb, psi

M, C, K = build_3DOF_matrices(m, c, c1, k, k1)
lamb, psi = calc_eigenvalues_eigenvectors(M, K)
f_res = np.sqrt(lamb) / (2 * np.pi)

n_modes = psi.shape[1]
modal_masses = [psi[:, i].T @ M @ psi[:, i] for i in range(n_modes)]
modal_stiffnesses = [psi[:, i].T @ K @ psi[:, i] for i in range(n_modes)]

f = np.linspace(0.001, 2, 400)
omega = 2 * np.pi * f

def transfer_function(selected_modes, out_dof, in_dof):
    H_modes = []
    H_total = np.zeros_like(omega, dtype=complex)
    contribution_factors = []
    
    for r, include in enumerate(selected_modes):
        mu = modal_masses[r]
        k_mod = modal_stiffnesses[r]
        factor = psi[out_dof, r] * psi[in_dof, r]
        contribution_factors.append(factor)
        H_mode = factor * (1.0 / (-omega**2 * mu + k_mod))
        H_modes.append(H_mode)
        if include:
            H_total += H_mode
    
    return H_modes, H_total, contribution_factors

fig, axes = plt.subplots(4, 1, gridspec_kw={'height_ratios': [5, 1, 1, 1]}, figsize=(10, 8))
def update_plot(mode1, mode2, mode3, out_dof, in_dof):
    selected = [mode1, mode2, mode3]
    H_modes, H_total, contribution_factors = transfer_function(selected, out_dof, in_dof)
    H_full = sum(H_modes)
    
    for ax in axes:
        ax.clear()
    
    colors = ['b', 'g', 'r']
    labels = ["Mode 1", "Mode 2", "Mode 3"]
    
    axes[0].semilogy(f, np.abs(H_full), color='gray', alpha=0.5, label='Full 3DOF FRF')
    for i in range(3):
        axes[0].semilogy(f, np.abs(H_modes[i]), colors[i]+'--', label=labels[i])
    axes[0].semilogy(f, np.abs(H_total), 'k-', linewidth=2, label='Sum of selected modes')
    
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude')
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)  # Light grid
    
    for i in range(3):
        factor = contribution_factors[i]
        axes[i+1].plot(f, np.angle(H_modes[i], deg=True), colors[i]+'--', label=f'{labels[i]} (Factor: {factor:.4f})')
        axes[i+1].set_xlabel('Frequency (Hz)')
        axes[i+1].set_ylabel('Phase (deg)')
        axes[i+1].legend()
        axes[i+1].grid(True, linestyle=':', linewidth=0.5, alpha=0.7)  # Light grid
    
    plt.tight_layout()
    plt.show()


mode1_cb = widgets.Checkbox(value=True, description='Mode 1', indent=False)
mode2_cb = widgets.Checkbox(value=True, description='Mode 2', indent=False)
mode3_cb = widgets.Checkbox(value=True, description='Mode 3', indent=False)

out_dof_rb = widgets.RadioButtons(options=[0, 1, 2], description='Output DOF', indent=False)
in_dof_rb = widgets.RadioButtons(options=[0, 1, 2], description='Input DOF', indent=False)

interactive_plot = widgets.interactive_output(update_plot, {
    'mode1': mode1_cb, 'mode2': mode2_cb, 'mode3': mode3_cb,
    'out_dof': out_dof_rb, 'in_dof': in_dof_rb
})

widgets_column = widgets.VBox([out_dof_rb, in_dof_rb, mode1_cb, mode2_cb, mode3_cb])
ui = widgets.HBox([interactive_plot,widgets_column])
display(ui)