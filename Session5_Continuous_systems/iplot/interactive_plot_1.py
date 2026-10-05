import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from ipywidgets import interact

# Define the system parameters
E=70e9
rho=2700
L=1
b=1e-2
h=1e-3

I=b*h**3/12
A=b*h
#utils
def modal_shape(x, L, n):
    """
    Compute the modal shape for a simply supported beam.
    
    Arguments:
    x (float) -- Position along the beam [m]
    L (float) -- Length of the beam [m]
    n (int)   -- Mode number
    
    Returns:
    psi (float) -- Modal shape
    """
    ### BEGIN SOLUTION
    psi = np.sin(n*np.pi*x/L)
    ### END SOLUTION
    return psi

def n_freq(E,rho,L,I,A,n):
    """ 
    Compute natural pulsations for a simply supported beam.
    
    Arguments:
    E (float)  -- Young's modulus [Pa]
    rho (float) -- Density [kg/m^3]
    L (float)  -- Length of the beam [m]
    I (float)  -- Bending inertia of the section [m^4]
    A (float)  -- Area of the section [m^2]
    n (array)  -- Indices of natural frequencies to compute

    Returns:
    w (array)  -- Array of natural pulsations [rad/s]
    """
    
    ### BEGIN SOLUTION
    # This equation is found on sl.20 of Vibrations_continuous.pdf
    w = (n**2 * np.pi**2 / L**2) * np.sqrt(E * I / (rho * A))
    ### END SOLUTION

    return w

def compute_FRF(omega_array, x_f, x_n, modal_masses, natural_pulsations, modal_damping):
    """
    Compute the Frequency Response Function (FRF) at a particular location due to an applied force at another location.

    Arguments:
    omega_array (array)       -- Array of angular frequencies [rad/s]
    x_f (float)               -- Position where the force is applied [m]
    x_m (float)               -- Position where the displacement is measured [m]
    modal_masses (array)      -- Modal masses [kg]
    natural_pulsations (array)-- Natural pulsations [rad/s]
    modal_damping (array)     -- Modal damping values (damping ratio)
    
    Returns:
    H (array)                 -- Frequency Response Function (complex array)
    """
    ### BEGIN SOLUTION
    U = np.zeros(np.size(omega_array), dtype=complex)
    for n, (mu_n,w_n, xi_n) in enumerate(zip(modal_masses, natural_pulsations, modal_damping)):# zip is used to iterate over multiple lists at the same time 
                                                                                            # enumerate is used to iterate over a list and return the index and the value
        # Modal projection at force location and measured location
        psi_i = modal_shape(x_f, L, n+1)
        psi_o = modal_shape(x_n, L, n+1)
        
        zi =  psi_i/ (mu_n*(-omega_array**2 + w_n**2 +2*1j*omega_array*w_n*xi_n))
        U += zi*psi_o
    ### END SOLUTION
    return U

#plot 
n = 10
nlist = np.arange(1, n+1)
wn = n_freq(E, rho, L, I, A, nlist)
xin = np.ones((n,))*0.00000001
xn = L / 3
xf = L / 2
mun = rho * A * L / 2 * np.ones((n, 1))

# Initial plot setup
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
fig.subplots_adjust(bottom=0.25)  
w_array = np.linspace(0, 80*2*np.pi, 1024)
H = compute_FRF(omega_array=w_array, x_f=xf, x_n=xn, modal_masses=mun, natural_pulsations=wn, modal_damping=xin)
f = w_array / (2 * np.pi)

# Magnitude Plot
axes[0].semilogy(f, np.abs(H), lw=2, color='lightgray')
line, = axes[0].semilogy(f, np.abs(H), lw=2, color='steelblue')
axes[0].set_title('Frequency Response Magnitude')
axes[0].set_ylabel('||Y(L/3) / F(L/2)||')

axes[0].grid(True)

# Phase Plot
axes[1].plot(f, np.angle(H, deg=True), color='lightgray')
line2, = axes[1].plot(f, np.angle(H, deg=True), color='steelblue')
axes[1].set_ylim([-200, 200])
axes[1].set_title('Frequency Response Phase')
axes[1].set_ylabel('Phase (Degrees)')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].grid(True)

# Sliders
axm1 = plt.axes([0.25, 0.15, 0.65, 0.03])
axm2 = plt.axes([0.25, 0.10, 0.65, 0.03])
axm5 = plt.axes([0.25, 0.05, 0.65, 0.03])

m1_slider = Slider(ax=axm1, label='xi mode 1', valmin=0, valmax=1, valinit=0)
m2_slider = Slider(ax=axm2, label='xi mode 2', valmin=0, valmax=1, valinit=0)
m5_slider = Slider(ax=axm5, label='xi mode 5', valmin=0, valmax=1, valinit=0)

def update(val):
    xin[0] = m1_slider.val
    xin[1] = m2_slider.val
    xin[4] = m5_slider.val
    H = compute_FRF(w_array, xf, xn, mun, wn, xin)
    line.set_ydata(np.abs(H))
    phase_deg = np.angle(H, deg=True)
    phase_deg = np.mod(phase_deg + 180, 360) - 180
    line2.set_ydata(phase_deg)
    fig.canvas.draw_idle()

m1_slider.on_changed(update)
m2_slider.on_changed(update)
m5_slider.on_changed(update)

# Reset Button
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    m1_slider.reset()
    m2_slider.reset()
    m5_slider.reset()
button.on_clicked(reset)

plt.show()