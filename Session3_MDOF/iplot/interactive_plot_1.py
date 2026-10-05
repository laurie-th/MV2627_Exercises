from ipywidgets import interact, fixed
from matplotlib.widgets import Slider, Button
from iplot.utils import *
# Define initial parameters
init_m = 1
init_k = 16
init_c = 0.1


init_M, init_C, init_K = build_3DOF_matrices(m=init_m, c1=init_c, c=init_c, k=init_k, k1=init_k)


# Create the figure and the line that we will manipulate

fig, axes = plt.subplots(2,1)

f= np.arange(0,2,0.01)
H = frequency_response_function(M=init_M,K=init_K,C=init_C,f=f,i=2,j=2)

# wn=calcW_n(k=init_k,m=init_m)
axes[0].semilogy(f,np.abs(H), lw=2, color='lightgray')

line, = axes[0].semilogy(f,np.abs(H), lw=2, color='steelblue')

axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('|H(f)|')
axes[0].grid(which='both', linestyle=':')

axes[1].plot(f,np.angle(H,deg=True), color='lightgray')

line2, = axes[1].plot(f,np.angle(H,deg=True), color='steelblue')

axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('angle(H(f))')
axes[1].set_ylim([-180,0])

axes[1].grid(which='both', linestyle=':')


# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.4)


axmass = plt.axes([0.25, 0.1, 0.65, 0.03])
m_slider = Slider(
    ax=axmass,
    label='Mass (kg)',
    valmin=0.1,
    valmax=10,
    valinit=init_m,
)
axstifness=plt.axes([0.25, 0.15, 0.65, 0.03])
k_slider = Slider(
    ax=axstifness,
    label="Stifness (N/m)",
    valmin=1,
    valmax=20,
    valinit=init_k,
)
axc = plt.axes([0.25, 0.2, 0.65, 0.03])
c_slider = Slider(
    ax=axc,
    label="damping (N/(m*s))",
    valmin=0.001,
    valmax=1,
    valinit=init_c,
)



# The function to be called anytime a slider's value changes
def update(fixed):
    
    M_u, C_u, K_u = build_3DOF_matrices(
        m=m_slider.val,
        c1=c_slider.val,
        c=c_slider.val,
        k=k_slider.val,
        k1=k_slider.val
    )

    H = frequency_response_function(M_u, C_u, K_u, f=f, i=2,j=2)
    
    line.set_ydata(np.abs(H))
    line2.set_ydata(np.angle(H, deg=True))

   
    
   
    fig.canvas.draw_idle()

# register the update function with each slider
m_slider.on_changed(update)
k_slider.on_changed(update)
c_slider.on_changed(update)


# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    m_slider.reset()
    k_slider.reset()
    c_slider.reset()
button.on_clicked(reset)
w = interact(update(fixed))