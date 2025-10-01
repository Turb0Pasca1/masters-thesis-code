# double_res_slider_tool.py
# adjusted to work with Schnall 1985 original circuit, presented in paper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
# only works with jupyter
# import ipywidgets as widgets 

# includes releveant functions from an external file to keep the code cleaner
import rf_functions as rf

# the tank circuit's inductance is limited by the available inductors 
# inductance and series resistance are stored in 'data/coilcraft_inductors.csv'
coil_data = rf.load_inductors()

# frequency range
f = np.linspace(1e6, 25e6, 10000)  
w = 2*np.pi*f

# frequencies of interest
f_H = 21.527 # in MHz
w_H = 1e6*f_H*2*np.pi # in rad/s
f_Na = 5.694 # in MHz
w_Na = 1e6*f_Na*2*np.pi # in rad/s

# initial values in SI units

# sample coil
L_S = 2.3e-6
R_S = 0.36
# arbitary pair of coil_data
L_1 = 18.5e-9
R_1 = 3.9e-3
# initial value calculation for components relevant for tuning
C_2 = 1 / (w_Na**2*L_S)
C_1 = L_1 / ((w_H**2*L_1)+R_1**2)
# initial guess
C_3 = 100e-12
L_3 = 43e-3
R_3 = 7.9e-3

# total impedance of circuit to plot
Z = rf.Schnall(w, C_1, C_2, C_3, L_1, R_1, L_3, R_3, L_S, R_S)

# create the plot
fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(12, 10))
fig.suptitle('Schnall 1985 double resonant circuit', fontsize=20)

# global impedance view
line01, = ax[0][0].plot(1e-6 * f, Z.real, label=r'$\Re(Z)$')
line02, = ax[0][0].plot(1e-6 * f, Z.imag, label=r'$\Im(Z)$')
ax[0][0].set_xlim([1,25])
ax[0][0].set_ylim([-100,100])
ax[0][0].set_xlabel('Frequency / MHz')
ax[0][0].set_ylabel(r'Z / $\Omega$')
ax[0][0].grid(True)
ax[0][0].legend()

# global return loss view
line03, = ax[0][1].plot(1e-6 * f, rf.RL(Z), label=r'$RL$')
ax[0][1].axvline(x=f_Na, color='grey', linestyle=':', linewidth=2, label=r'$f=5.694\,$MHz')
ax[0][1].axvline(x=f_H, color='black', linestyle=':', linewidth=2, label=r'$f=21.527\,$MHz')
ax[0][1].set_xlim([1,25])
ax[0][1].set_xlabel('Frequency / MHz')
ax[0][1].set_ylabel('RL / dB')
ax[0][1].grid(True)
ax[0][1].legend(loc='upper left')

# impedance zoomed in at f(Na)
line1, = ax[1][0].plot(1e-6 * f, Z.real, label=r'$\Re(Z)$')
line2, = ax[1][0].plot(1e-6 * f, Z.imag, label=r'$\Im(Z)$')
ax[1][0].axvline(x=f_Na, color='grey', linestyle=':', linewidth=2, label=r'$f=5.694\,$MHz')
ax[1][0].set_xlim([5.6,6])
ax[1][0].set_ylim([-100,200])
ax[1][0].set_xlabel('Frequency / MHz')
ax[1][0].set_ylabel(r'Z / $\Omega$')
ax[1][0].grid(True)
ax[1][0].legend()

# impedance zoomed in at f(H)
line3, = ax[1][1].plot(1e-6 * f, Z.real, label=r'$\Re(Z)$')
line4, = ax[1][1].plot(1e-6 * f, Z.imag, label=r'$\Im(Z)$')
ax[1][1].axvline(x=f_H, color='black', linestyle=':', linewidth=2, label=r'$f=21.527\,$MHz')
ax[1][1].set_xlim([21,22])
ax[1][1].set_ylim([-100,200])
ax[1][1].set_xlabel('Frequency / MHz')
ax[1][1].set_ylabel(r'Z / $\Omega$')
ax[1][1].grid(True)
ax[1][1].legend()

# generate text for label of Q
dip_freq, df, q_factor, idx = rf.calculate_q(1e-6*f, rf.RL(Z), prominence=0)  
q_texts = []
for i in range(len(dip_freq)):
    q_str = f"Q = {q_factor[i]:.2f}\n@f$_0$ = {dip_freq[i]:.2f} MHz"
    text = ax[0][1].text(
        dip_freq[i] + 0.1,
        rf.RL(Z)[idx[i]],
        q_str,
        fontsize=12,
        color='black',
        bbox=dict(facecolor="white", alpha=0.8)
    )
    q_texts.append(text)

# make space for slider
plt.subplots_adjust(bottom=0.3)

# slider axes
ax_L_1 = plt.axes([0.2, 0.21, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_L_3 = plt.axes([0.2, 0.16, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_C_3 = plt.axes([0.2, 0.11, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# sliders to change values
# makes the slider for the inductance equally spaced between each inductance value
n_steps = len(coil_data[1:,0])
L_1_slider = Slider(ax_L_1, r'L_1 / nH', 1, n_steps, valinit=6, valstep=1)
# index corresponds to the initial value chosen for L_3
L_1_slider.valtext.set_text(coil_data[6, 0])

L_3_slider = Slider(ax_L_3, r'L_3 / nH', 1, n_steps, valinit=14, valstep=1)
L_3_slider.valtext.set_text(coil_data[14, 0])

# slider bounds and initial values
C_3_slider = Slider(ax_C_3, r'C_3 / pF', .1e12*C_3, 4e12*C_3, valinit=1e12*C_3)

# update function
def update(val):
    # get slider values
    L_3_val = coil_data[int(L_3_slider.val),0]
    # overwrites the slider value (index value of the inductance) with the actual inductance value
    L_3_slider.valtext.set_text(coil_data[L_3_slider.val, 0])

    L_1_val = coil_data[int(L_1_slider.val),0]
    L_1_slider.valtext.set_text(coil_data[L_1_slider.val, 0])

    # adjustments for correct units
    C_3_val = 1e-12*C_3_slider.val
    R_3_val =  1e-3*coil_data[int(L_3_slider.val),1]
    R_1_val =  1e-3*coil_data[int(L_1_slider.val),1]

    L_3_val *= 1e-9 
    L_1_val *= 1e-9 
    # calculates the changed impedance by the slider values
    C_1_val = L_1_val / ((w_H**2*L_1_val)+R_1_val**2)

    Z_val = rf.Schnall(w, C_1_val, C_2, C_3_val, L_1_val, R_1_val, L_3_val, R_3_val, L_S, R_S)
    
    # sets the data
    y_data_1 = Z_val.real
    line1.set_ydata(y_data_1)
    y_data_2 = Z_val.imag
    line2.set_ydata(y_data_2)
    line3.set_ydata(y_data_1)
    line4.set_ydata(y_data_2)
    line01.set_ydata(y_data_1)
    line02.set_ydata(y_data_2)
    line03.set_ydata(rf.RL(Z_val))
    # self adjusting y span -> reduces visibility of changes
    # ax.set_ylim([np.min(y_data_1), np.max(y_data_1) * 1.1])

    # change text for  Q
    # access the text list to clear old ones
    global q_texts  
    # clear old annotations
    for text in q_texts:
        text.remove()
    q_texts.clear()
    dip_freq, df, q_factor, idx = rf.calculate_q(f * 1e-6, rf.RL(Z_val), prominence=0)  
    q_texts = []
    for i in range(len(dip_freq)):
        q_str = f"Q = {q_factor[i]:.2f}\n@f$_0$ = {dip_freq[i]:.2f} MHz"
        text = ax[0][1].text(
            dip_freq[i] + 0.1,
            rf.RL(Z_val)[idx[i]],
            q_str,
            fontsize=12,
            color='black',
            bbox=dict(facecolor="white", alpha=0.8)
        )
        q_texts.append(text)
    
    fig.canvas.draw_idle()

# link sliders to the update function
L_3_slider.on_changed(update)
C_3_slider.on_changed(update)
L_1_slider.on_changed(update)

# reset button
resetax = fig.add_axes([.92, .01, .05, .03])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    L_3_slider.reset()
    C_3_slider.reset()
    L_1_slider.reset()
button.on_clicked(reset)

# show the plot
plt.show()