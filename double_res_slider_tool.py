# double_res_slider_tool.py
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
# calculated to resonate at f_H
C_Thf = rf.C_T_opt(w_H, R_S, L_S)
C_Mhf = rf.C_M_opt(w_H, R_S, L_S, C_Thf)
# arbitary pair of coil_data
L_t1 = 18.5e-9
R_t1 = 3.9e-3
# calculated to resonate at f_H
C_t1 = rf.C_in_par(L_t1, R_t1, f_H)
# same design for second tank circuit
L_t2 = L_t1
R_t2 = R_t1
C_t2 = C_t1
# parallel capacitances add up as sums
# C_total = C_par1 + C_par2
# C_total is calculated to resonate at f_Na
C_totalT = rf.C_T_opt(w_Na, R_S, L_S)
C_totalM = rf.C_M_opt(w_Na, R_S, L_S, C_totalT)
C_Tlf = C_totalT - C_Thf
C_Mlf = C_totalM - C_Mhf

# total impedance of circuit to plot
Z = rf.Tadanki(w, C_Mhf, C_Mlf, C_Thf, C_Tlf, L_t1, R_t1, C_t1, L_t2, R_t2, C_t2, L_S, R_S)

# create the plot
fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(12, 10))
fig.suptitle('Tadanki 2012 double resonant circuit', fontsize=20)

# global impedance view
line01, = ax[0][0].plot(1e-6 * f, Z.real, label=r'$\Re(Z)$')
line02, = ax[0][0].plot(1e-6 * f, Z.imag, label=r'$\Im(Z)$')
ax[0][0].set_xlim([1,25])
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
ax[0][1].legend()

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
# variability is limited by assuming that 
# both tank circuits are identical
# R_t is connected to L_t
# the tank circuit should always resonate at f_H 
# and therefore C_t is caluclated from L_t and R_t

ax_L_t1 = plt.axes([0.2, 0.21, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_C_Thf = plt.axes([0.2, 0.16, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_C_Tlf = plt.axes([0.2, 0.11, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_C_Mhf = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_C_Mlf = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# sliders to change values
# makes the slider for the inductance of the tank circuit equally spaced between each inductance value
n_steps = len(coil_data[1:,0])
L_t1_slider = Slider(ax_L_t1, r'L_t1 / nH', 1, n_steps, valinit=6, valstep=1)
# index corresponds to the initial value chosen for L_t1
L_t1_slider.valtext.set_text(coil_data[6, 0])

# slider bounds and initial values
C_Thf_slider = Slider(ax_C_Thf, r'C_Thf / pF', .1e12*C_Thf, 4e12*C_Thf, valinit=1e12*C_Thf)
C_Tlf_slider = Slider(ax_C_Tlf, r'C_Tlf / pF', .1e12*C_Tlf, 4e12*C_Tlf, valinit=1e12*C_Tlf)
C_Mhf_slider = Slider(ax_C_Mhf, r'C_Mhf / pF', .1e12*C_Mhf, 4e12*C_Mhf, valinit=1e12*C_Mhf)
C_Mlf_slider = Slider(ax_C_Mlf, r'C_Mlf / pF', .1e12*C_Mlf, 4e12*C_Mlf, valinit=1e12*C_Mlf)

# update function
def update(val):
    # get slider values
    L_t1_val = coil_data[int(L_t1_slider.val),0]
    # overwrites the slider value (index value of the inductance) with the actual inductance value
    L_t1_slider.valtext.set_text(coil_data[L_t1_slider.val, 0])
    # adjustments for correct units
    C_Thf_val = 1e-12*C_Thf_slider.val
    C_Tlf_val = 1e-12*C_Tlf_slider.val
    C_Mhf_val = 1e-12*C_Mhf_slider.val
    C_Mlf_val = 1e-12*C_Mlf_slider.val
    R_t1_val =  1e-3*coil_data[int(L_t1_slider.val),1]
    # replaced by the line above
    # R_t1_val = 1e-3*get_R_for_L(L_t1_val)
    L_t1_val *= 1e-9 
    # calculates the changed impedance by the slider values
    C_t1_val = rf.C_in_par(L_t1_val, R_t1_val, f_H)
    L_t2_val, R_t2_val, C_t2_val = L_t1_val, R_t1_val, C_t1_val
    Z_val = rf.Tadanki(w, C_Mhf_val, C_Mlf_val, C_Thf_val, C_Tlf_val, L_t1_val, R_t1_val, C_t1_val, L_t2_val, R_t2_val, C_t2_val, L_S, R_S)
    
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
L_t1_slider.on_changed(update)
C_Thf_slider.on_changed(update)
C_Tlf_slider.on_changed(update)
C_Mhf_slider.on_changed(update)
C_Mlf_slider.on_changed(update)

# reset button
resetax = fig.add_axes([.92, .01, .05, .03])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    L_t1_slider.reset()
    C_Thf_slider.reset()
    C_Tlf_slider.reset()
    C_Mhf_slider.reset()
    C_Mlf_slider.reset()
button.on_clicked(reset)

# show the plot
plt.show()