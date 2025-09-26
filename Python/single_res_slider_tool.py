# single_res_slider_tool.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import rf_functions as rf

# frequnecy range
f = np.logspace(6, 8, 10000)  
w = 2 * np.pi * f

# frequencies of interest
f_Na = 5.694e6
f_H = 21.527e6
omega_Na = 2 * np.pi * f_Na
omega_H = 2 * np.pi * f_H

# initial values
# assumed parameters of custom 10mm solenoid coil
init_R = 0.23
init_L = 2.3e-6

Z_0 = 50

# follows the calculation of optimal C_T and C_M for and Z_0, omega
init_C_T = rf.C_T_opt(omega_Na, init_R, init_L)
init_C_M = rf.C_M_opt(omega_Na, init_R, init_L, init_C_T)

Z_init = rf.Z_M_T_Resonator(w, init_C_M, init_C_T, init_L, init_R)

# create the plot
fig, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(1e-6 * f, rf.RL(Z_init), label=r'$|S_{11}|$')
ax.axvline(x=1e-6*f_Na, color='grey', linestyle=':', linewidth=2, label=r'$f=5.694\,$MHz')
ax.axvline(x=1e-6*f_H, color='black', linestyle=':', linewidth=2, label=r'$f=21.53\,$MHz')
ax.set_xlim([1,30])
ax.set_ylim([-40,1])
ax.set_xlabel(r'$f$ / MHz')
ax.set_ylabel(r'$RL$ / dB')
ax.grid(True)
ax.legend()

# generate text for optimal values of C_T and C_M
C_T_Na = rf.C_T_opt(omega_Na, init_R, init_L)
C_M_Na = rf.C_M_opt(omega_Na, init_R, init_L, C_T_Na)
C_text = ax.text(15, -5, '', fontsize=12, color='gray', bbox=dict(facecolor="white", alpha=0.8))
C_text.set_text("opt. C for current L\n"f"C_T = {1e12*C_T_Na:.2f} pF\nC_M = {1e12*C_M_Na:.2f} pF")
C_text.set_position((5, 2))  

C_T_H = rf.C_T_opt(omega_H, init_R, init_L)
C_M_H =rf.C_M_opt(omega_H, init_R, init_L, C_T_H)
C_H_text = ax.text(15, -5, '', fontsize=12, color='gray', bbox=dict(facecolor="white", alpha=0.8))
C_H_text.set_text("opt. C for current L\n"f"C_T = {1e12*C_T_H:.2f} pF\nC_M = {1e12*C_M_H:.2f} pF")
C_H_text.set_position((21, 2))  

# generate text for label of Q
dip_freq, df, q_factor, idx = rf.calculate_q(1e-6*f, rf.RL(Z_init), prominence=0)  
q_texts = []
for i in range(len(dip_freq)):
    q_str = f"Q = {q_factor[i]:.2f}\n@f$_0$ = {dip_freq[i]:.2f} MHz"
    text = ax.text(
        dip_freq[i] + 0.1,
        rf.RL(Z_init)[idx[i]],
        q_str,
        fontsize=12,
        color='black',
        bbox=dict(facecolor="white", alpha=0.8)
    )
    q_texts.append(text)

plt.subplots_adjust(bottom=0.25)

# slider axes
ax_R = plt.axes([0.2, 0.16, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_L = plt.axes([0.2, 0.11, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_C_T = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_C_M = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# sliders to change values
R_slider = Slider(ax_R, r'Resistance $R$ / $\Omega$', 0.05*init_R, 2*init_R, valinit=init_R)
L_slider = Slider(ax_L, r'Inductance $L$ / $\mu$H', .05e6*init_L, 2e6*init_L, valinit=1e6*init_L)
C_T_slider = Slider(ax_C_T, r'Tune Capacitance $C_T$ / pF', .05e12*init_C_T, 2e12*init_C_T, valinit=1e12*init_C_T)
C_M_slider = Slider(ax_C_M, r'Match Capacitance $C_M$ / pF', .05e12*init_C_M, 2e12*init_C_M, valinit=1e12*init_C_M)

# update function
def update(val):
    # get slider values
    R = R_slider.val
    L = 1e-6*L_slider.val
    C_T = 1e-12*C_T_slider.val
    C_M = 1e-12*C_M_slider.val
    # set new values from slider
    Z = rf.Z_M_T_Resonator(w, C_M, C_T, L, R)
    y_data_1 = rf.RL(Z)
    line1.set_ydata(y_data_1)
    # self adjusting y span -> reduces visibility of changes
    # ax.set_ylim([np.min(y_data_1), np.max(y_data_1) * 1.1])

    # change text for  Q
    # access the text list to clear old ones
    global q_texts  
    # clear old annotations
    for text in q_texts:
        text.remove()
    q_texts.clear()
    dip_freq, df, q_factor, idx = rf.calculate_q(f * 1e-6, rf.RL(Z), prominence=0)  
    q_texts = []
    for i in range(len(dip_freq)):
        q_str = f"Q = {q_factor[i]:.2f}\n@f$_0$ = {dip_freq[i]:.2f} MHz"
        text = ax.text(
            dip_freq[i] + 0.1,
            rf.RL(Z)[idx[i]],
            q_str,
            fontsize=12,
            color='black',
            bbox=dict(facecolor="white", alpha=0.8)
        )
        q_texts.append(text)

    # change text for C_T and C_M
    C_T_calc = rf.C_T_opt(omega_Na, R, L)
    C_M_calc = rf.C_M_opt(omega_Na, R, L,  rf.C_T_opt(omega_Na, R, L))
    C_text.set_text("opt. C for curret L\n"f"C_T = {1e12*C_T_calc:.2f} pF\nC_M = {1e12*C_M_calc:.2f} pF")

    C_T_H = rf.C_T_opt(omega_H, R, L)
    C_M_H = rf.C_M_opt(omega_H, R, L, rf.C_T_opt(omega_H, R, L))
    C_H_text.set_text("opt. C for curret L\n"f"C_T = {1e12*C_T_H:.2f} pF\nC_M = {1e12*C_M_H:.2f} pF")
   
    fig.canvas.draw_idle()

# link sliders to the update function
R_slider.on_changed(update)
L_slider.on_changed(update)
C_T_slider.on_changed(update)
C_M_slider.on_changed(update)

# show the plot
plt.show()