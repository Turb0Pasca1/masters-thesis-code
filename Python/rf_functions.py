# rf_functions.py
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
import os
import skrf as rf

# loads inductor values of a csv table containing the inducatances and resistances in nH and mOhm
def load_inductors():
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'data', 'coilcraft_inductors.csv')
    return np.genfromtxt(data_path, delimiter=';')

def R_solenoid(N, di, do, ds, dw, r, Le, f, Xm=-9.7e-6, mu0=1.257e-6, sigma=59.52e6):
    '''
    returns resistance of a solenoid coil at frequency f and geometric properties
    N:      number of turns
    di:     inner coil diameter in m
    do:     outer coil diameter in m
    ds:     distance between each turn in m
    dw:     wire diameter in m
    r:      radius of conducting wire part in m
    Le:     length of external wires to connect the coil in m
    f:      operating frequency in Hz
    Xm:     magnetic susceptibility of copper
    mu0:    permeability of free space in Vs / A
    sigma:  conductivity of copper in S / m
    '''
    # effective radius of the coil
    reff = di/2 + (do-di)/4
    # pitch of the coil
    p = dw + ds
    # wire length of wound coil
    L = N * np.sqrt(4*(np.pi*reff)**2+p**2)
    # total wire length
    Ltot = Le + L
    mu = mu0*(1+Xm)
    # skin depth
    delta = 1 / np.sqrt(np.pi*f*mu*sigma)
    # coil crossectional area
    A = np.pi*(2*delta*r-delta**2)
    # resistance
    R = Ltot / (sigma*A)
    # resistance without skin effect
    R0 = Ltot / (sigma*np.pi*r**2)
    return {
        'reff': reff,
        'p': p,
        'L': L,
        'Ltot': Ltot,
        'delta': delta,
        'A': A,
        'R': R, 
        'R0': R0
    }

def Z_L(w, L, R):
    '''
    returns complex impedance of an inductor at w
    w:      angular frequency in rad/s
    L:      inductance in H
    R:      resistance of inductor in Ohm 
    '''
    return R + 1j*w*L

def Z_C(w, C):
    '''
    returns complex impedance of a capacitor at w
    w:      angular frequency in rad/s
    C:      capacitance in F
    '''
    return -1j/(w*C)

def LC_par(w, L, R, C):
    '''
    returns total impedance of a tank circuit - parallel LC at w
    w:      angular frequency in rad/s
    L:      inductance in H
    R:      resistance of inductor in Ohm 
    C:      capacitance in F
    '''
    return 1/(1/(Z_C(w, C))+1/(Z_L(w,L,R)))

def Z_par(Z_1, Z_2):
    '''
    returns total impedance of two parallel impedances Z_1 and Z_2
    Z_1:    impedance 1 in Ohm
    Z_2:    impedance 2 in Ohm
    '''
    return 1/(1/Z_1+1/Z_2)

def C_in_par(L, R, f):
    '''
    returns the capacitance of a capacitor in a tank circuit resonating at f
    f:      frequency in MHz
    L:      inductance in H
    R:      resistance of inductor in Ohm
    '''
    w = 2*np.pi*1e6*f
    return L/((w*L)**2+R**2)

def C_in_ser(L, f):
    '''
    returns the capacitance of a capacitor in a serial LC circuit resonating at f
    f:      frequency in Hz
    L:      inductance in H
    '''
    w = 2*np.pi*f
    return 1/(L*w**2)

def Z_M_T_Resonator(w, C_M, C_T, L, RL):
    '''
    returns the total impedance of a single resonant tuned and matched circuit at w
    w:      angular frequency in rad/s
    C_M:    match capacitance in F
    C_T:    tune capacitance in F
    L:      inductance in H
    RL:     resistance of inductor in Ohm  
    '''
    return Z_C(w, C_M) + Z_par(Z_C(w, C_T), Z_L(w, L, RL))

def calculate_L(f, C_T, C_M, R):
    '''
    returns inductance of a single resonant tuned and matched circuit
    f:      resonant frequency in Hz
    C_T:    parallel capacitance (tune capacitance) in F
    C_M:    series capacitance (match capacitance) in F
    R:      resistance of the inductor
    '''
    a = -w**4*C_T**2*C_M-w**3*C_T*C_M
    b = 2*w**2*C_T*C_M+w*C_M
    c = -C_M-R*w**2*C_T**2*C_M-R**2*w*C_T*C_M
    L1 = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    L2 = (-b-np.sqrt(b**2-4*a*c))/(2*a) 
    return [L1, L2]

def Tadanki(w, C_Mhf, C_Mlf, C_Thf, C_Tlf, L_t1, R_t1, C_t1, L_t2, R_t2, C_t2, L_S=2.3e-6, R_S=.36):
    '''
    returns total impedance of a double resonant circuit presented in Tadanki et al 2012 at w
    w:      angular frequency in rad/s
    C_Mhf:  match capacitance for high frequency resonance in F
    C_Mlf:  match capacitance for low frequency resonance in F
    C_Thf:  tune capacitance for high frequency resonance in F
    C_Tlf:  tune capacitance for low frequency resonance in F
    L_t1:   inductance of tank circuit 1 in H
    C_t1:   capacitance of tank circuit 1 in F
    R_t1:   resistance of inductor in tank circuit 1 in Ohm
    L_t2:   inductance of tank circuit 2 in H
    C_t2:   capacitance of tank circuit 2 in F
    R_t2:   resistance of inductor in tank circuit 2 in Ohm
    L_S:    inductance of sample coil in H
    R_S:    resistance of sample coil in Ohm
    '''
    # tank circuits
    Z_t1 = LC_par(w, L_t1, R_t1, C_t1)
    Z_t2 = LC_par(w, L_t2, R_t2, C_t2)
    # divides the circuit into calculable chunks
    Z1 = Z_t1 + Z_C(w, C_Mlf)
    Z2 = Z_par(Z1, Z_C(w, C_Mhf)) 
    Z3 = Z_t2 + Z_C(w, C_Tlf)
    Z4 = 1/(1/(Z_L(w, L_S, R_S))+1/(Z_C(w, C_Thf))+1/(Z3))
    return Z2 + Z4

def Schnall(w, C_1, C_2, C_3, L_1, R_1, L_3, R_3, L_S=2.3e-6, R_S=.2):
    '''
    returns total impedance of a double resonant circuit presented in Schnall et al 1985 at w
    w:      angular frequency in rad/s
    C_1:    capacitance of tank circuit in F
    C_2:    capacitance for serial resonator to reach low frequency resonance in F
    C_3:    capacitance of matching network in F
    L_1:    inductance of tank circuit in H
    R_1:    resistance of inductor in tank circuit in Ohm
    L_3:    inductance of matching network in H
    R_3:    resistance of inductor in matching network in Ohm
    L_S:    inductance of sample coil in H
    R_S:    resistance of sample coil in Ohm
    '''
    Z_M = LC_par(w, L_3, R_3, C_3)
    Z_1 = LC_par(w, L_1, R_1, C_1)
    Z_ser = Z_C(w, C_2) + Z_1 + Z_L(w, L_S, R_S)
    return Z_par(Z_M, Z_ser)

# extracts the matching value of R for a value of L in the list of inductors
# is obsolete since the slider works with indices
def get_R_for_L(L_value, data):
    results = [R for L, R in data if L == L_value]
    return results[0] if results else None

def S_11(Z, Z_0=50):
    '''
    returns S_11 parameter for Z using Z_0
    Z:      impedance in Ohm
    Z_0:    reference impedance in Ohm
    '''
    return (Z-Z_0)/(Z+Z_0)

def RL(Z, Z_0=50):
    '''
    returns return loss of Z using Z_0
    Z:      impedance in Ohm
    Z_0:    reference impedance in Ohm
    '''
    return 20*np.log10(np.abs(S_11(Z, Z_0)))

def RL_from_S(Re, Im):
    '''
    returns return loss of the real and imaginary part of S parameter
    Re:     real part of S parameter
    Im:     imaginary part of S parameter
    '''
    return 20*np.log10(np.abs(Re + 1j*Im))

def Z_from_S(S11, Z_0=50):
    '''
    returns impedance of S parameter
    S11:    S parameter (can be complex)
    Z_0:    reference impedance in Ohm
    '''
    return (Z_0*(1+S11))/(1-S11)

def C_T_opt(omega, R, L, Z_0=50):
    '''
    returns tune capacitance in F for a parallel LC resonator
    that has a resistance of Z_0 at omega
    omega:  angular frequency in rad/s
    R:      resistance of inductor in Ohm
    L:      inductance in H
    Z_0:    resistance of LC at omega
    '''
    return (L*omega - np.sqrt(2)*np.sqrt(R*(L**2*omega**2 + R**2 - Z_0*R))/10)/(omega*(L**2*omega**2 + R**2))

def C_M_opt(omega, R, L, C_T):
    '''
    returns match capacitance in F for a match capacitance in series to a LC resonator
    so that the entire configuration resonates at omega
    omega:  angular frequency in rad/s
    R:      resistance of inductor Ohm
    L:      inductance in H
    C_T:    capacitance of LC in F, likely calculated using C_T_opt()
    '''
    return (-C_T**2*R**2*omega**2 - (C_T*L*omega**2 - 1)**2)/(omega**2*(C_T*L**2*omega**2 + C_T*R**2 - L))

def calculate_q(f, arr, threshold=-3, prominence=1):
    '''
    returns resonant frequency, bandwidth, quality factor and resonant frequency index of return loss data
    data may include many resonant frequencies (dips)
    f:          one dimensional list of frequencies in Hz
    arr:        one dimensional list of return loss data in dB (same length as f)
    threshold:  value to evaluate bandwith at
    prominence: adjust the impact of a frequency dip
    '''
    # function to find peaks
    from scipy.signal import find_peaks
    # inversion of arr to find peaks instead of dips
    # prominence=1 to avoid finding smaller dips in noisy data
    dip_indices, _ = find_peaks(-arr, prominence=prominence)
    if len(dip_indices) < 1:
        raise ValueError('No frequency dip found')
    elif len(dip_indices) == 1:
        dip_freq = f[dip_indices]
        # starts to look at the dip frequency for the frequency where the data rises above the threshold 
        drop_index = dip_indices[0]
        # decreases the frequency as long as the data is below the threshold
        while drop_index > 0 and arr[drop_index] < threshold:
            drop_index -= 1
        # starts to look at the dip frequency for the frequency where the data rises above the threshold 
        rise_index = dip_indices[0]
        # increases the frequency as long as the data is below the threshold
        while rise_index < len(arr) - 1 and arr[rise_index] < threshold:
            rise_index += 1
        # bandwith is the difference between frequencies 
        # where the data rises above the threshold to the left and right of the dip frequency
        left_freq = f[drop_index]
        right_freq = f[rise_index]
        bandwidth = right_freq - left_freq
        q_factor = dip_freq / bandwidth if bandwidth > 0 else 0
        # uses lists to return all values to ensure the same data type as for multiple peaks
        dip_freq = [float(f[dip_indices[0]])]
        bandwidth = [float(right_freq - left_freq)]
        q_factor = [float(dip_freq[0] / bandwidth[0]) if bandwidth[0] > 0 else 0]
        dip_indices = [int(dip_indices[0])]
        return dip_freq, bandwidth, q_factor, dip_indices   
    # similar procedure but looped over all frequency dips
    else:
        sorted_dips = dip_indices[np.argsort(arr[dip_indices])]
        dip_freq = []
        bandwidth = []
        q_factor = []
        i = 0
        for dip_idx in sorted_dips:
            dip_freq.append(f[dip_idx])
            drop_index = dip_idx
            while drop_index > 0 and arr[drop_index] < threshold:
                drop_index -= 1
            rise_index = dip_idx
            while rise_index < len(arr) - 1 and arr[rise_index] < threshold:
                rise_index += 1
            left_freq = f[drop_index]
            right_freq = f[rise_index]
            bandwidth.append(right_freq - left_freq)
            q_factor.append(dip_freq[i] / bandwidth[i] if bandwidth[i] > 0 else 0)
            i += 1
        return dip_freq, bandwidth, q_factor, dip_indices
    
# plots VNA data
def visualize_VNA(datatype, data_fname, plot,  xlim=None, ylimZ=None, ylimRL=None, xticksint=False):

    error_msg_1 = ('ERROR: invalid datatype\n' 
                 'try NanoVNA, magspec, LTSpice or VNA')
    
    error_msg_2 = ('ERROR: invalid datatype\n' 
                 's1p file or csv file')
    
    # get f, S_Re and S_Im from data
    if datatype == 'NanoVNA':
        if os.path.splitext(data_fname)[1].lower() != '.s1p':
            return print(error_msg_2)
        data = pd.read_csv(data_fname, delimiter=' ')
        f = np.array(data)[:,:1][:,0]
        S_Re = np.array(data)[:,1:2][:,0]
        S_Im = np.array(data)[:,2:3][:,0]
        ntw = rf.Network(data_fname)
    elif datatype == 'magspec':
        if os.path.splitext(data_fname)[1].lower() != '.csv':
            return print(error_msg_2)
        data = pd.read_csv(data_fname, delimiter=',', names=['f', 'S'], skiprows=1)
        data['S'] = data['S'].apply(lambda x: complex(x.replace('i', 'j')))
        f = data['f']
        S_Re = np.real(data['S'])
        S_Im = np.imag(data['S'])
        # skrf network creation
        S = data['S']
        freq = rf.Frequency.from_f(f, unit='Hz')
        S_reshaped = np.array(S).reshape(-1, 1, 1)
        ntw = rf.Network(frequency=freq, s=S_reshaped, name='Network')
    elif datatype == 'LTSpice':
        if os.path.splitext(data_fname)[1].lower() != '.csv':
            return print(error_msg_2)
        data = pd.read_csv(data_fname, delimiter='\t', names=['Freq', 'S11'], skiprows=1)
        data['S11'] = data['S11'].apply(lambda x: complex(*map(float, x.split(','))))
        f = data['Freq']
        S_Re = np.real(data['S11'])
        S_Im = np.imag(data['S11'])
        # skrf network creation
        S = data['S11']
        freq = rf.Frequency.from_f(f, unit='Hz')
        S_reshaped = np.array(S).reshape(-1, 1, 1)
        ntw = rf.Network(frequency=freq, s=S_reshaped, name='Network')
    elif datatype == 'VNA':
        if os.path.splitext(data_fname)[1].lower() != '.csv':
            return print(error_msg_2)
        data = pd.read_csv(data_fname, delimiter=' ', skiprows=2)
        data = data.drop('Unnamed: 3', axis=1)
        f = data['freq[Hz]']
        Z_Re = data['re:Trc16_Z<-S11[Ohm]']
        Z_Im = data['im:Trc16_Z<-S11[Ohm]']
        S_Re = np.real(S_11(Z_Re+1j*Z_Im))
        S_Im = np.imag(S_11(Z_Re+1j*Z_Im))
        # skrf network creation
        S = S_Re + 1j*S_Im
        freq = rf.Frequency.from_f(f, unit='Hz')
        S_reshaped = S.reshape(-1, 1, 1)
        ntw = rf.Network(frequency=freq, s=S_reshaped, name='Network')
    else:
        return print(error_msg_1)

    Z_Re = np.real(Z_from_S(S_Re+1j*S_Im))
    Z_Im = np.imag(Z_from_S(S_Re+1j*S_Im))

    if plot == 'Z_and_RL':
        fig, ax = plt.subplots(ncols=2)
        ax[0].plot(1e-6*f, Z_Re, label=r'$\Re(Z)$')
        ax[0].plot(1e-6*f, Z_Im, label=r'$\Im(Z)$')
        ax[0].legend()
        ax[0].set_xlabel(r'$f$ / MHz')
        ax[0].set_ylabel(r'$Z$ / $\Omega$')
        ax[0].set_xlim(xlim)
        ax[0].set_ylim(ylimZ)

        ax[1].plot(1e-6*f, RL_from_S(S_Re, S_Im), label=r'$RL$')
        ax[1].set_xlabel(r'$f$ / MHz')
        ax[1].set_ylabel(r'$RL$ / dB')
        ax[1].set_xlim(xlim)
        ax[1].set_ylim(ylimRL)

        if xticksint:
            ax[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
            ax[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

        dip_freq, df, q_factor, idx = calculate_q(1e-6*f, RL_from_S(S_Re, S_Im))  
        q_texts = []
        for i in range(len(dip_freq)):
            if len(dip_freq) < 2:
                # calculates fitted Q factor -> only works for single res data
                # https://scikit-rf.readthedocs.io/en/latest/api/generated/skrf.qfactor.Qfactor.fit.html
                fit = rf.qfactor.Qfactor(ntw, 'reflection', Q_L0=q_factor[i]).fit(method='NLQFIT6')
                Q_fit = fit['Q_L']
                # calculates fitted S parameters -> only works for single res data
                # https://scikit-rf.readthedocs.io/en/latest/api/generated/skrf.qfactor.Qfactor.fitted_s.html
                S_fit = rf.qfactor.Qfactor(ntw, 'reflection', Q_L0=q_factor[i]).fitted_s(opt_res=fit)
                q_str = f'Q = {q_factor[i]:.2f}\nQfit = {Q_fit:.2f}\n@f$_0$ = {dip_freq[i]:.2f} MHz'
                text = ax[1].text(
                    dip_freq[i] + 0.1,
                    RL_from_S(S_Re, S_Im)[idx[i]],
                    q_str,
                    fontsize=10,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8)
                )
                # plot fitted RL curve
                ax[1].plot(1e-6*f, RL_from_S(np.real(S_fit), np.imag(S_fit)), label=r'$RL$ fit', linestyle='--')
                ax[1].legend()
            else:
                q_str = f'Q = {q_factor[i]:.2f}\n@f$_0$ = {dip_freq[i]:.2f} MHz'
                text = ax[1].text(
                    dip_freq[i] + 0.1,
                    RL_from_S(S_Re, S_Im)[idx[i]],
                    q_str,
                    fontsize=10,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8)
                )
            q_texts.append(text)
            print(f'df = {1e3*df[i]:.2f} kHz')  
            
    elif plot == 'RL':
        fig, ax = plt.subplots()

        ax.plot(1e-6*f, RL_from_S(S_Re, S_Im), label=r'$RL$')
        ax.set_xlabel(r'$f$ / MHz')
        ax.set_ylabel(r'$RL$ / dB')
        ax.set_xlim(xlim)
        ax.set_ylim(ylimRL)

        if xticksint:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

        dip_freq, df, q_factor, idx = calculate_q(1e-6*f, RL_from_S(S_Re, S_Im))  
        q_texts = []
        for i in range(len(dip_freq)):
            if len(dip_freq) < 2:
                # calculates fitted Q factor -> only works for single res data
                # https://scikit-rf.readthedocs.io/en/latest/api/generated/skrf.qfactor.Qfactor.fit.html
                fit = rf.qfactor.Qfactor(ntw, 'reflection', Q_L0=q_factor[i]).fit(method='NLQFIT6')
                Q_fit = fit['Q_L']
                # calculates fitted S parameters -> only works for single res data
                # https://scikit-rf.readthedocs.io/en/latest/api/generated/skrf.qfactor.Qfactor.fitted_s.html
                S_fit = rf.qfactor.Qfactor(ntw, 'reflection', Q_L0=q_factor[i]).fitted_s(opt_res=fit)
                q_str = f'Q = {q_factor[i]:.2f}\nQfit = {Q_fit:.2f}\n@f$_0$ = {dip_freq[i]:.2f} MHz'
                text = ax.text(
                    dip_freq[i] + 0.1,
                    RL_from_S(S_Re, S_Im)[idx[i]],
                    q_str,
                    fontsize=10,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8)
                )
                # plot fitted RL curve
                ax.plot(1e-6*f, RL_from_S(np.real(S_fit), np.imag(S_fit)), label=r'$RL$ fit', linestyle='--')
                ax.legend()
            else:
                q_str = f'Q = {q_factor[i]:.2f}\n@f$_0$ = {dip_freq[i]:.2f} MHz'
                text = ax.text(
                    dip_freq[i] + 0.1,
                    RL_from_S(S_Re, S_Im)[idx[i]],
                    q_str,
                    fontsize=10,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8)
                )
            q_texts.append(text)
            print(f'df = {1e3*df[i]:.2f} kHz')      
    else:
        return print('No valid template to plot from')

# plots VNA data
# different from visualize_VNA the data is not in the form of S_11  and
# not imported from external files
# used to plot simulation results
def visualize_Z_RL(f, Z, xlim=None, ylimZ=None, ylimRL=None, xticksint=False):
    f_Na = 5.694
    f_H = 21.527
    fig, ax = plt.subplots(ncols=2)
    
    ax[0].plot(1e-6*f, Z.real, label=r'$\Re(Z)$')
    ax[0].plot(1e-6*f, Z.imag, label=r'$\Im(Z)$')
    ax[0].axvline(x=f_Na, color='grey', linestyle=':', linewidth=2, label=r'$f=5.694\,$MHz')
    ax[0].axvline(x=f_H, color='black', linestyle=':', linewidth=2, label=r'$f=21.527\,$MHz')
    ax[0].legend(loc= 'upper right')
    ax[0].set_xlabel(r'$f$ / MHz')
    ax[0].set_ylabel(r'$Z$ / $\Omega$')
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylimZ)

    ax[1].plot(1e-6*f, RL(Z), label=r'$RL$')
    ax[1].axvline(x=f_Na, color='grey', linestyle=':', linewidth=2, label=r'$f=5.694\,$MHz')
    ax[1].axvline(x=f_H, color='black', linestyle=':', linewidth=2, label=r'$f=21.527\,$MHz')
    ax[1].legend(loc= 'upper right')
    ax[1].set_xlabel(r'$f$ / MHz')
    ax[1].set_ylabel(r'$RL$ / dB')
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylimRL)

    if xticksint:
        ax[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
        ax[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

    dip_freq, df, q_factor, idx = calculate_q(1e-6*f, RL(Z))  
    q_texts = []
    for i in range(len(dip_freq)):
        q_str = f'Q = {q_factor[i]:.2f}\n@f$_0$ = {dip_freq[i]:.2f} MHz'
        text = ax[1].text(
            dip_freq[i] + 0.1,
            RL(Z)[idx[i]],
            q_str,
            fontsize=8,
            color='black',
            bbox=dict(facecolor='white', alpha=0.8)
        )
        q_texts.append(text)

    return fig, ax
