# image_functions.py
import numpy as np
from matplotlib import pyplot as plt
import os
import re
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label

# load complex image data in numpy array
# sub folders of measurement data should be located in the structure
# -- dir xy
# ---- image_functions.py
# -- data
# ---- measurements
# ------ dataname
def load_data(dataname):
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'data', 'measurements', dataname)
    with open(data_path, 'r') as f:
        raw_data = f.read()
        # numbers in format 1.683422e-08-9.779471e-09i or 1.2e+01+3.4e+00i
        complex_strings = re.findall(r'[-+]?\d*\.\d+e[+-]?\d+[-+]\d*\.\d+e[+-]?\d+i', raw_data)
        complex_numbers = [complex(s.replace('i', 'j')) for s in complex_strings]
    return np.array(complex_numbers)

# create a circular region of interest
def circular_ROI(image_params, radius, center=None):
    '''
    returns mask of True and False in size of nread * nphase
    image_params:       list of [nread, nphase, FOVread in mm, FOVphase in mm]
    radius:             ROI radius in mm
    center:             optional 
                        center of ROI in mm as a list in the form of [read, phase]
    '''
    # unpack image_params
    nread, nphase, FOVread, FOVphase = image_params
    # coordinate mesh
    read = np.linspace(-FOVread/2, FOVread/2, nread)
    phase = np.linspace(-FOVphase/2, FOVphase/2, nphase)
    READ, PHASE = np.meshgrid(read, phase)
    # compute center of image if no center is specified for the ROI
    if center is None:
        cread, cphase = 0, 0
    else: 
        cread, cphase = center
        cphase = -cphase
    # transform cartesian grid to vector magnitudes to center
    transform = (READ - cread)**2 + (PHASE - cphase)**2
    # create mask of ROI for all points inside radius
    ROI = transform <= radius**2
    return ROI

# create a ring formed region of interest
def ring_ROI(image_params, radius, center=None):
    '''
    returns mask of True and False in size of nread * nphase
    image_params:       list of [nread, nphase, FOVread in mm, FOVphase in mm]
    radius:             list of ROI radii in mm [inner radius, outer radius]
    center:             optional 
                        center of ROI in mm as a list in the form of [read, phase]
    '''
    # unpack image_params
    nread, nphase, FOVread, FOVphase = image_params
    # unpack ring radii
    iradius, oradius = radius
    # coordinate mesh
    read = np.linspace(-FOVread/2, FOVread/2, nread)
    phase = np.linspace(-FOVphase/2, FOVphase/2, nphase)
    READ, PHASE = np.meshgrid(read, phase)
    # compute center of image if no center is specified for the ROI
    if center is None:
        cread, cphase = 0, 0
    else:
        cread, cphase = center
        cphase = -cphase
    # transform cartesian grid to vector magnitudes to center
    transform = (READ - cread)**2 + (PHASE - cphase)**2
    # create mask of ROI for all points outside iradius and inside oradius 
    ROI = (transform >= iradius**2) & (transform <= oradius**2)
    return ROI

# semi automatic ROI finding
# works best with phase image
def find_ROI(image, lthreshold, uthreshold, sigma=.4):
    '''
    returns mask of True and False in size of image
    image:          2d gray scale image as numpy array
    lthreshold:     lower threshold of image amplitude
    uthreshold:     upper threshold of image amplitude
    sigma:          optional, default =.4
                    width of gaussian filter smoothening
    '''
    # threshold mask
    binary_mask = (image >= lthreshold) & (image <= uthreshold)
    # smoothen result with Gauss filter
    # beneficial as most ROI are round
    circular_mask = gaussian_filter(binary_mask.astype(float), sigma=sigma)
    circular_mask = circular_mask > 0.1
    # remove smaller regions outside main ROI
    # check for connected regions
    labeled_mask, _ = label(circular_mask)
    # largest region (= most pixels)
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0  
    largest_label = sizes.argmax()
    ROI = labeled_mask == largest_label
    return ROI

def get_noise(noise_data, image_params, ROI_fix_params=None):
    '''
    plots noise image and histogram
    fits Gauss distribution to histogram
    returns RMS of complex std
    
    noise_image:        complex 2d array of noise measurement in image dimensions
    image_params:       list of [nread, nphase, FOVread in mm, FOVphase in mm]
    ROI_fix_params:     optional
                        list of circular ROI parameters to calculate std of
                        radius:             ROI radius in mm, 
                                            if radius is a list [inner radius, outer radius]
                                            ring_ROI is used
                        center:             optional 
                                            center of ROI in mm as a list in the form of [read, phase]
    '''
    # reshape data to correct dimensions
    noise_image = np.array(noise_data).reshape((image_params[1], image_params[0]))

    # convert to nT
    noise_image *= 1e9

    noise_image_real = np.real(noise_image)
    noise_image_imag = np.imag(noise_image)
    # used for imshow to scale image axis to mm instead of samples
    # image center at (0,0)
    extent = [-image_params[2]/2, image_params[2]/2, -image_params[3]/2, image_params[3]/2]

    fig, ax = plt.subplots(ncols=2, nrows=2)
    
    # if no ROI is defined, the std is calculated for the entire image
    # images
    ax[0][0].grid(False, which='both')
    ax[1][0].grid(False, which='both')
    # real part of image 
    h_noise_image_real = ax[0][0].imshow(noise_image_real, extent=extent, cmap='gray')
    ax[0][0].set_title('real Noise', pad=10)
    ax[0][0].set_xlabel('z read / mm')
    ax[0][0].set_ylabel('-x phase / mm')
    fig.colorbar(h_noise_image_real, ax=ax[0][0])
    # imag part of image
    h_noise_image_imag = ax[1][0].imshow(noise_image_imag, extent=extent, cmap='gray')
    ax[1][0].set_title('imag Noise', pad=10)
    ax[1][0].set_xlabel('z read / mm')
    ax[1][0].set_ylabel('-x phase / mm')
    fig.colorbar(h_noise_image_imag, ax=ax[1][0])
    
    # histograms
    noise_data_real = noise_image_real.flatten()
    noise_data_imag = noise_image_imag.flatten()
    # real part histogram
    _, bins_real, _ = ax[0][1].hist(noise_data_real, bins=100, density=True, alpha=0.3, color='k')
    bin_width_real = bins_real[1] - bins_real[0]
    mu_real, std_real = norm.fit(noise_data_real)
    x = np.linspace(bins_real[0], bins_real[-1], 500)
    p = norm.pdf(x, mu_real, std_real)
    ax[0][1].plot(x, p, 'k', label=fr"$\mu={mu_real:.2f}$, $\sigma={std_real:.2f}$")
    ax[0][1].set_xlabel('Amplitude / nT')
    ax[0][1].set_ylabel('PDF')
    ax[0][1].set_title(r'real Noise Histogram', pad=10)
    ax[0][1].legend()
    
    # imag part histogram
    _, bins_imag, _ = ax[1][1].hist(noise_data_imag, bins=100, density=True, alpha=0.3, color='k')
    bin_width_imag = bins_imag[1] - bins_imag[0]
    mu_imag, std_imag = norm.fit(noise_data_imag)
    x = np.linspace(bins_imag[0], bins_imag[-1], 500)
    p = norm.pdf(x, mu_imag, std_imag)
    ax[1][1].plot(x, p, 'k', label=fr"$\mu={mu_imag:.2f}$, $\sigma={std_imag:.2f}$")
    ax[1][1].set_xlabel('Amplitude / nT')
    ax[1][1].set_ylabel('PDF')
    ax[1][1].set_title(r'imag Noise Histogram', pad=10)
    ax[1][1].legend()

    total_noise = np.sqrt((std_real**2+std_imag**2)/2)
    # if ROI is specified
    if ROI_fix_params:
        # unpack ROI parameters
        radius, *rest = ROI_fix_params
        # if no center is specified
        center = rest[0] if rest else None
        # ROI selection
        if isinstance(radius, (list, tuple)):
            ROI = ring_ROI(image_params, radius=radius, center=center)
        else:
            ROI = circular_ROI(image_params, radius=radius, center=center)

        # grid for plotting the ROI
        read = np.linspace(-image_params[2]/2, image_params[2]/2, image_params[0])  
        phase = np.linspace(-image_params[3]/2, image_params[3]/2, image_params[1])  
        READ, PHASE = np.meshgrid(read, phase)

        ax[0][0].contour(READ, -PHASE, ROI, colors='tab:blue')
        ax[1][0].contour(READ, -PHASE, ROI, colors='tab:orange')

        noise_data_real_roi = noise_image_real[ROI].flatten()
        noise_data_imag_roi = noise_image_imag[ROI].flatten()

        roi_counts_real, _ = np.histogram(noise_data_real_roi, bins=bins_real)
        roi_pdf_real = roi_counts_real / (len(noise_data_real) * bin_width_real)
        ax[0][1].bar(bins_real[:-1], roi_pdf_real, width=bin_width_real, align='edge', alpha=0.6, color='tab:blue', label='ROI')
        # fit Gauss to histogram
        mu_real_roi, std_real_roi = norm.fit(noise_data_real_roi)
        x = np.linspace(bins_real[0], bins_real[-1], 500)
        scaling_factor_real = len(noise_data_real_roi) / len(noise_data_real)
        pdf_fit_scaled_real = norm.pdf(x, mu_real_roi, std_real_roi) * scaling_factor_real
        ax[0][1].plot(x, pdf_fit_scaled_real, color='tab:blue', linestyle='--', label=fr"$\mu={mu_real_roi:.2f}$, $\sigma={std_real_roi:.2f}$")
        ax[0][1].legend()
        
        roi_counts_imag, _ = np.histogram(noise_data_imag_roi, bins=bins_imag)
        roi_pdf_imag = roi_counts_imag / (len(noise_data_imag) * bin_width_imag)
        ax[1][1].bar(bins_imag[:-1], roi_pdf_imag, width=bin_width_imag, align='edge', alpha=0.6, color='tab:orange', label='ROI')
        # fit Gauss to histogram
        mu_imag_roi, std_imag_roi = norm.fit(noise_data_imag_roi)
        x = np.linspace(bins_imag[0], bins_imag[-1], 500)
        scaling_factor_imag = len(noise_data_imag_roi) / len(noise_data_imag)
        pdf_fit_scaled_imag = norm.pdf(x, mu_imag_roi, std_imag_roi) * scaling_factor_imag
        ax[1][1].plot(x, pdf_fit_scaled_imag, color='tab:orange', linestyle='--', label=fr"$\mu={mu_imag_roi:.2f}$, $\sigma={std_imag_roi:.2f}$")
        ax[1][1].legend()

        total_noise_roi = np.sqrt((std_real_roi**2+std_imag_roi**2)/2)
        return [total_noise, total_noise_roi]
    return [total_noise]

def image_eval(data, image_params, ROI_fix_params, ROI_fit_params, ROI_noise_params, fit=False):
    '''
    plots signal image and histogram
    fits Gauss distribution to histogram
    signal and noise data, defined by two different ROIs
    returns total noise, total mean signal, total SNR

    signal_image:       complex 2d array of signal measurement in image dimensions
    image_params:       list of [nread, nphase, FOVread in mm, FOVphase in mm]
    ROI_fix_params:     list of ROI parameters 
                        radius:             ROI radius in mm, 
                                            if radius is a list [inner radius, outer radius]
                                            ring_ROI is used
                        center:             optional 
                                            center of ROI in mm as a list in the form of [read, phase]
    ROI_fit_params:     list of ROI fit parameters
                        lthreshold:         lower threshold of image amplitude
                        uthreshold:         upper threshold of image amplitude
                        sigma:              optional, default =.4
                                            width of gaussian filter smoothening
    ROI_noise_params:   list of circular ROI parameters to calculate std of
                        radius:             ROI radius in mm
                        center:             optional 
                                            center of ROI in mm as a list in the form of [read, phase]
    fit:                Bool if fitting ROI is tried                 
    '''

    # reshape data to correct dimensions
    signal_image = np.array(data).reshape((image_params[1], image_params[0]))

    # convert to nT
    signal_image *= 1e9

    # calculate real, imag, magnitude and phase of complex data
    signal_image_real = np.real(signal_image)
    signal_image_imag = np.imag(signal_image)
    signal_image_abs = np.abs(signal_image)
    signal_image_phase = np.angle(signal_image)

    # used for imshow to scale image axis to mm instead of samples
    extent = [-image_params[2]/2, image_params[2]/2, -image_params[3]/2, image_params[3]/2]

    # grid for plotting the ROI
    read = np.linspace(-image_params[2]/2, image_params[2]/2, image_params[0])  
    phase = np.linspace(-image_params[3]/2, image_params[3]/2, image_params[1])  
    READ, PHASE = np.meshgrid(read, phase)

    radius, *rest = ROI_fix_params
    center = rest[0] if rest else None
    
    lthreshold, uthreshold, sigma = ROI_fit_params

    radius_noise, *rest_noise = ROI_noise_params
    center_noise = rest[0] if rest else None

    # ROI selection
    if isinstance(radius, (list, tuple)):
        ROI_fixed = ring_ROI(image_params, radius=radius, center=center)
    else:
        ROI_fixed = circular_ROI(image_params, radius=radius, center=center)

    ROI_fit = find_ROI(signal_image_phase, lthreshold=lthreshold, uthreshold=uthreshold, sigma=sigma)
    ROI = [ROI_fixed, ROI_fit]
    ROI_noise = ~circular_ROI(image_params, radius=radius_noise, center=center_noise)

    # data for different plots
    image       = [signal_image_real, signal_image_imag, signal_image_abs, signal_image_phase]
    title_image = ['real Signal', 'imag Signal', 'abs Signal', 'phase Signal']
    signal_data = [signal_image_real.flatten(), signal_image_imag.flatten(), np.abs(signal_image).flatten(), np.angle(signal_image).flatten()]
    title_hist  = ['real Signal Histogram', 'imag Signal Histogram', 'abs Signal Histogram', 'phase Signal Histogram']
    hist_xlabel = ['Amplitude / nT', 'Amplitude / nT', 'Amplitude / nT', 'Phase / rad']
    label_ROI   = ['ROI fixed', 'ROI fit']
    signal_roi  = [[signal_image_real[ROI_fixed].flatten(), signal_image_imag[ROI_fixed].flatten(), np.abs(signal_image)[ROI_fixed].flatten(), np.angle(signal_image)[ROI_fixed].flatten()],
                [signal_image_real[ROI_fit].flatten(), signal_image_imag[ROI_fit].flatten(), np.abs(signal_image)[ROI_fit].flatten(), np.angle(signal_image)[ROI_fit].flatten()]]
    noise_roi   = [signal_image_real[ROI_noise].flatten(), signal_image_imag[ROI_noise].flatten()]
    color       = [['tab:blue', 'tab:orange', 'tab:green', 'tab:red'],
                ['tab:cyan', 'tab:red', 'tab:olive', 'tab:pink']]

    std_noise_list = []
    # plots
    fig, ax = plt.subplots(ncols=2, nrows=4)

    # loop for each subplot row (real, imag, abs, phase) 
    for i in np.arange(4):
        # disable grid for images
        ax[i][0].grid(False, which='both')
        # add handle to image, to plot a colorbar
        # plot image
        h_image = ax[i][0].imshow(image[i], extent=extent, cmap='gray')
        fig.colorbar(h_image, ax=ax[i][0])
        ax[i][0].set_title(title_image[i], pad=10)
        ax[i][0].set_xlabel('z read / mm')
        ax[i][0].set_ylabel('-x phase / mm')

        # plot full histogram
        full_data = signal_data[i]
        counts, bins, _ = ax[i][1].hist(full_data, bins=100, density=True, alpha=0.6, color='gray')
        bin_width = bins[1] - bins[0]
        ax[i][1].set_xlabel(hist_xlabel[i])
        ax[i][1].set_ylabel('PDF')
        ax[i][1].set_title(title_hist[i], pad=10)

        # plot ROI
        if fit:
            fit_h = 2
        else:
            fit_h = 1
        for j in np.arange(fit_h):
            roi = ROI[j]
            ax[i][0].contour(READ, -PHASE, roi, colors=color[j][i])
            roi_data = signal_roi[j][i]
            # calculate mean signal amplitude of ROI
            # same value as by fitting norm.fit to the histogram data
            mean = np.mean(signal_roi[j][i])
            # label position
            if fit:
                y_pos = 0.2-0.13*j
            else:
                y_pos = 0.2-0.13
            ax[i][0].text(
                0.05, y_pos,              
                fr"mean = {mean:.2f}",             
                transform=ax[i][0].transAxes,   
                fontsize=10,
                color=color[j][i],
                bbox=dict(facecolor='white', alpha=1) 
            )
            # rescale roi histogram to original histogram
            roi_counts, _ = np.histogram(roi_data, bins=bins)
            roi_pdf = roi_counts / (len(full_data) * bin_width)
            ax[i][1].bar(bins[:-1], roi_pdf, width=bin_width, align='edge', alpha=0.3, color=color[j][i], label=label_ROI[j])
            # fit Gauss to histogram
            mu, std = norm.fit(roi_data)
            x = np.linspace(bins[0], bins[-1], 500)
            scaling_factor = len(roi_data) / len(full_data)
            pdf_fit_scaled = norm.pdf(x, mu, std) * scaling_factor
            ax[i][1].plot(x, pdf_fit_scaled, color=color[j][i], linestyle='--',
                        label=fr"$\mu={mu:.2f}$, $\sigma={std:.2f}$")
            if fit and (i==3):
                ax[i][1].vlines([lthreshold, uthreshold], 0, ax[i][1].get_ylim()[1], color=color[1][i], linewidth=1, linestyle='-.')
        # noise
        if i < 2:
            noise_data = noise_roi[i]
            ax[i][0].contour(READ, -PHASE, ROI_noise, colors='k')
            noise_counts, _ = np.histogram(noise_data, bins=bins)
            noise_pdf = noise_counts / (len(full_data) * bin_width)
            ax[i][1].bar(bins[:-1], noise_pdf, width=bin_width, align='edge', alpha=0.5, color='k', label='ROI noise')
            # fit Gauss to histogram
            mu_noise, std_noise = norm.fit(noise_data)
            x = np.linspace(bins[0], bins[-1], 500)
            scaling_factor_noise = len(noise_data) / len(full_data)
            pdf_fit_scaled_noise = norm.pdf(x, mu_noise, std_noise) * scaling_factor_noise
            ax[i][1].plot(x, pdf_fit_scaled_noise, color='k', linestyle='--',
                        label=fr"$\mu={mu_noise:.2f}$, $\sigma={std_noise:.2f}$")
            std_noise_list.append(std_noise)
        ax[i][1].legend(fontsize=8)

    total_mean = [np.abs(np.mean(signal_image_real[ROI_fixed].flatten() + 1j*signal_image_imag[ROI_fixed].flatten())),
                  np.abs(np.mean(signal_image_real[ROI_fit].flatten() + 1j*signal_image_imag[ROI_fit].flatten()))]
    total_noise = np.sqrt((std_noise_list[0]**2+std_noise_list[1]**2)/2)
    snr = total_mean / total_noise
    return [total_noise, total_mean, snr]

def abs_image(data, image_params):

    # reshape data to correct dimensions
    signal_image = np.array(data).reshape((image_params[1], image_params[0]))

    # convert to nT
    signal_image *= 1e9

    # used for imshow to scale image axis to mm instead of samples
    extent = [-image_params[2]/2, image_params[2]/2, -image_params[3]/2, image_params[3]/2]
    fig, ax = plt.subplots()
    # disable grid for images
    ax.grid(False, which='both')
    # add handle to image, to plot a colorbar
    image = np.abs(signal_image)
    # plot image
    h_image = ax.imshow(image, extent=extent, cmap='gray')
    fig.colorbar(h_image, ax=ax)
    ax.set_title('abs Signal', pad=10)
    ax.set_xlabel('z read / mm')
    ax.set_ylabel('-x phase / mm')