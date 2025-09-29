# image_functions.py
import numpy as np
from matplotlib import pyplot as plt
import os
import re
from scipy.stats import norm, sem
from scipy.optimize import curve_fit
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
    with scipy.stats.norm.fit and
    scipy.optimize.curve_fit
    returns RMS of complex std and uncertainty
    
    noise_data:         complex noise measurement data
    image_params:       list of [nread, nphase, FOVread in mm, FOVphase in mm]
    ROI_fix_params:     optional
                        list of circular ROI parameters to calculate std of
                        radius:             ROI radius in mm, 
                                            if radius is a list [inner radius, outer radius]
                                            ring_ROI is used
                        center:             optional 
                                            center of ROI in mm as a list in the form of [read, phase]
    '''
    # convert to nT
    noise_data *= 1e9

    # reshape data to correct image dimensions
    noise_image = np.array(noise_data).reshape((image_params[1], image_params[0]))

    # used for imshow to scale image axis to mm instead of samples
    # image center at (0,0)
    extent      = [-image_params[2]/2, image_params[2]/2, -image_params[3]/2, image_params[3]/2]

    noise_image = [np.real(noise_image), np.imag(noise_image)]
    noise_data  = [noise_image[0].flatten(), noise_image[1].flatten()]
    image_title = ['real Noise', 'imag Noise']
    hist_title  = ['real Noise Histogram', 'imag Noise Histogram']

    total_noise = []

    # if ROI is specified
    if ROI_fix_params:
        # unpack ROI parameters
        radius, *rest   = ROI_fix_params
        # if no center is specified
        center          = rest[0] if rest else None
        # ROI selection
        if isinstance(radius, (list, tuple)):
            ROI = ring_ROI(image_params, radius=radius, center=center)
        else:
            ROI         = circular_ROI(image_params, radius=radius, center=center)

        # grid for plotting the ROI
        read        = np.linspace(-image_params[2]/2, image_params[2]/2, image_params[0])  
        phase       = np.linspace(-image_params[3]/2, image_params[3]/2, image_params[1])  
        READ, PHASE = np.meshgrid(read, phase)

        colors      = ['tab:blue', 'tab:orange']
        colors2     = ['blue', 'orange']
        colors3     = ['cyan', 'red']

    fig, ax = plt.subplots(ncols=2, nrows=2)
    
    for i in np.arange(2):

        ax[i][0].grid(False, which='both')
        h_image = ax[i][0].imshow(noise_image[i], extent=extent, cmap='gray')
        ax[i][0].set_title(image_title[i], pad=10)
        ax[i][0].set_xlabel('z read / mm')
        ax[i][0].set_ylabel('-x phase / mm')
        fig.colorbar(h_image, ax=ax[i][0])
        
        # histogram (PDF normalized)
        n, bins, _ = ax[i][1].hist(noise_data[i], bins=100, density=True, alpha=0.3, color='k')
        bin_width = bins[1] - bins[0]
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # scipy.stats.norm.fit
        # calculate mu and std with scipy.stats.norm (does not include uncertainty calculation)
        # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit
        # Returns
        #   Estimates for any shape parameters (if applicable), 
        #   followed by those for location and scale. 
        #   For most random variables, shape statistics will be returned, 
        #   but there are exceptions (e.g. norm).
        mu, std = norm.fit(noise_data[i])
        x = np.linspace(bins[0], bins[-1], 200)
        p = norm.pdf(x, mu, std)
        ax[i][1].plot(x, p, 'k', label=fr'norm.fit $\mu={mu:.2f}$nT, $\sigma={std:.2f}$nT')

        # scipy.optimize.curve_fit
        pars, cov = curve_fit(
            lambda x, mu, std: norm.pdf(x, loc=mu, scale=std),
            bin_centers, n, p0=[0, 1]
        )
        # extract uncertainties from covarience matrix
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        mu2, umu2, std2, ustd2 = pars[0], np.sqrt(cov[0,0]), pars[1], np.sqrt(cov[1,1])
        ax[i][1].plot(x, norm.pdf(x, *pars), color='gray', linestyle='dotted',
                      label=fr'curve_fit $\mu=({mu2:.2f}\pm{umu2:.2f})$nT, $\sigma=({std2:.2f}\pm{ustd2:.2f})$nT')

        ax[i][1].set_xlabel('Amplitude / nT')
        ax[i][1].set_ylabel('PDF')
        ax[i][1].set_title(hist_title[i], pad=10)

        total_noise.append([[mu, std], [mu2, umu2, std2, ustd2]])

        if ROI_fix_params:
            ax[i][0].contour(READ, -PHASE, ROI, colors=colors[i])
            noise_data_roi = noise_image[i][ROI].flatten()

            roi_counts, _ = np.histogram(noise_data_roi, bins=bins)

            # scale to fit selected pixels of original histogram
            scaling_factor = len(noise_data_roi) / len(noise_data[i])
            roi_pdf = roi_counts / (len(noise_data[i]) * bin_width)

            # plot scaled ROI histogram
            ax[i][1].bar(bins[:-1], roi_pdf, width=bin_width, align='edge',
                        alpha=0.6, color=colors[i], label='ROI')

            # fit Gauss with norm.fit
            mu_roi, std_roi = norm.fit(noise_data_roi)
            pdf_fit_scaled = norm.pdf(x, mu_roi, std_roi) * scaling_factor
            ax[i][1].plot(x, pdf_fit_scaled, color=colors2[i], linestyle='-',
                        label=fr'ROI norm.fit $\mu={mu_roi:.2f}$nT, $\sigma={std_roi:.2f}$nT')

            # fit Gauss with curve_fit 
            pars_roi, cov_roi = curve_fit(
                lambda xx, mu, std: norm.pdf(xx, mu, std) * scaling_factor,
                bin_centers, roi_pdf, p0=[0, 1]
            )
            mu2_roi, umu2_roi, std2_roi, ustd2_roi = (
                pars_roi[0], np.sqrt(cov_roi[0,0]),
                pars_roi[1], np.sqrt(cov_roi[1,1])
            )
            ax[i][1].plot(x, norm.pdf(x, *pars_roi) * scaling_factor, linestyle='dotted', color=colors3[i],
                        label=fr'ROI curve_fit $\mu=({mu2_roi:.2f}\pm{umu2_roi:.2f})$nT, $\sigma=({std2_roi:.2f}\pm{ustd2_roi:.2f})$nT')

            total_noise.append([[mu_roi, std_roi], [mu2_roi, umu2_roi, std2_roi, ustd2_roi]])
        
        # adjust legend position
        handles_row0, labels_row0 = [], []
        handles_row1, labels_row1 = [], []
        for a in ax[0]:
            h, l = a.get_legend_handles_labels()
            handles_row0.extend(h)
            labels_row0.extend(l)
        for a in ax[1]:
            h, l = a.get_legend_handles_labels()
            handles_row1.extend(h)
            labels_row1.extend(l)
        fig.subplots_adjust(right=0.75)
        fig.legend(handles_row0, labels_row0, loc='center left', bbox_to_anchor=(0.9, 0.85), ncol=1)
        fig.legend(handles_row1, labels_row1, loc='center left', bbox_to_anchor=(0.9, 0.35), ncol=1)

    # return correct RMS values of noise parameters
    RMS_noise = []
    # scipy.stats.norm.fit
    RMS_noise.append(np.sqrt(np.array(total_noise[0][0])**2 + np.array(total_noise[2][0])**2)) # RMS norm: [mu, std]
    RMS_noise.append(np.sqrt(np.array(total_noise[1][0])**2 + np.array(total_noise[3][0])**2)) # RMS norm ROI: [mu, std]
    # scipy.optimize.curve_fit
    # full image
    RMS_curve_mu = np.sqrt(total_noise[0][1][0]**2 + total_noise[2][1][0]**2) 
    # propagation of uncertainty for values determined with curve fit
    u_RMS_curve_mu = np.sqrt((total_noise[0][1][0]/RMS_curve_mu*total_noise[0][1][1])**2+(total_noise[2][1][0]/RMS_curve_mu*total_noise[2][1][1])**2)
    RMS_curve_std = np.sqrt(total_noise[0][1][2]**2 + total_noise[2][1][2]**2)             
    u_RMS_curve_std = np.sqrt((total_noise[0][1][2]/RMS_curve_std*total_noise[0][1][3])**2+(total_noise[2][1][2]/RMS_curve_std*total_noise[2][1][3])**2)
    RMS_noise.append([RMS_curve_mu, u_RMS_curve_mu, RMS_curve_std, u_RMS_curve_std])
    # ROI
    RMS_curve_mu_ROI = np.sqrt(total_noise[1][1][0]**2 + total_noise[3][1][0]**2) 
    u_RMS_curve_mu_ROI = np.sqrt((total_noise[1][1][0]/RMS_curve_mu_ROI*total_noise[1][1][1])**2+(total_noise[3][1][0]/RMS_curve_mu_ROI*total_noise[3][1][1])**2)
    RMS_curve_std_ROI = np.sqrt(total_noise[1][1][2]**2 + total_noise[3][1][2]**2)             
    u_RMS_curve_std_ROI = np.sqrt((total_noise[1][1][2]/RMS_curve_std_ROI*total_noise[1][1][3])**2+(total_noise[3][1][2]/RMS_curve_std_ROI*total_noise[3][1][3])**2)
    RMS_noise.append([RMS_curve_mu_ROI, u_RMS_curve_mu_ROI, RMS_curve_std_ROI, u_RMS_curve_std_ROI])

    # RMS norm:         [[mu, std],
    # RMS norm ROI:     [mu, std]
    # RMS curve:        [mu, umu, std, ustd],
    # RMS curve ROI:    [mu, umu, std, ustd]]

    # std, ustd for ROI using curve fit
    print(f'std = ({RMS_noise[-1][2]:.2f} +- {RMS_noise[-1][3]:.2f})nT')

    return RMS_noise

def image_eval(data, image_params, ROI_fix_params, ROI_fit_params, ROI_noise_params, fit=False):
    '''
    plots signal image and histogram
    fits Gauss distribution to histogram with scipy.stats.norm.fit
    signal and noise data can be defined by two different ROIs
    returns total noise, total mean signal, uncertainty of mean signal, total SNR

    signal_data:        complex signal measurement data
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
                                            width of Gaussian filter smoothening
    ROI_noise_params:   list of circular ROI parameters to calculate std of
                        radius:             ROI radius in mm
                        center:             optional 
                                            center of ROI in mm as a list in the form of [read, phase]
    fit:                Bool:               whether fitted ROI is plotted               
    '''

    # reshape data to correct dimensions
    signal_image = np.array(data).reshape((image_params[1], image_params[0]))

    # convert to nT
    signal_image *= 1e9

    # calculate real, imag, magnitude and phase of complex data
    signal_image_real   = np.real(signal_image)
    signal_image_imag   = np.imag(signal_image)
    signal_image_abs    = np.abs(signal_image)
    signal_image_phase  = np.angle(signal_image)

    # used for imshow to scale image axis to mm instead of samples
    extent = [-image_params[2]/2, image_params[2]/2, -image_params[3]/2, image_params[3]/2]

    # grid for plotting the ROI
    read        = np.linspace(-image_params[2]/2, image_params[2]/2, image_params[0])  
    phase       = np.linspace(-image_params[3]/2, image_params[3]/2, image_params[1])  
    READ, PHASE = np.meshgrid(read, phase)

    radius, *rest = ROI_fix_params
    center = rest[0] if rest else None
    
    lthreshold, uthreshold, sigma = ROI_fit_params

    radius_noise, *rest_noise = ROI_noise_params
    center_noise = rest_noise[0] if rest_noise else None

    # ROI selection
    if isinstance(radius, (list, tuple)):
        ROI_fixed = ring_ROI(image_params, radius=radius, center=center)
    else:
        ROI_fixed = circular_ROI(image_params, radius=radius, center=center)

    ROI_fit     = find_ROI(signal_image_phase, lthreshold=lthreshold, uthreshold=uthreshold, sigma=sigma)
    ROI         = [ROI_fixed, ROI_fit]
    # inverse of circular ROI
    ROI_noise   = ~circular_ROI(image_params, radius=radius_noise, center=center_noise)

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
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        x = np.linspace(bins[0], bins[-1], 200)

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
            umean = sem(signal_roi[j][i])
            # label position
            if fit:
                y_pos = 0.2-0.13*j
            else:
                y_pos = 0.2-0.13
            ax[i][0].text(
                0.05, y_pos,              
                fr'mean = {mean:.2f} $\pm$ {umean:.2f}',             
                transform=ax[i][0].transAxes,   
                fontsize=10,
                color=color[j][i],
                bbox=dict(facecolor='white', alpha=1) 
            )

            # rescale roi histogram to original histogram
            roi_counts, _ = np.histogram(roi_data, bins=bins)
            roi_pdf = roi_counts / (len(full_data) * bin_width)
            ax[i][1].bar(bins[:-1], roi_pdf, width=bin_width, align='edge', 
                         alpha=0.3, color=color[j][i], label=label_ROI[j])
            # fit Gauss to histogram
            mu, std = norm.fit(roi_data)
            scaling_factor = len(roi_data) / len(full_data)
            pdf_fit_scaled = norm.pdf(x, mu, std) * scaling_factor
            ax[i][1].plot(x, pdf_fit_scaled, color=color[j][i], linestyle='--',
                         label=fr"$\mu={mu:.2f}$, $\sigma={std:.2f}$")

            # scipy.optimize.curve_fit creates many issues with fitting the imaginary data
            # fit curve does not fit the rescaled histogram
            # use scipy.stats.norm.fit instead
            # scipy.optimize.curve_fit
            # pars, cov = curve_fit(
            #     lambda xx, mu, std: norm.pdf(xx, mu, std) * scaling_factor,
            #     bin_centers, roi_pdf, p0=[0, 1]
            # )
            # mu, umu, std, ustd = (
            #     pars[0], np.sqrt(cov[0,0]),
            #     pars[1], np.sqrt(cov[1,1])
            # )
            # ax[i][1].plot(x, norm.pdf(x, mu, std) * scaling_factor, linestyle='dotted', color=color[j][i],
            #            label=fr'$\mu=({mu:.2f}\pm{umu:.2f})$nT, $\sigma=({std:.2f}\pm{ustd:.2f})$nT')
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
            
            scaling_factor_noise = len(noise_data) / len(full_data)
            pdf_fit_scaled_noise = norm.pdf(x, mu_noise, std_noise) * scaling_factor_noise
            ax[i][1].plot(x, pdf_fit_scaled_noise, color='k', linestyle='--',
                         label=fr"$\mu={mu_noise:.2f}$, $\sigma={std_noise:.2f}$")
            # adjustments for scipy.optimize.curve_fit
            # pars_noise, cov_noise = curve_fit(
            #     lambda xx, mu, std: norm.pdf(xx, mu, std) * scaling_factor_noise,
            #     bin_centers, noise_pdf, p0=[0, 1]
            # )
            # mu_noise, umu_noise, std_noise, ustd_noise = (
            #     pars_noise[0], np.sqrt(cov_noise[0,0]),
            #     pars_noise[1], np.sqrt(cov_noise[1,1])
            # )
            # ax[i][1].plot(x, norm.pdf(x, mu_noise, std_noise) * scaling_factor_noise, linestyle='--', color='k',
            #             label=fr'$\mu=({mu_noise:.2f}\pm{umu_noise:.2f})$nT, $\sigma=({std_noise:.2f}\pm{ustd_noise:.2f})$nT')
            std_noise_list.append(std_noise)
        ax[i][1].legend(fontsize=8)

    total_mean = [np.abs(np.mean(signal_image_real[ROI_fixed].flatten() + 1j*signal_image_imag[ROI_fixed].flatten())),
                  np.abs(np.mean(signal_image_real[ROI_fit].flatten() + 1j*signal_image_imag[ROI_fit].flatten()))]
    # uncertainty of total mean
    # standard error of the mean sem:
    umean = [sem(np.abs(signal_image_real[ROI_fixed].flatten() + 1j*signal_image_imag[ROI_fixed].flatten())),
             sem(np.abs(signal_image_real[ROI_fit].flatten() + 1j*signal_image_imag[ROI_fit].flatten()))]
    total_noise = np.sqrt(std_noise_list[0]**2+std_noise_list[1]**2)
    snr = total_mean / total_noise
    # as no uncertainty of the noise is calculated with scipy.stats.norm
    # -> no uncertainty of snr is returned
    # -> the noise is determined in a different ROI than the signal 
    # the determined noise level with get_noise() shows for some measurements a significant
    # difference between the ROI and the total image
    # -> noise might not be completely random for the entire image
    return [total_noise, total_mean, umean, snr]

def abs_image(data, image_params):
    '''
    plots gray scale magnitude image of complex measurement data

    data:               complex measurement data
    image_params:       list of [nread, nphase, FOVread in mm, FOVphase in mm]
    '''
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