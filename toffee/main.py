import numpy as np
import wotan
import matplotlib.pyplot as plt
from matplotlib import cm 
from scipy.stats import binom,norm
import scipy
from scipy.special import factorial as factorial
import scipy.integrate as integrate
from sklearn.metrics import r2_score
import lightkurve as lk # make sure this is the latest version, v.2.4 - you get it with pip install lightkurve
from lightkurve import search_targetpixelfile
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import LombScargle
from astroquery.mast import Catalogs
import pandas as pd
import os
import requests
from pathlib import Path
from .get_tess_orbit import download_csv_file


def flatten(lc_t, raw_lc, raw_lc_errs, plot_results=False, short_window=None, periodogram=None):
    ########################## Argument Info ########################
    # lc_t, raw_lc, and raw_lc_errs are light curve time, flux, and errors of equal length
    # plot_result is bool to plot quadratic, wotan, and sine trends. Default is False
    # short_window is int for wotan flattening window. Default is None (no flattening with wotan)
    # periodogram is list/array of two numbers corresponding to frequency range for periodogram to search over. Default is None (Does not attempt periodogram)
    #################################################################
    # Import Tess sector time data, to know when observation periods for the
    # sectors start and end.
    #get from tess.mit
    tess_orbit_time_url = 'https://tess.mit.edu/public/files/TESS_orbit_times.csv'
    Tess_orbit_times = download_csv_file(tess_orbit_time_url)
    Tess_sector_times= Table.read(Tess_orbit_times, format='ascii.csv')
    Tess_start_times = Time(Tess_sector_times['Start of Orbit'], format='iso', scale='utc')
    Tess_end_times = Time(Tess_sector_times['End of Orbit'], format='iso', scale='utc')
    tess_start_times_tdb = Tess_start_times.tdb
    tess_end_times_tdb = Tess_end_times.tdb
    tess_start_times_bjd =  tess_start_times_tdb.jd - 2457000
    tess_end_times_bjd =  tess_end_times_tdb.jd - 2457000
    orbit_t = np.array([])
    orbit_lcs = np.array([])
    orbit_lc_errs = np.array([])
    orbit_trends = np.array([])
    orbit_masks = []

    #Polyfit won't work if there are any nan values in the fluxes, so we'll need to get rid of those

    lc_t = lc_t[np.isnan(raw_lc) == False]
    raw_lc_errs = raw_lc_errs[np.isnan(raw_lc) == False]
    raw_lc = raw_lc[np.isnan(raw_lc) == False]
    
    tot_mask = np.full(len(lc_t), False)
    # Loop through TESS orbit times and select data within each orbit
    for ii in range(0, len(tess_start_times_bjd)):
        orbit_up = lc_t < tess_end_times_bjd[ii]
        orbit_down = lc_t > tess_start_times_bjd[ii]
        orbit_mask = orbit_up & orbit_down
        tot_mask = tot_mask | orbit_mask
        if True not in orbit_mask: # Skip if no data in 
            continue
        orbit_masks.append(orbit_mask)
    # if plot_results == True:
        ##### Plotter to find points not belonging to any TESS Orbit ######
        # if len(lc_t[~tot_mask]) != 0:
        #     fig, ax = plt.subplots(1, 1, figsize=(10,10))
        #     ax.errorbar(lc_t[~tot_mask], raw_lc[~tot_mask], raw_lc_errs[~tot_mask], color='red', zorder=3, fmt='.')
        #     ax.errorbar(lc_t[tot_mask], raw_lc[tot_mask], raw_lc_errs[tot_mask], color='green', zorder=1, fmt='.')
        #     ax.set_xlim(np.min(lc_t[~tot_mask])-1, np.max(lc_t[~tot_mask])+1)
        #     print(len(lc_t[~tot_mask]))
        #     plt.show()
        #     return
        
        # fig, axs = plt.subplots(len(orbit_masks), 1, figsize=(10,4*len(orbit_masks)))
        # fig.suptitle('Quadratic Trend per TESS Orbit', fontsize=16, color='blue')
    for ii in range(0, len(orbit_masks)):
        # Fit quadratic trend to orbit data
        coeff = np.polyfit(lc_t[orbit_masks[ii]], raw_lc[orbit_masks[ii]], 2)
        test_trend = coeff[0]*lc_t[orbit_masks[ii]]**2 + coeff[1]*lc_t[orbit_masks[ii]] + coeff[2]
        test_lc = np.array(raw_lc[orbit_masks[ii]])
        test_t = lc_t[orbit_masks[ii]]
        test_errs = raw_lc_errs[orbit_masks[ii]]
        ##### Add orbit data and trends together (some data falls under no orbit, causing array length mismatch) #####
        orbit_lcs = np.append(orbit_lcs, test_lc)
        orbit_t = np.append(orbit_t, test_t)
        orbit_lc_errs = np.append(orbit_lc_errs, test_errs)
        orbit_trends = np.append(orbit_trends, test_trend)
        # if plot_results == True:
        #     ax = axs[ii]
        #     ax.errorbar(test_t, test_lc, test_errs, fmt='.')
        #     ax.plot(test_t, test_trend, color='r', alpha=0.7, zorder=3)
        #     plt.show()
    # Store quadratic-removed light curves
    lc_long,long_trend = orbit_lcs / orbit_trends, orbit_trends
    lc_errs_long = orbit_lc_errs / orbit_trends # normalise the errors too
    lc_quad = np.array([lc_long, lc_errs_long, long_trend])
    lc_working, lc_errs_working = lc_long, lc_errs_long
    
    # Wotan flatten and store flattened light curve
    if short_window != None:
        lc_short, short_trend = wotan.flatten(orbit_t,lc_working,window_length=short_window,return_trend=True)
        lc_errs_short = lc_errs_working / short_trend
        if plot_results == True:
            fig, axs = plt.subplots(2, 1, figsize=(10,4))
            fig.suptitle('Wotan Trend over TESS Orbit', fontsize=12, color='blue')
            ax = axs[0]
            ax.plot(orbit_t, short_trend)
            ax.errorbar(orbit_t, lc_working, lc_errs_working, fmt='.')
            ax.set_xlim(orbit_t[-1]-5, orbit_t[-1])
            ax = axs[1]
            ax.errorbar(orbit_t, lc_short, lc_errs_short, fmt='.')
            ax.plot(orbit_t, scipy.ndimage.uniform_filter1d(lc_short, size=10), c='orange', zorder=3, linewidth=1)
            plt.show()
            
        lc_wotan = np.array([lc_short, lc_errs_short, short_trend])
        
        lc_working, lc_errs_working = lc_short, lc_errs_short
    # Run periodogram 
    periodic = False
    if periodogram != None:
        lc_working, lc_errs_working = lc_short, lc_errs_short
        # # initialize sine light curves as wotan flattened light curves
        lc_sine, lc_errs_sine = lc_working, lc_errs_working
        sine_trend = np.full(len(lc_sine), 1)
        # lc_sine, sine_trend = wotan.flatten(orbit_t,lc_working,window_length=0.2,kernel_size=5,method='gp',kernel='periodic_auto',return_trend=True)
        # lc_sine_wotan, sine_trend = wotan.flatten(orbit_t,lc_working,window_length=0.15,method='lowess',return_trend=True)
        # lc_errs_sine = lc_errs_working / sine_trend
        if plot_results == True:
            wind_up, wind_down = 1.05*np.max(lc_sine[np.abs(orbit_t - orbit_t[-1]) < 5]), 0.9*np.min(lc_sine[np.abs(orbit_t - orbit_t[-1]) < 5])
            fig, axs = plt.subplots(2, 1, figsize=(10,10))
            fig.suptitle('Wotan Sine Trend over TESS Orbit', fontsize=12, color='blue')
            ax = axs[0]
            ax.plot(orbit_t, sine_trend,zorder=3, c='orange')
            ax.errorbar(orbit_t, lc_working, lc_errs_working, fmt='.', c='blue', zorder=1)
            ax.set_xlim(orbit_t[-1]-5, orbit_t[-1])
            # ax.set_xlim(2494-5, 2494)
            ax.set_ylim(wind_down, wind_up)
            ax = axs[1]
            ax.errorbar(orbit_t, lc_sine, lc_errs_sine, fmt='.')
            ax.plot(orbit_t, scipy.ndimage.uniform_filter1d(lc_sine, size=10), c='orange', zorder=3, linewidth=1)
            ax.set_xlim(orbit_t[-1]-5, orbit_t[-1])
            # ax.set_xlim(2494-5, 2494)
            ax.set_ylim(wind_down, wind_up)
            plt.show()
        lc_flat = np.array([lc_sine, lc_errs_sine, sine_trend])
        
        
        # # Create sine function for scipy.curve_fit
        # def sine_func(x, A, B, C, D, E, F, G, H, I, J, K):
        #         return (A*x**2+B*x+C) * np.sin(D * x + E) + (F*x**2+G*x+H) * np.sin(I * x + J) + K
            
        # # Loop through TESS orbit times and select data within each orbit
        # sine_trend = np.array([])
        for ii in range(0, len(tess_start_times_bjd)):
            orbit_up = orbit_t < tess_end_times_bjd[ii]
            orbit_down = orbit_t > tess_start_times_bjd[ii]
            orbit_mask = orbit_up & orbit_down
            if True not in orbit_mask: # Skip if no data in orbit
                continue
            orb_trend = np.full(len(orbit_t[orbit_mask]), 1)
            # Conduct periodograms and fit sine if signal is found, then repeat
            # for jj in range(0,1):
            frequency = np.linspace(periodogram[0], periodogram[1], 100000)
            for jj in range(0,8):
                
                ls = LombScargle(orbit_t[orbit_mask], lc_sine[orbit_mask], lc_errs_sine[orbit_mask])
                power = ls.power(frequency)

                if power[500] == np.nan:

                    return 'Broken!', 'Broken!', 'Broken!', 'Broken!', 'Broken!', 'Broken!'
                    
                prob_false = ls.false_alarm_probability(power.max())
                if prob_false > 0.2: # No sinusoidal signal, exit loop
                    break
                elif prob_false < 0.2:
                    periodic = True
                    # else:
                        # print("SINE FLATTENED", end='\n')
                        
                    peak_freq = frequency[np.where(power==power.max())[0][0]]
                if plot_results == True:
                    plot_t = orbit_t[orbit_mask]
                    plot_mask = np.abs(orbit_t - plot_t[-1]) < 5/peak_freq
                    mask = plot_mask & orbit_mask
                    fig = plt.figure(figsize=(10,5))
                    plt.errorbar(orbit_t[mask], lc_sine[mask], yerr=lc_errs_sine[mask], fmt='.')
                    plt.xlabel('Time (BJD - 2,7457,000)')
                    plt.ylabel('Normalised And Flattened flux')
                    # plt.plot(orbit_t[mask], scipy.ndimage.uniform_filter1d(lc_sine, size=10)[mask], c='orange', zorder=3, linewidth=3)
                    half_window = int(0.5/peak_freq * 24 * 3600 / 120)
                    lc_sine_wotan, sine_trend = wotan.flatten(orbit_t,lc_working,window_length=half_window*120/3600/24,
                                                              method='median',return_trend=True)
                    if half_window % 2 == 0:
                        window_median = scipy.signal.medfilt(lc_sine, kernel_size=half_window+1)
                        plt.plot(orbit_t[mask], window_median[mask], c='orange', zorder=3, linewidth=3)
                    else:
                        window_median = scipy.signal.medfilt(lc_sine, kernel_size=half_window)
                        plt.plot(orbit_t[mask], window_median[mask], c='orange', zorder=3, linewidth=3)
                    plt.plot(orbit_t[mask], sine_trend[mask], c='red', zorder=2, linewidth=3)
                    lc_sine[orbit_mask] = lc_sine[orbit_mask] / window_median[orbit_mask]
                    lc_errs_sine[orbit_mask] = lc_errs_sine[orbit_mask] / window_median[orbit_mask]
                    
                    plt.show()
        #     # print(sine_trend)
        # lc_flat = np.array([lc_sine, lc_errs_sine, sine_trend])
    if short_window == None:
        return orbit_t, lc_quad, periodic
    elif periodogram == None:
        return orbit_t, lc_quad, lc_wotan, periodic
    else:
        return orbit_t, lc_quad, lc_wotan, lc_flat, periodic
    

############### Flare Masking ###############
# Find the TOFFEE-identified flare duration per flare (flare_dur)
# then define a "mask_extra" duration that is half of flare_dur, 
# to a minimum of 15 data points in 2 minute cadence.
# The mask of the flare durations then includes data points 2*mask_extra after
# the TOFFEE-identified end, and 0.5*mask_extra before the start.
def make_flare_mask(lc_t, benchmark, data_min = 15):
    days2sec = 86400
    #calculate the cadence
    cadence = np.median(np.diff(lc_t))
    flare_mask = np.full(len(lc_t), True)
    for ii in range(0, len(benchmark)):
        flare_dur = benchmark.iloc[ii]["T_end"] - benchmark.iloc[ii]["T_start"]
        if 0.5*flare_dur < data_min * cadence/days2sec:
            mask_extra = data_min * cadence/days2sec
        else:
            mask_extra = 0.5*flare_dur

        up_mask = lc_t < benchmark.iloc[ii]["T_end"] + 2*mask_extra
        low_mask = lc_t > benchmark.iloc[ii]["T_start"] - 0.5*mask_extra
        temp_mask = up_mask & low_mask
        flare_mask = flare_mask & ~temp_mask
        # print("Flare Masks Complete: " + str(ii +1), end='\r')

    return flare_mask

################### GAP MASKING ###################
# Find gaps in data greater than 1/4 of the LS period, then identify the gaps before and after each flare
# creating a gap_mask to ignore all other light curve points beyond the adjacent gaps to ensure flares
# are not fitted across gaps.
def make_gap_mask(lc_t, flare_peak, gap_times):
    if len(gap_times) != 0:
        # Find gap_times the flare peak is between, i.e. the closest two gaps to the flare
        flare_gap = np.searchsorted(gap_times, flare_peak)
        
        # Edge case 1: No gap before the flare
        if flare_gap == 0: 
            low_mask = np.full(len(lc_t), True)
            up_mask = lc_t < gap_times[flare_gap]
        # Edge case 2: No gap after the flare
        elif flare_gap == len(gap_times): 
            up_mask = np.full(len(lc_t), True)
            low_mask = lc_t > gap_times[flare_gap-1]
        else:
            low_mask = lc_t > gap_times[flare_gap-1]
            up_mask = lc_t < gap_times[flare_gap]
        gap_mask = low_mask & up_mask
    else:
        gap_mask = np.full(len(lc_t), True)  
        
    return gap_mask

def make_window_mask(lc_t, lc_flux, lc_errs, flare_peak, window_size):
    ################### Window Mask #####################
    # Define frequency range for periodograms to search for
    # The below corresponds to a period in the range of 0.1 to 7 days
    frequency = np.linspace(1/7, 1/0.1, 100000)
    
    window_mask, window_mask_left, window_mask_right, peak_freq, success = None, None, None, None, False
    # Define a window mask around T_peak equal to a multiple of the global LS period
    ls_mask = np.abs(lc_t - flare_peak) < window_size
    
    # If no light curve points qualify the flare, gap, and window masks
    # skip the flare, entering sentinel values instead
    if len(lc_t[ls_mask]) != 0:
        # Conduct a local LS on the window around the selected flare
        ls = LombScargle(lc_t[ls_mask], lc_flux[ls_mask], lc_errs[ls_mask])
        power = ls.power(frequency)
        # frequency, power = ls.autopower()
        prob_false = ls.false_alarm_probability(power.max())
        peak_freq = frequency[np.where(power==power.max())[0][0]]
    
        # The false alarm probability is above 5%, so periodicity is not strong enough: skip the flare
        if prob_false <= 0.05:
            # Now recreate the window, but now with the local LS period. Also create duplicate windows
            # shifted left and right used for checking fit quality and agreement.
            window_mask = np.abs(lc_t - flare_peak) < window_size
            window_mask_left = np.abs(lc_t - flare_peak + 0.5*window_size) < window_size
            window_mask_right = np.abs(lc_t - flare_peak - 0.5*window_size) < window_size
        
            # If any of the windows contain no qualifying points, skip the flare
            if True not in window_mask or True not in window_mask_right or True not in window_mask_left:
                pass
            else:
                success = True
    return ls_mask, window_mask, window_mask_left, window_mask_right, peak_freq, success
    
def spot_fitting(lc_t, lc_flux, lc_errs, flare_mask, init):
    # Quadratic ampltiude double sine function
    def sine_func(x, A, B, C, D, E, F, G, H, I, J, K):
        return (A*x**2+B*x+C) * np.sin(D * x + E) + (F*x**2+G*x+H) * np.sin(I * x + J) + K

    fail = False
    try:
        popt, pcov = scipy.optimize.curve_fit(sine_func, lc_t[flare_mask], lc_flux[flare_mask], sigma=lc_errs[flare_mask],
                                              nan_policy='omit', p0=init, maxfev=20000, method='trf')
    except KeyboardInterrupt:
        raise ValueError('User Interrupt')
    except:
        fail = True
        print('\n')
        print("FAILED FLARE")
        popt = init
        pcov = False
        spot_fit = False
    finally:
        spot_fit = sine_func(lc_t[flare_mask], *popt)

    return popt, pcov, spot_fit, fail

def spot_mod(lc_t, lc_flux, lc_errs, benchmark, plot=False, params=None, rand=False, period_num=3):
    ############### NOTE ##############
    # lc_t, lc_flux, lc_errs are the quadratic-trend-flattened light curves

    # the spot mod at flare times for the centered, left shifted, and right shifted flares, and corresponding reduced Chi2
    # red_chi2 is a N x 3 matrix where N is the # of flares. Indices 0, 1, and 2 of each row correspond to the center, left, and right fits.
    flare_spot, flare_spot_left, flare_spot_right, red_chi2 = np.array([]), np.array([]), np.array([]), []

    # Boolean array with entries for each of the flares in benchmark to track existence and agreement of spot fittings
    spot_exists = np.full(len(benchmark), True)
    spot_agreement = np.full(len(benchmark), True)

    # Define frequency range for periodograms to search for
    # The below corresponds to a period in the range of 0.1 to 7 days
    frequency = np.linspace(1/7, 1/0.1, 100000)

    # Quadratic ampltiude double sine function
    def sine_func(x, A, B, C, D, E, F, G, H, I, J, K):
            return (A*x**2+B*x+C) * np.sin(D * x + E) + (F*x**2+G*x+H) * np.sin(I * x + J) + K

    spot_mask = make_flare_mask(lc_t, benchmark)
    
    # Conduct a LS over the lightcurve with flares masked
    full_ls = LombScargle(lc_t[spot_mask], lc_flux[spot_mask], lc_errs[spot_mask])
    power = full_ls.power(frequency)
    # frequency, power = ls.autopower()
    prob_false = full_ls.false_alarm_probability(power.max())
    full_peak_freq = frequency[np.where(power==power.max())[0][0]]
    window_size = period_num/full_peak_freq
    del_time = lc_t - scipy.ndimage.shift(lc_t, 1) # time from previous data point
    gap_index = np.where(del_time > 0.25/full_peak_freq)
    gap_times = lc_t[gap_index]

    # print("LS Complete")
    for kk in range(0, len(benchmark)):
        flare_peak = np.array(benchmark["T_peak"])[kk]
        gap_mask = make_gap_mask(lc_t, flare_peak, gap_times)

        # Create light curve segment from flare and gap masks
        orbit_time = lc_t[spot_mask & gap_mask]
        orbit_lc = lc_flux[spot_mask & gap_mask]
        orbit_errs = lc_errs[spot_mask & gap_mask]

        ################### Window Mask #####################
        ls_mask, window_mask, window_mask_left, window_mask_right, peak_freq, success = make_window_mask(orbit_time, orbit_lc, orbit_errs, flare_peak, window_size)
        
        # If window masking fails, skip the flare and enter sentinel values
        if success == False:
            # print("passed flare LS" + str(kk))
            flare_spot = np.concatenate((flare_spot, np.array([10])))
            flare_spot_left = np.concatenate((flare_spot_left, np.array([10])))
            flare_spot_right = np.concatenate((flare_spot_right, np.array([10])))
            red_chi2.append(np.array([-1,-1,-1]))
            continue

        ################### Spot Fitting #####################
        # Estimate amplitude of the sinusoidal signal
        amp = (np.percentile(orbit_lc[window_mask],99) - np.percentile(orbit_lc[window_mask],1))/2

        # Create two sets of initial values for the quadratic amplitude sinusoids. The primary sine frequency is identical,
        # but the secondary frequencies are double/half respectively to ensure quality of fit
        init_double = np.array([0, 0, amp, 2*np.pi*peak_freq, 0, 0, 0, amp, 4*np.pi*peak_freq, np.pi, 1])
        init_half = np.array([0, 0, amp, 2*np.pi*peak_freq, 0, 0, 0, amp, np.pi*peak_freq, np.pi, 1])

        fail = False
        m_index = 0
        masks = [window_mask, window_mask_left, window_mask_right]
        popt, curve_opt = [], []
        run_red_chi = []
        for ii in range(0,len(masks)):
            popt_double, pcov, sine_double, fit_1 = spot_fitting(orbit_time, orbit_lc, orbit_errs, masks[ii], init_double)
            popt_half, pcov, sine_half, fit_2 = spot_fitting(orbit_time, orbit_lc, orbit_errs, masks[ii], init_half)
            if fit_1 == True or fit_2 == True:
                spot_exists[kk] = False
                spot_agreement[kk] = False
                fail = True
                break

            ############### Reduced Chi-Squared ##############
            # Choose the best fit for each of the windows using Chi2 (note: only difference was initial values)
            if np.sum(np.abs(orbit_lc[masks[ii]] - sine_double)**2) > np.sum(np.abs(orbit_lc[masks[ii]] - sine_half)**2):
                popt.append(popt_half)
                curve_opt.append(sine_half)
            else:
                popt.append(popt_double)
                curve_opt.append(sine_double)
            run_red_chi.append(np.sum(np.abs(orbit_lc[masks[ii]] - curve_opt[-1])**2 / orbit_errs[masks[ii]]**2) / (len(curve_opt[-1]) - len(popt[-1])))
        if fail == True:
            flare_spot = np.concatenate((flare_spot, np.array([10])))
            flare_spot_left = np.concatenate((flare_spot_left, np.array([10])))
            flare_spot_right = np.concatenate((flare_spot_right, np.array([10])))
            red_chi2.append(np.array([-1,-1,-1]))
            continue
        # Record the set of reduced chi squareds
        red_chi2.append(np.array(run_red_chi))
    
        
        # Recreate light curve segment using gap and window masks, this time without masking flares
        orbit_time = lc_t[gap_mask]
        orbit_lc = lc_flux[gap_mask]
        orbit_errs = lc_errs[gap_mask]
        
        ls_mask, window_mask, window_mask_left, window_mask_right, peak_freq, success = make_window_mask(orbit_time, orbit_lc, orbit_errs, flare_peak, window_size)
        masks = [window_mask, window_mask_left, window_mask_right]
        # Recreate spot modulation curve, this time interpolating during flares
        fit_spot = []
        sine_opts = []
        for ii in range(0, len(masks)):
            sine_opt = sine_func(orbit_time[masks[ii]], *popt[ii])
            sine_opts.append(sine_opt)
            # Calculate spot mod at flare peaks (-1 means flare peaks at minimum spot mod, 1 means flare peaks at max)
            fit_spot.append(2*((sine_func(flare_peak, *popt[ii]) - np.min(sine_opt)) / (np.max(sine_opt) - np.min(sine_opt))) - 1)
        # Calculate spot mod at flare peaks (-1 means flare peaks at minimum spot mod, 1 means flare peaks at max)
        flare_spot = np.concatenate((flare_spot, np.array([fit_spot[0]])))
        flare_spot_left = np.concatenate((flare_spot_left, np.array([fit_spot[1]])))
        flare_spot_right = np.concatenate((flare_spot_right, np.array([fit_spot[2]])))
        # Check for agreement of fits based on the sign of spot mod
        flare_sign = np.sign(np.array([flare_spot[-1], flare_spot_left[-1], flare_spot_right[-1]]))
        if 1 in flare_sign and -1 in flare_sign:
            spot_agreement[kk] = False

        flare_percentage = (flare_spot[-1] + flare_spot_left[-1] + flare_spot_right[-1]) / 3

    
        if plot == True:# and np.mean(run_red_chi) > 100:#(1 in flare_sign and -1 in flare_sign):# and (np.abs(flare_spot[-1]) > 0.99 or np.abs(flare_spot_left[-1]) > 0.99
        #               or np.abs(flare_spot_right[-1]) > 0.99): #np.mean(run_red_chi) > 5:# and np.abs(flare_percentage) > 0.99:
            
            fig, ax = plt.subplots(1,1,figsize=(10,4))
            plot_text = 'spot_fits: ' + str([flare_spot[-1], flare_spot_left[-1], flare_spot_right[-1]]) + '  red_chi: ' + str(np.round(np.mean(run_red_chi), decimals=3))
            # plt.text(.01, .99, plot_text, ha='left', va='top', transform=ax.transAxes)
            # # fig, ax = plt.subplots(1,1,figsize=(6.4,4.8))
            # ax.errorbar(orbit_time[flare_mask], orbit_lc[flare_mask], np.abs(orbit_errs[flare_mask]), c='#BF527F', zorder=1, ls='none')
            # ax.errorbar(lc_t[spot_mask], lc_flux[spot_mask], np.abs(lc_errs[spot_mask]), zorder=2, ls='none', c='#3b3b3b')
            ax.scatter(orbit_time[masks[0]], orbit_lc[masks[0]], c='#BF527F', zorder=1, s=10)
            ax.scatter(lc_t[spot_mask], lc_flux[spot_mask], zorder=2, c='black', s=10)
            if flare_spot[-1] > 0:
                ax.plot(orbit_time[masks[0]], sine_opts[0], c='#F28C28', lw=3, zorder=3)
            else:
                ax.plot(orbit_time[masks[0]], sine_opts[0], c='#96DED1', lw=3, zorder=3)
            if flare_spot_left[-1] > 0:
                ax.plot(orbit_time[masks[1]], sine_opts[1], c='#FF5F1F', lw=3, zorder=3)
            else:
                ax.plot(orbit_time[masks[1]], sine_opts[1], c='#00FFFF', lw=3, zorder=3)
            if flare_spot_right[-1] > 0:
                ax.plot(orbit_time[masks[2]], sine_opts[2], c='#FFAC1C', lw=3, zorder=3)
            else:
                ax.plot(orbit_time[masks[2]], sine_opts[2], c='#87CEEB', lw=3, zorder=3)
            ax.axhline(y=1, zorder=3, ls='--', c='black')
            ax.set_xlim(np.min(orbit_time[masks[0]]), np.max(orbit_time[masks[0]]))
            ax.set_ylim(np.min(orbit_lc[masks[0]]), np.max(orbit_lc[masks[0]]))
            # ax.axvline(x=np.array(benchmark["T_peak"])[kk], zorder=2, lw=2, c='orange', alpha=.5)
            # plt.savefig(str(self.star_name) + '_flare_' + str(kk+1) + '.png')
            # plt.close(fig)
            # snapshot = tracemalloc.take_snapshot()
            # print("PLOTTER:")
            # for stat in snapshot.statistics("lineno")[:5]:
            #     pass
            #     print(stat)
            plt.show()

        # tracemalloc.stop()

    if len(flare_spot) != len(spot_exists):
        print(flare_spot)
        print(spot_exists)
        raise ValueError("DIMENSION OF SPOT MOD VALUES ARRAY vs. SPOT AGREEMENT ARRAY MISMATCH")
    return flare_spot, flare_spot_left, flare_spot_right, np.array(red_chi2).T, spot_exists, spot_agreement


def get_rot_period(lc_t, lc_flux, lc_errs, benchmark=None):
    ######## PLACEHOLDER: INSERT TOFFEE FLATTEN FUNCTION ##########
    flat_t, flat_flux, flat_errs = np.full(len(lc_t), True), np.full(len(lc_t), True), np.full(len(lc_t), True)

    if benchmark:
        spot_mask = make_flare_mask(flat_t, benchmark)
    else:
        ### Replace with flare detection output
        spot_mask = np.full(len(flat_t), True)

    # Conduct a LS over the lightcurve with flares masked
    full_ls = LombScargle(flat_t[spot_mask], flat_flux[spot_mask], flat_errs[spot_mask])
    power = full_ls.power(frequency)
    prob_false = full_ls.false_alarm_probability(power.max())
    full_peak_freq = frequency[np.where(power==power.max())[0][0]]
    return 1/full_peak_freq, prob_false


def break_finder(time, flux, min_break = 0.25):
    #find the differences of all the time coordinates and determine if
    #they're long enough to be called a break from min_break
    time_diffs = np.append(False, np.diff(time) > min_break)

    #find where that break is, this will tabulate the right hand side of the break(s)
    end_of_time_breaks = np.where(time_diffs == True)[0]

    #and the beginning of the breaks which come right before
    lightcurve_break_index = end_of_time_breaks -1

    #determine which belongs to the orbit break
    #############INITIALIZE ARRAYS TO HOLD BREAKS AND TYPE OF BREAK#############
    break_indices = []
    break_start_time = []
    break_end_time = []
    break_type = []

    #find the time of the middle of the sector
    sector_mid_time = (max(time) + min(time))/2

    #loop through breaks and find which one contains this time
    for break_index in lightcurve_break_index:
        #add the features of this break to the lists
        break_indices.append(break_index)
        break_start_time.append(time[break_index])
        break_end_time.append(time[break_index + 1])

    #convert to dataframe
    sector_break_frame = pd.DataFrame({'Break_Index': pd.Series(break_indices, dtype = int),
                                       'Break_Start_Time': pd.Series(break_start_time, dtype = float),
                                       'Break_End_Time': pd.Series(break_end_time, dtype = float)})
    
    return sector_break_frame





def light_curve_mask(time, flux, min_break = 0.25, clip_breaks = 200):
    #####################find the breaks###################
    #Use breakfinder to find the break(s) from TESS sector and clip off cadences
    #from either side of the breaks
    sector_break_frame = break_finder(time, flux, min_break = min_break)

    #pull out indices of breaks
    break_index = sector_break_frame['Break_Index']

    #convert to numpy array
    break_index = np.array(break_index)

    #loop through and build boolean array of the points we want to keep in the lightcurve
    #need to account for the breaks that are smaller than the number of points we want to clip
    light_curve_break_mask = np.full(len(time), 1, dtype = bool)

    #boolean arguments to see if we need to clip anything at the beginning or end or if it's already
    #been done
    clip_start = True
    clip_end = True
    j = 0 

    while j < len(break_index):
        index = break_index[j]

        #check to the left to see if the length between the left side of the
        #first break isn't close to the beginning of the curve or close to the other breaks
        if j == 0:
            if break_index[j] <= 2 * clip_breaks:

                #if close to beginning clip everyhing up to the start of the break
                clip_left = index

                #also set boolean argument to clip the first cadences ofhe lightcurve to
                #false so we don't clip anything else
                clip_start = False

        #otherwise use normal amount
            else:
                clip_left = clip_breaks
                
        #check to the left, ideally this was already taken care of this in the previous
        #iteration when looking to the right
        if j > 0:
            #we want to make this twice the length of the break_clips argument
            #because we're reaching some length to the right of one breaks AND
            #some length of cadences to the left of the other to check if they overlap
            if break_index[j] - break_index[j - 1] <= 2 * clip_breaks:
                #clip nothing, should've already been done
                clip_left = 0
                
            #otherwise use normal amount
            else:
                clip_left = clip_breaks

        #Now look to the right to see if we're close to the end of the light curve
        #or another break in the curve
        if j == len(break_index) - 1:
            if break_index[j] > len(time) - (2 * clip_breaks):

                #if close to the end for the last break then clip everything to the end
                clip_right = len(time) - clip_breaks

                #also set boolean argument to clip the last cadences of the lightcurve
                #to false
                clip_end = False

            else:
                clip_right = clip_breaks

        #check to see if we're close to another break
        if j < len(break_index) - 1:
            #we want to make this twice the length of the break_clips argument
            #because we're reaching some length to the right of one breaks AND
            #some length of cadences to the left of the other to check if they overlap
            if break_index[j + 1] - break_index[j] <= 2 * clip_breaks:
                #clip everyhing between them
                clip_right = break_index[j + 1] - break_index[j]

            else:
                clip_right = clip_breaks

        #add these indices to the mask as false values
        #clip stuff to the left of the break
        mask_left_start = index - clip_left
        light_curve_break_mask[mask_left_start:index + 1] = False
        mask_right_end = index + clip_right
        light_curve_break_mask[index:mask_right_end] = False
        j += 1

    #Now clip the beginning and end if still needed
    if clip_start == True:
        light_curve_break_mask[0:clip_breaks] = False
    if clip_end == True:
        light_curve_break_mask[len(time) - clip_breaks:len(time)] = False
    return light_curve_break_mask


#Input lightcurve information and get out times and fluxes of bright points
#above a given threshold
def find_candidates(time, flux, flux_std, threshold_std):

    '''Arguments:

    time: time coordinates of the lightcurve
    flux: flux values in the lightcurve corresponding to the times
    flux_std: global spead of flux values, sets σ
    threshold_std: the threshold in σ of what defines a bright point
    '''

    '''Outputs:
    time_candidates: 1D array holding the times for points that lie above the flux threshold
    bright_points: 1D array holding the fluxes for the points above the threshold corresponding to the times
    '''
    #Find flux threshold
    flux_threshold = 1 + (threshold_std * flux_std)

    #Find bright points above the threshold and their times
    bright_points = flux[flux > flux_threshold]
    time_candidates = time[flux > flux_threshold]
            
    #sort them in descending order with highest fluxes first
    
    #we call three separate numpy functions to sort the times
    #we want it so that we preserve the times corresponding to each flare canditate
    #once the flares are sorted. searchsorted tells us the indices of the sorted array of
    #flare candidates and thus the corresponding indices of the times. We then need to reverse
    #to be in descending order from brightest to dimmest

    time_candidates = np.flip(time_candidates[np.argsort(bright_points)])
    bright_points = np.flip(np.sort(bright_points))

    return time_candidates, bright_points


#Function to find the start and end times of a bright epoch to later test if it's a flare
def start_end_time(time, flux, time_candidate, flux_std,
                           prim_marginal_threshold, num_below_threshold):

    '''Arguments
    For this version of the start_end finder we'll be looking at how many consecutive points reside well below the threshold
    
    time: 1D array of ALL the time coordinates in the flattened light curve
    flux: 1D array of ALL the flux coordinates in the flattened light curve
    time_candidate: time coordinate of the peak of the bright epoch
    flare_candidate: flux coordinate of the peak of the bright epoch, potential flare peak vale
    flux_std: Global spread of detrended flux values
    prim_marginal_threshold: The sigma threshold considered for a point to be dim enough to longer be associated with a flare.
                             Default value of 2 means that we're looking for flux values below 2σ to end the flare
    num_below_threshold: the number of consecutive points below prim_marginal_threshold telling us that a flare has for sure
                         begun or ended. Default value of three tells us that three consecutive points below the threshold before
                         the first point in the flare or after the last point in the flare.

    Returns:
        index_start: Index of time/flux/flux_error of bright epoch
        index_end: Index of time/flux/flux_error of the bright epoch

    '''

    #Since in the flare_finder code we'll be sorting the flare candidates from brightest to dimmest, the flare_candidate
    #that is passed as an argument should be a peak of a flare with the rise and decay to either side (if it is indeed a flare)

    #Alright, let's filter through these arrays and get rid of the redundant measurements
    #We'll run through it until a certain break condition
    time_peak_flare_index = np.where(time == time_candidate)[0][0] #identify the index of the flare peak

    #for later clarity
    flare_peak_time = time_candidate

    #find median flux, spread, and threshold for primary flare detection
    median_flux = np.nanmedian(flux)
    # calculate threshold from the photometric error of the photometry point
    one_sigma_percentile = 84
    flux_std = np.nanpercentile(flux - 1, one_sigma_percentile)
    
    #look around this time to see if any of the time candidates belong to the same flare
    #look to the left to find where the beginning is

    #Define function to count the number of consecutive preceeding flux values
    #That are below a given flux threshold. Used for determining the beginning/end of flares
    def count_left(time, flux, j, time_peak_flare_index, flux_std, prim_marginal_threshold):

        '''Arguments
        time: 1D array of ALL the time coordinates in the flattened light curve
        flux: 1D array of ALL the flux coordinates in the flattened light curve
        j: A index value telling us how displaced we are from the peak of the flare. Potentially the index for
            the beginning/end of the flare
        time_peak_flare_index: Index in time (and flux) of the peak of the flare
        flux_std: Global spread of detrended flux values
        prim_marginal_threshold: The sigma threshold considered for a point to be dim enough to longer be associated with a flare.
                                 Default value of 2 means that we're looking for flux values below 2σ to end the flare

        Returns:
        count: the number of consecutive points preceding flux[time_peak_flare_index - j] that are below the threshold

        '''
        #initialize count
        count = 0
        #initialize end condition for the loop telling us the next point
        terminate = False

        while terminate == False:
        
            #if we're at the first index, end the program and just return the current count
            if time_peak_flare_index - j - count <= 0:
                return count
            
            #look left to see if the next value is below the threshold 
            if flux[time_peak_flare_index - j - count] <= 1 + (prim_marginal_threshold * flux_std):
                count += 1

            else:
                terminate = True
                return count
                

    #initialize count
    count = 0
    j = -1 #intialize index
    
    while count <= num_below_threshold:
        j += 1
        count = count_left(time, flux, j, time_peak_flare_index, flux_std, prim_marginal_threshold)
        if time_peak_flare_index - j - count <= 0: #condition to break if we're at the end of the lightcurve
            break

    #Hopefullt with the count loop completed, j now tells us the displaced index from the peak
    #To the beginning of the flare

    index_start = time_peak_flare_index - j


    ####Find flare End Time

    #Define function to count the number of consecutive following flux values
    #That are below a given flux threshold. Used for determining the end of flares
    #literally the same as count_left but with the signs flipped
    def count_right(time, flux, j, time_peak_flare_index, flux_std, prim_marginal_threshold):

        '''Arguments
        time: 1D array of ALL the time coordinates in the flattened light curve
        flux: 1D array of ALL the flux coordinates in the flattened light curve
        j: A index value telling us how displaced we are from the peak of the flare. Potentially the index for
            the beginning/end of the flare
        time_peak_flare_index: Index in time (and flux) of the peak of the flare
        flux_std: Global spread of detrended flux values
        prim_marginal_threshold: The sigma threshold considered for a point to be dim enough to longer be associated with a flare.
                                 Default value of 2 means that we're looking for flux values below 2σ to end the flare

        '''
        #initialize count
        count = 0
        #initialize end condition for the loop telling us the next point
        terminate = False

        while terminate == False:
        
            #if we're at the last index, end the program and just return the current count
            if time_peak_flare_index + j + count >= len(time) - 1:
                terminate = True
                return count
            
            #look left to see if the next value is below the threshold 
            if flux[time_peak_flare_index + j + count] <= 1 + (prim_marginal_threshold * flux_std):
                count += 1

            else:
                terminate = True
                return count

    #initialize count
    count = 0
    j = -1 #intialize index
    
    while count <= num_below_threshold:
        j += 1
        count = count_right(time, flux, j, time_peak_flare_index, flux_std, prim_marginal_threshold)
        if time_peak_flare_index + j + count >= len(time) - 1: #condition to break if we're at the end of the lightcurve
            break

    #Hopefullt with the count loop completed, j now tells us the displaced index from the peak
    #To the beginning of the flare

    index_end = time_peak_flare_index + j

    return index_start, index_end

#for a bright epoch find the number of consecutive points

def consec_finder(time, flux, time_candidate, flux_std, threshold_std,
                           prim_marginal_threshold, num_below_threshold):

    '''Arguments
        time: 1D array of ALL the time coordinates in the flattened light curve
        flux: 1D array of ALL the flux coordinates in the flattened light curve
        time_candidate: Index in time (and flux) of the peak of the flare
        flux_std: Global spread of detrended flux values
        threshold_std: Desired sigma of the lightcurve to find a flare
        prim_marginal_threshold: The sigma threshold considered for a point to be dim enough to longer be associated with a flare.
                                 Default value of 2 means that we're looking for flux values below 2σ to end the flare
        num_below_threshold: the number of consecutive points below prim_marginal_threshold telling us that a flare has for sure
                         begun or ended. Default value of three tells us that three consecutive points below the threshold before
                         the first point in the flare or after the last point in the flare.

        '''

    #find start and end time
    index_start, index_end = start_end_time(time, flux, time_candidate, flux_std,
                           prim_marginal_threshold, num_below_threshold)

    #find indices between start and end that are above threshold
    threshold_flux = 1 + flux_std * threshold_std
    bright_points = np.where(flux[index_start:index_end+1] >= threshold_flux)[0]
    #Example:
    #[100,102,103,104]

    #split up the bright_points list into portions that are consecutive by cutting
    #them into pieces where adjacent index values do not have a difference of one
    consec_portions = np.split(bright_points,
                               np.where(np.diff(bright_points) != 1)[0] + 1)
    #Ex: [[100],[102,103,104]]
    #find the lengths of each split array
    lengths = []
    for arr in consec_portions:
       lengths.append(len(arr))

    #Return the max value of lengths, the largest number of
    #consecutive points in the bright epoch
    num_consec_points = max(lengths)
    return num_consec_points


def declare_flare(time, flux, time_candidate, flux_std, threshold_std, num_consec,
                           prim_marginal_threshold, num_below_threshold):
    '''Arguments
        time: 1D array of ALL the time coordinates in the flattened light curve
        flux: 1D array of ALL the flux coordinates in the flattened light curve
        time_candidate: Index in time (and flux) of the peak of the flare
        flux_std: Global spread of detrended flux values
        threshold_std: Desired sigma of the lightcurve to find a flare
        num_required: Number of required consecutive points to be considered a flare. Equal to the value
                      described in equation 3d from Chang 2015
        prim_marginal_threshold: The sigma threshold considered for a point to be dim enough to longer be associated with a flare.
                                 Default value of 2 means that we're looking for flux values below 2σ to end the flare
    '''
    time_peak_flare_index = np.where(time == time_candidate)[0][0] #identify the index of the flare peak
    num_consec_points = consec_finder(time, flux, time_candidate, flux_std, threshold_std,
                           prim_marginal_threshold, num_below_threshold)

    #If this value is greater than the required then we have a flare!!
    if num_consec_points >= num_consec:
        flare = True
        #calculate flare characteristics
        flare_peak = time_candidate
        #find start and end
        index_start, index_end = start_end_time(time, flux, time_candidate, flux_std, prim_marginal_threshold, num_below_threshold)
        flare_start = time[index_start]
        flare_end = time[index_end]
        flare_amp = flux[time_peak_flare_index] - 1
        flare_ED = np.trapz(flux[index_start:index_end+1] - 1, x = time[index_start:index_end+1])
        #covert to seconds
        days_to_seconds = 86400
        flare_ED = flare_ED * 86400
        flare_type = 'primary'
        num_points = index_end - index_start + 1
        num_abv_threshold = len(np.where(flux[index_start:index_end+1] > (1 + threshold_std * flux_std))[0])
        flare_amp_sigma = flare_amp/flux_std
        return (flare_peak, flare_start, flare_end, flare_amp, flare_ED, flare_type,
                num_points, num_abv_threshold, flare_amp_sigma)
        

    else:
        flare = False
        return flare


#Function for fitting any general function to lightcurve data
def model_fit(func, time, flux, flux_err, p0, loss = 'huber', bounds = (0, np.inf)):
    '''
    Inputs:
    func: function being fit to the data. Should be defined elsewhere in the code
    time: time coordinates from the lightcurve of the flare
    flux: flux values from the lightcurve from the flare
    flux_err: error values of the flux measurements from the flare.
              We're leaving it defaulted to None in case people just want to fit the data
              under the (mostly good) assumption that the error is uniform
    p0: initial guesses of the parameters for the function
    loss: loss function used to fit the function
    bounds: 2-tuple object holding the bounds for the passed parameters. If None the default bounds of (-inf, inf)
            will be assumed for all variables.

    Returns:
    params_opt: 1D array of the optimal parameters for the fit listed in the order as
                they're defined in the original func
    perr: error on the parameter estimates, again listed in the order as they're defined in the original func
    '''

    #We have the function for the model, but we need to calculate
    #The residuals that will be minimized in the least squares
    #Specifically: it minimizes sum(res**2)
    def func_residuals(params, time, flux, flux_err):
        model_flux = func(time, *params)
        if flux_err is not None:
            residuals = (model_flux - flux)/flux_err
        else:
            residuals = (model_flux - flux)
        return residuals

    #Now with residuals we can run least squares
    if bounds == None:
        least_squares_fit = scipy.optimize.least_squares(func_residuals, p0, args=(time, flux, flux_err), loss = loss)
    else:
        least_squares_fit = scipy.optimize.least_squares(func_residuals, p0, args=(time, flux, flux_err),
                                                         loss = loss, bounds = bounds)

    #Pull out best fit parameters
    params_opt = least_squares_fit.x

    #calculate their errors from the Jacobian
    from scipy.linalg import svd
    _, s, VT = svd(least_squares_fit.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(least_squares_fit.jac.shape) * s[0]
    s = s[s > threshold]
    vt = VT[:len(s)]
    pcov = np.dot(vt.T / s**2, vt)
    perr = np.sqrt(np.diag(pcov))
    return params_opt, perr


#find the residuals between the best fit function and the flux points
def flare_residuals(func, time, flux, flux_err, p0, loss = 'huber', bounds = (0, np.inf)):
    #run best fit
    params_opt = model_fit(func, time, flux, flux_err, p0, loss = loss, bounds = bounds)[0]
    #find model fluxes
    model_fluxes = func(time, *params_opt)
    #find residuals
    residuals = flux - model_fluxes
    return residuals


#find the residuals between the best fit function and the flux points
def flare_residuals(func, time, flux, flux_err, p0, loss = 'huber'):
    #run best fit
    params_opt = model_fit(func, time, flux, flux_err, p0, loss)[0]
    #find model fluxes
    model_fluxes = func(time, *params_opt)
    #find residuals
    residuals = flux - model_fluxes
    return residuals

#Function to adjust the initial guesses of a fit if they're not
#in the bounds
def adjust_bounds(p_0, bounds):
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    
    #check the bounds are above the lower bounds and below the upper bound
    above_lower = np.all((p_0 >= lower_bounds))
    below_upper = np.all((p_0 <= upper_bounds))
    
    if above_lower == False: #if there's an error
        if type(lower_bounds) != list: #if there's just broad error, not itemized by input
            bounds_mask = p_0 < lower_bounds #True values correspond to problematic values
            p_0[bounds_mask] = lower_bounds + 1 #adjust values to be above minimum
        else: #for list of lower bounds
            for i in range(len(p_0)): #iterate through bounds and adjust them
                if p_0[i] < lower_bounds[i]: #if the initial guess is too small
                    p_0[i] = lower_bounds[i] + 1 #adjust it to be greater than lower bound
    
    if below_upper == False: #if there's an error
        if type(upper_bounds) != list: #if there's just broad error, not itemized by input
            bounds_mask = p_0 > upper_bounds #True values correspond to problematic values
            p_0[bounds_mask] = upper_bounds - 1 #adjust values to be below maximum
        else: #for list of lower bounds
            for i in range(len(p_0)): #iterate through bounds and adjust them
                if p_0[i] > upper_bounds[i]: #if the initial guess is too big
                    p_0[i] = upper_bounds[i] - 1 #adjust it to be less than than upper bound
    return p_0

def fit_gaussian_rise(time, flux, flux_err, flare_peak_time, loss = 'huber', bounds = None):
    #define function
    def guassian_rise(x, alpha, sigma, c):
        return alpha * np.exp(-(x - flare_peak_time)**2 / (2 * (sigma)**2)) + c

    #set initial guesses for guassian rise
    alpha_i = flux[-1]
    sigma_i = (time[-1] - time[0])/2
    c_i = 1
    
    p0 = np.array([alpha_i, sigma_i, c_i])

    #Check these initial guesses are within the bounds of fitting
    if bounds != None:
        p0 = adjust_bounds(p0, bounds)

    #set sigma of the peak point to be super small for force fit to go through it
    sigma = flux_err
    #sigma[-1] = sigma[-1] * 1e-3 #make super small
    #sigma[0] = sigma[0] * 1e-3

    #fit for parameters and residuals
    params_opt, perr = model_fit(guassian_rise, time, flux, sigma, p0, loss = loss, bounds = bounds)
    residuals = flare_residuals(guassian_rise, time, flux, sigma, p0, loss = loss, bounds = bounds)
    return params_opt, perr, residuals

# First, create one for double exponential decay
def fit_dbl_exp_decay(time, flux, flux_err, flare_peak_time, loss = 'huber', bounds = (0, np.inf)):
    #define double exponential decay function for fitting
    def dbl_exp_decay(x, alpha_0, beta_0, alpha_1, beta_1, C):
        return (alpha_0 * np.exp(- beta_0 * (x - flare_peak_time)) +
            alpha_1 * np.exp(- beta_1 * (x - flare_peak_time))  + C)

    #initial parameter guess
    #alphas should add up to flare flux
    alpha_0_i = 0.67 * flux[0]
    alpha_1_i = 0.33 * flux[0]
    
    #betas, got no idea, but it's like in the hundreds and one of them should be much smaller
    beta_0_i = 2000
    beta_1_i = 5
    C_i = 1
    #intial guess array
    p0 = np.array([alpha_0_i, beta_0_i, alpha_1_i, beta_1_i, C_i])

    #we need a dummy check to make sure the initial guesses are within the set
    #bounds if we have any
    if bounds != None:
        p0 = adjust_bounds(p0, bounds)

    #set sigma of the first point to be super small for force fit to go through it
    sigma = flux_err
    #sigma[0] = sigma[0] * 1e-3 #make super small
    #sigma[-1] = sigma[-1] * 1e-3
    
    #fit for parameters and residuals
    params_opt, perr = model_fit(dbl_exp_decay, time, flux, sigma, p0, loss = loss, bounds = bounds)
    residuals = flare_residuals(dbl_exp_decay, time, flux, sigma, p0, loss = loss, bounds = bounds)
    #done
    return params_opt, perr, residuals


#create routine for fitting with product gompertz
def fit_w_gompertz_decay(time, flux, flux_err, flare_peak_time, loss = 'huber', bounds = (0, np.inf)):
    #define the function
    def product_gompertz(x, alpha_0, beta_0, alpha_1, beta_1, C):
    #alpha_0, beta_0, alpha_1, beta_1, C = theta
        return (np.exp(alpha_0 * np.exp(- beta_0 * (x - flare_peak_time))
                       + alpha_1 * np.exp(- beta_1 * (x - flare_peak_time))) + C)

    #log of the alphas should add up to flare flux
    A = np.log(flux[0]) #since everything in log initializing the leading coefficients is odd
    alpha_0_i = 0.67 * A
    alpha_1_i = 0.33 * A
    
    #betas, got no idea, but it's like in the hundreds or thousands and one of them should be much smaller
    #to capture the fact there are two timescales at play
    beta_0_i = 2000
    beta_1_i = 5
    
    C_i = 0
    
    p0_prd_gomp = np.array([alpha_0_i, beta_0_i, alpha_1_i, beta_1_i, C_i])

    #Check these initial guesses are within the bounds of fitting
    if bounds != None:
        p0_prd_gomp = adjust_bounds(p0_prd_gomp, bounds)
    

    #set sigma of the first point to be super small for force fit to go through it
    sigma = flux_err
    #sigma[0] = sigma[0] * 1e-3 #make super small
    #sigma[-1] = sigma[-1] * 1e-3

    #fit for parameters and residuals
    params_opt, perr = model_fit(product_gompertz, time, flux, sigma, p0_prd_gomp, loss = loss, bounds = bounds)
    residuals = flare_residuals(product_gompertz, time, flux, sigma, p0_prd_gomp, loss = loss, bounds = bounds)
    #done
    return params_opt, perr, residuals


#################ALT GOMP: LOGISTIC DECAY########################

def fit_w_gompertz_logistic(time, flux, flux_err, flare_peak_time, loss = 'huber', bounds = (0, np.inf)):
    #define the function
    def product_gompertz(x, A, alpha_0, beta_0, alpha_1, beta_1, C):
        #alpha_0, beta_0, alpha_1, beta_1, C = theta
        return (A * np.exp(alpha_0 * np.exp(- beta_0 * (x - flare_peak_time))
                           + alpha_1 * np.exp(- beta_1 * (x - flare_peak_time))) + C)

    #alphas should add up to flare flux
    A_i = -10
    alpha_0_i = -0.67 * flux[0]
    alpha_1_i = -0.33 * flux[0]
    
    #betas, got no idea, but it's like in the hundreds and one of them should be much smaller
    beta_0_i = 500
    beta_1_i = 100
    
    C_i = 0
    
    p0_prd_gomp = np.array([A_i, alpha_0_i, beta_0_i, alpha_1_i, beta_1_i, C_i])

    #Check these initial guesses are within the bounds of fitting
    if bounds != None:
        p0_prd_gomp = adjust_bounds(p0_prd_gomp, bounds)

    #set sigma of the first point to be super small for force fit to go through it
    sigma = flux_err
    #sigma[0] = sigma[0] * 1e-3 #make super small
    #sigma[-1] = sigma[-1] * 1e-3

    #fit for parameters and residuals
    params_opt, perr = model_fit(product_gompertz, time, flux, sigma, p0_prd_gomp, loss = loss, bounds = bounds)
    residuals = flare_residuals(product_gompertz, time, flux, sigma, p0_prd_gomp, loss = loss, bounds = bounds)
    #done
    return params_opt, perr, residuals

#Function that calls models to run through model_fit and flare_residuals
def call_decay_model(time, flux, flux_err, flare_peak_time, loss = 'huber', bounds = None, p0 = None,
                model = 'dbl_exp_decay', fit_peak = False):
    
    ##This function is essentially a bunch of check to see which model will actually be applied
    ## And returns to appropriate residuals and ideal params for the fit

    #Check if we're doing robust or super robust fitting
    if model == 'robust':
        params_opt, perr, residuals = robust_fitting(time, flux, flux_err, flare_peak_time, loss = loss, bounds = bounds, p0 = p0,
                   super_robust = False, fit_peak = fit_peak)
        return params_opt, perr, residuals
    
    if model == 'super_robust':
        params_opt, perr, residuals = robust_fitting(time, flux, flux_err, flare_peak_time, loss = loss, bounds = bounds, p0 = p0,
                   super_robust = True, fit_peak = fit_peak)
        return params_opt, perr, residuals
    
    #Selection if we're just doing one fitting

    #Check if we're fitting the peak at all
    if fit_peak == False:
        add_nan = False #initialize flag to see if we need to add nan values to resiudals
        if len(time) > 3: #only cut if the tail is long enough
            #trim off the first two cadences of the time, flux, flux_err
            time, flux, flux_err = time[2:], flux[2:], flux_err[2:]
            #Flag to see if we need to add nan values to residuals
            add_nan = True

    #Now go through the models to decide the fit
    if model == 'dbl_exp_decay':
        #Set recommended bounds if none are passed
        if bounds == None:
            bounds = (0, np.inf)
         #fit double exponential, we only care about the residuals here
        params_opt, perr, residuals = fit_dbl_exp_decay(time, flux, flux_err, flare_peak_time,
                                                                loss = loss, bounds = bounds)
        
    if model == 'gomp_decay':
        #Set recommended bounds if none are passed
        if bounds == None:
            bounds = (0, np.inf)
         #fit gompertz decay, we only care about the residuals here
        params_opt, perr, residuals = fit_w_gompertz_decay(time, flux, flux_err, flare_peak_time,
                                                                loss = loss, bounds = bounds)
        
    if model == 'gomp_logistic':
        #Set recommended bounds if none are passed
        if bounds == None:
            bounds = (-np.inf, np.inf)
         #fit gompertz decay, we only care about the residuals here
        params_opt, perr, residuals = fit_w_gompertz_logistic(time, flux, flux_err, flare_peak_time,
                                                                loss = loss, bounds = bounds)
        
    #Custom routine for new models if desired, pass the name of the func for the decay
    #BOUNDS AND P0 CANNOT BE NONE VALUE, THEY MUST BE PASSED FOR LEAST SQUARES TO RUN  
    if type(model) != str:
        #shift the times to be centered around zero for easy fitting
        time = time - flare_peak_time
        #fit the function to get ideal params and residuals
        params_opt, perr = model_fit(model, time, flux, flux_err, p0, loss = loss, bounds = bounds)
        residuals = flare_residuals(model, time, flux, flux_err, p0, loss = loss, bounds = bounds)
        #shift the times back the real times
        time = time + flare_peak_time

    #Check later if we fit without the peak to inser np.nan into the first two indices of the residuals
    if fit_peak == False:
        if add_nan == True:
            #now, notice the length residual array is short of the actual fluxes in the decay by two
            #we have no residuals for the first two cadences, we cut them out
            #insert nan values for first two residuals to match residuals with actual times and fluxes
            residuals = np.insert(residuals, [0, 0], np.nan)

    return params_opt, perr, residuals

#Special Routines for super robust fitting which fits four functions. Created for streamlined approach to later function
def super_fitting(time, flux, flux_err, flare_peak_time, loss = 'huber', bounds = None, p0 = None):
    #Fit both gompertz decay and double exponential decay with the full deacy
    
    #Set recommended bounds if none are passed
    if bounds == None:
        bounds = (0, np.inf)
        #fit double exponential, we only care about the residuals here
    params_opt_exp, perr_exp, residuals_exp = fit_dbl_exp_decay(time, flux, flux_err, flare_peak_time,
                                                            loss = loss, bounds = bounds)
    #fit gompertz decay, we only care about the residuals here
    params_opt_gomp, perr_gomp, residuals_gomp = fit_w_gompertz_decay(time, flux, flux_err, flare_peak_time,
                                                            loss = loss, bounds = bounds)
    #combine the results
    params_opt, perr, residuals = (np.array([params_opt_exp, params_opt_gomp]),
                                    np.array([perr_exp, perr_gomp]),
                                    np.array([residuals_exp, residuals_gomp]))

    #Fit without the peak
    #If the flare is short there's no point and we'll just return the full residuals as above
    if len(time) <= 3:
        return params_opt, perr, residuals

    #If long enough, trim off the first two cadences of the time, flux, flux_err
    time, flux, flux_err = time[2:], flux[2:], flux_err[2:]
    #fir again, without the peaks this time
    params_opt_exp_wo_peak, perr_exp_wo_peak, residuals_exp_wo_peak = fit_dbl_exp_decay(time, flux, flux_err, flare_peak_time,
                                                            loss = loss, bounds = bounds)
    #insert nan values for the first two indices to match length of fitting with peak
    residuals_exp_wo_peak = np.insert(residuals_exp_wo_peak, [0, 0], np.nan)

    #fit gompertz decay, we only care about the residuals here
    params_opt_gomp_wo_peak, perr_gomp_wo_peak, residuals_gomp_wo_peak = fit_w_gompertz_decay(time, flux, flux_err, flare_peak_time,
                                                            loss = loss, bounds = bounds)
    #insert nan values for the first two indices to match length of fitting with peak
    residuals_gomp_wo_peak = np.insert(residuals_gomp_wo_peak, [0, 0], np.nan)
    
    #combine the results
    params_opt, perr, residuals = (np.array([params_opt_exp, params_opt_gomp, params_opt_exp_wo_peak, params_opt_gomp_wo_peak]),
                                    np.array([perr_exp, perr_gomp, perr_exp_wo_peak, perr_gomp_wo_peak]),
                                    np.array([residuals_exp, residuals_gomp, residuals_exp_wo_peak, residuals_gomp_wo_peak]),)
    
    return params_opt, perr, residuals
    

#Special Routines for robust fitting which fits multiple functions
def robust_fitting(time, flux, flux_err, flare_peak_time, loss = 'huber', bounds = None, p0 = None,
                   super_robust = False, fit_peak = False):
    #This function fits multiple functions and returns all their residuals and ideal
    #params for fitting

    #Check whether we're doing super fitting and fitting gompertz decay and double exponential
    #both with and without the peaks. Four total fittings
    if super_robust == True:
        params_opt, perr, residuals = super_fitting(time, flux, flux_err, flare_peak_time, loss = 'huber', bounds = None, p0 = None)
        return params_opt, perr, residuals
    
    #second, check if we're fitting the peak at all
    if fit_peak == False:
        add_nan = False #initialize flag to see if we need to add nan values to resiudals
        if len(time) > 3: #only cut if the tail is long enough
            #trim off the first two cadences of the time, flux, flux_err
            time, flux, flux_err = time[2:], flux[2:], flux_err[2:]
            #Flag to see if we need to add nan values to residuals
            add_nan = True

    #Set recommended bounds if none are passed
    if bounds == None:
        bounds = (0, np.inf)
        #fit double exponential, we only care about the residuals here
    params_opt_exp, perr_exp, residuals_exp = fit_dbl_exp_decay(time, flux, flux_err, flare_peak_time,
                                                            loss = loss, bounds = bounds)
    #fit gompertz decay, we only care about the residuals here
    params_opt_gomp, perr_gomp, residuals_gomp = fit_w_gompertz_decay(time, flux, flux_err, flare_peak_time,
                                                            loss = loss, bounds = bounds)
    #combine the results
    params_opt, perr, residuals = (np.array([params_opt_exp, params_opt_gomp]),
                                   np.array([perr_exp, perr_gomp]),
                                   np.array([residuals_exp, residuals_gomp]))

    #Check if we fit without the peak to inser np.nan into the first two indices of the residuals
    if fit_peak == False:
        if add_nan == True:
            #insert nan values for first two residuals to match residuals with actual times and fluxes
            #for this we need to create two dimensional array of nans to add to the left of the residuals
            nans = np.full((residuals.shape[0], 2), np.nan)
            residuals = np.hstack([nans, residuals])

    return params_opt, perr, residuals

#Look through the residuals
def find_secondaries(residuals, flux_std, secondary_threshold, num_consec_sec, dt,
                     fit_multiple_secs = False):
    '''Inputs
    resdiuals: 1D array-like holding the residuals between the measured fluxes and the best fit models
    flux_std: Value for the flux spread
    secondary_threshold: sigma threshold for detecting secondary flare event in the residuals
    num_consec_sec: the number of consecutive points required in the residuals above the threshold for a detection
    dt = difference in time coordinates, set by the cadence
    '''

    if len(residuals.shape) == 1: #if the array is just one dimensional
        #Find points above the threshold
        bright_res = np.where(residuals > secondary_threshold * flux_std)[0]

    #If we're comparing the fits from multiple models we'll have multiple residuals
    #We need to find the bright residuals for each and find the common values where
    #all fits agree
    else:
        indices = [np.where(row > secondary_threshold * flux_std)[0].tolist() for row in residuals]
        #Bright points indices found for all fits
        bright_res = reduce(np.intersect1d, indices)
        #We need to reduce the residuals down to one list of values
        #We'll take the averages of the values
        residuals = np.mean(residuals, axis = 0)

    #Split this array into the individual bright epochs of consecutive points
    res_splits = np.split(bright_res, np.where(np.diff(bright_res) != 1)[0] + 1)

    #initialize secondary detection
    secondary = False
    #Iterate through splits to see if they're flares
    flare_peaks = []
    flare_starts = []
    flare_ends = []
    flare_amps = []
    flare_EDs = []
    flare_types = []
    num_points = []
    num_abv_threshold = []
    amp_sigmas = []
    for epoch in res_splits:
        if len(epoch) >= num_consec_sec: #if it's long enough we have a flare!
            #The exact start, end, peak time need to be found later with the known peak time
            #And whether this flare has been found to the left or right of the peak
            #these starts and ends are indices in the residuals and are thus relative to the peak index
            flare_peaks.append(np.argmax(epoch))
            flare_starts.append(min(epoch))
            flare_ends.append(max(epoch))
            #But we can find the non-time related quantities easily!
            flare_amps.append(max(residuals[epoch]))
            flare_ED = np.trapz(residuals[epoch], x = np.arange(0, len(epoch) * dt, dt))
            #covert to seconds
            days_to_seconds = 86400
            flare_ED = flare_ED * 86400
            flare_EDs.append(flare_ED)
            if secondary == True: #if we've previously found a secondary this must be a higher order flare
                flare_types.append('tertiary')
            else:
                flare_types.append('secondary')
            num_points.append(len(epoch))
            num_abv_threshold.append(len(epoch))
            amp_sigmas.append(max(residuals[epoch])/flux_std)
            secondary = True

    #If we don't want to include multuple secondaries then let's only keep the LARGEST found flare
    #by amplitude
    if fit_multiple_secs == False:
        if secondary == True:
            #Find which flare had the largest amplitude
            biggest_sec = np.argmax(flare_amps)
            #Only keep the flare signature from this flare
            flare_peaks = [flare_peaks[biggest_sec]]
            flare_starts = [flare_starts[biggest_sec]]
            flare_ends = [flare_ends[biggest_sec]]
            flare_amps = [flare_amps[biggest_sec]]
            flare_EDs = [flare_EDs[biggest_sec]]
            flare_types = ['secondary']
            num_points = [num_points[biggest_sec]]
            num_abv_threshold = [num_abv_threshold[biggest_sec]]
            amp_sigmas = [amp_sigmas[biggest_sec]]
            
    #if there was a flare return it's attributes
    if secondary == True:
        return flare_peaks, flare_starts, flare_ends, flare_amps, flare_EDs, flare_types, num_points, num_abv_threshold, amp_sigmas

    #If no flare was found then return a false reading
    else:
        return False


def detect_flares(time, flux, flux_err, quality, primary_threshold = 3.0, secondary_threshold = 3.0,
                  num_consec = 3, num_consec_sec = 3, loss = 'huber',
                  rise_bounds = None, rise_func = 'gaussian',
                  decay_func = 'dbl_exp_decay', fit_peak = False, p0_decay = None, decay_bounds = None,
                  fit_multiple_secs = True,
                  prim_marginal_threshold = 3.0, num_below_threshold = 3,
                  min_break = 0.25, clip_breaks = 200, flag_values = [0]):
    
    # Yes, it's a lot of inputs, but most of them are defaulted to pretty good values. I'd recommend only toying
    #With primary_threshold, secondary_threshold, num_consec, & num_consec_sec. 
    #You can also change flag values to [0,512] which isn't a bad idea

    '''All in one function to take detrended lightcurve and return the detected flares

    LIGHTCURVE INPUTS:
    time: the time coordinates of the lightcurve in BJD. The detrending utilizes the orbit timing so these should be
          in the default units from lightkurve.
    flux: 1D array of Photometric flux. Ideally, detrended. The code will RUN on raw data but it won't work well.
    flux_err: 1D array of Photometric error on the cadences. Make sure to have these be normalized if you've detrended!
    quality: quality flag of the cadence from TESS. We utilize filtering to remove poor quality readings from the
             lightcurves to ensure more accurate detection. If you want to skip this step then you can just make
             an array of equal length to the time, flux, and flux_err arrays and fill it with the value 0.
             
    DETECTION:
    primary_threshold: Integer or float value greater than 0. Sets the threshold for detection of primary flares
                       in terms of sigma of the global spread of photometry points. Default value is set to 3σ,
                       which is the common value for threshold-based detection.
    secondary_threshold: Integer or float value greater than 0. Sets the sigma threshold for detection of secondary flares
                         within the residuals between the primary and best-fit model. Default value is set to 2σ.
    num_consec: Number of consecutive points required to be above the primary detection threshold to be called a primary flare.
                Same as N_3 from equation 3d) from Chang et al 2015
    num_consec: Number of consecutive points required to be above the secondary detection threshold to be called
                a secondary flare. Same as num_consec but specifically for finding secondaries in the residuals

    FLARE MODELING:
    rise_func: function to model the rise portion of a flare.
               Default value corresponds to default use of gaussian rise.
    rise_bounds: Bounds for the chosen rise function. Use 2-tuple bounds object or array like with shape consistent
                 with the number of free parameters in the function chosen.
    decay_func: function to model the decay portion of a flare.
               Default value corresponds to default use of double exponential decay on full decay.
               Other elligible values are gomp_decay, gomp_logistic, robust, and super_robust.
    decay_bounds: Bounds for the chosen rise function. Use 2-tuple bounds object or array like with shape consistent
                  with the number of free parameters in the function chosen. Default ensures all the values are positive
                  but None can also be set if no bounds are desired.

    SETTING BEGINNING AND END TIMES OF FLARES:
    prim_marginal_threshold: Sets the threshold to consider the start and end of a flare. A default value of 3 corresponds
                             to points needing to be 3σ or larger from the median to be considered a part of a flare. A value
                             of 2.0 or 2.5 aren't bad ideas.
    num_below_threshold: Sets how many points need to be below prim_marginal_threshold in order for a bright
                         epoch to be considered returned to quiescent level. A default value of 3 means there must be
                         3 consecutive points below prim_marginal_threshold to mark the beginning and end of a flare.
                         Many people adopt the value of 2.
             
    LIGHTCURVE PREP:
    min_break: float value represening the smallest duration of a break in the lightcurve (in days) for which TOFFEE will
               mask out points on either side. Default is set to 0.25, the same value as the default window length for Wotan.
               In Pratt et al 2025 we used a value of 0.025 days to be super conservative.
    clip_breaks: The number of cadences to clip out on either side of a break detected by min_break. Default is 100 cadences
                 which corresponds to 200 minutes of data in 120 sec cadence data from TESS.
    flag_values: A list of the acceptable flag values you want to include in the analysis. By default we only
                 permit cadences with a perfect quality of zero into the analysis. However it has been shown that
                 points with a flag value of 512 can correspond to the peaks of flares and so can be kept in. For that
                 you can pass flag_values = [0, 512] to include them. The other will be masked out.
    '''

    #################LIGHTCURVE PREP#################

    ######REMOVE THE LOW QUALITY POINTS######
    #perfect points is the quality mask only containing cadences with the desired quality flag(s)
    perfect_points = []

    #loop through quality of points
    for q in quality:
        #if the quality of this cadence is acceptable
        if q in flag_values:
            #don't mask out
            perfect_points.append(True)
        else:
            #mask out poor points
            perfect_points.append(False)

    #unpack good values of flux and time for the star
    flux = flux[perfect_points]
    time = time[perfect_points][np.isnan(flux) == False]
    flux_err = flux_err[perfect_points][np.isnan(flux) == False]
    flux = flux[np.isnan(flux) == False]
    
    ######CLIP OFF THE BREAKS######

    #Apply a mask to cut out points on either side of the breaks
    if clip_breaks != None:
        break_mask = light_curve_mask(time, flux, min_break = min_break, clip_breaks = clip_breaks)
        #apply to light curve
        flux = flux[break_mask]
        time = time[break_mask]
        flux_err = flux_err[break_mask]


    #################IDENTIFY BRIGHT POINTS AS POTENTIAL FLARES##################

    #Find the spread of the flux
    one_sigma_percentile = 84
    flux_std = np.nanpercentile(flux - 1, one_sigma_percentile)

    time_candidates, bright_points = find_candidates(time, flux, flux_std, primary_threshold)


    #################WITH THE TIMES AND FLUXES OF BRIGHT POINTS ITERATE AND DETECT FLARES#################
    #Initialize arrays for the flare attributes
    flare_peak_times = []
    flare_start_times = []
    flare_end_times = []
    flare_amps = []
    flare_equivalent_durations = []
    primary_or_secondary = []
    points_in_flare = []
    points_abv_threshold = []
    amp_sigma = []

    i = 0
    while len(time_candidates) > 2: #Keep running until we exhaust all the time candidates
        #################FIND THE START AND END TIMES#################
        #This is done to remove redundant points even if it's not a flare event
        #pull out peak time
        flare_peak_time = time_candidates[i]
        flare_peak_index = np.where(time == flare_peak_time)[0][0]

        #start and end times via their indices
        index_start, index_end = start_end_time(time, flux, flare_peak_time, flux_std,
                           prim_marginal_threshold, num_below_threshold)
        flare_start_time = time[index_start]
        flare_end_time = time[index_end]
        #also useful now to pull out the flare times, fluxes, and flux_err
        flare_times = time[index_start:index_end+1]
        flare_fluxes = flux[index_start:index_end+1]
        flare_flux_errs = flux_err[index_start:index_end+1]

        ## All the points between the start and end are redundant and need to be tossed from the
        #candidates
        redundant_points = np.where((time_candidates <= flare_end_time) &
                                    (time_candidates >= flare_start_time))[0]
        
        #Cut these points from the candidates
        time_candidates = np.delete(time_candidates, redundant_points)
        bright_points = np.delete(bright_points, redundant_points)

        #################DETERMINE IF THIS CANDIDATE IS A FLARE#################
        
        start_end_time(time, flux, flare_peak_time, flux_std,
                           prim_marginal_threshold, num_below_threshold)
        consec_points = consec_finder(time, flux, flare_peak_time, flux_std, primary_threshold,
                           prim_marginal_threshold, num_below_threshold)
        flare_result = declare_flare(time, flux, flare_peak_time, flux_std, primary_threshold, num_consec,
                           prim_marginal_threshold, num_below_threshold)
        
        #Flare result of False means nothing was found, let's pull out the attributes
        if flare_result != False:
            flare_peak_times.append(flare_result[0])
            flare_start_times.append(flare_result[1])
            flare_end_times.append(flare_result[2])
            flare_amps.append(flare_result[3])
            flare_equivalent_durations.append(flare_result[4])
            primary_or_secondary.append(flare_result[5])
            points_in_flare.append(flare_result[6])
            points_abv_threshold.append(flare_result[7])
            amp_sigma.append(flare_result[8])

        #if there's no flare, continue to next candidate, we already tossed the redundant points
        if flare_result == False:
            #key point, DONT increase the index, we're tossing
            #out this data point so we want to re-evaluate the index i
            i = i
            continue

        #print(flare_result)
        dt = flare_times[1] - flare_times[0]

        #################LEFT SEARCH FOR SECONDARIES#################
        rise_time = time[index_start:flare_peak_index+1]
        rise_flux = flux[index_start:flare_peak_index+1]
        rise_flux_err = flux_err[index_start:flare_peak_index+1]
        flare_amp = flare_result[3] #from the flare result of the primary
        
        #If the peak is the first point in the flare, then obviously there are no
        #secondaries to the left
        if flare_peak_index - index_start == 0:
            rise_residuals = np.array([0])

        else:
        #go through fitting depending on function    
            if rise_func == 'gaussian':
                #fit gaussian, we only care about the residuals here
                params_opt, perr, rise_residuals = fit_gaussian_rise(rise_time, rise_flux, rise_flux_err,
                                                                     flare_peak_time, loss = loss, bounds = rise_bounds)
        
        
        ###LOOK FOR SECONDARIES###
        #When looking for secondaries, let's just use the median flux error as the threshold since
        #global systematics shouldn't really change flare shape
        rise_flux_err_med = np.nanmedian(rise_flux_err)
        sec_results = find_secondaries(rise_residuals, rise_flux_err_med, secondary_threshold, num_consec_sec, dt,
                      fit_multiple_secs = fit_multiple_secs)

        if sec_results != False:
            peaks, starts, ends, sec_amps, flare_EDs, flare_types, num_points, num_abv_thresh, amp_sigmas = sec_results
            #Still need to shift the peak, start and end times to the correct times. The values above are only
            #the indices in the residuals
            peaks = flare_peak_index - len(rise_residuals) + peaks
            peaks = time[peaks]
            starts = flare_peak_index - len(rise_residuals) + starts
            starts = time[starts]
            ends = flare_peak_index - len(rise_residuals) + ends
            ends = time[ends]

            #Additional Detail, subtract these EDs from the primary
            flare_equivalent_durations[-1] = flare_equivalent_durations[-1] - np.sum(flare_EDs)

            #Add these flares to the total
            flare_peak_times.extend(peaks)
            flare_start_times.extend(starts)
            flare_end_times.extend(ends)
            flare_amps.extend(sec_amps)
            flare_equivalent_durations.extend(flare_EDs)
            primary_or_secondary.extend(flare_types)
            points_in_flare.extend(num_points)
            points_abv_threshold.extend(num_abv_thresh)
            amp_sigma.extend(amp_sigmas)
            

        #################RIGHT SEARCH FOR SECONDARIES#################
        decay_time = time[flare_peak_index:index_end+1]
        decay_flux = flux[flare_peak_index:index_end+1]
        decay_flux_err = flux_err[flare_peak_index:index_end+1]
        peak_flux = flux[flare_peak_index]

        #Perform the fitting and get the residuals for the default model of the flare decay
            
        params_opt, perr, decay_residuals = call_decay_model(decay_time, decay_flux, decay_flux_err, flare_peak_time,
                                                              loss = loss, bounds = decay_bounds, p0 = p0_decay,
                                                              model = decay_func, fit_peak = fit_peak)

        ###LOOK FOR SECONDARIES###
        #When looking for secondaries, let's just use the median flux error as the threshold since
        #global systematics shouldn't really change flare shape
        decay_flux_err_med = np.nanmedian(decay_flux_err)
        sec_results = find_secondaries(decay_residuals, decay_flux_err_med, secondary_threshold, num_consec_sec, dt,
                     fit_multiple_secs = fit_multiple_secs)

        if sec_results != False:
            peaks, starts, ends, sec_amps, flare_EDs, flare_types, num_points, num_abv_thresh, amp_sigmas = sec_results
            #Still need to shift the peak, start and end times to the correct times. The values above are only
            #the indices in the residuals
            peaks = flare_peak_index + peaks
            peaks =  time[peaks]
            starts = flare_peak_index + starts
            starts = time[starts]
            ends = flare_peak_index + ends
            ends = time[ends]
    
            #Additional Detail, subtract these EDs from the primary
            #A bit tougher now, find the last thing labeled a primary
            last_prim = [primary_or_secondary == 'primary'][-1]
            flare_equivalent_durations[last_prim] = flare_equivalent_durations[last_prim] - np.sum(flare_EDs)
    
            #Add these flares to the total
            flare_peak_times.extend(peaks)
            flare_start_times.extend(starts)
            flare_end_times.extend(ends)
            flare_amps.extend(sec_amps)
            flare_equivalent_durations.extend(flare_EDs)
            primary_or_secondary.extend(flare_types)
            points_in_flare.extend(num_points)
            points_abv_threshold.extend(num_abv_thresh)
            amp_sigma.extend(amp_sigmas)

        #condition to break, if the next iteration will have an i that is out of
        #range of the array, we'll end the for loop
        if i >= (len(time_candidates)):
            break

    #convert everything to numpy array
    #as a dummy check we're not double counting anything let's use np.unique to
    #only get unique entries
    flare_peak_times, unique_index = np.unique(np.array(flare_peak_times), return_index = True)
    flare_start_times = np.array(flare_start_times)[unique_index]
    #print('End Times: ', flare_end_times)
    flare_end_times = np.array(flare_end_times)[unique_index]
    #print('Amps: ', flare_amps)
    #print('Types: ', primary_or_secondary)
    flare_amps = np.array(flare_amps)[unique_index]
    flare_equivalent_durations = np.array(flare_equivalent_durations)[unique_index]
    primary_or_secondary = np.array(primary_or_secondary)[unique_index]
    points_in_flare = np.array(points_in_flare)[unique_index]
    points_abv_threshold = np.array(points_abv_threshold)[unique_index]
    amp_sigma = np.array(amp_sigma)[unique_index]

    #return as pandas data frame
    #Make as dictionary
    flare_results = {'T_peak' : flare_peak_times,
                     'T_start' : flare_start_times,
                     'T_end' : flare_end_times,
                     'Amplitude' : flare_amps,
                     'ED' : flare_equivalent_durations,
                     'Primary_or_Secondary' : primary_or_secondary,
                     'Num_Points' : points_in_flare,
                     'Num_Abv_Threshold' : points_abv_threshold,
                     'Amp_Sigma' : amp_sigma
    }
    #Transform into DataFrame
    flare_results = pd.DataFrame(flare_results)

    return flare_results


def visualizer(time, flux, flux_err, quality, primary_threshold = 3.0, secondary_threshold = 3.0,
               num_consec = 3, num_consec_sec = 3, loss = 'huber',
               rise_bounds = None, rise_func = 'gaussian',
               decay_func = 'dbl_exp_decay', fit_peak = False, p0_decay = None, decay_bounds = None,
               fit_multiple_secs = True,
               prim_marginal_threshold = 3.0, num_below_threshold = 3,
               min_break = 0.25, clip_breaks = 200, flag_values = [0],
               primary_color = 'red', secondary_color = 'blue', tertiary_color = 'green', cadence_color = 'black',
               fontsize = 24, labelsize = 18, Target = None, Sector = None):

    mask = light_curve_mask(time, flux, min_break = 0.25, clip_breaks = 200)
    flux_std = np.nanpercentile(flux - 1, 84)
    
    #find flares
    flare_results = detect_flares(time, flux, flux_err, quality, primary_threshold = primary_threshold,
                                  secondary_threshold = secondary_threshold,
                                  num_consec = num_consec, num_consec_sec = num_consec_sec, loss = loss,
                                  rise_bounds = rise_bounds, rise_func = rise_func,
                                  decay_func = decay_func, fit_peak = fit_peak, p0_decay = p0_decay, decay_bounds = decay_bounds,
                                  fit_multiple_secs = fit_multiple_secs,
                                  prim_marginal_threshold = prim_marginal_threshold,
                                  num_below_threshold = num_below_threshold,
                                  min_break = min_break, clip_breaks = clip_breaks, flag_values = flag_values)
    
    #flare peak times
    flare_peak_times = flare_results['T_peak']
    #flare start times
    flare_start_times = flare_results['T_start']
    #flare end times
    flare_end_times = flare_results['T_end']
    #flare amps
    flare_amps = flare_results['Amplitude']
    #flare eds
    flare_eds = flare_results['ED']
    #flare type
    flare_type = flare_results['Primary_or_Secondary']
    #number of points associated with a flare
    num_points_in_flare = flare_results['Num_Points']
    #number of points above with a threshold
    num_abv_threshold = flare_results['Num_Abv_Threshold']
    #sigma amplitude
    amp_sigma = flare_results['Amp_Sigma']
                    
    ###########Let's find the points belonging to the flares###########
    
    #For primaries
    #array holding the times of the flare points
    times_of_primary_flares = np.array([])
    #array holding associated fluxes
    fluxes_of_primary_flares = np.array([])
    #array holding peak times
    peak_time_of_primary = np.array([])
    #array holding peak flux
    peak_flux_of_primary = np.array([])
    
    #array holding the times of the flare points
    times_of_secondary_flares = np.array([])
    #array holding associated fluxes
    fluxes_of_secondary_flares = np.array([])
    #array holding peak times
    peak_time_of_secondary = np.array([])
    #array holding peak flux
    peak_flux_of_secondary = np.array([])
    
    #array holding the times of the flare points
    times_of_tertiary_flares = np.array([])
    #array holding associated fluxes
    fluxes_of_tertiary_flares = np.array([])
    #array holding peak times
    peak_time_of_tertiary = np.array([])
    #array holding peak flux
    peak_flux_of_tertiary = np.array([])
    
    #iterate through the flares and find the relevant times
    for i in range(len(flare_start_times)):
        start = flare_start_times[i]
        end = flare_end_times[i]
        #find indices of flux points between these values
        flare_flux_points = np.where((time >= start) & (time <= end))[0]
        #and log those times in the flare
        flare_times = time[flare_flux_points]
        #and log those fluxes
        flare_fluxes = flux[flare_flux_points]
        #find the peak flux
        peak_flux = np.max(flare_fluxes)
        #and associated time
        peak_time = flare_times[np.argmax(flare_fluxes)]
        #sort depending on type
        if (flare_type[i] == 'primary'):  
            #add to the list
            times_of_primary_flares = np.append(times_of_primary_flares, flare_times)
            fluxes_of_primary_flares = np.append(fluxes_of_primary_flares, flare_fluxes)
            peak_flux_of_primary = np.append(peak_flux_of_primary, peak_flux)
            peak_time_of_primary = np.append(peak_time_of_primary, peak_time)
    
        elif (flare_type[i] == 'secondary'):
            #add to the list
            times_of_secondary_flares = np.append(times_of_secondary_flares, flare_times)
            fluxes_of_secondary_flares = np.append(fluxes_of_secondary_flares, flare_fluxes)
            peak_flux_of_secondary = np.append(peak_flux_of_secondary, peak_flux)
            peak_time_of_secondary = np.append(peak_time_of_secondary, peak_time)
    
        elif (flare_type[i] == 'tertiary'):
            #add to the list
            times_of_tertiary_flares = np.append(times_of_tertiary_flares, flare_times)
            fluxes_of_tertiary_flares = np.append(fluxes_of_tertiary_flares, flare_fluxes)
            peak_flux_of_tertiary = np.append(peak_flux_of_tertiary, peak_flux)
            peak_time_of_tertiary = np.append(peak_time_of_tertiary, peak_time)
    
    #First plot: Show the flattened lightcurve with the flares found and colored in
    plt.figure(figsize = (10,8))
    plt.scatter(time, flux, s = 6, color = 'gray')
    #plot all the good flux points not masked 
    
    plt.scatter(time[mask], flux[mask], s = 6, color = cadence_color, label = 'Used')
    #the plot for the residual periodogram result
    #plt.plot(time, flatt_trend, color = 'yellow')
    
    #color the points around the primary flares
    plt.scatter(times_of_primary_flares, fluxes_of_primary_flares, color = primary_color, s = 6)
    #color the points around the secondary flares
    plt.scatter(times_of_secondary_flares, fluxes_of_secondary_flares, color = secondary_color, s = 6, zorder = 10)
    #color the points around the tertiary flares
    plt.scatter(times_of_tertiary_flares, fluxes_of_tertiary_flares, color = tertiary_color, s = 6, zorder = 10)
    
    #add primary flare peaks
    plt.scatter(peak_time_of_primary, peak_flux_of_primary, marker = '*', s = 200, color = primary_color)
    #add secondary flare peaks
    plt.scatter(peak_time_of_secondary, peak_flux_of_secondary, marker = '*', s = 200, color = secondary_color)
    #add tertiary flare peaks
    plt.scatter(peak_time_of_tertiary, peak_flux_of_tertiary, marker = '*', s = 200, color = tertiary_color)
    #add three sigma line
    plt.hlines(primary_threshold * flux_std + 1, min(time), max(time), color = tertiary_color, linewidth = 2)
    plt.xlabel('Time (BJD)', fontsize = 24, font = 'Serif')
    plt.ylabel(r'Normalized Flux', fontsize = 24, font = 'Serif')
    plt.ylim(0.9, 1.25)
    plt.tick_params(direction = 'in', labelsize = 20, width = 3, length = 10, labelfontfamily = 'Serif')
    #plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = 24)
    plt.show()
    
    #################PRINT FLARE MORPHOLOGY AND FITTING#################
    
    for i in range(len(flare_peak_times)):
    
        if flare_type[i] == 'primary':
    
            #Unpack indices of the flare
            #flare peak time index
            flare_peak_index = np.where(time == flare_peak_times[i])[0][0]
            #flare start time index
            index_start = np.where(time == flare_start_times[i])[0][0]
            #flare end time index
            index_end = np.where(time == flare_end_times[i])[0][0]
        
            flare_peak_time = flare_peak_times[i]
            flare_times = time[index_start:index_end]
        
            #Unpack flare fluxes
            flare_peak_flux = flare_amps[i] + 1
        
            #################LEFT FIT#################
            #Perform model fit
            rise_time = time[index_start:flare_peak_index+1]
            rise_flux = flux[index_start:flare_peak_index+1]
            rise_flux_err = flux_err[index_start:flare_peak_index+1]

            #figure out which function to plot
            if rise_func == 'gaussian':
                def guassian_rise(x, alpha, sigma, c):
                    return alpha * np.exp(-(x - flare_peak_time)**2 / (2 * (sigma)**2)) + c
                #fit gaussian to the rise, get parameters and residuals
                p_opt_rise, perr, rise_residuals = fit_gaussian_rise(rise_time, rise_flux, rise_flux_err,
                                                                     flare_peak_time, loss = loss, bounds = rise_bounds)
            #set flux threshold for plotting
            flux_threshold = primary_threshold * flux_std + 1
            
            #Plot all flux points
            plt.figure(figsize = (10,8))
            plt.scatter(time, flux, s = 10, color = cadence_color, label = 'Background Points')
            plt.errorbar(time, flux, yerr = flux_err, linestyle = '', color = cadence_color, capsize = 3)
            #plot flare points
            plt.scatter(time[index_start:index_end+1], flux[index_start:index_end+1], s = 10,
                        color = primary_color, label = 'Primary Flare')
            plt.errorbar(time[index_start:index_end+1], flux[index_start:index_end+1],
                         yerr = flux_err[index_start:index_end+1], linestyle = '', color = primary_color, capsize = 3)
            
            #plot fitted function
            if rise_func == 'gaussian':
                plt.plot(rise_time, guassian_rise(rise_time, *p_opt_rise), label = 'Rise Fit',
                         color = secondary_color, linewidth = 2)
        
            plt.hlines(primary_threshold * flux_std + 1, flare_peak_time - 0.025, flare_peak_time + 0.1,
                       linewidth = 3, linestyle = '--', color = 'gray')
            plt.xlabel('Time (BJD)', fontsize = fontsize)
            plt.ylabel('Detrended flux', fontsize = fontsize)
            #Print title of target at sector for record keeping
            if (Target != None) and (Sector != None):
                plt.title('Target: ' + str(Target) + ' Sector: ' + str(Sector), fontsize = fontsize)
            if (Target == None) and (Sector != None):
                plt.title('Sector: ' + str(Sector), fontsize = fontsize)
            if (Target != None) and (Sector == None):
                plt.title('Target: ' + str(Target), fontsize = fontsize)
            
            plt.ylim(1 - flux_std, flare_peak_flux + 0.05)
            plt.xlim(flare_peak_time - 0.025, flare_peak_time + 0.1)
            plt.xticks([flare_peak_time - 0.02, flare_peak_time + 0.04, flare_peak_time + 0.1],
                       np.round([flare_peak_time - 0.02, flare_peak_time + 0.04, flare_peak_time + 0.1], 2))
            plt.tick_params(direction = 'in', labelsize = labelsize)
            plt.legend(fontsize = fontsize)
            plt.show()
        
            #Find the residuals            
            median_flux_err = np.nanmedian(flux_err)
            median_flux = np.nanmedian(flux)
            threshold = secondary_threshold * median_flux_err
            
            plt.figure(figsize = (10,8))
            plt.scatter(rise_time, rise_residuals, color = cadence_color)
            plt.errorbar(rise_time, rise_residuals, yerr = median_flux_err, color = cadence_color, capsize = 4)
            plt.hlines(threshold, rise_time[0], rise_time[-1],
                      linewidth = 3, linestyle = '--', color = 'gray')
            plt.xlabel('Time (BJD)', fontsize = fontsize)
            plt.ylabel('Residuals', fontsize = fontsize)
            plt.tick_params(direction = 'in', labelsize = labelsize)
            plt.show()
        
            #################RIGHT FIT#################
            decay_time = time[flare_peak_index:index_end+1]
            decay_flux = flux[flare_peak_index:index_end+1]
            decay_flux_err = flux_err[flare_peak_index:index_end+1]

            def dbl_exp_decay(x, alpha_0, beta_0, alpha_1, beta_1, C):
                #alpha_0, beta_0, alpha_1, beta_1, C = theta
                return (alpha_0 * np.exp(- beta_0 * (x - flare_peak_time)) +
                        alpha_1 * np.exp(- beta_1 * (x - flare_peak_time))  + C)
            def gompertz_decay(x, alpha_0, beta_0, alpha_1, beta_1, C):
                #alpha_0, beta_0, alpha_1, beta_1, C = theta
                return (np.exp(alpha_0 * np.exp(- beta_0 * (x - flare_peak_time))
                                + alpha_1 * np.exp(- beta_1 * (x - flare_peak_time))) + C)
            def gompertz_logistic(x, A, alpha_0, beta_0, alpha_1, beta_1, C):
                #alpha_0, beta_0, alpha_1, beta_1, C = theta
                return (A * np.exp(alpha_0 * np.exp(- beta_0 * (x - flare_peak_time))
                                + alpha_1 * np.exp(- beta_1 * (x - flare_peak_time))) + C)
            
            #figure out which function to plot
            p_opt_decay, perr, decay_residuals = call_decay_model(decay_time, decay_flux, decay_flux_err, flare_peak_time,
                                                              loss = loss, bounds = decay_bounds, p0 = p0_decay,
                                                              model = decay_func, fit_peak = fit_peak)
            
            ###########ROBUST FITTING###########
            if decay_func == 'robust':
                ####Fit regular double exponential decay###
                p_opt_dbl_exp, perr, dbl_exp_residuals = p_opt_decay[0], perr[0], decay_residuals[0]
                ####Fit product gompertz###
                p_opt_gompertz, perr, gompertz_residuals = p_opt_decay[1], perr[1], decay_residuals[1]
                
            if decay_func == 'super_robust':
                ####Fit regular double exponential decay###
                p_opt_dbl_exp, perr_dbl_exp, dbl_exp_residuals = p_opt_decay[0], perr[0], decay_residuals[0]
                ####Fit product gompertz###
                p_opt_gompertz, perr_gomp, gompertz_residuals = p_opt_decay[1], perr[1], decay_residuals[1]
                if len(p_opt_decay) > 2: #for short flares we didn't do the fitting without the peaks
                    ####Fit dbl exponential w/o peak#
                    p_opt_dbl_exp_wo_peak, perr_dbl_exp_wo_peak, dbl_exp_residuals_wo_peak =  p_opt_decay[2], perr[2], decay_residuals[2]
                    ####Fit product gompertz w/o peak#
                    p_opt_gompertz_wo_peak, perr_gomp_wo_peak, gompertz_residuals_wo_peak = p_opt_decay[3], perr[3], decay_residuals[3]
            
            #Set threshold for plotting
            flux_threshold = primary_threshold * flux_std + 1

            #Plot all flux points
            plt.figure(figsize = (10,8))
            plt.scatter(time, flux, s = 10, color = cadence_color, label = 'Background Points')
            plt.errorbar(time, flux, yerr = flux_err, linestyle = '', color = cadence_color, capsize = 3)
            plt.scatter(time[index_start:index_end+1], flux[index_start:index_end+1], s = 10,
                        color = primary_color, label = 'Flare Points')
            plt.errorbar(time[index_start:index_end+1], flux[index_start:index_end+1],
                         yerr = flux_err[index_start:index_end+1], linestyle = '', color = primary_color, capsize = 3)
            
            #plot the decay fit
            if decay_func == 'dbl_exp_decay':
                plt.plot(decay_time,
                     dbl_exp_decay(decay_time, *p_opt_decay), label = 'Decay Fit', linewidth = 2, color = 'red')

            if decay_func == 'gomp_decay':
                plt.plot(decay_time,
                     gompertz_decay(decay_time, *p_opt_decay), label = 'Decay Fit', linewidth = 2, color = 'red')
                
            if decay_func == 'gomp_logistic':
                plt.plot(decay_time,
                     gompertz_logistic(decay_time, *p_opt_decay), label = 'Decay Fit', linewidth = 2, color = 'red')

            #plotting all three
            if decay_func == 'robust':
                plt.plot(decay_time,
                     dbl_exp_decay(decay_time, *p_opt_dbl_exp), label = 'Double Exp',
                         linewidth = 2, color = 'black', linestyle = '-')
                plt.plot(decay_time,
                     gompertz_decay(decay_time, *p_opt_gompertz), label = 'Product Gompertz',
                         linewidth = 2, color = 'green', linestyle = '-.')
                
            #plotting all three
            if decay_func == 'super_robust':
                plt.plot(decay_time,
                     dbl_exp_decay(decay_time, *p_opt_dbl_exp), label = 'Double Exp',
                         linewidth = 2, color = 'black', linestyle = '-')
                plt.plot(decay_time,
                     gompertz_decay(decay_time, *p_opt_gompertz), label = 'Product Gompertz',
                         linewidth = 2, color = 'green', linestyle = '-.')
                if len(p_opt_decay) > 2:
                    plt.plot(decay_time,
                        dbl_exp_decay(decay_time, *p_opt_dbl_exp_wo_peak), label = 'Double Exp w/o Peak',
                            linewidth = 2, color = 'blue', linestyle = '--')
                    plt.plot(decay_time,
                        gompertz_decay(decay_time, *p_opt_gompertz_wo_peak), label = 'Product Gompertz w/o Peak',
                            linewidth = 2, color = 'red', linestyle = ':')
                
            
            plt.hlines(primary_threshold * flux_std + 1, flare_peak_time - 0.025, flare_peak_time + 0.1,
                      linewidth = 3, linestyle = '--', color = 'gray')
            plt.xlabel('Time (days)', fontsize = fontsize)
            plt.ylabel('Detrended flux', fontsize = fontsize)
            #Print title of target at sector for record keeping
            if (Target != None) and (Sector != None):
                plt.title('Target: ' + str(Target) + ' Sector: ' + str(Sector), fontsize = fontsize)
            if (Target == None) and (Sector != None):
                plt.title('Sector: ' + str(Sector), fontsize = fontsize)
            if (Target != None) and (Sector == None):
                plt.title('Target: ' + str(Target), fontsize = fontsize)
            
            plt.ylim(1 - flux_std, flare_peak_flux + 0.05)
            plt.xlim(flare_peak_time - 0.025, flare_peak_time + 0.1)
            plt.xticks([flare_peak_time - 0.02, flare_peak_time + 0.04, flare_peak_time + 0.1],
                       np.round([flare_peak_time - 0.02, flare_peak_time + 0.04, flare_peak_time + 0.1], 2))
            plt.tick_params(direction = 'in', labelsize = labelsize)
            plt.legend(fontsize = fontsize)
            plt.show()

            #Plot the residuals
            
            median_flux_err = np.nanmedian(flux_err)
            median_flux = np.nanmedian(flux)
            threshold = secondary_threshold * median_flux_err
            
            plt.figure(figsize = (10,8))

            #if the decay function is anything but robust then we only have one set of residuals
            if decay_func == 'robust':
                #plot residuals from double exponential decay
                plt.scatter(decay_time, dbl_exp_residuals, color = 'black')
                plt.errorbar(decay_time, dbl_exp_residuals, yerr = median_flux_err,
                             color = 'black', capsize = 4, linestyle = '-', label = 'Double Exp')
                #plot residuals from product gompertz decay without the peak
                plt.scatter(decay_time, gompertz_residuals, color = 'green')
                plt.errorbar(decay_time, gompertz_residuals, yerr = median_flux_err,
                             color = 'green', capsize = 4, linestyle = '-.', label = 'Product Gompertz')
                
            elif decay_func == 'super_robust':
                #plot residuals from double exponential decay
                plt.scatter(decay_time, dbl_exp_residuals, color = 'black')
                plt.errorbar(decay_time, dbl_exp_residuals, yerr = median_flux_err,
                             color = 'black', capsize = 4, linestyle = '-', label = 'Double Exp')
                #plot residuals from product gompertz decay without the peak
                plt.scatter(decay_time, gompertz_residuals, color = 'green')
                plt.errorbar(decay_time, gompertz_residuals, yerr = median_flux_err,
                             color = 'green', capsize = 4, linestyle = '-.', label = 'Product Gompertz')
                if len(p_opt_decay) > 2:
                    #plot residuals from double exponential decay w/o the peak
                    plt.scatter(decay_time, dbl_exp_residuals_wo_peak, color = 'blue')
                    plt.errorbar(decay_time, dbl_exp_residuals_wo_peak, yerr = median_flux_err,
                                color = 'blue', capsize = 4, linestyle = '--', label = 'Double Exp w/o Peak')
                    #plot residuals from product gompertz decay without the peak
                    plt.scatter(decay_time, gompertz_residuals_wo_peak, color = 'red')
                    plt.errorbar(decay_time, gompertz_residuals_wo_peak, yerr = median_flux_err,
                                color = 'red', capsize = 4, linestyle = ':', label = 'Product Gompertz w/o Peak')
                
            else:
                plt.scatter(decay_time, decay_residuals, color = cadence_color)
                plt.errorbar(decay_time, decay_residuals, yerr = median_flux_err, color = cadence_color, capsize = 4)
            
            plt.hlines(threshold, decay_time[0], decay_time[-1],
                      linewidth = 3, linestyle = '--', color = 'gray')
            plt.xlabel('Time (BJD)', fontsize = fontsize)
            plt.ylabel('Residuals', fontsize = fontsize)
            plt.tick_params(direction = 'in', labelsize = labelsize)
            plt.show()

            #Now plot the secondaries in the time series if they were found
        if flare_type[i] == 'secondary':
    
            #Unpack indices of the flare
            #flare peak time index
            flare_peak_index = np.where(time == flare_peak_times[i])[0][0]
            #flare start time index
            index_start = np.where(time == flare_start_times[i])[0][0]
            #flare end time index
            index_end = np.where(time == flare_end_times[i])[0][0]
        
            flare_peak_time = flare_peak_times[i]
            flare_times = time[index_start:index_end]
        
            #Unpack flare fluxes
            flare_peak_flux = flux[np.where(time == flare_peak_time)[0]]
    
            #Plot all flux points
            plt.figure(figsize = (10,8))
            plt.scatter(time, flux, s = 10, color = cadence_color, label = 'Background Points')
            plt.errorbar(time, flux, yerr = flux_err, linestyle = '', color = cadence_color, capsize = 3)
            #plot flare points
            plt.scatter(time[index_start:index_end+1], flux[index_start:index_end+1], s = 10,
                        color = secondary_color, label = 'Secondary Flare')
            plt.errorbar(time[index_start:index_end+1], flux[index_start:index_end+1],
                         yerr = flux_err[index_start:index_end+1], linestyle = '', color = secondary_color, capsize = 3)
            plt.hlines(primary_threshold * flux_std + 1, flare_peak_time - 0.025, flare_peak_time + 0.1,
                       linewidth = 3, linestyle = '--', color = 'gray')
            plt.xlabel('Time (BJD)', fontsize = fontsize)
            plt.ylabel('Detrended flux', fontsize = fontsize)
            #Print title of target at sector for record keeping
            if (Target != None) and (Sector != None):
                plt.title('Target: ' + str(Target) + ' Sector: ' + str(Sector), fontsize = fontsize)
            if (Target == None) and (Sector != None):
                plt.title('Sector: ' + str(Sector), fontsize = fontsize)
            if (Target != None) and (Sector == None):
                plt.title('Target: ' + str(Target), fontsize = fontsize)
        
            plt.ylim(1 - flux_std, flare_peak_flux + 0.05)
            plt.xlim(flare_peak_time - 0.025, flare_peak_time + 0.1)
            plt.xticks([flare_peak_time - 0.02, flare_peak_time + 0.04, flare_peak_time + 0.1],
                       np.round([flare_peak_time - 0.02, flare_peak_time + 0.04, flare_peak_time + 0.1], 2))
            plt.tick_params(direction = 'in', labelsize = labelsize)
            plt.legend(fontsize = fontsize)
            plt.show()

        if flare_type[i] == 'tertiary':
    
            #Unpack indices of the flare
            #flare peak time index
            flare_peak_index = np.where(time == flare_peak_times[i])[0][0]
            #flare start time index
            index_start = np.where(time == flare_start_times[i])[0][0]
            #flare end time index
            index_end = np.where(time == flare_end_times[i])[0][0]
        
            flare_peak_time = flare_peak_times[i]
            flare_times = time[index_start:index_end]
        
            #Unpack flare fluxes
            flare_peak_flux = flux[np.where(time == flare_peak_time)[0]]
    
            #Plot all flux points
            plt.figure(figsize = (10,8))
            plt.scatter(time, flux, s = 10, color = cadence_color, label = 'Background Points')
            plt.errorbar(time, flux, yerr = flux_err, linestyle = '', color = cadence_color, capsize = 3)
            #plot flare points
            plt.scatter(time[index_start:index_end+1], flux[index_start:index_end+1], s = 10,
                        color = tertiary_color, label = 'Tertiary Flare')
            plt.errorbar(time[index_start:index_end+1], flux[index_start:index_end+1],
                         yerr = flux_err[index_start:index_end+1], linestyle = '', color = tertiary_color, capsize = 3)
            plt.hlines(primary_threshold * flux_std + 1, flare_peak_time - 0.025, flare_peak_time + 0.1,
                       linewidth = 3, linestyle = '--', color = 'gray')
            plt.xlabel('Time (BJD)', fontsize = fontsize)
            plt.ylabel('Detrended flux', fontsize = fontsize)
            #Print title of target at sector for record keeping
            if (Target != None) and (Sector != None):
                plt.title('Target: ' + str(Target) + ' Sector: ' + str(Sector), fontsize = fontsize)
            if (Target == None) and (Sector != None):
                plt.title('Sector: ' + str(Sector), fontsize = fontsize)
            if (Target != None) and (Sector == None):
                plt.title('Target: ' + str(Target), fontsize = fontsize)
            plt.ylim(1 - flux_std, flare_peak_flux + 0.05)
            plt.xlim(flare_peak_time - 0.025, flare_peak_time + 0.1)
            plt.xticks([flare_peak_time - 0.02, flare_peak_time + 0.04, flare_peak_time + 0.1],
                       np.round([flare_peak_time - 0.02, flare_peak_time + 0.04, flare_peak_time + 0.1], 2))
            plt.tick_params(direction = 'in', labelsize = labelsize)
            plt.legend(fontsize = fontsize)
            plt.show()


def flare_energy_calc(star_luminosity, equivalent_duration):
    #units of energy depend on stellar luminosity input
    #Input of W will give Joules for Ebol, input of erg/s will give erg
    #equation used in Howard and MacGregor 2022
    flare_energy = equivalent_duration * star_luminosity
    return flare_energy
