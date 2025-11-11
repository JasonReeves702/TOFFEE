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
    tess_orbit_time_url = 'https://tess.mit.edu/public/files/TESS_FFI_observation_times.csv'
    Tess_orbit_times = download_csv_file(tess_orbit_time_url)
    #import
    Tess_sector_times= Table.read(Tess_orbit_times, format='ascii.csv')
    Tess_start_times = Time(Tess_sector_times['Start Time'], format='iso', scale='utc')
    Tess_end_times = Time(Tess_sector_times['End Time'], format='iso', scale='utc')
    tess_start_times_tdb = Tess_start_times.tdb
    tess_end_times_tdb = Tess_end_times.tdb
    tess_start_times_bjd =  tess_start_times_tdb.jd - 2457000
    tess_end_times_bjd =  tess_end_times_tdb.jd - 2457000
    orbit_t = np.array([])
    orbit_lcs = np.array([])
    orbit_lc_errs = np.array([])
    orbit_trends = np.array([])
    orbit_masks = []
    
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
                    lc_sine_wotan, sine_trend = wotan.flatten(orbit_t,lc_working,window_length=half_window*120/3600/24,method='median',return_trend=True)
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

        #figure out if its an orbit break or not
        #if the middle of the sector is sandwhiched by the beginning and end of the break
        #then it's the orbit break

        if (time[break_index] < sector_mid_time) & (time[break_index + 1] > sector_mid_time):

            break_type.append('Orbit')

        else:

            break_type.append('Reduction')


    #After looping go back through, if there's only one break it's def the orbit??

    if len(break_indices) == 1:

        #if the only break is reduction, it's actually orbit

        if break_type == ['Reduction']:

            break_type = ['Orbit']

    #convert to dataframe

    sector_break_frame = pd.DataFrame({'Break_Index': pd.Series(break_indices, dtype = int),
                                       'Break_Start_Time': pd.Series(break_start_time, dtype = float),
                                       'Break_End_Time': pd.Series(break_end_time, dtype = float),
                                       'Break_Type': pd.Series(break_type, dtype = str)})
    
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






def flare_finder(time, flux, flux_err, quality, magic_flare_std = 3.0, sec_flare_std = 2.0,
                 prim_marg_rate = 0.67, sec_marg_rate = 0.75, consecutive = False,  rise_func = 'gaussian', visualize_fit = False,
                 fit_twice = False, rate_above_threshold = 0.75, fit_multiple_secs = False, flag_values = [0], detrend = True,
                 window_length = 0.25, periodogram = None, clip_breaks = 100, min_break = 0.25,
                 primary_color = 'red', secondary_color = 'blue', tertiary_color = 'green',
                 cadence_color = 'black', threshold_color = 'red', fit_color = 'black',
                 fontsize = 14, labelsize = 12, TIC_number = 0, TESS_sector = 0):

    '''INTRO
    
    Welcome, this is the TESS Overlapping Flare Finder and Energy Evaluator
    (TOFFEE) flare finder

    This code works by taking a raw lightcurve, detrending it, then searching for
    primary flares and secondary flares within the primary flares. It's been trained
    for 120 second data and so it's advised to use the same data when running the program.

    We call this a "top-down" approach as instead of scanning a lightcurve
    from left to right, looking at every single photometric point, we only look
    at all the points above a given flux threshold. 
    
    For those points we start at the brightest points, starting from the peaks and modeling
    around it to find the beginning and the end of the flare using a series of conditions
    if there are sufficient enough points above the threshold it's counted as a flare and added
    to the catalog. We call these flares primaries as they're found first

    If the flare has a good number of points in the rise to the left of the peak and in the decay
    to the right of the peak it's worthwhile to search for a secondary. To the rise we can fit a variety
    of models such as a quadratic or gaussian (although we find gaussian to work much better for 120 second
    TESS data). It fits a double exponential to the decay (no other model worked well at all). In the residuals,
    defined as the difference of the flux points and the value of the flux from the fitted model, we try to find major
    deviations from the expected behaviors. If we see three or more points above a threshold, boom, another flare
    has been found within the first one. We call this the secondary flare becuase it was found second.
    
    Note we call it a secondary regardless of whether or not it occurs before or after the primary. It's just named based
    on the order we found the flares and the secondary flares always have a lower
    peak flux (not necessarily amplitude) than the primary.
    
    All the other bright points above the threshold that were found to belong
    to this flare are thrown out and the code moves onto the next peak and starts again.

    '''


    '''ARGUMENTS

    These arguments should be 1D array-like objects of equal size

    time: the time coordinates of the lightcurve in BJD. The detrending utilizes the orbit timing so these should be
          in the default units from lightkurve.

    flux: Photometric flux. Can either be in raw units of e/sec becuase we'll normalized later or can be normalized already.

    flux_err: Photometric error on the cadences.
              Can either be in raw units of e/sec becuase we'll normalize later or can be normalized already.

    quality: quality flag of the cadence from TESS. We utilize filtering to remove poor quality readings from the
             lightcurves to ensure more accurate detection. If you want to skip this step then you can just make
             an array of equal length to the time, flux, and flux_err arrays and fill it with the value 0.

    LIGHTCURVE PREPARATION

    flag_values: A list of the acceptable flag values you want to include in the analysis. By default we only
                 permit cadences with a perfect quality of zero into the analysis. However it has been shown that
                 points with a flag value of 512 can correspond to the peaks of flares and so can be kept in. For that
                 you can pass flag_values = [0, 512] to include them. The other will be masked out.

    detrend: boolean value representing whether or not to detrend the lightcurve. 
    
             If True TOFFEE will detrend using biwieght detrending. If not, TOFFEE will skip this
             and go straight to detection. TOFFEE relies on detrending to find flare signals however
             if the lightcurve has been detrended beforehand using a preferred method you
             may want to skip detrending again which may eat into the flare signal.

    window_length: float value representing the window length of wotan detrending (Hippke et al. (2019, AJ, 158, 143)).

    periodogram: None value or list of length 2 in the form [min_period, max_period]. Periodogram iteratively runs LS test
                 on lightcurve post Wotan detrend to find residual periodic red noise and flatten it using sine fit.
                 The periods of such fits are in the range (min_period, max_period) such that min_period is the smallest period
                 in days considered for the fit and max_period is the maximum period tested. In our analysis we find [0.01, 10] to 
                 work effectively, needing to capture the rapid rotators that majorly affect fitting.

    TOFFEE also clips all breaks in the lightcurve to avoid edge effects of detrending creating false positve signals.

    min_break: float value represening the smallest duration of a break in the lightcurve (in days) for which TOFFEE will
               mask out points on either side. Default is set to 0.25, the same value as the default window length for Wotan.
               In Pratt et al 2025 we used a value of 0.025 days to be super conservative.

    clip_breaks: The number of cadences to clip out on either side of a break detected by min_break. Default is 100 cadences
                 which corresponds to 200 minutes of data in 120 sec cadence data from TESS.

    

    PRIMARY FLARE DETECTION ARGUMENTS

    magic_flare_std: Integer or float value greater than 0. Sets the threshold for detection of primary flares in terms of sigma of the global
                     spread of photometry points. Default value is set to 3σ which is the common value for threshold-based detection.

    consecutive: boolean value representing how a primary flare will be determined. 
    
                 If true a flare will be counted only if three consecutive points are above the threshold.
                 This is consistent with many other threshold-based methods. It's easy to follow and generally
                 works well. However, for some detrendended lightcurves certain artifacts of the detrending
                 may pop up as false-positive signals.
                 
                 If false a flare will be counted even if not every point is above the threshold.

                 The non-consecutive case which we call the 'three in four rule' works to both include low amp
                 flares that don't have three consecutive points above the threshold and rule out signals that
                 result from detrending artifacts. It works by starting from the peak of the flare and looking
                 to the left to find the start of the flare. A flare is considered to have begun when the next
                 three points to the left of some cadence all lie well under the threshold (1σ less than the threshold).
                 Then the code looks to the right, saying a flare has ended when the next three points to the right of some cadence
                 are all well below the threshold (1σ less than the threshold). In this way we try to say with confidence that
                 a flare event has truly begun and ended and can include points below the threshold.

                 Then TOFFEE reads where the first point associated with the flare was above the threshold and where
                 the last point associated with the flare was above the threshold. If 75% of the points between them are
                 above the flux threshold then we call it a flare as we see a true signal. So if there's a small signal where
                 there are three cadences above the threshold separated by one cadence below the threshold it's still a flare.
                 In practice this one cadence below the threshold is only just below the threshold of detection so visually
                 it still passes as a flare.

                 While catching low amp flares it also catches incomplete detrending where there is residual modulation.
                 Leftover red noise can occasianally rise above the flux threshold being falsly flagged as a flare. This condition
                 helps to eliminate such false-positives because they tend to fluctuate around the threshold for a while, causing
                 the fraction of points above the threshold as opposed to below the threshold to be lower than 75%.

                 The only catch is the case where a flare occurs during this positive modulation. In that case it's quite tricky
                 to recover them. To save quality signals we still recover the flares if the amplitude is sufficiently high.
                 Sufficiently high is defined as being 1.5x the original flux threshold for detection. This typically finds obvious
                 flares while keeping away false-positive detection when using the typical threshold value of 3σ.

    prim_marg_rate: float value from 0.0 to 1.0 representing the threshold value for a cadence to be
                    consideres a part of a flare. Rather than being the sigma threshold itslef it's a
                    fraction of the primary flare sigma. For example, with the default primary flare threshold
                    of 3σ and the default prim_marg_rate of 0.67 cadences flux points with values of 2σ can be
                    considered a part of the flare. You want something low enough that you can catch the epochs of the
                    lightcurve above the threshold that may be separated by one or two other cadences but no too low that
                    an unreasonable number of points are included in the flare and the start and end times are poor.

    rate_above_threshold: float value between 0.0 and 1.0. If using the nonconsecutve detection for primary flares
                          (consecutive = False) rate_above_threshold determines the minimum fraction of points above
                          the flux threshold compared to the total number of points in the flare to be called a flare.
                          The default is 0.75 where the 'three in four rule' gets its name but can be altered depending on
                          the purpose of the user. Higher values will result in stricter requirements (0.9 requires 90% of the
                          flux points to be above the threshold) and lower values loosen requirements. We find 0.75 to be the
                          minimum while comfortably getting quality flares as opposed to 0.6 (or a 'three in five rule').

    SECONDARY FLARE DETECTION ARGUMENTS

    sec_flare_std: Integer or float value greater than 0. Sets the threshold for detection of primary flares in terms of sigma of the global
                   spread of photometry points. Default value is set to 2σ.

    rise_func: str variable of value 'gaussian' or 'quad' determining the model used to fit the rise. If 'gaussian' is applied
               then the rise will be modeled by a guassian rise function. If 'quad' is applied then a quadratic rise will be used.
               'gaussian' is recommended for 120 sec data

    sec_marg_rate:  float value from 0.0 to 1.0 representing the threshold value for a cadence to be
                    consideres a part of a fsecondary lare. Rather than being the sigma threshold itslef it's a
                    fraction of the secondary flare sigma. For example, with the default primary flare threshold
                    of 2σ and the default prim_marg_rate of 0.75 cadences flux points with values of 1.5σ can be
                    considered a part of the secondary flare. It's used to determine whether or not a secondary flare
                    was missed because of it's affect on the fit. A default value of sec_marg_rate = 0.75 with
                    sec_flare_std = 2.0 means that if there are three points 1.5σ above the fitted model then
                    there is a possible secondary and it's worth refitting without those points.

    fit_twice: boolean value representing whether or not to attempt refitting of the decay in the case where a potential
               secondary flare was missed because of it's affect in lifting up the fit. A marginal flare is defined as an
               event in the residual where three consecutive points are at least 75% of the desired threshold.

    fit_multiple_secs: boolean value representing whether or not to search for more than one secondary flare event in the residual
                       of the decay of a flare, finding possible teritiary and even quartic flares. We initially leave this off
                       as it often picks up Quasi-Periodic Pulsations as these tertiary flares but it could be useful if that's
                       your goal.


    PLOTTING FLARE FITS ARGUMENTS
    TOFFEE also has built in plotting for the fittings of flares and finding secondary flares. If you don't want to see them
    you can leave visualize_fit in the default state of False. If you do then set visualize_fit = True but you don't need to mess with most of
    the arguments unless you really want a different color pallette or the TIC and Sector plotted as well.

    visualize_fit: boolean value representing whether or not to plot each attempted fit for primary flares along with residuals
                   and overplotting of primary and secondary flares. Does not show ALL flares, only ones where a fit was attempted.
                   If true fits and residuals will be plotted, if false this step will be skipped.

    IF visualize_fit = True, these variables will be passed

    primary_color = string variable for the color of the points associated with a primary flare event.

    secondary_color = string variable for the color of the points associated with a secondary flare event.

    terciary_color = string variable for the color of the points associated with a terciary flare event.

    cadence_color = string variable for the color of the TESS photometry points not associated with flares.

    threshold_color = string variable for the color of the horizontal line representing the flux threshold for primary flare detection.

    fit_color = string variable for the color of the parameterized curves for the rise and decay of the flares.

    fontsize = integer variable for the fontsize of the axes labels and titles of the plots from visualize_fit

    labelsize = integer variable for the fontsize of the labels on the axes

    TIC_number = integer variable for the TIC Number of the star from which the lightcurve comes from. Useful when plotting
                 the flare fits from many stars.

    TESS_sector = integer variable for the TESS Sector Number from which the lightcurve comes from. Useful when plotting the flare fits
                  for multiple sectors of a star.

    

    '''



    '''RETURNS

    The returns of the code are a suite of physical and data based measurements:

    flare_peak_times, flare_start_times, flare_end_times, flare_amps, flare_equivalent_durations,
    primary_or_secondary, points_in_flare, points_abv_threshold, amp_sigma


    flare_peak_times: Times (in BTJD), of the peaks of the flares, the moments of brightest emission

    
    flare_start_times: Time marking the beginning of the flares before the sudden rise

    
    flare_end_times: Time marking the end of the flares after the decay

    
    NOTE: The flare start and end times are a bit finnicky based on how you define it in the
          code. We advise using the peak times as a better characterization of when the flare occurs.

    
    flare_amps: the flux of the peak emission normalized to the median emission for the star. This is with
                the background emission of the star subtracted out so it's just the emission of the flare

    
    flare_equivalent_durations: Used for finding the energies. Defined as the equivalent amount of time (in seconds)
                                that the star would need to shine in order to emit the same amount of energy as the flare.
                                Found by integrating the flux of the flare with respect to time using the trapezoidal rule
                                after subtracting the background emission of the star. Multiplying the ED by the luminosity
                                of the star finds the energy of the flare.

    
    primary_or_secondary: Label telling how this flare was found. Comes in four flavors.
    
                          "primary": Means this is a flare that was found where the peak was identified,
                                     it was modeled, and passed the checks. There may or may not have been
                                     a fit applied to the rise and decay. There may or may nor have been a
                                     secondary found to this flare.
                          
                          "primary_dbl_fit_failed": A special type of primary flare in which a double exponential
                                                    was fit to the decay with no secondary found. However it seemed
                                                    like there was possibly another flare that was just missed so it
                                                    tried to refit the decay and still didn't find a secondary. We keep
                                                    track of these to see the relative success of these attempted refittings
                                                    to find more secodnary flares.

                         "secondary": Means this flare was found within a primary flare. It may be in the rise or the decay.
                                      It was found on the first attempt of fitting a model to the rise and decay with no need
                                      for a refit

                         "secondary_second_try": A special type of secondary that was found in the decay of a flare only
                                                 after refitting the model without brighter points that affected the first
                                                 fit


    points_in_flare: Number of flux points identified with the flare.

    
    points_abv_threshold: Not all points associated with the primary flare are necessarily above the threshold, the conditions
                          are complicated depending on what setting you use to identify the beginning and end. So we
                          also keep track of the number of points in the flare above the detection threshold.


    amp_sigma: The amplitude of the flare divided by the spread of the photometric points. Tells us how much the peak of the
               flare rises relative to the noise of the lightcurve.

    
    '''

    '''FUTURE WORK
    
    TOFFEE is always looking for improvement in performance and flexibility. There are a few outstanding challenges for the
    project in terms of secondary detection. 
    
    1) Seeking tertiary and other high order flares. The issue comes
    from the effect of Quasi-Periodic Pulsations that get flagged as loads of secondary flares. While this is perhaps a
    cool way to find such events in a large sample it's not good when you want quality sources. One solution is to test
    for periodicity in the events (whether or not the primary and secondary are equally spaced in time to the secondary and
    tertiary). 
    
    2) The conditions for secondary detection are still a little rigid. It has been shown limiting detection
    to only two cadences above 3σ is good for finding flares. Or maybe you want super obvious flares so you want 4 cadences above
    5σ. We're adding that flexibility as well.

    3) As best as we tried the detrending is not perfect. For pulsating stars in particular the fits are not great yet
    when the program lands on a solution TOFFEE just goes ahead to detect flare signals.
    We're adding filtering so you can only search for flares on lightcurves for which the detrending 
    actually flattens the light curve.

    '''
    
    #######################Unpack Array#############################

    #set what flags you want to be in the analysis 

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

    #we're also going to chop off the first and last 200 cadences following Feinstein

    flux = flux[perfect_points]

    time = time[perfect_points][np.isnan(flux) == False]

    flux_err = flux_err[perfect_points][np.isnan(flux) == False]

    flux = flux[np.isnan(flux) == False]

    



    ###################Flatten the lightcurve###########################

    ##If you passed a raw lightcurve and want it detrended go through this
    #Patch of code, if not go stright to next section

    if detrend == True:

        #if you are using a periodogram and passed the range of frequencies
        #we want to use the flattened light curve
        #after taking out possible residual oscillations
    
        #if not then we'll just use wotan
    
        if periodogram != None:
        
            t, lc_quad, lc_wotan, lc_flat, periodic = flatten(time, flux, flux_err, plot_results=False,
                                                               short_window=window_length, periodogram= periodogram)
    
            #sometimes the size of the arrays spat out by the flattening function is not the same as the light
            #curve and thus not the same as the break_mask. So we need another mask to ensure they're the same
    
            flattening_mask = np.full(len(time), 1, dtype = bool)
    
            for index in range(len(time)):
    
                if time[index] in t:
    
                    continue
    
                else:
    
                    flattening_mask[index] = False
    
            #convert this mask to indices to include in the mask
    
            flattening_mask = np.arange(len(time))[flattening_mask]
    
            #done for convience
    
            time = t
    
                
    
            #pull out flattened flux and flux_err values
    
            normalized_flux, flux_err, sine_trend = lc_flat
    
            #redundancy step, normalize again
    
            normalized_flux = normalized_flux/np.median(normalized_flux)
    
            flux_err = flux_err/np.median(normalized_flux)

            #find the 84th percentile of the detrended flux, or 1σ spread
    
            flux_std = np.nanpercentile(normalized_flux - 1, 84)
    
        else:
    
            t, lc_quad, lc_wotan, periodic = flatten(time, flux, flux_err, plot_results=False,
                                                               short_window=window_length, periodogram = periodogram)
    
    
            #sometimes the size of the arrays spat out by the flattening function is not the same as the light
            #curve and thus not the same as the break_mask. So we need another mask to ensure they're the same
    
            flattening_mask = np.full(len(time), 1, dtype = bool)
    
            for index in range(len(time)):
    
                if time[index] in t:
    
                    continue
    
                else:
    
                    flattening_mask[index] = False
    
            #convert this mask to indices to include in the mask
    
            flattening_mask = np.arange(len(time))[flattening_mask]
    
            #done for convience
    
            time = t
    
            #pull out flattened flux and flux_err values
    
            normalized_flux, flux_err, sine_trend = lc_wotan
    
            #redundancy step, normalize again
    
            normalized_flux = normalized_flux/np.median(normalized_flux)
    
            flux_err = flux_err/np.median(normalized_flux)
    
            #find the 84th percentile of the detrended flux, or 1σ spread
    
            flux_std = np.nanpercentile(normalized_flux - 1, 84)


    #if we elected not to detrend we still need something to fill int he flattening_mask
    #this should basically mask out none of the points
    else:

        flattening_mask = np.full(len(time), 1, dtype = bool)

        #and the normalized fluxes and flux errors

        normalized_flux = flux/np.nanmedian(flux)
    
        flux_err = flux_err/np.nanmedian(flux)

        #find the 84th percentile of the detrended flux, or 1σ spread
    
        flux_std = np.nanpercentile(normalized_flux - 1, 84)


    
    
    #####################Find the break and apply mask###################

    #Use breakfinder to find the break(s) from TESS sector and clip off cadences
    #from either side of the breaks

    #Apply a mask to cut out points on either side of the breaks

    if clip_breaks != None:

        break_mask = light_curve_mask(time, flux, min_break = min_break, clip_breaks = clip_breaks)

        #apply flattening mask

        break_mask = break_mask[flattening_mask]

        #apply to light curve

        normalized_flux = normalized_flux[break_mask]

        time = time[break_mask]

        flux_err = flux_err[break_mask]


    


    #################IDENTIFY BRIGHT POINTS AS POTENTIAL FLARES##################


    #identify really bright fluxes and their associated times a certain number of
    #standard deviations from the median


    #There ARE redundant fluxes in these measurements, we'll need to filter those out

    median_flux = 1

    flux_threshold = 1 + (magic_flare_std * flux_std)

    flare_candidates = normalized_flux[normalized_flux > flux_threshold]

    time_candidates = time[normalized_flux > flux_threshold]

    #set a marginal threshold to help count the points that belong to a flare

    prim_marginal_threshold = prim_marg_rate * magic_flare_std

    sec_marginal_threshold = sec_marg_rate * sec_flare_std

            

    #sort them in descending order with highest fluxes first
    
    #we call three separate numpy functions to sort the times
    #we want it so that we preserve the times corresponding to each flare canditate
    #once the flares are sorted. searchsorted tells us the indices of the sorted array of
    #flare candidates and thus the corresponding indices of the times. We then need to reverse
    #to be in descending order

    time_candidates = np.flip(time_candidates[np.argsort(flare_candidates)])

    flare_candidates = np.flip(np.sort(flare_candidates))

    #empty array to hold flare amplitudes

    flare_amps = []

    #empty array to hold the flare equivalent durations

    flare_equivalent_durations = []

    #flare times at peak flux

    flare_peak_times = []

    #flare start times

    flare_start_times = []

    #flare end times

    flare_end_times = []

    #######INITIALIZE FLAGS TELLING US HOW WE FOUND THESE FLARES########

    #Was this a primary or secondary detection, secondary means found in the residuals

    primary_or_secondary = []


    #How many points are associated with this flare, both above and below the threshold?
    
    points_in_flare = []
    
    
    #How many points above the threshold were there?

    points_abv_threshold = []

    #What's the amplitude of the flare in terms of sigma?

    amp_sigma = []

    i = 0

    #So basically we're going to go through the list from the largest fluxes down
    #if there are other listed candidates near the flux we're testing they're probably a part of this
    #flare

    while i < len(time_candidates):

        ###########CLASSIFY FLARE ESSENTIAL PARAMETERS##############

        #this guy is for SURE a flare candidate
        flare_flux = flare_candidates[i]

        #this is the associated time
        flare_time = time_candidates[i]

            
        #Alright, let's filter through these arrays and get rid of the redundant measurements
        #We'll run through it until a certain break condition
    
        time_peak_flare_index = np.where(time == flare_time)[0][0]

        
        #for later clarity

        flare_peak_time = flare_time

        #find median flux, spread, and threshold for primary flare detection

        median_flux = 1

        # calculate threshold from the photometric error of the photometry point
        one_sigma_percentile = 84

        flux_std = np.nanpercentile(normalized_flux - 1, one_sigma_percentile)

        #and the flux threshold

        flux_threshold = median_flux + (magic_flare_std * flux_std)

        
        
        
        #look around this time to see if any of the time candidates belong to the same flare

        #look to the left to find where the beginning is

        if time_peak_flare_index + 3 >= len(normalized_flux) - 1:

            j = 0

        else:
        
            j = 0
    
            #We want three consecutive points to be below the threshold to be sure the flare has finished off
            #before this point
    
            while (normalized_flux[time_peak_flare_index - j - 1] > median_flux + (prim_marginal_threshold * flux_std) or
                   normalized_flux[time_peak_flare_index - j - 2] > median_flux + (prim_marginal_threshold * flux_std) or
                   normalized_flux[time_peak_flare_index - j - 3] > median_flux + (prim_marginal_threshold * flux_std)):

                if time_peak_flare_index - j - 3 <= 0:
    
                    break
    
                j += 1


        #once it's done we have the beginning time of flare

        time_start_flare_index = time_peak_flare_index - j

        flare_start_time = time[time_peak_flare_index - j]


        if time_peak_flare_index + 3 >= len(normalized_flux) - 1:

            j = 0

        #Now for the end time

        else:

            #look to the right to find where the end is
    
            j = 0
    
            #We want two consecutive points to be below the threshold to be sure the flare has finished off
            #after this point
    
            while (normalized_flux[time_peak_flare_index + j + 1] > median_flux + (prim_marginal_threshold * flux_std) or
                   normalized_flux[time_peak_flare_index + j + 2] > median_flux + (prim_marginal_threshold * flux_std) or
                   normalized_flux[time_peak_flare_index + j + 3] > median_flux + (prim_marginal_threshold * flux_std)):
    
                if time_peak_flare_index + j + 3 >= len(normalized_flux) - 1:
    
                    break
    
                j += 1


        #once it's done we have the beginning time of flare

        time_end_flare_index = time_peak_flare_index + j

        flare_end_time = time[time_peak_flare_index + j]

        

        #find all the points above the threshold that are considered a part of this flare
        redundant_points = np.where((time_candidates <= flare_end_time) &
                                    (time_candidates >= flare_start_time) &
                                    (time_candidates != flare_time))[0]


        ##################DETERMINING WHETHER THERE'S SIGNAL########################

        if consecutive == True:
            
            #See if we should keep the flare if there are two more points around it
            #if there are less than two redundant points it's probably noise and we should
            #dump it
    
            if len(redundant_points) >= 2:
    
                #We still need some consecutive points to be above threshold
    
                #Create boolean conditions for consecutive bright points
    
                #two points to the left are above threshold
    
                two_points_left = ((normalized_flux[time_peak_flare_index - 2] > flux_threshold) &
                                   (normalized_flux[time_peak_flare_index - 1] > flux_threshold))
    
                #two surrounding points are above threshold
    
                two_surrounding_points = ((normalized_flux[time_peak_flare_index - 1] > flux_threshold) &
                                   (normalized_flux[time_peak_flare_index + 1] > flux_threshold))
            
    
                #two points to the right
    
                two_points_right = ((normalized_flux[time_peak_flare_index + 2] > median_flux + (magic_flare_std * flux_std)) &
                                   (normalized_flux[time_peak_flare_index + 1] > median_flux + (magic_flare_std * flux_std)))
    
    
                #If at least one of these conditions are met then this flare is TRUE
                
                if two_points_left == True or two_surrounding_points == True or two_points_right == True:
    
                    sufficient_points = True

                    #track the total number of points we found above the threshold

                    num_past_threshold = len(np.where(normalized_flux[time_start_flare_index:time_end_flare_index + 1] >
                                              median_flux + (magic_flare_std * flux_std)))
    
                #If none of the conditions are met then this flare is FAKE
    
                else:
    
                    sufficient_points = False
    
            else:
                        
                sufficient_points = False


        
        
        
        elif consecutive == False:

            #See if we should keep the flare if there are at least 3 points above the threshold
    
            if time_end_flare_index - time_start_flare_index >= 2:
    
                #find the number of points above the threshold
    
                num_past_threshold = np.where(normalized_flux[time_start_flare_index:time_end_flare_index + 1] >
                                              median_flux + (magic_flare_std * flux_std))[0]

                if len(num_past_threshold) > 0:

                    #We want the bright points to cluster together, we want the bright points above the threshold
                    #to occurr 3 every 4 points
    
                    bright_range = max(num_past_threshold) -  min(num_past_threshold)
    
                    #number of points above threshold in range of bright points
    
                    bright_ratio = len(num_past_threshold)/bright_range
    
                    if (len(num_past_threshold) >=3) & (bright_ratio > rate_above_threshold):
        
                        sufficient_points = True
    
                        num_past_threshold = len(num_past_threshold)

                    #so if the flare failed at this step it may be a flare
                    #that failed because it occurred on the up of a flare
                    #modulation so let's log that and see if it's something with the large amplitude
    
                    else:

                        #new threshold, two times the amount we normally look for

                        firm_flare_std = 1.5 * magic_flare_std

                        #We need some consecutive points to be above threshold
    
                        #Create boolean conditions for consecutive bright points
            
                        #two points to the left are above threshold
            
                        two_points_left = ((normalized_flux[time_peak_flare_index - 2] > median_flux + (firm_flare_std * flux_std)) & 
                                           (normalized_flux[time_peak_flare_index - 1] > median_flux + (firm_flare_std * flux_std)))
            
                        #two surrounding points are above threshold
            
                        two_surrounding_points = ((normalized_flux[time_peak_flare_index - 1] > median_flux +
                                                   (firm_flare_std * flux_std)) &
                                           (normalized_flux[time_peak_flare_index + 1] > median_flux + (firm_flare_std * flux_std)))
                    
            
                        #two points to the right
            
                        two_points_right = ((normalized_flux[time_peak_flare_index + 2] > median_flux +
                                             (firm_flare_std * flux_std)) &
                                           (normalized_flux[time_peak_flare_index + 1] > median_flux + (firm_flare_std * flux_std)))
            
            
                        #If at least one of these conditions are met then this flare is TRUE
                        
                        if two_points_left == True or two_surrounding_points == True or two_points_right == True:

                            #so we have three points above a much stronger threshold, it's definitely a flare
            
                            sufficient_points = True

                            num_past_threshold = len(num_past_threshold)

                        else:
                            
                            sufficient_points = False

                else:

                    sufficient_points = False
    
            else:
                        
                sufficient_points = False

        #If it IS a real flare
        #delete these redundant points in the arrays
        

        if sufficient_points == True:

            #############BEGIN LOOKING FOR POSSIBLE SECONDARIES############

            #look left and right for another peak

            #For high amplitude flares the inner complexities of the flare show up
            #as flare-like signals so let's raise the threshold for secondaries if the amplitude is large

            if flare_flux >= 3:

                sec_flare_std = 10.0

            else:

                sec_flare_std = sec_flare_std


            ############LEFT SEARCH###############

            #If there are at least two points to the left of the peak fit a quadratic and see
            #if it's poor. Pull out the residual as ways to find the amplitude of peak

            #if the R Square really sucks then refit the line only taking the points that
            #decrease from the peak and find the flux of the largest flux to the left
            #we may be looking at a large connected flare to the left.


            #Find the points to fit the line to, search to the left until at least two points regress
            #to a little below the threshold


            #look for something interesting inbetween the peak and the start point we found earlier

            #Fit a line with all the points between the two ends if there are more than three points left
            #of the peak

            if time_peak_flare_index - time_start_flare_index >= 5:

                rise_indices = np.arange(time_start_flare_index, time_peak_flare_index + 1)

                rise_time = time[time_start_flare_index:time_peak_flare_index + 1]

                rise_flux = normalized_flux[time_start_flare_index:time_peak_flare_index + 1]

                rise_flux_err = flux_err[time_start_flare_index:time_peak_flare_index + 1]

                #use curve-fit to fit a functional form to the rise of the flare to test for secondaries

                #using keywork arg for which form to use

                if rise_func == 'gaussian':

                    def guassian_rise(x, alpha, sigma, c):

                        return alpha * np.exp(-(x - flare_peak_time)**2 / (2 * sigma)**2) + c

                    #set initial guesses for guassian rise
    
                    alpha_i = flare_flux
    
                    sigma = flare_peak_time - flare_start_time
    
                    c_i = 1

                    p0 = [alpha_i, sigma, c_i]

                    #set weight to have to go through peak point and first point

                    sigma = rise_flux_err
                    sigma[-1] = 0.001
    
    
                    #set initial guess for guassian rise
    
    
                    params, cov = scipy.optimize.curve_fit(guassian_rise, rise_time, rise_flux, sigma = sigma, 
                                                           p0 = p0, bounds = ([0, 0, 0],
                                                                     [np.inf, np.inf, np.inf]),
                                                           maxfev = 10000)
    
    
                    alpha, sigma, c = params
    
                    #find residuals
    
                    res = rise_flux - guassian_rise(rise_time, alpha, sigma, c)
                
                
                if rise_func == 'quad':

                    def quad(x, a, b, c):
                        return a * (x - flare_peak_time)**2 + b * (x - flare_peak_time) + c
                
                    #set initial guesses for quad
    
                    a_i = 0
    
                    b_i = 1
    
                    c_i = 1

                    p0 = [a_i, b_i, c_i]

                    #set weight to have to go through peak point and first point

                    sigma = rise_flux_err
                    sigma[-1] = 0.001 * sigma[-1]
    
    
                    #set initial guess for guassian rise
    
    
                    params, cov = scipy.optimize.curve_fit(quad, rise_time, rise_flux, sigma = sigma, 
                                                           p0 = p0, bounds = ([0, 0, median_flux - magic_flare_std * flux_std],
                                                                     [np.inf, np.inf, median_flux + magic_flare_std * flux_std]))
    
    
                    a, b, c = params
    
                    #find residuals
    
                    res = rise_flux - quad(rise_time, a, b, c)

                #we want to check these residuals for three consecutive values above
                #2σ, goes all the way to the peak, maybe worth revision

                #initialize secondary condition

                left_secondary = False

                for k in range(len(res) - 3):

                    if ((res[k] > sec_flare_std * flux_std) and
                        (res[k + 1] > sec_flare_std * flux_std) and
                        (res[k + 2] > sec_flare_std * flux_std)):

                        #three consecutive points, there's a flare

                        left_secondary = True


                

                
                
                #################VISUALIZING THE FIT################

                if visualize_fit == True:

                    if rise_func == 'gaussian':


                        #Plot all flux points
            
                        plt.figure(figsize = (10,8))
                        
                        plt.scatter(time, normalized_flux, s = 10, color = cadence_color, label = 'Background Points')
                        plt.errorbar(time, normalized_flux, yerr = flux_std, linestyle = '', color = cadence_color)

                        #Plot all points belonging to the flare

                        plt.scatter(time[time_start_flare_index:time_end_flare_index + 1],
                                    normalized_flux[time_start_flare_index:time_end_flare_index + 1], s = 10,
                                    color = primary_color, label = 'Flare Points')
                        plt.errorbar(time[time_start_flare_index:time_end_flare_index + 1],
                                     normalized_flux[time_start_flare_index:time_end_flare_index + 1],
                                     yerr = flux_std, linestyle = '', color = primary_color)

                        #plot flares
                        
                        plt.scatter(flare_time, flare_flux, marker = '*', s = 100, color = primary_color, label = 'Flare',
                                   zorder = 10)


                        #plot threshold line
                        
                            
                        plt.plot(time, time * 0 + median_flux + (magic_flare_std * flux_std), color = threshold_color,
                                 linewidth = 3, label = str(magic_flare_std) + 'σ')
        
                        #plot quadratic
        
                        plt.plot(rise_time, guassian_rise(rise_time, alpha, sigma, c), color = fit_color)
                        plt.xlabel('Time (days)', fontsize = fontsize)
                        plt.ylabel('Detrended flux', fontsize = fontsize)
                        plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = fontsize)
                        plt.xlim(flare_time - 0.025, flare_time + 0.1)
                        plt.ylim(0.98, flare_flux + 0.05)
                        plt.xticks([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1],
                                   np.round([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1], 2))
                        plt.tick_params(direction = 'in', labelsize = labelsize)
                        plt.legend(fontsize = fontsize)
                        plt.show()
        
        
                        #plot the residuals
        
                        plt.figure(figsize = (10,8))
        
                        plt.errorbar(np.arange(0, len(res)), res, yerr = flux_std, color = cadence_color)
                        plt.scatter(np.arange(0, len(res)), res, color = cadence_color)
                            
                        plt.hlines(sec_flare_std * flux_std, 0, len(res), color = threshold_color,
                                 linewidth = 3, linestyle = '--')
        
                        plt.xlabel('Rise Point Index', fontsize = fontsize)
                        plt.ylabel('Residual Flux', fontsize = fontsize)
                        plt.ylim(min(res) - 0.05, max(res) + 0.05)
                        plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = fontsize)
                        plt.tick_params(direction = 'in', labelsize = labelsize)
                        #plt.legend(fontsize = fontsize)
                        plt.show()
                        

                    if rise_func == 'quad':


                        #Plot all flux points
            
                        plt.figure(figsize = (10,8))
                        
                        plt.scatter(time, normalized_flux, s = 10, color = cadence_color, label = 'Background Points')
                        plt.errorbar(time, normalized_flux, yerr = flux_std, linestyle = '', color = cadence_color)

                        #Plot all points belonging to the flare

                        plt.scatter(time[time_start_flare_index:time_end_flare_index + 1],
                                    normalized_flux[time_start_flare_index:time_end_flare_index + 1], s = 10,
                                    color = primary_color, label = 'Flare Points')
                        plt.errorbar(time[time_start_flare_index:time_end_flare_index + 1],
                                     normalized_flux[time_start_flare_index:time_end_flare_index + 1],
                                     yerr = flux_std, linestyle = '', color = primary_color)

                        #plot flares
                        
                        plt.scatter(flare_time, flare_flux, marker = '*', s = 100, color = primary_color, label = 'Flare',
                                   zorder = 10)
                        
                            
                        plt.plot(time, time * 0 + median_flux + (magic_flare_std * flux_std), color = threshold_color,
                                 linewidth = 3, label = str(magic_flare_std) + 'σ')
        
                        #plot quadratic
        
                        plt.plot(rise_time, quad(rise_time, a, b, c), color = fit_color)
                        plt.xlabel('Time (days)', fontsize = fontsize)
                        plt.ylabel('Detrended flux', fontsize = fontsize)
                        plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = fontsize)
                        plt.xlim(flare_time - 0.025, flare_time + 0.1)
                        plt.ylim(0.98, flare_flux + 0.05)
                        plt.xticks([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1],
                               np.round([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1], 2))
                        plt.tick_params(direction = 'in', labelsize = labelsize)
                        plt.legend(fontsize = fontsize)
                        plt.show()
        
        
                        #plot the residuals
        
                        plt.figure(figsize = (10,8))
        
                        plt.errorbar(np.arange(0, len(res)), res, yerr = flux_std, color = cadence_color)
                        plt.scatter(np.arange(0, len(res)), res, color = cadence_color)
                            
                        plt.hlines(sec_flare_std * flux_std, 0, len(res), color = threshold_color,
                                 linewidth = 3, linestyle = '--')
        
                        plt.xlabel('Rise Point Index', fontsize = fontsize)
                        plt.ylabel('Residual Flux', fontsize = fontsize)
                        plt.ylim(min(res) - 0.05, max(res) + 0.05)
                        plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = fontsize)
                        plt.tick_params(direction = 'in', labelsize = labelsize)
                        #plt.legend(fontsize = fontsize)
                        plt.show()






                ###################FINDING OUTLIER POINTS###################

                #if we have multiple consecutive points of high res we have a flare
                #now find the peak and add it to list

                if left_secondary == True:

                    #the peak of the secondary is the one of highest flux with high residual
                    #and two other points above the residual

                    #Find the points of high residuals, uses lowered threshold
                    #we want to use points at least three points to the left

                    high_res_point = res > sec_flare_std * flux_std

                    #find all the points that have two neighboring points of high residual

                    bright_neighbors = np.full(len(res), 0, dtype = bool)

                    #again, start at fourth point and keep previous points false

                    for k in range(len(res)):

                        #initialize neighboring conditions to see if next three points are
                        #above the threshold

                        three_right = False

                        #or if the 1st point itself is the beginning and there are two more to the right

                        two_right = False


                        three_right = ((res[k+1] > sec_flare_std * flux_std) &
                                     (res[k+2] > sec_flare_std * flux_std) &
                                     (res[k+3] > sec_flare_std * flux_std))

                        two_right = ((res[k+1] > sec_flare_std * flux_std) &
                                     (res[k+2] > sec_flare_std * flux_std) &
                                     (res[k] > sec_flare_std * flux_std))

                        #if any of these are true we have three consecutive bright points

                        if two_right == True:

                            #tally of how many points to the right of k are bright, can assume it's at least two
    
                            extra = 2

                            #keep searching until they dip back down or we reach the edge
    
                            while res[k + extra] >= sec_flare_std * flux_std:

                                #check to see if adding one more index will cause overflow
    
                                if k + extra >= len(res) - 3:
    
                                    break

                                extra += 1
    
                            secondary_indices = np.arange(k, k + extra)

                            #exit the loop

                            break

                        

                        elif three_right == True:

                            #tally of how many points to the right of k are bright, can assume it's at least two
    
                            extra = 2

                            #keep searching until they dip back down or we reach the edge
    
                            while res[k + 1 + extra] >= sec_flare_std * flux_std:

                                #check to see if adding one more index will cause overflow
    
                                if k + extra >= len(res) - 3:
    
                                    break

                                extra += 1
    
                            secondary_indices = np.arange(k + 1, k + extra)

                            #exit the loop

                            break


                    #now find the point meeting these two conditions with the highest flux

                    #we need to be careful about the indices here, high_res_points is just looking
                    #at the indices between the beginning and peak of the flare, if we add the index of 
                    #the beginning we'll get the index in the lightcurve

                    secondary_peak_index = np.where((normalized_flux[rise_indices] ==
                                                     max(normalized_flux[rise_indices][secondary_indices])))[0]

                    #quickly pull out the amp of the secondary
                    
                    sec_flare_amp = res[secondary_peak_index][0]

                    #and the amp in terms of sigma

                    sec_flare_amp_sigma = sec_flare_amp/(flux_std)

                    #also pull out the equivalent duration

                    sec_equivalent_duration = np.trapz(res[secondary_indices],
                                                       x = time[secondary_indices + time_start_flare_index])

                    #convert to seconds

                    sec_equivalent_duration *= 86400

                    

                    #set the index of the secondary to be along the whole lightcurve
                    #rather than just in the rise

                    secondary_peak_index += time_start_flare_index

                    secondary_indices += time_start_flare_index


                    

                    if normalized_flux[secondary_peak_index][0] > median_flux + magic_flare_std * flux_std:

                        #now we can get the amp and the time
                        #right now it's pretty rudimentary and we just have the beginning and end in the same
                        #timw as the peak with no calculated equivalent duration, but that will be later work
    
                        flare_peak_times.append(time[secondary_peak_index][0])
            
                        flare_start_times.append(time[min(secondary_indices)])
            
                        flare_end_times.append(time[max(secondary_indices)])
            
                        #make sure to subtract one from the fluxes to isolate
                        #the amp of the flare
            
                        flare_amps.append(sec_flare_amp)
    
                        #########LATER WORK: CALCULATE THE ENERGY FOR THESE FLARES#######
            
                        flare_equivalent_durations.append(sec_equivalent_duration)

                        ############ADD THE FLAGS#############

                        #this was a secondary detection

                        primary_or_secondary.append('secondary')


                        #there are at least three points in the flare, it's given by the bright points argument

                        num_in_sec = len(np.where((normalized_flux[secondary_indices])[0]))

                        points_in_flare.append(num_in_sec)


                        ## All of these points should be above the threshold

                        points_abv_threshold.append(num_in_sec)


                        ##And the amp sigma

                        amp_sigma.append(sec_flare_amp_sigma)


                        #################VISUALIZING THE FIT################

                        if visualize_fit == True:


                            if rise_func == 'gaussian':


                                #Plot all flux points
            
                                plt.figure(figsize = (10,8))
                                
                                plt.scatter(time, normalized_flux, s = 10, color = cadence_color, label = 'Background Points')
                                plt.errorbar(time, normalized_flux, yerr = flux_std, linestyle = '', color = cadence_color)
        
                                #Plot all points belonging to the flare
        
                                plt.scatter(time[time_start_flare_index:time_end_flare_index + 1],
                                            normalized_flux[time_start_flare_index:time_end_flare_index + 1], s = 10,
                                            color = primary_color, label = 'Flare Points')
                                plt.errorbar(time[time_start_flare_index:time_end_flare_index + 1],
                                             normalized_flux[time_start_flare_index:time_end_flare_index + 1],
                                             yerr = flux_std, linestyle = '', color = primary_color)
        
                                #plot flares
                                
                                plt.scatter(flare_time, flare_flux, marker = '*', s = 100,
                                            color = primary_color, label = 'Primary Flare', zorder = 10)
                                plt.scatter(time[secondary_peak_index][0], normalized_flux[secondary_peak_index][0],
                                            marker = '*', s = 100, color = secondary_color, label = 'Secondary Flare',
                                   zorder = 10)
                                
                                    
                                plt.plot(time, time * 0 + median_flux + (magic_flare_std * flux_std), color = threshold_color,
                                         linewidth = 3, label = str(magic_flare_std) + 'σ')
                
                                #plot gaussian
                
                                #plt.plot(rise_time, guassian_rise(rise_time, alpha, sigma, c), color = fit_color)
                                plt.xlabel('Time (days)', fontsize = fontsize)
                                plt.ylabel('Detrended flux', fontsize = fontsize)
                                plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = fontsize)
                                plt.xlim(flare_time - 0.025, flare_time + 0.1)
                                plt.ylim(0.98, flare_flux + 0.05)
                                plt.tick_params(direction = 'in', labelsize = labelsize)
                                plt.xticks([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1],
                                            np.round([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1], 2))
                                plt.legend(fontsize = fontsize)
                                plt.show()
                        

                            if rise_func == 'quad':
        
        
                                #Plot all flux points
            
                                plt.figure(figsize = (10,8))
                                
                                plt.scatter(time, normalized_flux, s = 10, color = cadence_color, label = 'Background Points')
                                plt.errorbar(time, normalized_flux, yerr = flux_std, linestyle = '', color = cadence_color)
        
                                #Plot all points belonging to the flare
        
                                plt.scatter(time[time_start_flare_index:time_end_flare_index + 1],
                                            normalized_flux[time_start_flare_index:time_end_flare_index + 1], s = 10,
                                            color = primary_color, label = 'Flare Points')
                                plt.errorbar(time[time_start_flare_index:time_end_flare_index + 1],
                                             normalized_flux[time_start_flare_index:time_end_flare_index + 1],
                                             yerr = flux_std, linestyle = '', color = primary_color)
        
                                #plot flares
                                
                                plt.scatter(flare_time, flare_flux, marker = '*', s = 100,
                                            color = primary_color, label = 'Primary Flare',
                                           zorder = 10)
                                plt.scatter(time[secondary_peak_index][0], normalized_flux[secondary_peak_index][0],
                                            marker = '*', s = 100, color = secondary_color, label = 'Secondary Flare',
                                   zorder = 10)
                                
                                    
                                plt.plot(time, time * 0 + median_flux + (magic_flare_std * flux_std), color = threshold_color,
                                         linewidth = 3, label = str(magic_flare_std) + 'σ')
                
                                #plot quad
                
                                #plt.plot(rise_time, quad(rise_time, a, b, c), color = fit_color)
                                plt.xlabel('Time (days)', fontsize = fontsize)
                                plt.ylabel('Detrended flux', fontsize = fontsize)
                                plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = fontsize)
                                plt.xlim(flare_time - 0.025, flare_time + 0.1)
                                plt.ylim(0.98, flare_flux + 0.05)
                                plt.xticks([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1],
                                           np.round([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1], 2))
                                plt.tick_params(direction = 'in', labelsize = labelsize)
                                plt.legend(fontsize = fontsize)
                                plt.show()



            ####WE ALSO NEED TO MAKE A CONDITION THAT IF THE FIT IS AWFUL WE'LL REFIT THE LINE
            #AND ONLY INCLUDE THE POINTS THAT MONOTONICALLY DECREASE TO SEE IF THERES A LARGE COMPLEX
            #TO THE LEFT



            ######################RIGHT SEARCH#######################


            #Now comes the harder one in principle, we need to fit an exponential to the decay and look for
            #high res points. We should really only bother doing this if we have enough points. We'll be discounting
            #the residuals of the first four points and looking for at least three to be above the secondary threshold
            #so we need at least six points to the right of the peak

            if time_end_flare_index - time_peak_flare_index >= 6:

                #define a double exponential decay that starts at the flare peak time

                def dbl_exp_decay(x, alpha_0, beta_0, alpha_1, beta_1, C):

                    return (alpha_0 * np.exp(- beta_0 * (x - flare_peak_time)) +
                            alpha_1 * np.exp(- beta_1 * (x - flare_peak_time))  + C)

                #run a curve fit on the decay portion of the curve with the error

                #pull out the indices involved

                decay_indices = np.arange(time_peak_flare_index, time_end_flare_index+1)

                #pull out the times involved

                decay_times = time[time_peak_flare_index:time_end_flare_index+1]

                #and the fluxes

                decay_fluxes = normalized_flux[time_peak_flare_index:time_end_flare_index+1]

                #and their errors

                decay_flux_err = flux_err[time_peak_flare_index:time_end_flare_index+1]

                #initial parameter guess

                #alphas should add up to flare flux

                alpha_0_i = 0.67 * flare_flux

                alpha_1_i = 0.33 * flare_flux

                #betas, got no idea, but it's like in the hundreds and one of them should be much smaller

                beta_0_i = 500

                beta_1_i = 100

                C_i = 1

                p0 = [alpha_0_i, beta_0_i, alpha_1_i, beta_1_i, C_i]

                sigma = decay_flux_err
                #sigma[0] = 0.01 * sigma[0]


                

                #curve fit

                params, cov = scipy.optimize.curve_fit(dbl_exp_decay, decay_times, decay_fluxes,
                                                       sigma = sigma, p0 = p0, maxfev=1000000,
                                                       bounds = (0, [np.inf, np.inf, np.inf, np.inf, np.inf]))



                #Find the residuals

                alpha_0, beta_0, alpha_1, beta_1, C = params

                #make a copy

                alpha_0_first, beta_0_first, alpha_1_first, beta_1_first, C_first = params

                res = decay_fluxes - dbl_exp_decay(decay_times, alpha_0, beta_0, alpha_1, beta_1, C)

                #we want to check these residuals for three consecutive values above
                #2σ, starts at the 4th point from the peak, we don't expect one earlier 

                #initialize secondary condition

                right_secondary = False

                for k in range(4, len(res) - 2):

                    if ((res[k] > sec_flare_std * flux_std) and
                        (res[k + 1] > sec_flare_std * flux_std) and
                        (res[k + 2] > sec_flare_std * flux_std)):

                        #three consecutive points, there's a flare

                        right_secondary = True


                #################VISUALIZING THE FIT################

                if visualize_fit == True:


                    #Plot all flux points
            
                    plt.figure(figsize = (10,8))
                    
                    plt.scatter(time, normalized_flux, s = 10, color = cadence_color, label = 'Background Points')
                    plt.errorbar(time, normalized_flux, yerr = flux_std, linestyle = '', color = cadence_color)
    
                    #Plot all points belonging to the flare
    
                    plt.scatter(time[time_start_flare_index:time_end_flare_index + 1],
                                normalized_flux[time_start_flare_index:time_end_flare_index + 1], s = 10,
                                color = primary_color, label = 'Flare Points')
                    plt.errorbar(time[time_start_flare_index:time_end_flare_index + 1],
                                 normalized_flux[time_start_flare_index:time_end_flare_index + 1],
                                 yerr = flux_std, linestyle = '', color = primary_color)
    
                    #plot flares
                    
                    plt.scatter(flare_time, flare_flux, marker = '*', s = 100, color = primary_color, label = 'Flare',
                               zorder = 10)
                    
                        
                    plt.plot(time, time * 0 + median_flux + (magic_flare_std * flux_std), color = threshold_color,
                             linewidth = 3, label = str(magic_flare_std) + 'σ')
    
                    #plot dbl exp fit
    
                    plt.plot(decay_times, dbl_exp_decay(decay_times, alpha_0, beta_0, alpha_1, beta_1, C),
                             color = fit_color)
                    plt.xlabel('Time (days)', fontsize = fontsize)
                    plt.ylabel('Detrended flux', fontsize = fontsize)
                    plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = fontsize)
                    plt.xlim(flare_time - 0.025, flare_time + 0.1)
                    plt.ylim(0.98, flare_flux + 0.05)
                    plt.xticks([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1],
                               np.round([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1], 2))
                    plt.tick_params(direction = 'in', labelsize = labelsize)
                    plt.legend(fontsize = fontsize)
                    plt.show()
    
                    #plot the residuals
    
                    plt.figure(figsize = (10,8))
    
                    plt.errorbar(np.arange(0, len(res)), res, yerr = flux_std, color = cadence_color)
                    plt.scatter(np.arange(0, len(res)), res, color = cadence_color)
                        
                    plt.hlines(sec_flare_std * flux_std, 0, len(res), color = threshold_color,
                             linewidth = 3, linestyle = '--')
    
                    plt.xlabel('Decay Point Index', fontsize = fontsize)
                    plt.ylabel('Residual Flux', fontsize = fontsize)
                    plt.ylim(min(res) - 0.05, max(res) + 0.05)
                    plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = fontsize)
                    plt.tick_params(direction = 'in', labelsize = labelsize)
                    #plt.legend(fontsize = fontsize)
                    plt.show()


                ###################FINDING OUTLIER POINTS###################         

                #if we have multiple consecutive points of high res we have a flare
                #now find the peak and add it to list

                if right_secondary == True:

                    #drop the first four points of the decay features

                    res = res[4:]

                    decay_indices = decay_indices[4:]

                    #the peak of the secondary is the one of highest flux with high residual
                    #and two other points above the residual

                    #Find the points of high residuals, uses lowered threshold
                    #we want to use points at least three points to the left

                    high_res_point = res > sec_flare_std * flux_std

                    #find all the points that have two neighboring points of high residual

                    bright_neighbors = np.full(len(res), 0, dtype = bool)

                    #again, start at fourth point and keep previous points false

                    for k in range(len(res)):

                        #initialize neighboring conditions to see if next three points are
                        #above the threshold

                        three_right = False

                        #or if the 4th point itself is the beginning and there are two more to the right

                        two_right = False

                        #if there are only three points in the residual then it must be the two_right case

                        if len(res) >= 4:

                            three_right = ((res[k+1] > sec_flare_std * flux_std) &
                                         (res[k+2] > sec_flare_std * flux_std) &
                                         (res[k+3] > sec_flare_std * flux_std))
                            

                        two_right = ((res[k+1] > sec_flare_std * flux_std) &
                                     (res[k+2] > sec_flare_std * flux_std) &
                                     (res[k] > sec_flare_std * flux_std))

                        #if any of these are true we have three consecutive bright points

                        if two_right == True:

                            #tally of how many points to the right of k are bright, can assume it's at least two
    
                            extra = 2

                            #keep searching until they dip back down or we reach the edge
    
                            while res[k + extra] >= sec_flare_std * flux_std:

                                #check to see if adding one more index will cause overflow
    
                                if k + extra >= len(res) - 3:
    
                                    break

                                #add extra index

                                extra += 1
    
                            secondary_indices = np.arange(k, k + extra)

                            #exit the loop

                            break

                        

                        elif three_right == True:

                            #tally of how many points to the right of k are bright, can assume it's at least two
    
                            extra = 2

                            #keep searching until they dip back down or we reach the edge
    
                            while res[k + 1 + extra] >= sec_flare_std * flux_std:

                                #check to see if adding one more index will cause overflow
    
                                if k + extra >= len(res) - 3:
    
                                    break

                                #add extra index

                                extra += 1
    
                            secondary_indices = np.arange(k + 1, k + extra)

                            #exit the loop

                            break

                            

                    #now find the point meeting these two conditions with the highest flux

                    #we need to be careful about the indices here, high_res_points is just looking
                    #at the indices between the beginning and peak of the flare, if we add the index of 
                    #the beginning we'll get the index in the lightcurve

                    secondary_peak_index = np.where((normalized_flux[decay_indices] ==
                                                     max(normalized_flux[decay_indices][secondary_indices])))[0]


                    #quickly pull out the amp of the secondary
                    
                    sec_flare_amp = res[secondary_peak_index][0]

                    #and the amp in terms of sigma

                    sec_flare_amp_sigma = sec_flare_amp/(flux_std)

                    #also pull out the equivalent duration

                    sec_equivalent_duration = np.trapz(res[secondary_indices],
                                                       x = time[secondary_indices + time_peak_flare_index])

                    #convert to seconds

                    sec_equivalent_duration *= 86400

                    

                    #set the index of the secondary to be along the whole lightcurve
                    #rather than just in the rise

                    secondary_peak_index += time_peak_flare_index + 4

                    secondary_indices += time_peak_flare_index + 4

                    
                    if normalized_flux[secondary_peak_index][0] > median_flux + magic_flare_std * flux_std:

                        #now we can get the amp and the time
                        #right now it's pretty rudimentary and we just have the beginning and end in the same
                        #timw as the peak with no calculated equivalent duration, but that will be later work
    
                        flare_peak_times.append(time[secondary_peak_index][0])
            
                        flare_start_times.append(time[min(secondary_indices)])
            
                        flare_end_times.append(time[max(secondary_indices)])
            
                        #make sure to subtract one from the fluxes to isolate
                        #the amp of the flare
            
                        flare_amps.append(sec_flare_amp)
    
                        #########LATER WORK: CALCULATE THE ENERGY FOR THESE FLARES#######
            
                        flare_equivalent_durations.append(sec_equivalent_duration)


                        ############ADD THE FLAGS#############

                        #this was a secondary detection

                        primary_or_secondary.append('secondary')


                        #there are at least three points in the flare, it's given by the bright points argument

                        num_in_sec = len(np.where((normalized_flux[secondary_peak_index]))[0])

                        points_in_flare.append(num_in_sec)


                        ## All of these points should be above the threshold

                        points_abv_threshold.append(num_in_sec)


                        ##And the amp sigma

                        amp_sigma.append(sec_flare_amp_sigma)



                        #################VISUALIZING THE FIT################

                        if visualize_fit == True:

    
                            #Plot all flux points
            
                            plt.figure(figsize = (10,8))
                            
                            plt.scatter(time, normalized_flux, s = 10, color = cadence_color, label = 'Background Points')
                            plt.errorbar(time, normalized_flux, yerr = flux_std, linestyle = '', color = cadence_color)
    
                            #Plot all points belonging to the flare
    
                            plt.scatter(time[time_start_flare_index:time_end_flare_index + 1],
                                        normalized_flux[time_start_flare_index:time_end_flare_index + 1], s = 10,
                                        color = primary_color, label = 'Flare Points')
                            plt.errorbar(time[time_start_flare_index:time_end_flare_index + 1],
                                         normalized_flux[time_start_flare_index:time_end_flare_index + 1],
                                         yerr = flux_std, linestyle = '', color = primary_color)
    
                            #plot flares
                            
                            plt.scatter(flare_time, flare_flux, marker = '*', s = 100, color = primary_color, label = 'Primary Flare',
                                       zorder = 10)
    
                            plt.scatter(time[secondary_peak_index][0], normalized_flux[secondary_peak_index][0],
                                        marker = '*', s = 100, color = secondary_color, zorder = 9, label = 'Secondary Flare')
                            
                                
                            plt.plot(time, time * 0 + median_flux + (magic_flare_std * flux_std), color = threshold_color,
                                     linewidth = 3, label = str(magic_flare_std) + 'σ')
            
                            #plot double exponential
            
                            #plt.plot(decay_times, dbl_exp_decay(decay_times, alpha_0, beta_0, alpha_1, beta_1, C), color = 'blue')
                            plt.xlabel('Time (days)', fontsize = fontsize)
                            plt.ylabel('Detrended flux', fontsize = fontsize)
                            plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = fontsize)
                            plt.xlim(flare_time - 0.025, flare_time + 0.1)
                            plt.ylim(0.98, flare_flux + 0.05)
                            plt.xticks([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1],
                                       np.round([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1], 2))
                            plt.tick_params(direction = 'in', labelsize = labelsize)
                            plt.legend(fontsize = fontsize)
                            plt.show()


                    ###################SEE IF THERE ARE MORE SECONDARIES###############

                    #We've found one secondary in the decay, but let's see if there's at
                    #least one more!

                    if fit_multiple_secs == True:

                        #keep looking AFTER the indices of the secondaries to see if
                        #there are more

                        #the comb of high_res_point & bright_neighbors tells us
                        #where they are

                        #search in the boolean array to find the features (where the value
                        #is True. They're each labeled with a unique label. We can pull out
                        #the labels with at least three entries. The first one is the one we
                        #identified. The rest we didn't.

                        #recall we already chopped off the first three cadences in the residual
                        #so this boolean list is only for that fourth index and beyond

                        #convery boolean array into 1's and 0's

                        #the peak of the secondary is the one of highest flux with high residual
                    
                        #and two other points above the residual

                        #Find the points of high residuals, uses lowered threshold
                        #we want to use points at least three points to the left
    
                        high_res_point = res > sec_flare_std * flux_std
    
                        #find all the points that have two neighboring points of high residual
    
                        bright_neighbors = np.full(len(res), 0, dtype = bool)
    
                        for k in range(len(res)):
    
                            #initialize neighboring conditions
    
                            two_left = False
    
                            two_surrounding = False
    
                            two_right = False
    
                            #list the conditions for neighboring
    
                            if k >= 2:
    
                                #the two points to the left are bright
    
                                two_left = ((res[k-2] > sec_flare_std * flux_std) &
                                            (res[k-1] > sec_flare_std * flux_std))
    
                            if k >= 1 and k <= len(res) - 2:
    
                                #the two surrounding points are bright
    
                                two_surrounding = ((res[k-1] > sec_flare_std * flux_std) &
                                                   (res[k+1] > sec_flare_std * flux_std))
    
                            if k <= len(res) - 3:
    
                                #the two points to the right are bright
    
                                two_right = ((res[k+1] > sec_flare_std * flux_std) &
                                             (res[k+2] > sec_flare_std * flux_std))
    
                            #if any of these are true we have three consecutive bright points
    
                            if (two_left == True) or (two_surrounding == True) or (two_right == True):
    
                                bright_neighbors[k] = True

                        
                        #convery boolean array into 1's and 0's                                

                        int_feature = np.asarray([high_res_point & bright_neighbors], dtype = int)[0]

                        #find features

                        labeled_array, num_features = scipy.ndimage.label(int_feature)

                        #if there are two features then try for second
                        #JUST BECAUSE THERE ARE TWO FEATURES DOESN'T MEAN THERE ARE TWO SECONDARIES!!!!

                        if num_features > 1:

                            #make sure we have a condition telling us whether we already
                            #found the first secondary
    
                            first_sec_found = False
    
                            for value in range(1, max(np.unique(labeled_array)) + 1):
    
                                #see if there are three entries for this
                                #value in the features
    
                                if len(np.where((labeled_array == value))[0]) >= 3:

                                    #if this is the first time we come across this that
                                    #means it's the secondary we already found

                                    if first_sec_found == False:

                                        first_sec_found = True

                                        continue

                                    #if not, then we found another secondary!

                                    #indices of the secondary secondary flare
                                    
                                    terciary_flare_indices = np.where((labeled_array == value))[0]
                                    

                                    #find amp and t-peak

                                    terciary_peak_index = np.where((normalized_flux[decay_indices] == 
                                                                 max(normalized_flux[decay_indices][terciary_flare_indices])))[0]

                                    #quickly pull out the amp of the secondary
                    
                                    terciary_flare_amp = res[terciary_peak_index][0]
                
                                    #and the amplitude in terms of sigma
                
                                    terciary_flare_amp_sigma = terciary_flare_amp/(flux_std)

                                    #with the residual we can also calculate the equivalent duration

                                    terciary_equivalent_duration = np.trapz(res[terciary_flare_indices],
                                             x = time[terciary_flare_indices + time_peak_flare_index + 4])

                                    terciary_equivalent_duration *= 86400

                                    #set the index of the terciary to be along the whole lightcurve
                                    #rather than just in the rise


                                    terciary_flare_indices += time_peak_flare_index + 4

                                    terciary_peak_index += time_peak_flare_index + 4
                

                                    #print(time[terciary_peak_index])
                
                                    
                                    if normalized_flux[terciary_peak_index][0] > median_flux + magic_flare_std * flux_std:
                
                                        #now we can get the amp and the time
                                        #right now it's pretty rudimentary and we just have the beginning and end in the same
                                        #timw as the peak with no calculated equivalent duration, but that will be later work
                    
                                        flare_peak_times.append(time[terciary_peak_index][0])
                            
                                        flare_start_times.append(time[min(terciary_flare_indices)])
                            
                                        flare_end_times.append(time[max(terciary_peak_index)])
                            
                                        #make sure to subtract one from the fluxes to isolate
                                        #the amp of the flare
                            
                                        flare_amps.append(terciary_flare_amp)
                    
                                        #########LATER WORK: CALCULATE THE ENERGY FOR THESE FLARES#######
                            
                                        flare_equivalent_durations.append(terciary_equivalent_duration)
                
                
                                        ############ADD THE FLAGS#############
                
                                        #this was a secondary detection
                
                                        primary_or_secondary.append('terciary')
                
                
                                        #there are at least three points in the flare, it's given by the bright points argument
                
                                        num_in_sec = len(np.where((normalized_flux[terciary_peak_index]))[0])
                
                                        points_in_flare.append(num_in_sec)
                
                
                                        ## All of these points should be above the threshold
                
                                        points_abv_threshold.append(num_in_sec)
                
                
                                        ##And the amp sigma
                
                                        amp_sigma.append(terciary_flare_amp_sigma)

                                        #################VISUALIZING THE FIT################

                                        if visualize_fit == True:
                
                    
                                            #Plot all flux points
                        
                                            plt.figure(figsize = (10,8))
                                            
                                            plt.scatter(time, normalized_flux, s = 10, color = cadence_color,
                                                        label = 'Background Points')
                                            plt.errorbar(time, normalized_flux, yerr = flux_std, linestyle = '', color = cadence_color)
                    
                                            #Plot all points belonging to the flare
                    
                                            plt.scatter(time[time_start_flare_index:time_end_flare_index + 1],
                                                        normalized_flux[time_start_flare_index:time_end_flare_index + 1], s = 10,
                                                        color = primary_color, label = 'Flare Points')
                                            plt.errorbar(time[time_start_flare_index:time_end_flare_index + 1],
                                                         normalized_flux[time_start_flare_index:time_end_flare_index + 1],
                                                         yerr = flux_std, linestyle = '', color = primary_color)
                    
                                            #plot flares
                                            
                                            plt.scatter(flare_time, flare_flux, marker = '*', s = 100, color = primary_color,
                                                        label = 'Primary Flare', zorder = 10)
                    
                                            plt.scatter(time[secondary_peak_index][0], normalized_flux[secondary_peak_index][0],
                                                        marker = '*', s = 100, color = secondary_color,
                                                        zorder = 6, label = 'Secondary Flare')

                                            plt.scatter(time[terciary_peak_index][0], normalized_flux[terciary_peak_index][0],
                                                        marker = '*', s = 100, color = tertiary_color,
                                                        zorder = 10, label = 'Terciary Flare')
                                            
                                                
                                            plt.plot(time, time * 0 + median_flux + (magic_flare_std * flux_std),
                                                     color = threshold_color,
                                                     linewidth = 3, label = str(magic_flare_std) + 'σ')
                            
                                            #plot decay
                            
                                            #plt.plot(decay_times, dbl_exp_decay(decay_times, alpha_0,
                                            #                                    beta_0, alpha_1, beta_1, C), color = fit_color)
                                            plt.xlabel('Time (days)', fontsize = fontsize)
                                            plt.ylabel('Detrended flux', fontsize = fontsize)
                                            plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = fontsize)
                                            plt.xlim(flare_time - 0.025, flare_time + 0.1)
                                            plt.ylim(0.98, flare_flux + 0.05)
                                            plt.xticks([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1],
                                                        np.round([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1], 2))
                                            plt.tick_params(direction = 'in', labelsize = labelsize)
                                            plt.legend(fontsize = fontsize)
                                            plt.show()

                                    

                        

                

            
                ####################FIT TWICE########################

                # Let's see if we should try this fit again if there are
                # some marginal non-detections

                #drop the first four points of the decay features

                if (fit_twice == True) and (right_secondary == False):

                    marginal_flare = False

                    #see if there are marginal non-detections of a little less
                    #than the secondary sigma threshold

                    for k in range(4, len(res) - 2):

                        #test if there are three points above a slightly
                        #lowered threshold

                        if ((res[k] > sec_marginal_threshold * flux_std) and
                            (res[k + 1] > sec_marginal_threshold * flux_std) and
                            (res[k + 2] > sec_marginal_threshold * flux_std)):
    
                                marginal_flare = True
                            

                    #if there is one let's clip those points and refit!

                    if marginal_flare == True:

                        #res = res[4:]

                        #decay_indices = decay_indices[4:]

                        #find the consecutive points in question

                        #loop through the residual points and count how many
                        #in a row are above the marginal threshold

                        marginal_indices = np.where((res[4:] > sec_marginal_threshold * flux_std))[0]

                        #go through these indices above the threshold and see in the next
                        #two points to the right are above the threshold

                        little_flare_indices = []

                        for index in marginal_indices:

                            #see how many more to the right are above the threshold

                            extra = 3

                            #hopefully this condition is never used because we
                            #know there is a little flare so we never loop to the end

                            if index >= len(res) - 3:

                                break

                            #if the next three points are above mini threshold we've
                            #found it!

                            if ((res[index] >  sec_marginal_threshold * flux_std) &
                                (res[index + 1] >  sec_marginal_threshold * flux_std) &
                                (res[index + 2] >  sec_marginal_threshold * flux_std)):

                                #fun edge cases to make sure we don't go over the extent of the flare

                                if index == len(res) - 3:

                                    break
                                    

                                while res[index + extra] >  sec_marginal_threshold * flux_std:

                                    #if we're at the end of the flare stop looking

                                    if index + extra >= len(res) - 1:

                                        break

                                    #add one to keep searching to the right

                                    extra += 1

                                #okay we have the indices, we don't need to loop anymore!

                                break

                            little_flare_indices = np.arange(index, index + extra)

                        #clip those guys and redo the fit

                        #indices we WILL use in refit

                        #refit_indices = (np.arange(0, len(res)) != little_flare_indices)

                        refit_indices = np.setdiff1d(np.arange(0, len(res)), little_flare_indices)

                        #run a curve fit on the decay portion of the curve with the error

                        #pull out the indices involved
        
                        decay_indices = decay_indices[refit_indices]
        
                        #pull out the times involved
        
                        decay_times = decay_times[refit_indices]
        
                        #and the fluxes
        
                        decay_fluxes = decay_fluxes[refit_indices]
        
                        #and their errors
        
                        decay_flux_err = decay_flux_err[refit_indices]
        
                        #initial parameter guess
        
                        #alphas should add up to flare flux
        
                        alpha_0_i = 0.67 * flare_flux
        
                        alpha_1_i = 0.33 * flare_flux
        
                        #betas, got no idea, but it's like in the hundreds and one of them should be much smaller
        
                        beta_0_i = 500
        
                        beta_1_i = 100
        
                        C_i = 1
        
                        p0 = [alpha_0_i, beta_0_i, alpha_1_i, beta_1_i, C_i]
        
                        sigma = decay_flux_err
                        sigma[0] = 0.01
        
        
                        #curve fit
        
                        params, cov = scipy.optimize.curve_fit(dbl_exp_decay, decay_times, decay_fluxes,
                                                               sigma = sigma, p0 = p0, maxfev=1000000,
                                                               bounds = (0, [np.inf, np.inf, np.inf, np.inf, np.inf]))
        
        
        
                        #Find the residuals
        
                        alpha_0, beta_0, alpha_1, beta_1, C = params

                        #Add the clipped points back in

                        #pull out the indices involved
        
                        decay_indices = np.arange(time_peak_flare_index, time_end_flare_index+1)
        
                        #pull out the times involved
        
                        decay_times = time[time_peak_flare_index:time_end_flare_index+1]
        
                        #and the fluxes
        
                        decay_fluxes = normalized_flux[time_peak_flare_index:time_end_flare_index+1]
        
                        #and their errors
        
                        decay_flux_err = flux_err[time_peak_flare_index:time_end_flare_index+1]
        
                        res = decay_fluxes - dbl_exp_decay(decay_times, alpha_0, beta_0, alpha_1, beta_1, C)

                        #now do a real secondary search

                        #we want to check these residuals for three consecutive values above
                        #2σ, starts at the 4th point from the peak, we don't expect one earlier 
        
                        #initialize secondary condition
        
                        right_secondary = False
        
                        for k in range(4, len(res) - 2):
        
                            if ((res[k] > sec_flare_std * flux_std) and
                                (res[k + 1] > sec_flare_std * flux_std) and
                                (res[k + 2] > sec_flare_std * flux_std)):
        
                                #three consecutive points, there's a flare
        
                                right_secondary = True
        
        
                        #################VISUALIZING THE FIT################
        
                        if visualize_fit == True:
        
        
                            #Plot all flux points
            
                            plt.figure(figsize = (10,8))
                            
                            plt.scatter(time, normalized_flux, s = 10, color = cadence_color, label = 'Background Points')
                            plt.errorbar(time, normalized_flux, yerr = flux_std, linestyle = '', color = cadence_color)
    
                            #Plot all points belonging to the flare
    
                            plt.scatter(time[time_start_flare_index:time_end_flare_index + 1],
                                        normalized_flux[time_start_flare_index:time_end_flare_index + 1], s = 10,
                                        color = primary_color, label = 'Flare Points')
                            plt.errorbar(time[time_start_flare_index:time_end_flare_index + 1],
                                         normalized_flux[time_start_flare_index:time_end_flare_index + 1],
                                         yerr = flux_std, linestyle = '', color = primary_color)
    
                            #plot flares
                            
                            plt.scatter(flare_time, flare_flux, marker = '*', s = 100, color = primary_color, label = 'Flare',
                                       zorder = 10)
                            
                                
                            plt.plot(time, time * 0 + median_flux + (magic_flare_std * flux_std), color = threshold_color,
                                     linewidth = 3, label = str(magic_flare_std) + 'σ')
            
                            #plot dbl exp fit
            
                            plt.plot(decay_times, dbl_exp_decay(decay_times, alpha_0, beta_0, alpha_1, beta_1, C),
                                     color = fit_color, linestyle = '--')
                            plt.xlabel('Time (days)', fontsize = fontsize)
                            plt.ylabel('Detrended flux', fontsize = fontsize)
                            plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector) + ' Second Try', fontsize = fontsize)
                            plt.xlim(flare_time - 0.025, flare_time + 0.1)
                            plt.ylim(0.98, flare_flux + 0.05)
                            plt.xticks([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1],
                                       np.round([flare_time - 0.02, flare_time + 0.04, flare_time + 0.1], 2))
                            plt.tick_params(direction = 'in', labelsize = labelsize)
                            plt.legend(fontsize = fontsize)
                            plt.show()
            
                            #plot the residuals
        
                            plt.figure(figsize = (10,8))
            
                            plt.errorbar(np.arange(0, len(res)), res, yerr = flux_std, color = cadence_color)
                            plt.scatter(np.arange(0, len(res)), res, color = cadence_color)
                                
                            plt.hlines(sec_flare_std * flux_std, 0, len(res), color = threshold_color,
                                     linewidth = 3, linestyle = '--')
            
                            plt.xlabel('Decay Point Index', fontsize = fontsize)
                            plt.ylabel('Residual Flux', fontsize = fontsize)
                            plt.ylim(min(res) - 0.05, max(res) + 0.05)
                            plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = fontsize)
                            plt.tick_params(direction = 'in', labelsize = labelsize)
                            #plt.legend(fontsize = fontsize)
                            plt.show()
        
        
        
                        ###################FINDING OUTLIER POINTS###################         
        
                        #if we have multiple consecutive points of high res we have a flare
                        #now find the peak and add it to list
        
                        if right_secondary == True:
        
                            #drop the first four points of the decay features
        
                            res = res[4:]
        
                            decay_indices = decay_indices[4:]
        
                            #the peak of the secondary is the one of highest flux with high residual
                            #and two other points above the residual
        
                            #Find the points of high residuals, uses lowered threshold
                            #we want to use points at least three points to the left
        
                            high_res_point = res > sec_flare_std * flux_std
        
                            #find all the points that have two neighboring points of high residual
        
                            bright_neighbors = np.full(len(res), 0, dtype = bool)
        
                            #again, start at fourth point and keep previous points false
        
                            for k in range(len(res)):

                                #initialize neighboring conditions to see if next three points are
                                #above the threshold
        
                                three_right = False
        
                                #or if the 4th point itself is the beginning and there are two more to the right
        
                                two_right = False
        
        
                                #if there are only three points in the residual then it must be the two_right case

                                if len(res) >= 4:
        
                                    three_right = ((res[k+1] > sec_flare_std * flux_std) &
                                                 (res[k+2] > sec_flare_std * flux_std) &
                                                 (res[k+3] > sec_flare_std * flux_std))
                                    
        
                                two_right = ((res[k+1] > sec_flare_std * flux_std) &
                                             (res[k+2] > sec_flare_std * flux_std) &
                                             (res[k] > sec_flare_std * flux_std))
        
                                #if any of these are true we have three consecutive bright points
        
                                if two_right == True:
        
                                    #tally of how many points to the right of k are bright, can assume it's at least two
            
                                    extra = 2
        
                                    #keep searching until they dip back down or we reach the edge
            
                                    while res[k + extra] >= sec_flare_std * flux_std:

                                        #check to see if adding one index will cause overflow
            
                                        if k + extra >= len(res) - 3:
            
                                            break

                                        #add index

                                        extra += 1
            
                                    secondary_indices = np.arange(k, k + extra +1)
        
                                    #exit the loop
        
                                    break
        
                                
        
                                elif three_right == True:
        
                                    #tally of how many points to the right of k are bright, can assume it's at least two
            
                                    extra = 2
        
                                    #keep searching until they dip back down or we reach the edge
            
                                    while res[k + 1 + extra] >= sec_flare_std * flux_std:

                                        #check to see if one more index will cause overflow
            
                                        if k + extra >= len(res) - 3:
            
                                            break

                                        #add one index

                                        extra += 1
            
                                    secondary_indices = np.arange(k + 1, k + extra + 1)
        
                                    #exit the loop
        
                                    break
                                    
        
                            #now find the point meeting these two conditions with the highest flux

                            #we need to be careful about the indices here, high_res_points is just looking
                            #at the indices between the beginning and peak of the flare, if we add the index of 
                            #the beginning we'll get the index in the lightcurve
        
                            secondary_peak_index = np.where((normalized_flux[decay_indices] ==
                                                             max(normalized_flux[decay_indices][secondary_indices])))[0]
        
        
                            #quickly pull out the amp of the secondary
                            
                            sec_flare_amp = res[secondary_peak_index][0]
        
                            #and the amp in terms of sigma
        
                            sec_flare_amp_sigma = sec_flare_amp/(flux_std)
        
                            #also pull out the equivalent duration
        
                            sec_equivalent_duration = np.trapz(res[secondary_indices],
                                                               x = time[secondary_indices + time_peak_flare_index])
        
                            #convert to seconds
        
                            sec_equivalent_duration *= 86400
        
                            
        
                            #set the index of the secondary to be along the whole lightcurve
                            #rather than just in the rise
        
                            secondary_peak_index += time_peak_flare_index + 4
        
                            secondary_indices += time_peak_flare_index + 4
        
                            
                            if normalized_flux[secondary_peak_index][0] > median_flux + magic_flare_std * flux_std:
        
                                #now we can get the amp and the time
                                #right now it's pretty rudimentary and we just have the beginning and end in the same
                                #timw as the peak with no calculated equivalent duration, but that will be later work
            
                                flare_peak_times.append(time[secondary_peak_index][0])
                    
                                flare_start_times.append(time[min(secondary_indices)])
                    
                                flare_end_times.append(time[max(secondary_indices)])
                    
                                #make sure to subtract one from the fluxes to isolate
                                #the amp of the flare
                    
                                flare_amps.append(sec_flare_amp)
            
                                #########LATER WORK: CALCULATE THE ENERGY FOR THESE FLARES#######
                    
                                flare_equivalent_durations.append(sec_equivalent_duration)
        
        
                                ############ADD THE FLAGS#############
        
                                #this was a secondary detection
        
                                primary_or_secondary.append('secondary_second_try')
        
        
                                #there are at least three points in the flare, it's given by the bright points argument
        
                                num_in_sec = len(np.where((normalized_flux[secondary_indices]))[0])
        
                                points_in_flare.append(num_in_sec)
        
        
                                ## All of these points should be above the threshold
        
                                points_abv_threshold.append(num_in_sec)
        
        
                                ##And the amp sigma
        
                                amp_sigma.append(sec_flare_amp_sigma)
        
        
        
                                #################VISUALIZING THE FIT################
        
                                if visualize_fit == True:
        
            
                                    #Plot all flux points
            
                                    plt.figure(figsize = (10,8))
                                    
                                    plt.scatter(time, normalized_flux, s = 10, color = cadence_color, label = 'Background Points')
                                    plt.errorbar(time, normalized_flux, yerr = flux_std, linestyle = '', color = cadence_color)
            
                                    #Plot all points belonging to the flare

                                    #primary flare
            
                                    plt.scatter(time[time_start_flare_index:time_end_flare_index + 1],
                                                normalized_flux[time_start_flare_index:time_end_flare_index + 1], s = 10,
                                                color = primary_color, label = 'Flare Points')
                                    plt.errorbar(time[time_start_flare_index:time_end_flare_index + 1],
                                                 normalized_flux[time_start_flare_index:time_end_flare_index + 1],
                                                 yerr = flux_std, linestyle = '', capsize = 4, color = primary_color)

                                    #secondary flare

                                    plt.scatter(time[secondary_indices],
                                                normalized_flux[secondary_indices], s = 10,
                                                color = secondary_color, label = 'Flare Points')
                                    plt.errorbar(time[secondary_indices],
                                                 normalized_flux[secondary_indices],
                                                 yerr = flux_std, linestyle = '', capsize = 4, color = secondary_color)
            
                                    #plot flares
                                    
                                    plt.scatter(flare_time, flare_flux, marker = '*',
                                                color = primary_color, label = 'Primary Flare',
                                               zorder = 10, s = 100)
            
                                    plt.scatter(time[secondary_peak_index][0], normalized_flux[secondary_peak_index][0],
                                                marker = '*', color = secondary_color, zorder = 8,
                                                s = 100, label  = 'Secondary Flare')
                                    
                                        
                                    plt.plot(time, time * 0 + median_flux + (magic_flare_std * flux_std), color = threshold_color,
                                             linewidth = 3, label = str(magic_flare_std) + 'σ')
                    
                                    #plot decay
                    
                                    plt.plot(decay_times, dbl_exp_decay(decay_times, alpha_0, beta_0,
                                                                        alpha_1, beta_1, C),
                                             linestyle = '--', color = fit_color)

                                    #plot decay of first attempt

                                    plt.plot(decay_times, dbl_exp_decay(decay_times, alpha_0_first, beta_0_first,
                                                                        alpha_1_first, beta_1_first,
                                                                        C_first),
                                             linestyle = '-', color = fit_color)

                                    
                                    plt.xlabel('Time (days)', fontsize = fontsize, font = 'Serif')
                                    plt.ylabel('Detrended flux', fontsize = fontsize, font = 'Serif')
                                    #plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector),
                                    #          fontsize = fontsize, font = 'Serif')
                                    plt.xlim(flare_time - 0.025, flare_time + 0.1)
                                    plt.ylim(0.98, flare_flux + 0.05)
                                    plt.xticks([flare_time - 0.01, flare_time + 0.04, flare_time + 0.1],
                                               np.round([flare_time - 0.01, flare_time + 0.04, flare_time + 0.1], 2))
                                    plt.tick_params(direction = 'in', labelsize = labelsize, 
                                                   labelfontfamily = 'Serif')
                                    #plt.legend(fontsize = fontsize)
                                    plt.show()
                            
                        #if we didn't find another secondary then we'll label the primary as a failed second fit    
                        
                        else:
    
                            primary_or_secondary.append('primary_failed_dbl_fit')

                
            

                

            #find the flare energy by subtracting 1 from each point to elliminate the underlying
            #continuum and integrate between the points
            #we'll approximate the area between points as a trapazoid with bases equal to fluxes
            #and height equal to difference in time coords
        

            #find area under curve as equivalent duration using trapezoidal rule

            #Subract one to isolate contribution from flare

            equivalent_duration = np.trapz(normalized_flux[time_start_flare_index:time_end_flare_index] - median_flux,
                                           x = time[time_start_flare_index:time_end_flare_index])

            #covert to seconds

            days_to_seconds = 86400

            equivalent_duration = equivalent_duration * 86400



            ##### DELETE REDUNDANT POINTS BELONGING TO SAME FLARE########

            time_candidates = np.delete(time_candidates, redundant_points)

            flare_candidates = np.delete(flare_candidates, redundant_points)


            #add these new characteristics to the running tally

            flare_peak_times.append(flare_time)

            flare_start_times.append(flare_start_time)

            flare_end_times.append(flare_end_time)

            #make sure to subtract one from the fluxes to isolate
            #the amp of the flare

            flare_amps.append(flare_flux - median_flux)

            flare_equivalent_durations.append(equivalent_duration)


            #########ADD THE FLAGS##########

            #this was a primary flare
            #if we tried double fitting and didn't find anything then the length of the
            #list will have one too many entries and be the same length as the above flare amp list

            #if they're not the same length then we still havent added it

            if len(primary_or_secondary) != len(flare_equivalent_durations):

                primary_or_secondary.append('primary')


            #the points associated with the flare is given by the end and start indices

            points_in_flare.append(time_end_flare_index - time_start_flare_index)


            #and the number above threshold was given when checking for sufficient points

            points_abv_threshold.append(num_past_threshold)


            #and amp sigma

            amp_sigma.append((flare_flux - median_flux)/(flux_std))

            i = i + 1

        #if it's not a true flare
        
        else:

            time_candidates = np.delete(time_candidates, i)

            flare_candidates = np.delete(flare_candidates, i)

            #key point, DONT increase the index, we're tossing
            #out this data point so we want to re-evaluate the index i

            i = i
        
        #condition to break, if the next iteration will have an i that is out of
        #range of the array, we'll end the for loop

        if i >= (len(time_candidates)):

            break

    #convert everything to numpy array
    #as a dummy check we're not double counting anything let's use np.unique to
    #only get unique entries
    
    flare_peak_times, unique_index = np.unique(np.array(flare_peak_times), return_index = True)

    flare_start_times = np.array(flare_start_times)[unique_index]

    flare_end_times = np.array(flare_end_times)[unique_index]

    flare_amps = np.array(flare_amps)[unique_index]

    flare_equivalent_durations = np.array(flare_equivalent_durations)[unique_index]

    primary_or_secondary = np.array(primary_or_secondary)[unique_index]

    points_in_flare = np.array(points_in_flare)[unique_index]

    points_abv_threshold = np.array(points_abv_threshold)[unique_index]

    amp_sigma = np.array(amp_sigma)[unique_index]

    return (flare_peak_times, flare_start_times, flare_end_times, flare_amps, flare_equivalent_durations,
    primary_or_secondary, points_in_flare, points_abv_threshold, amp_sigma)




def flare_energy_calc(star_luminosity, equivalent_duration):

    c = 0.19 #value adopted from Petrucci etal 2024

    #YES, I KNOW THEY SAY IT'S 0.19 BUT THAT'S JUST NOT TRUE
    #I CHECKED THEIR NUMBERS AND C MUST BE 1. WHYYYYYYY?

    equivalent_duration_secs = equivalent_duration

    #equation used in Petrucci etal 2024

    flare_energy = equivalent_duration_secs * star_luminosity / c #in erg/s

    return flare_energy