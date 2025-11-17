import numpy as np
from .main import flatten, break_finder, light_curve_mask, flare_finder

#A function for plotting flares, labels all the points in a lightcurve belonging to a flare and separates them
#By primary and secondary detection for purpose of coloring them differently

def flare_color(time, flux, flare_results):

    flare_peak_times, flare_start_times, flare_end_times, flare_amps, flare_type, num_points_in_flare = (flare_results[0],
                                                                                                     flare_results[1],
                                                                                                     flare_results[2],
                                                                                                     flare_results[3],
                                                                                                     flare_results[5],
                                                                                                     flare_results[6])
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
        if (flare_type[i] == 'primary') or (flare_type[i] == 'primary_failed_dbl_fit'):
            #add to the list
            times_of_primary_flares = np.append(times_of_primary_flares, flare_times)
            fluxes_of_primary_flares = np.append(fluxes_of_primary_flares, flare_fluxes)
            peak_flux_of_primary = np.append(peak_flux_of_primary, peak_flux)
            peak_time_of_primary = np.append(peak_time_of_primary, peak_time)
            
        else:
            #add to the list
            times_of_secondary_flares = np.append(times_of_secondary_flares, flare_times)
            fluxes_of_secondary_flares = np.append(fluxes_of_secondary_flares, flare_fluxes)
            peak_flux_of_secondary = np.append(peak_flux_of_secondary, peak_flux)
            peak_time_of_secondary = np.append(peak_time_of_secondary, peak_time)


    return (times_of_primary_flares, fluxes_of_primary_flares, peak_flux_of_primary, peak_time_of_primary,
            times_of_secondary_flares, fluxes_of_secondary_flares, peak_flux_of_secondary, peak_time_of_secondary)




#plotting function that automatically detrends a lightcurve, finds the flares, and makes a plot
#of the detrended lightcurve with the flares labeled by type

def detrended_sector_plot(time, flux, flux_err, quality, detrend = True, visualize_fit = False,
                          min_break = 0.025, clip_breaks = 100, periodogram = [0.01, 10], short_window = 0.25,
                          rise_func = 'gaussian', prim_marg_rate = 0.67, sec_marg_rate = 0.75,
                          consecutive = False, fit_twice = True, fit_multiple_secs = False,
                          flag_values = [0], primary_color = 'red', secondary_color = 'blue', tertiary_color = 'green',
                          cadence_color = 'black', threshold_color = 'red', fit_color = 'black',
                          fontsize = 14, labelsize = 12, TIC_number = 0, TESS_sector = 0):

    if detrend == True:

        #Detrend
    
        t_curve, quadratic, wotan_fit, flatt, periodic = main.flatten(time, flux, flux_err, plot_results=False,
                                                      short_window=short_window, periodogram=periodogram)
        
        #sometimes the size of the arrays spat out by the flattening function is not the same as the light
        #curve and thus not the same as the break_mask. So we need another mask to ensure they're the same
        
        flattening_mask = np.full(len(time), 1, dtype = bool)
        
        for index in range(len(time)):
            if time[index] in t_curve:
                continue
            else:
                flattening_mask[index] = False
        
        #convert this mask to indices to include in the mask
        flattening_mask = np.arange(len(time))[flattening_mask]
        
        #find the breaks
        sector_break_frame = main.break_finder(time, flux)
        #Now color in the masked points in a different color
        mask = main.light_curve_mask(time, flux, min_break = min_break, clip_breaks = clip_breaks)
        #apply flattening mask
        mask = mask[flattening_mask]
        #unpack flux after running periodogram and eliminating residual
        #sinusoidal noise
        flatt_trend = flatt[2]
        flatt_flux = flatt[0]

    else: 

        t_curve, flatt_flux = time, flux
    
    ###############Now is when we need to find the flares to plot them#############
    flare_results = main.flare_finder(time, flux, flux_err, quality,
                                        visualize_fit = visualize_fit, rise_func = rise_func,
                                        consecutive = consecutive, prim_marg_rate = prim_marg_rate, sec_marg_rate = sec_marg_rate,
                                        detrend = detrend, periodogram = periodogram, clip_breaks = clip_breaks,
                                        min_break = min_break, fit_twice = fit_twice, fit_multiple_secs = fit_multiple_secs,
                                        flag_values = flag_values, primary_color = primary_color,
                                        secondary_color = secondary_color, tertiary_color = tertiary_color,
                                        cadence_color = cadence_color, threshold_color = threshold_color, fit_color = fit_color,
                                        fontsize = fontsize, labelsize = labelsize, TIC_number = TIC_number, TESS_sector = TESS_sector)

    
    primary_secondary = flare_color(flare_results)

    times_of_primary_flares, fluxes_of_primary_flares, peak_flux_of_primary, peak_time_of_primary = (flare_color(flare_results)[0],
                                                                                                     flare_color(flare_results)[1],
                                                                                                     flare_color(flare_results)[2],
                                                                                                     flare_color(flare_results)[3])
    
    
    times_of_secondary_flares, fluxes_of_secondary_flares, peak_flux_of_secondary, peak_time_of_secondary = (flare_color(flare_results)[4],
                                                                                                     flare_color(flare_results)[5],
                                                                                                     flare_color(flare_results)[6],
                                                                                                     flare_color(flare_results)[7])

    #Third plot: Show the flattened lightcurve with the flares found and colored in
    plt.figure(figsize = (10,8))
    plt.scatter(t_curve, flatt_flux, s = 4, color = 'gray')
    #plot all the good flux points not masked 
    plt.scatter(t_curve[mask], flatt_flux[mask], s = 4, color = cadence_color, label = 'Used')
    #the plot for the residual periodogram result
    #plt.plot(time, flatt_trend, color = 'yellow')
    #color the points around the primary flares
    plt.scatter(times_of_primary_flares, fluxes_of_primary_flares, color = primary_color, s = 6)
    #color the points around the secondary flares
    plt.scatter(times_of_secondary_flares, fluxes_of_secondary_flares, color = secondary_color, s = 6, zorder = 10)
    #add primary flare peaks
    plt.scatter(peak_time_of_primary, peak_flux_of_primary, marker = '*', s = 200, color = primary_color)       
    #add secondary flare peaks
    plt.scatter(peak_time_of_secondary, peak_flux_of_secondary, marker = '*', s = 200, color = secondary_color)
    #add three sigma line
    plt.hlines(3 * (np.nanpercentile(flatt_flux, 84) - 1) + 1, min(time), max(time), color = threshold_color, linewidth = 2)
    plt.xlabel('Time (BJD)', fontsize = 24)
    plt.ylabel(r'Normalized Flux', fontsize = 24)
    plt.ylim(0.9, 1.25)
    plt.tick_params(direction = 'in', labelsize = 20)
    plt.title('TIC ' + str(TIC_number) + ' Sector ' + str(TESS_sector), fontsize = 24)
    plt.show()