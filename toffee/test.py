import .toffee
import numpy as np
import pytest



def test_flare_detection(plot = False):

    np.random.seed(42)
    
    #Function that creates simple lightcurve and tests various aspects
    #of TOFFEE to detect them

    
    #Simple, detrended lightcurve

    time = np.linspace(10, 30, 10000) #just some time labels for TOFFEE

    flux = 1 + np.random.normal(0, 10**-2, 10000) #Note we set the spread of the points to 10**-2

    flux_err = np.full(10000, 10**-2) #same as spread of points

    quality = np.full(10000, 0) #state all qualities are good


    toffee.flatten(time, flux, flux_err)

    
    #define shape for the decay of a flare given from double exponential decay
    def dbl_exp_decay(x, start_time, alpha_0, beta_0, alpha_1, beta_1, C):
    
        return (alpha_0 * np.exp(- beta_0 * (x - start_time)) +
                alpha_1 * np.exp(- beta_1 * (x - start_time))  + C)



    #Add simple impulse with a little noise

    print('Testing with simple impulse...')

    print('         Test with consecutive rule...')
    

    flux[20:40] = dbl_exp_decay(time[20:40], time[20], 0.1, 50, 0.1, 100, 1) + np.random.normal(0, 10**-2, 20)

    if plot == True:

        plt.scatter(time[20:40], flux[20:40], color = 'black', s = 20)
        plt.xlabel('Time (days)', fontsize = 14)
        plt.ylabel('Detrended Flux', fontsize = 14)
        plt.tick_params(direction = 'in', labelsize = 12)
        plt.show()

    #plt.scatter(time[20:40], flux[20:40])

    #Run through TOFFEE

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = False,
                                                clip_breaks = 0, consecutive = True)

    detected_peak_time = flare_characteristics[0]
    true_peak_time = time[20]

    #See if the times of the peaks are the same

    assert true_peak_time == detected_peak_time

    print('         Testing with 3 in 4 rule...')

    #Run through TOFFEE

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = False,
                                                clip_breaks = 0, consecutive = False)

    detected_peak_time = flare_characteristics[0]
    true_peak_time = time[20]

    assert true_peak_time == detected_peak_time

    print('Testing with secondary flare to the left...')

    #add another flare later in the curve

    flux[1040:1070] = (dbl_exp_decay(time[1040:1070], time[1040], 0.075, 100, 0.075, 150, 0) +
                       np.random.normal(0, 10**-2, 30) + 1)

    #and then a larger one immediately to the right

    flux[1048:1070] = flux[1048:1070] + dbl_exp_decay(time[1048:1070], time[1048], 0.1, 150, 0.1, 200, 0)
    
    if plot == True:

        plt.scatter(time[1040:1070], flux[1040:1070], color = 'black', s = 20)
        plt.xlabel('Time (days)', fontsize = 14)
        plt.ylabel('Detrended Flux', fontsize = 14)
        plt.tick_params(direction = 'in', labelsize = 12)
        plt.show()

    
    print('         Test with consecutive rule...')

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = False,
                                                clip_breaks = 0, consecutive = True, visualize_fit = False)

    detected_peak_times = flare_characteristics[0]
    true_peak_times = np.array([time[20], time[1040], time[1048]])

    np.testing.assert_array_equal(detected_peak_times,true_peak_times)
    


    print('         Testing with 3 in 4 rule...')

    #Run through TOFFEE

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = False,
                                                clip_breaks = 0, consecutive = False, visualize_fit = False)

    detected_peak_times = flare_characteristics[0]
    true_peak_times = np.array([time[20], time[1040], time[1048]])

    np.testing.assert_array_equal(detected_peak_times,true_peak_times)


    print('Testing with secondary flare to the right...')

    #add a large flare later in the curve

    flux[2100:2150] = (dbl_exp_decay(time[2100:2150], time[2100], 0.7, 100, 0.07, 150, 0) +
                       np.random.normal(0, 10**-2, 50) + 1)

    #and then a smaller one immediately to the right

    flux[2110:2150] = flux[2110:2150] + dbl_exp_decay(time[2110:2150], time[2110], 0.05, 150, 0.05, 200, 0)

    if plot == True:

        plt.scatter(time[2100:2150], flux[2100:2150], color = 'black', s = 20)
        plt.xlabel('Time (days)', fontsize = 14)
        plt.ylabel('Detrended Flux', fontsize = 14)
        plt.tick_params(direction = 'in', labelsize = 12)
        plt.show()


    print('         Test with consecutive rule...')


    #Run through TOFFEE

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = False,
                                                clip_breaks = 0, consecutive = True, visualize_fit = False)
    
    detected_peak_times = flare_characteristics[0]
    true_peak_times = np.array([time[20], time[1040], time[1048], time[2100], time[2110]])

    np.testing.assert_array_equal(detected_peak_times,true_peak_times)



    print('         Testing with 3 in 4 rule...')


    #Run through TOFFEE

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = False,
                                                clip_breaks = 0, consecutive = False, visualize_fit = False)
    
    detected_peak_times = flare_characteristics[0]
    true_peak_times = np.array([time[20], time[1040], time[1048], time[2100], time[2110]])

    np.testing.assert_array_equal(detected_peak_times,true_peak_times)



    print('Testing tertiary flare detection...')


    #return to that flare we added

    flux[2100:2150] = (dbl_exp_decay(time[2100:2150], time[2100], 0.5, 75, 0.5, 100, 0) +
                       np.random.normal(0, 10**-2, 50) + 1)

    #and the smaller one immediately to the right

    flux[2110:2150] = flux[2110:2150] + dbl_exp_decay(time[2110:2150], time[2110], 0.1, 120, 0.1, 200, 0)

    #and then another small one to the right

    flux[2120:2150] = flux[2120:2150] + dbl_exp_decay(time[2120:2150], time[2120], 0.08, 100, 0.08, 200, 0)

    if plot == True:
        plt.scatter(time[2100:2150], flux[2100:2150], color = 'black', s = 20)
        plt.xlabel('Time (days)', fontsize = 14)
        plt.ylabel('Detrended Flux', fontsize = 14)
        plt.tick_params(direction = 'in', labelsize = 12)
        plt.show()


    print('         Test with consecutive rule...')


    #Run through TOFFEE

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = False,
                                                clip_breaks = 0, consecutive = True,
                                                visualize_fit = False, fit_multiple_secs = True)
    
    detected_peak_times = flare_characteristics[0]
    true_peak_times = np.array([time[20], time[1040], time[1048], time[2100], time[2110], time[2120]])
    np.testing.assert_array_equal(detected_peak_times,true_peak_times)

    #Test that they're labeled correctly
    detected_flare_types = flare_characteristics[5]
    true_flare_types = ['primary', 'secondary', 'primary', 'primary', 'secondary', 'tertiary']
    np.testing.assert_array_equal(detected_flare_types,true_flare_types)



    print('         Testing with 3 in 4 rule...')


    #Run through TOFFEE

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = False,
                                                clip_breaks = 0, consecutive = False,
                                                visualize_fit = False, fit_multiple_secs = True)
    
    detected_peak_times = flare_characteristics[0]
    true_peak_times = np.array([time[20], time[1040], time[1048], time[2100], time[2110], time[2120]])
    np.testing.assert_array_equal(detected_peak_times,true_peak_times)

    #Test that they're labeled correctly
    detected_flare_types = flare_characteristics[5]
    true_flare_types = ['primary', 'secondary', 'primary', 'primary', 'secondary', 'tertiary']
    np.testing.assert_array_equal(detected_flare_types,true_flare_types)


    print('Testing Double Fitting...')


    #return to that flare we added

    flux[2500:2550] = (dbl_exp_decay(time[2500:2550], time[2500], 0.5, 60, 0.5, 100, 0) +
                       np.random.normal(0, 10**-2, 50) + 1)

    #and the smaller one immediately to the right

    flux[2510:2550] = flux[2510:2550] + dbl_exp_decay(time[2510:2550], time[2510], 0.04, 150, 0.04, 200, 0)

    if plot == True:
        plt.scatter(time[2500:2550], flux[2500:2550], color = 'black', s = 20)
        plt.xlabel('Time (days)', fontsize = 14)
        plt.ylabel('Detrended Flux', fontsize = 14)
        plt.tick_params(direction = 'in', labelsize = 12)
        plt.show()


    print('         Test with consecutive rule...')


    #Run through TOFFEE

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = False,
                                                clip_breaks = 0, consecutive = True,
                                                visualize_fit = False, fit_multiple_secs = True,
                                                fit_twice = True)
    
    detected_peak_times = flare_characteristics[0]
    true_peak_times = np.array([time[20], time[1040], time[1048], time[2100],
                                time[2110], time[2120], time[2500], time[2510]])
    np.testing.assert_array_equal(detected_peak_times,true_peak_times)

    #Test that they're labeled correctly
    detected_flare_types = flare_characteristics[5]
    true_flare_types = ['primary', 'secondary', 'primary', 'primary', 'secondary', 'tertiary',
                        'primary', 'secondary_second_try']
    np.testing.assert_array_equal(detected_flare_types,true_flare_types)



    print('         Testing with 3 in 4 rule...')


    #Run through TOFFEE

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = False,
                                                clip_breaks = 0, consecutive = False,
                                                visualize_fit = False, fit_multiple_secs = True,
                                                fit_twice = True)
    
    detected_peak_times = flare_characteristics[0]
    true_peak_times = np.array([time[20], time[1040], time[1048], time[2100],
                                time[2110], time[2120], time[2500], time[2510]])
    np.testing.assert_array_equal(detected_peak_times,true_peak_times)

    #Test that they're labeled correctly
    detected_flare_types = flare_characteristics[5]
    true_flare_types = ['primary', 'secondary', 'primary', 'primary', 'secondary', 'tertiary',
                        'primary', 'secondary_second_try']
    np.testing.assert_array_equal(detected_flare_types,true_flare_types)
    

    return 'Done!'




#Meant to cover basic modulation, double sines, and quickly rotating stars with large modulation

cases = [(2, 0.5, 0, 0, 0),
         (1.5, 0.8, 2.5, 1.4, 0.01),
         (3, 0.5, 1.2, 1.1, -0.01),
         (1.75, 2.5, 1, 1.1, -0.005),
         (1.2, 2.1, 0.6, 0.8, -0.005),
         (2, 5.1, 0.7, 1.2, 0.005),
         (5, 6, 0.5, 1.1, 0.005)
        ]


@pytest.mark.parametrize("A_1, omega_1, A_2, omega_2", cases)
def test_detrend(A_1, omega_1, A_2, omega_2, quad):

    #Create synthetic lightcurve modeled by two sines with quadratic given by
    #A_1, omega_1, A_2, omega_2, and quad
    #add some noise to it as well
    
    #Let's make our fake orbit run over sector 1 of tess observation
    orbit_1_start = 1325.31677296
    orbit_1_end = 1338.52857851
    orbit_2_start = 1339.66052295
    orbit_2_end = 1353.1848285
    
    time = np.linspace(orbit_1_start, orbit_2_end, 18000) #time across entire sector 1 times

    flux = 100 + A_1*np.sin(omega_1 * time) + A_2*np.sin(omega_2 * time) + quad*(time - 1340) #basic modulation

    flux = flux + np.random.normal(0, 1, 18000) #add noise

    flux_err = np.full(18000, 1) #same as spread of points
    
    #Add a long break in the middle to replicate TESS Orbit break
    break_mask = ((time > orbit_1_end) & (time < orbit_2_start)) == False
    time = time[break_mask]
    flux = flux[break_mask]
    flux_err = flux_err[break_mask]

    #detrend

    orbit_t, lc_quad, lc_wotan, periodic = toffee.flatten(time, flux, flux_err, plot_results=False,
                                                          short_window=0.25, periodogram=None)

    #check that the spread of the detrended lightcurve is equal to the normalized flux_err
    #or flux_err/median_flux

    norm_flux_err = np.median(flux_err/np.nanmedian(flux))

    detrended_spread = np.std(lc_wotan[0])

    assert detrended_spread == pytest.approx(norm_flux_err, abs = 1e-3) #ensure it's accurate down to 0.1% error

    #detrend

    orbit_t, lc_quad, lc_wotan, lc_flat, periodic = toffee.flatten(time, flux, flux_err, plot_results=False,
                                                          short_window=0.25, periodogram=[0.01,10])

    #check that the spread of the detrended lightcurve is equal to the normalized flux_err
    #or flux_err/median_flux

    norm_flux_err = np.median(flux_err/np.nanmedian(flux))

    detrended_spread = np.std(lc_flat[0])

    assert detrended_spread == pytest.approx(norm_flux_err, abs = 1e-3)



# Or if you don't want to use pytest

def test_detrend_cases(cases):
    
    for case in cases:
    
        A_1, omega_1, A_2, omega_2, quad = case
    
        test_detrend(A_1, omega_1, A_2, omega_2, quad)

    print('Done!')




def test_end_to_end(A_1, omega_1, A_2, omega_2, quad):

    #Create synthetic lightcurve modeled by two sines with quadratic given by
    #A_1, omega_1, A_2, omega_2, and quad
    #add some noise to it as well
    
    #Let's make our fake orbit run over sector 1 of tess observation
    orbit_1_start = 1325.31677296
    orbit_1_end = 1338.52857851
    orbit_2_start = 1339.66052295
    orbit_2_end = 1353.1848285
    
    time = np.linspace(orbit_1_start, orbit_2_end, 18000) #time across entire sector 1 times

    flux = 100 + A_1*np.sin(omega_1 * time) + A_2*np.sin(omega_2 * time) + quad*(time - 1340) #basic modulation

    flux = flux + np.random.normal(0, 1, 18000) #add noise

    flux_err = np.full(18000, 1) #same as spread of points

    quality = np.full(18000, 0)
    
    #Add a long break in the middle to replicate TESS Orbit break
    break_mask = ((time > orbit_1_end) & (time < orbit_2_start)) == False
    time = time[break_mask]
    flux = flux[break_mask]
    flux_err = flux_err[break_mask]
    quality = quality[break_mask]

    
    #define shape for the decay of a flare given from double exponential decay
    def dbl_exp_decay(x, start_time, alpha_0, beta_0, alpha_1, beta_1, C):
    
        return (alpha_0 * np.exp(- beta_0 * (x - start_time)) +
                alpha_1 * np.exp(- beta_1 * (x - start_time))  + C)

    #Add small flare

    flux[200:220] = flux[200:220] + dbl_exp_decay(time[200:220], time[200], 10, 50, 10, 100, 1)

    
    #add a large flare later in the curve

    flux[2100:2150] = flux[2100:2150] + dbl_exp_decay(time[2100:2150], time[2100], 70, 100, 70, 150, 0)

    #and then a smaller one immediately to the right that should be a secondary

    flux[2112:2152] = flux[2112:2152] + dbl_exp_decay(time[2112:2152], time[2112], 7, 150, 7, 200, 0)

    #Run through TOFFEE

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = True, window_length=0.25,
                                                periodogram=[0.01,10], clip_breaks = 0, consecutive = False, visualize_fit = False)

    #Test that it's finding the flares
    detected_peak_times = flare_characteristics[0]
    true_peak_times = np.array([time[200], time[2100], time[2112]]) 
    #np.testing.assert_array_equal(detected_peak_times,true_peak_times)
    np.testing.assert_array_almost_equal(detected_peak_times,true_peak_times, 2)

    #Test that they're labeled correctly
    detected_flare_types = flare_characteristics[5]
    true_flare_types = ['primary', 'primary', 'secondary']
    np.testing.assert_array_equal(detected_flare_types,true_flare_types)

    return flare_characteristics


# Or if you don't want to use pytest

def test_end_to_end_cases(cases):
    
    for case in cases:
    
        A_1, omega_1, A_2, omega_2, quad = case
    
        test_end_to_end(A_1, omega_1, A_2, omega_2, quad)

    print('Done!')



#Test whether ED is accurate within 5%
def test_equivalent_duration():
    np.random.seed(42)

    #Let's make our fake orbit run over sector 1 of tess observation
    orbit_1_start = 0
    orbit_1_end = 10
    
    time = np.linspace(0, 10, 1001) #time across entire sector 1 times

    flux = np.full(1001, 1) + np.random.normal(0, 10**-2, 1001)

    flux_err = np.full(1001, 10**-2) #same as spread of points

    quality = np.full(1001, 0)
    
    flux[300:310] = np.linspace(1, 10, 10) * -1 + 11

    #triangle has height of 9 (when you subtract background) and length of 9*0.01 = 0.09, area of 0.405 (days)
    true_area =  0.405 #days
    true_area = true_area * 86400 #seconds

    #run through TOFFEE

    flare_characteristics = toffee.flare_finder(time, flux, flux_err, quality, detrend = False,
                                                clip_breaks = 0, consecutive = True)
    
    measured_area = flare_characteristics[4][0]

    #Test that it's accurate to within 5% error
    assert np.isclose(true_area, measured_area, rtol = 0.05)

    return 'Done!'



def test_break_finder():

    #create flat lightcurve with various breaks
    time = np.linspace(0, 100, 1001) #time across entire sector 1 times

    flux = np.full(1001, 1)

    #Add a break 25 cadences wide, 2.5 days
    break_mask_25 = ((time > time[100]) & (time < time[126])) == False

    #Add a break 100 cadences wide, 10 days
    break_mask_100 = ((time > time[200]) & (time < time[301])) == False

    #Add a break 250 cadences wide, 25 days
    break_mask_250 = ((time > time[500]) & (time < time[751])) == False

    #Add breaks masks to lightcurve
    time = time[break_mask_25 & break_mask_100 & break_mask_250]
    flux = flux[break_mask_25 & break_mask_100 & break_mask_250]

    #Run through break detector to only find break of 250 cadences
    sector_break_frame = toffee.break_finder(time, flux, min_break = 25)

    #Test the break is at the right place
    true_break = np.array([375, 50.0, 75.1]) #recall we cut out 125 cadences before index 500 so it should start at 375
    detected_break = np.array([sector_break_frame['Break_Index'],
                               sector_break_frame['Break_Start_Time'],
                               sector_break_frame['Break_End_Time']]).flatten()

    np.testing.assert_array_almost_equal(detected_break,true_break) #Using almost equal due to some floating point error


    #Run through break detector to find breaks of 250 cadences and 100 cadences
    sector_break_frame = toffee.break_finder(time, flux, min_break = 10)

    #Test the break is at the right place
    true_break = np.array([175, 375, 20.0, 50.0, 30.1, 75.1])
    detected_break = np.array([sector_break_frame['Break_Index'],
                               sector_break_frame['Break_Start_Time'],
                               sector_break_frame['Break_End_Time']]).flatten()

    np.testing.assert_array_almost_equal(detected_break,true_break)


    #Run through break detector to find all breaks
    sector_break_frame = toffee.break_finder(time, flux, min_break = 2.5)

    #Test the break is at the right place
    true_break = np.array([100, 175, 375, 10.0, 20.0, 50.0, 12.6, 30.1, 75.1])
    detected_break = np.array([sector_break_frame['Break_Index'],
                               sector_break_frame['Break_Start_Time'],
                               sector_break_frame['Break_End_Time']]).flatten()

    np.testing.assert_array_almost_equal(detected_break,true_break)


    return 'Done!'




def test_break_masking():

    #create flat lightcurve with various breaks
    time = np.linspace(0, 100, 1001) #time across entire sector 1 times

    flux = np.full(1001, 1)

    #Add a break 25 cadences wide, 2.5 days
    break_mask_25 = ((time > time[100]) & (time < time[126])) == False

    #Add a break 100 cadences wide, 10 days
    break_mask_100 = ((time > time[200]) & (time < time[301])) == False

    #Add a break 250 cadences wide, 25 days
    break_mask_250 = ((time > time[500]) & (time < time[751])) == False

    #Add breaks masks to lightcurve
    time = time[break_mask_25 & break_mask_100 & break_mask_250]
    flux = flux[break_mask_25 & break_mask_100 & break_mask_250]

    #For simplicity we'll only clip 10 points

    #Run through break detector to only find break of 250 cadences 
    detected_mask = toffee.light_curve_mask(time, flux, min_break = 25, clip_breaks = 10)

    #For true mask 10 indices left of index 375 and 10 right of 375 and first and last 10 cadences
    #Avoiding an off by one error
    true_mask_break = ((time > time[364]) & (time < time[385])) == False
    true_mask_start = time > time[9]
    true_mask_end = time < time[616]
    true_mask = true_mask_break & true_mask_start & true_mask_end
    np.testing.assert_array_equal(detected_mask,true_mask)

    
    #Run through break detector to only find break of 100 cadences 
    detected_mask = toffee.light_curve_mask(time, flux, min_break = 10, clip_breaks = 10)

    true_mask_break_100 = ((time > time[164]) & (time < time[185])) == False
    true_mask_break_250 = ((time > time[364]) & (time < time[385])) == False
    true_mask_start = time > time[9]
    true_mask_end = time < time[616]

    true_mask = true_mask_break_100 & true_mask_break_250 & true_mask_start & true_mask_end

    np.testing.assert_array_equal(detected_mask,true_mask)


    #Run through break detector to only find break of 100 cadences 
    detected_mask = toffee.light_curve_mask(time, flux, min_break = 2.5, clip_breaks = 10)
    
    true_mask_break_25 = ((time > time[89]) & (time < time[110])) == False
    true_mask_break_100 = ((time > time[164]) & (time < time[185])) == False
    true_mask_break_250 = ((time > time[364]) & (time < time[385])) == False
    true_mask_start = time > time[9]
    true_mask_end = time < time[616]

    true_mask = true_mask_break_25 & true_mask_break_100 & true_mask_break_250 & true_mask_start & true_mask_end

    np.testing.assert_array_equal(detected_mask,true_mask)

    return 'Done!'