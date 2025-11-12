import .toffee
import numpy as np
import pytest



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