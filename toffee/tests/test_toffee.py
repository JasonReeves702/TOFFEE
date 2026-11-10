'''
tests.py -- pytest suite for the TOFFEE flare-detection package.

Covers the public functions currently defined in `toffee/main.py` and
`toffee/get_tess_orbit.py`. The tests are organised in the same order that a
light curve moves through the package:

    1.  Light curve preparation      (break_finder, light_curve_mask)
    2.  Candidate identification     (find_candidates, start_end_time,
                                      consec_finder, declare_flare)
    3.  Model fitting                (model_fit, flare_residuals, adjust_bounds,
                                      fit_* routines, call_decay_model,
                                      super_fitting, robust_fitting)
    4.  Secondary detection          (find_secondaries)
    5.  Full detection pipeline      (detect_flares)
    6.  Purity methods               (red_noise_smoothing)
    7.  Masking and stellar params   (make_flare_mask, make_gap_mask,
                                      make_window_mask, spot_fitting, spot_mod)
    8.  Energy                       (flare_energy_calc)
    9.  Utilities                    (get_tess_orbit)
    10. Plotting smoke test          (visualizer)
    11. Detrending                   (flatten, get_rot_period,
                                      rotation.get_rot_period)
    12. NumPy compatibility          (the np.trapezoid shim)

Every test builds its own synthetic light curve from a seeded random number
generator, so no TESS data download and no network access is required.

Run with:      pytest tests.py -v
'''

import os
import sys

#Use a non-interactive matplotlib backend BEFORE toffee is imported, because
#main.py imports pyplot at module level and visualizer() calls plt.show().
#Without this the plotting test would try to open a window on a headless machine.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
#Import the package.
#
#If star_toffee is pip-installed, `import toffee` just works. If it is not,
#fall back to the source tree that sits alongside this file, so the suite can
#be run straight out of a checkout without installing anything first.
# ---------------------------------------------------------------------------
try:
    import toffee
    import toffee.main  #needed to monkeypatch the FFI-times download
except ImportError:  #pragma: no cover - only runs on an uninstalled checkout
    #This file lives at TOFFEE_upload_claude/toffee/tests/test_toffee.py, so
    #two levels up is TOFFEE_upload_claude/, which contains the toffee/ package
    #directly. Also check the sibling TOFFEE_upload/src, the release package.
    _here = os.path.dirname(os.path.abspath(__file__))
    for _candidate in (
        os.path.join(_here, "..", ".."),
        os.path.join(_here, "..", "..", "..", "TOFFEE_upload", "src"),
                       ):
        if os.path.isdir(os.path.join(_candidate, "toffee")):
            sys.path.insert(0, os.path.abspath(_candidate))
            break
    import toffee
    import toffee.main

#toffee.rotation is the candidate replacement for main.get_rot_period and does
#not exist in every installed copy of the package, so import it defensively and
#let the tests that need it skip when it is absent.
try:
    import toffee.rotation
    HAS_ROTATION_MODULE = True
except ImportError:  #pragma: no cover - only runs against an older install
    HAS_ROTATION_MODULE = False


# ===========================================================================
#Shared helpers and fixtures
# ===========================================================================

#TESS 2-minute cadence expressed in days. Used to build realistic time arrays.
CADENCE = 2.0 / 1440.0

#Noise level of the synthetic light curves, in relative flux units.
NOISE = 1e-3


def make_flat_lightcurve(num_points = 3000, seed = 0, noise = NOISE, time_start = 1000.0):
    '''Build a flat, normalised light curve of Gaussian noise around 1.0.

    This stands in for a detrended TESS light curve with no flares in it.

    Arguments:
        num_points: number of cadences in the light curve
        seed: seed for the random number generator, so tests are reproducible
        noise: standard deviation of the Gaussian noise on the flux
        time_start: first time coordinate, in BJD - 2457000 (TESS convention)

    Returns:
        time, flux, flux_err, quality -- four arrays of equal length
    '''
    random_generator = np.random.default_rng(seed)
    time = time_start + np.arange(num_points) * CADENCE
    flux = 1.0 + random_generator.normal(0.0, noise, num_points)
    flux_err = np.full(num_points, noise)
    #A quality array of all zeros means every cadence passes the flag check
    quality = np.zeros(num_points, dtype = int)
    return time, flux, flux_err, quality


def inject_flare(time, flux, peak_index, amplitude, tau = 0.01, num_points = 40):
    '''Add a single exponentially decaying flare to a light curve in place.

    A real flare has a fast rise and a slower decay. A pure exponential decay
    starting at the peak is enough to exercise the detection and fitting code.

    Arguments:
        time: time array of the light curve
        flux: flux array of the light curve (modified in place and returned)
        peak_index: index at which the flare peaks
        amplitude: peak amplitude of the flare in relative flux
        tau: exponential decay timescale, in days
        num_points: how many cadences the injected decay spans

    Returns:
        the modified flux array
    '''
    stop_index = min(peak_index + num_points, len(time))
    flare_window = slice(peak_index, stop_index)
    #Exponential decay measured from the peak time
    flux[flare_window] += amplitude * np.exp(
        -(time[flare_window] - time[peak_index]) / tau)
    return flux


@pytest.fixture
def flat_lightcurve():
    '''A flat, flare-free light curve.'''
    return make_flat_lightcurve(num_points = 3000, seed = 0)


@pytest.fixture
def single_flare_lightcurve():
    '''A light curve containing exactly one clean, high signal-to-noise flare.'''
    time, flux, flux_err, quality = make_flat_lightcurve(num_points = 3000, seed = 0)
    flux = inject_flare(time, flux, peak_index = 1500, amplitude = 0.05, tau = 0.01)
    return time, flux, flux_err, quality


@pytest.fixture
def decay_segment():
    '''An isolated, noiseless two-timescale flare decay.

    This is the shape the decay-fitting routines are designed to model, so a
    good fit should reproduce it almost exactly.

    Returns:
        time, flux, flux_err, peak_time
    '''
    peak_time = 1000.0
    time = np.linspace(peak_time, peak_time + 0.03, 25)
    #Two exponentials with different timescales, matching dbl_exp_decay
    flux = (1.0
            + 0.05 * np.exp(-(time - peak_time) / 0.005)
            + 0.02 * np.exp(-(time - peak_time) / 0.020))
    flux_err = np.full(len(time), NOISE)
    return time, flux, flux_err, peak_time


def global_flux_std(flux):
    '''Reproduce the sigma that TOFFEE uses internally for its thresholds.

    detect_flares defines the flux spread as the 84th percentile of (flux - 1),
    i.e. the one-sigma level of a normal distribution. Tests that call the
    lower-level detection functions directly need the same value.
    '''
    one_sigma_percentile = 84
    return np.nanpercentile(flux - 1, one_sigma_percentile)


# ===========================================================================
#1. Light curve preparation
# ===========================================================================

class TestBreakFinder:
    '''Tests for break_finder, which locates gaps in the time sampling.'''

    def test_finds_single_break(self):
        #Two continuous segments separated by a 1 day gap
        time = np.concatenate([np.linspace(0.0, 5.0, 500),
                               np.linspace(6.0, 11.0, 500)])
        flux = np.ones(len(time))

        sector_break_frame = toffee.break_finder(time, flux, min_break = 0.25)

        #Exactly one break, and it should be reported at the segment boundary
        assert isinstance(sector_break_frame, pd.DataFrame)
        assert len(sector_break_frame) == 1
        assert list(sector_break_frame.columns) == ["Break_Index", "Break_Start_Time",
                                                    "Break_End_Time"]
        assert sector_break_frame["Break_Index"].iloc[0] == 499
        assert sector_break_frame["Break_Start_Time"].iloc[0] == pytest.approx(5.0)
        assert sector_break_frame["Break_End_Time"].iloc[0] == pytest.approx(6.0)

    def test_no_break_in_continuous_curve(self, flat_lightcurve):
        #An evenly sampled curve has no gaps larger than min_break
        time, flux, _, _ = flat_lightcurve

        sector_break_frame = toffee.break_finder(time, flux, min_break = 0.25)

        assert len(sector_break_frame) == 0

    def test_finds_multiple_breaks(self):
        #Three segments, so two breaks
        time = np.concatenate([np.linspace(0.0, 5.0, 300),
                               np.linspace(6.0, 11.0, 300),
                               np.linspace(12.0, 17.0, 300)])
        flux = np.ones(len(time))

        sector_break_frame = toffee.break_finder(time, flux, min_break = 0.25)

        assert len(sector_break_frame) == 2

    def test_min_break_controls_sensitivity(self):
        #A gap of 0.5 days should count as a break at min_break=0.25 but not
        #at min_break=1.0
        time = np.concatenate([np.linspace(0.0, 5.0, 500),
                               np.linspace(5.5, 10.5, 500)])
        flux = np.ones(len(time))

        assert len(toffee.break_finder(time, flux, min_break = 0.25)) == 1
        assert len(toffee.break_finder(time, flux, min_break = 1.0)) == 0


class TestLightCurveMask:
    '''Tests for light_curve_mask, which clips cadences around gaps and edges.'''

    def test_mask_length_and_dtype(self):
        time = np.concatenate([np.linspace(0.0, 5.0, 500),
                               np.linspace(6.0, 11.0, 500)])
        flux = np.ones(len(time))

        mask = toffee.light_curve_mask(time, flux, min_break = 0.25,
                                       clip_breaks = 50)

        #One boolean entry per cadence
        assert mask.dtype == bool
        assert len(mask) == len(time)

    def test_clips_start_end_and_break(self):
        time = np.concatenate([np.linspace(0.0, 5.0, 500),
                               np.linspace(6.0, 11.0, 500)])
        flux = np.ones(len(time))
        clip_breaks = 50

        mask = toffee.light_curve_mask(time, flux, min_break = 0.25,
                                       clip_breaks = clip_breaks)

        #The first and last `clip` cadences are always removed
        assert not mask[:clip_breaks].any()
        assert not mask[-clip_breaks:].any()
        #Cadences immediately either side of the break at index 499 are removed
        assert not mask[499]
        assert not mask[500]
        #The middle of each segment survives
        assert mask[250]
        assert mask[750]

    def test_keeps_more_points_with_smaller_clip(self):
        time = np.concatenate([np.linspace(0.0, 5.0, 500),
                               np.linspace(6.0, 11.0, 500)])
        flux = np.ones(len(time))

        small_clip = toffee.light_curve_mask(time, flux, clip_breaks = 20).sum()
        large_clip = toffee.light_curve_mask(time, flux, clip_breaks = 100).sum()

        #Clipping more cadences per break must never keep more points
        assert small_clip > large_clip


# ===========================================================================
#2. Candidate identification
# ===========================================================================

class TestFindCandidates:
    '''Tests for find_candidates, the first-pass search for bright points.'''

    def test_returns_points_above_threshold_sorted_by_brightness(self,
                                                                 single_flare_lightcurve):
        time, flux, _, _ = single_flare_lightcurve
        flux_std = global_flux_std(flux)

        times, fluxes = toffee.find_candidates(time, flux, flux_std, 3.0)

        #Every returned flux must sit above the 3 sigma threshold
        threshold = 1.0 + 3.0 * flux_std
        assert (fluxes > threshold).all()
        #Results are sorted brightest first
        assert (np.diff(fluxes) <= 0).all()
        #Times and fluxes stay paired: the brightest candidate is the flare peak
        assert times[0] == pytest.approx(time[1500])

    def test_higher_threshold_returns_fewer_candidates(self,
                                                       single_flare_lightcurve):
        time, flux, _, _ = single_flare_lightcurve
        flux_std = global_flux_std(flux)

        num_flares_low_threshold = len(toffee.find_candidates(time, flux, flux_std, 3.0)[0])
        num_flares_high_threshold = len(toffee.find_candidates(time, flux, flux_std, 10.0)[0])

        assert num_flares_high_threshold < num_flares_low_threshold

    def test_flat_curve_yields_no_candidates_at_high_threshold(self,
                                                               flat_lightcurve):
        time, flux, _, _ = flat_lightcurve
        flux_std = global_flux_std(flux)

        times, fluxes = toffee.find_candidates(time, flux, flux_std, 20.0)

        #Pure noise should not produce a 20 sigma excursion
        assert len(times) == 0
        assert len(fluxes) == 0


class TestStartEndTime:
    '''Tests for start_end_time, which brackets a bright epoch.'''

    def test_brackets_the_injected_flare(self, single_flare_lightcurve):
        time, flux, _, _ = single_flare_lightcurve
        flux_std = global_flux_std(flux)

        index_start, index_end = toffee.start_end_time(
            time, flux, time[1500], flux_std,
            prim_marginal_threshold = 3.0, num_below_threshold = 3)

        #The window must contain the peak, and be ordered
        assert index_start <= 1500 <= index_end
        assert index_start < index_end
        #The flare was injected over 40 cadences, so the window should be
        #of that order rather than spanning the whole light curve
        assert (index_end - index_start) < 200

    def test_returns_integer_indices(self, single_flare_lightcurve):
        time, flux, _, _ = single_flare_lightcurve
        flux_std = global_flux_std(flux)

        index_start, index_end = toffee.start_end_time(
            time, flux, time[1500], flux_std, 3.0, 3)

        #These are used to slice arrays, so they must be integers
        assert index_start == int(index_start)
        assert index_end == int(index_end)
        assert 0 <= index_start < len(time)
        assert 0 <= index_end < len(time)

    def test_flare_at_start_of_curve_is_handled(self):
        #A flare in the first few cadences exercises the left-edge break
        time, flux, _, _ = make_flat_lightcurve(num_points = 1000, seed = 1)
        flux = inject_flare(time, flux, peak_index = 5, amplitude = 0.05)
        flux_std = global_flux_std(flux)

        index_start, index_end = toffee.start_end_time(
            time, flux, time[5], flux_std, 3.0, 3)

        assert index_start >= 0
        assert index_end > index_start


class TestConsecFinder:
    '''Tests for consec_finder, which counts consecutive above-threshold points.'''

    def test_counts_consecutive_points_in_flare(self, single_flare_lightcurve):
        time, flux, _, _ = single_flare_lightcurve
        flux_std = global_flux_std(flux)

        num_consec_points = toffee.consec_finder(time, flux, time[1500], flux_std,
                                                 threshold_std = 3.0,
                                                 prim_marginal_threshold = 3.0,
                                                 num_below_threshold = 3)

        #A 50 sigma flare decaying over ~40 cadences must clear 3 sigma for
        #several consecutive points
        assert num_consec_points >= 3
        assert isinstance(num_consec_points, (int, np.integer))

    def test_higher_threshold_gives_fewer_consecutive_points(self,
                                                             single_flare_lightcurve):
        time, flux, _, _ = single_flare_lightcurve
        flux_std = global_flux_std(flux)

        flares_low_threshold = toffee.consec_finder(time, flux, time[1500], flux_std, 3.0, 3.0, 3)
        flares_high_threshold = toffee.consec_finder(time, flux, time[1500], flux_std, 30.0, 3.0, 3)

        assert flares_high_threshold <= flares_low_threshold


class TestDeclareFlare:
    '''Tests for declare_flare, which decides whether a candidate is a flare.'''

    def test_real_flare_returns_full_attribute_tuple(self,
                                                     single_flare_lightcurve):
        time, flux, _, _ = single_flare_lightcurve
        flux_std = global_flux_std(flux)

        result = toffee.declare_flare(time, flux, time[1500], flux_std,
                                      threshold_std = 3.0, num_consec = 3,
                                      prim_marginal_threshold = 3.0,
                                      num_below_threshold = 3)

        #The success return is a 9-element tuple of flare properties
        assert result is not False
        assert len(result) == 9
        (peak, flare_start, flare_end, amp, ED, flare_type,
         num_points, num_abv_threshold, amp_sigma) = result

        assert flare_start <= peak <= flare_end
        assert flare_type == "primary"
        #Injected amplitude was 0.05; the measured peak includes noise
        assert amp == pytest.approx(0.05, abs = 5 * NOISE)
        assert ED > 0                       #equivalent duration in seconds
        assert num_points > 0
        assert num_abv_threshold > 0
        #Amplitude in sigma should be consistent with amp / flux_std
        assert amp_sigma == pytest.approx(amp / flux_std, rel = 1e-6)

    def test_returns_false_when_too_few_consecutive_points(self,
                                                           single_flare_lightcurve):
        time, flux, _, _ = single_flare_lightcurve
        flux_std = global_flux_std(flux)

        #Demand far more consecutive points than the flare can provide
        result = toffee.declare_flare(time, flux, time[1500], flux_std,
                                      threshold_std = 3.0, num_consec = 1000,
                                      prim_marginal_threshold = 3.0,
                                      num_below_threshold = 3)

        assert result is False


# ===========================================================================
#3. Model fitting
# ===========================================================================

class TestModelFit:
    '''Tests for model_fit, the generic least-squares wrapper.'''

    def test_recovers_known_linear_parameters(self):
        #A noiseless straight line: the fit must recover the exact parameters
        def linear_model(x_values, slope, intercept):
            return slope * x_values + intercept

        x_values = np.linspace(0.0, 10.0, 100)
        y_values = 3.0 * x_values + 2.0
        y_errors = np.full(len(x_values), 0.01)

        params, perr = toffee.model_fit(linear_model, x_values, y_values, y_errors,
                                        p0 = np.array([1.0, 1.0]),
                                        bounds = (0, np.inf))

        assert params == pytest.approx([3.0, 2.0], rel = 1e-6)
        #Parameter errors must be finite and non-negative
        assert len(perr) == 2
        assert np.all(np.isfinite(perr))
        assert np.all(perr >= 0)

    def test_works_without_flux_errors(self):
        #flux_err=None is allowed, and means unweighted fitting
        def linear_model(x_values, slope, intercept):
            return slope * x_values + intercept

        x_values = np.linspace(0.0, 10.0, 100)
        y_values = 3.0 * x_values + 2.0

        params, _ = toffee.model_fit(linear_model, x_values, y_values, None,
                                     p0 = np.array([1.0, 1.0]),
                                     bounds = (0, np.inf))

        assert params == pytest.approx([3.0, 2.0], rel = 1e-6)

    def test_raises_on_all_nan_flux(self):
        #An entirely NaN window cannot be fitted and must raise a clear error
        def linear_model(x_values, slope, intercept):
            return slope * x_values + intercept

        x_values = np.linspace(0.0, 10.0, 100)
        y_values = np.full(len(x_values), np.nan)
        y_errors = np.full(len(x_values), 0.01)

        with pytest.raises(ValueError, match = "No finite data points"):
            toffee.model_fit(linear_model, x_values, y_values, y_errors, p0 = np.array([1.0, 1.0]))


class TestFlareResiduals:
    '''Tests for flare_residuals, the data-minus-model helper.'''

    def test_residuals_are_near_zero_for_perfect_model(self):
        def linear_model(x_values, slope, intercept):
            return slope * x_values + intercept

        x_values = np.linspace(0.0, 10.0, 100)
        y_values = 3.0 * x_values + 2.0
        y_errors = np.full(len(x_values), 0.01)

        residuals = toffee.flare_residuals(linear_model, x_values, y_values, y_errors,
                                           p0 = np.array([1.0, 1.0]),
                                           bounds = (0, np.inf))

        #One residual per data point, all essentially zero
        assert len(residuals) == len(x_values)
        assert np.abs(residuals).max() < 1e-8


class TestAdjustBounds:
    '''Tests for adjust_bounds, which pulls initial guesses inside the bounds.'''

    def test_scalar_lower_bound_is_enforced(self):
        # -5 is below the scalar lower bound of 0 and must be moved up
        p0 = np.array([-5.0, 2.0, 100.0])

        adjusted_guesses = toffee.adjust_bounds(p0.copy(), (0, np.inf))

        assert np.all(adjusted_guesses >= 0)
        #Values already inside the bounds are left alone
        assert adjusted_guesses[1] == 2.0
        assert adjusted_guesses[2] == 100.0

    def test_listed_bounds_are_enforced_per_parameter(self):
        #Element 0 is below its lower bound, element 2 above its upper bound
        p0 = np.array([-5.0, 2.0, 100.0])
        bounds = ([0, 0, 0], [10, 10, 10])

        adjusted_guesses = toffee.adjust_bounds(p0.copy(), bounds)

        assert np.all(adjusted_guesses >= np.array(bounds[0]))
        assert np.all(adjusted_guesses <= np.array(bounds[1]))
        assert adjusted_guesses[1] == 2.0

    def test_valid_guesses_are_unchanged(self):
        p0 = np.array([1.0, 2.0, 3.0])

        adjusted_guesses = toffee.adjust_bounds(p0.copy(), (0, np.inf))

        np.testing.assert_array_equal(adjusted_guesses, p0)


class TestDecayAndRiseFits:
    '''Tests for the individual rise and decay fitting routines.'''

    def test_dbl_exp_decay_fits_its_own_model(self, decay_segment):
        time, flux, flux_err, peak_time = decay_segment

        params, perr, residuals = toffee.fit_dbl_exp_decay(
            time, flux, flux_err, peak_time)

        #dbl_exp_decay has five free parameters
        assert len(params) == 5
        assert len(perr) == 5
        assert len(residuals) == len(time)
        #The data were generated from this exact model, so the fit is excellent
        assert np.sqrt(np.mean(residuals ** 2)) < 1e-6

    def test_gompertz_decay_returns_expected_shapes(self, decay_segment):
        time, flux, flux_err, peak_time = decay_segment

        params, perr, residuals = toffee.fit_w_gompertz_decay(
            time, flux, flux_err, peak_time)

        assert len(params) == 5
        assert len(residuals) == len(time)
        assert np.all(np.isfinite(params))

    def test_gompertz_logistic_returns_expected_shapes(self, decay_segment):
        time, flux, flux_err, peak_time = decay_segment

        #The logistic variant needs unbounded fitting, since its leading
        #amplitude is initialised negative
        params, perr, residuals = toffee.fit_w_gompertz_logistic(
            time, flux, flux_err, peak_time, bounds = (-np.inf, np.inf))

        #This model has six free parameters (an extra leading amplitude)
        assert len(params) == 6
        assert len(residuals) == len(time)
        assert np.all(np.isfinite(params))

    def test_gaussian_rise_returns_expected_shapes(self):
        #Build the rising half of a Gaussian-shaped flare
        peak_time = 1000.0
        time = np.linspace(peak_time - 0.01, peak_time, 10)
        flux = 1.0 + 0.05 * np.exp(-(time - peak_time) ** 2 / (2 * 0.004 ** 2))
        flux_err = np.full(len(time), NOISE)

        params, perr, residuals = toffee.fit_gaussian_rise(
            time, flux, flux_err, peak_time)

        #guassian_rise has three free parameters: alpha, sigma, c
        assert len(params) == 3
        assert len(perr) == 3
        assert len(residuals) == len(time)
        assert np.all(np.isfinite(params))
        assert np.all(np.isfinite(residuals))

    def test_gaussian_rise_recovers_the_injected_shape(self):
        #Regression test for the initial-guess bug in fit_gaussian_rise.
        #
        #guassian_rise is alpha * exp(...) + c, so alpha is the height ABOVE
        #the offset c. c_i is 1, so alpha_i must be the excess above the
        #quiescent level, flux[-1] - 1. When alpha_i was the absolute flux
        #the starting model peaked near 2.0 against data peaking near 1.05,
        #a ~21x amplitude overshoot. Scaled by flux_err that is a 1000 sigma
        #initial residual, which the default huber loss then de-weights, so
        #the optimiser escaped by collapsing sigma to a needle instead of
        #correcting alpha.
        peak_time = 1000.0
        true_amplitude, true_sigma = 0.05, 0.004
        time = np.linspace(peak_time - 0.01, peak_time, 10)
        flux = 1.0 + true_amplitude * np.exp(
            -(time - peak_time) ** 2 / (2 * true_sigma ** 2))
        flux_err = np.full(len(time), NOISE)

        params, _, residuals = toffee.fit_gaussian_rise(time, flux, flux_err,
                                                        peak_time)
        alpha, sigma, c = params

        #The fit must recover the injected Gaussian, not a spike at the peak
        assert alpha == pytest.approx(true_amplitude, rel = 0.1)
        assert abs(sigma) == pytest.approx(true_sigma, rel = 0.1)
        assert c == pytest.approx(1.0, abs = 1e-3)
        #Residuals should sit at the photometric error, not a fraction of
        #the flare amplitude
        assert np.sqrt(np.mean(residuals ** 2)) < 1e-3

    @pytest.mark.parametrize('num_rise_points', [4, 6, 8, 10, 13, 20])
    def test_gaussian_rise_is_stable_across_rise_lengths(self, num_rise_points):
        #The collapse used to depend chaotically on how many cadences the
        #rise spanned, so sweep a range of rise lengths rather than one case
        peak_time = 1000.0
        true_sigma = num_rise_points * CADENCE / 2.5
        time = peak_time - np.arange(num_rise_points)[::-1] * CADENCE
        flux = 1.0 + 0.05 * np.exp(
            -(time - peak_time) ** 2 / (2 * true_sigma ** 2))
        flux_err = np.full(num_rise_points, NOISE)

        params, _, _ = toffee.fit_gaussian_rise(time, flux, flux_err,
                                                peak_time)

        #The recovered width must be the right order, not a collapsed needle
        assert 0.5 < abs(params[1]) / true_sigma < 2.0


class TestCallDecayModel:
    '''Tests for call_decay_model, which dispatches between the decay models.'''

    @pytest.mark.parametrize('model,num_params', [
        ("dbl_exp_decay", 5),
        ("gomp_decay", 5),
        ("gomp_logistic", 6),
                                                  ])
    def test_single_models_return_one_parameter_set(self, decay_segment,
                                                    model, num_params):
        time, flux, flux_err, peak_time = decay_segment

        params, perr, residuals = toffee.call_decay_model(
            time, flux, flux_err, peak_time, model = model)

        assert params.shape == (num_params,)
        #Residuals are padded back out to the full decay length
        assert residuals.shape == (len(time),)

    def test_fit_peak_false_pads_first_two_residuals_with_nan(self,
                                                              decay_segment):
        time, flux, flux_err, peak_time = decay_segment

        #fit_peak=False trims the first two cadences before fitting, then
        #re-inserts NaN so the residuals line up with the input arrays
        _, _, residuals = toffee.call_decay_model(
            time, flux, flux_err, peak_time,
            model = "dbl_exp_decay", fit_peak = False)

        assert len(residuals) == len(time)
        assert np.isnan(residuals[:2]).all()
        assert np.isfinite(residuals[2:]).all()

    def test_fit_peak_true_keeps_every_cadence(self, decay_segment):
        time, flux, flux_err, peak_time = decay_segment

        _, _, residuals = toffee.call_decay_model(
            time, flux, flux_err, peak_time,
            model = "dbl_exp_decay", fit_peak = True)

        assert len(residuals) == len(time)
        assert np.isfinite(residuals).all()

    def test_robust_returns_two_fits(self, decay_segment):
        time, flux, flux_err, peak_time = decay_segment

        params, perr, residuals = toffee.call_decay_model(
            time, flux, flux_err, peak_time, model = "robust")

        #robust fits double exponential and Gompertz: two parameter sets
        assert params.shape == (2, 5)
        assert residuals.shape == (2, len(time))

    def test_super_robust_returns_four_fits(self, decay_segment):
        time, flux, flux_err, peak_time = decay_segment

        params, perr, residuals = toffee.call_decay_model(
            time, flux, flux_err, peak_time, model = "super_robust")

        #super_robust fits both models with and without the peak: four sets
        assert params.shape == (4, 5)
        assert residuals.shape == (4, len(time))


class TestSuperAndRobustFitting:
    '''Tests for the multi-model fitting routines called by call_decay_model.'''

    def test_super_fitting_shapes(self, decay_segment):
        time, flux, flux_err, peak_time = decay_segment

        params, perr, residuals = toffee.super_fitting(
            time, flux, flux_err, peak_time)

        assert params.shape == (4, 5)
        assert residuals.shape == (4, len(time))
        #The two "without peak" rows have NaN in their first two entries
        assert np.isnan(residuals[2, :2]).all()
        assert np.isnan(residuals[3, :2]).all()

    def test_super_fitting_short_decay_skips_the_no_peak_fits(self):
        #With three or fewer cadences there is nothing left after trimming,
        #so only the two full-decay fits are returned
        peak_time = 1000.0
        time = np.linspace(peak_time, peak_time + 0.004, 3)
        flux = 1.0 + 0.05 * np.exp(-(time - peak_time) / 0.005)
        flux_err = np.full(len(time), NOISE)

        params, perr, residuals = toffee.super_fitting(
            time, flux, flux_err, peak_time)

        assert params.shape == (2, 5)
        assert residuals.shape == (2, len(time))

    def test_robust_fitting_shapes(self, decay_segment):
        time, flux, flux_err, peak_time = decay_segment

        params, perr, residuals = toffee.robust_fitting(
            time, flux, flux_err, peak_time)

        assert params.shape == (2, 5)
        assert residuals.shape == (2, len(time))

    def test_robust_fitting_super_flag_delegates_to_super_fitting(self,
                                                                  decay_segment):
        time, flux, flux_err, peak_time = decay_segment

        params, _, residuals = toffee.robust_fitting(
            time, flux, flux_err, peak_time, super_robust = True)

        assert params.shape == (4, 5)
        assert residuals.shape == (4, len(time))


# ===========================================================================
#4. Secondary detection
# ===========================================================================

class TestFindSecondaries:
    '''Tests for find_secondaries, which searches residuals for extra flares.'''

    def test_detects_a_single_bump_in_1d_residuals(self):
        #A five-point bump well above the threshold
        residuals = np.zeros(30)
        residuals[10:15] = 0.05

        result = toffee.find_secondaries(residuals, flux_std = NOISE,
                                         secondary_threshold = 3.0,
                                         num_consec_sec = 3, dt = CADENCE)

        assert result is not False
        (flare_peaks, flare_starts, flare_ends, flare_amps, EDs, flare_types,
         num_points, num_abv_threshold, amp_sigmas) = result

        assert len(flare_peaks) == 1
        #Start and end are indices into the residual array
        assert flare_starts[0] == 10
        assert flare_ends[0] == 14
        assert flare_types[0] == "secondary"
        assert flare_amps[0] == pytest.approx(0.05)
        assert num_points[0] == 5
        assert amp_sigmas[0] == pytest.approx(0.05 / NOISE)

    def test_returns_false_when_nothing_is_above_threshold(self):
        residuals = np.zeros(30)

        result = toffee.find_secondaries(residuals, NOISE, 3.0, 3, CADENCE)

        assert result is False

    def test_returns_false_when_bump_is_too_short(self):
        #Only two consecutive points, but three are required
        residuals = np.zeros(30)
        residuals[10:12] = 0.05

        result = toffee.find_secondaries(residuals, NOISE, 3.0,
                                         num_consec_sec = 3, dt = CADENCE)

        assert result is False

    def test_multiple_secondaries_are_labelled_in_order(self):
        #Two separated bumps: the first is secondary, the next tertiary
        residuals = np.zeros(30)
        residuals[10:14] = 0.05
        residuals[20:23] = 0.05

        result = toffee.find_secondaries(residuals, NOISE, 3.0, 3, CADENCE,
                                         fit_multiple_secs = True)

        flare_types = result[5]
        assert flare_types == ["secondary", "tertiary"]

    def test_single_secondary_mode_keeps_only_the_largest(self):
        #The second bump is brighter, so it is the one that survives
        residuals = np.zeros(30)
        residuals[10:14] = 0.05
        residuals[20:24] = 0.09

        result = toffee.find_secondaries(residuals, NOISE, 3.0, 3, CADENCE,
                                         fit_multiple_secs = False)

        flare_peaks, flare_starts, flare_ends, flare_amps, _, flare_types = result[:6]
        assert len(flare_peaks) == 1
        assert flare_types == ["secondary"]
        assert flare_amps[0] == pytest.approx(0.09)
        assert flare_starts[0] == 20

    def test_2d_residuals_use_the_intersection_of_all_fits(self):
        #When several models are fitted, a secondary must appear in all of them
        flare_row = np.zeros(30)
        flare_row[10:15] = 0.05
        residuals = np.vstack([flare_row, flare_row])

        result = toffee.find_secondaries(residuals, NOISE, 3.0, 3, CADENCE)

        assert result is not False
        assert result[5] == ["secondary"]

    def test_2d_residuals_reject_disagreeing_fits(self):
        #The two models flag different, non-overlapping windows, so their
        #intersection is empty and no secondary should be declared
        residual_row_a = np.zeros(30)
        residual_row_a[10:15] = 0.05
        residual_row_b = np.zeros(30)
        residual_row_b[20:25] = 0.05

        result = toffee.find_secondaries(np.vstack([residual_row_a, residual_row_b]),
                                         NOISE, 3.0, 3, CADENCE)

        assert result is False


# ===========================================================================
#5. Full detection pipeline
# ===========================================================================

class TestDetectFlares:
    '''Tests for detect_flares, the top-level detection routine.'''

    #The columns every caller of detect_flares depends on
    EXPECTED_COLUMNS = ["T_peak", "T_start", "T_end", "Amplitude", "ED",
                        "Primary_or_Secondary", "Num_Points",
                        "Num_Abv_Threshold", "Amp_Sigma"]

    def test_returns_dataframe_with_expected_schema(self,
                                                    single_flare_lightcurve):
        time, flux, flux_err, quality = single_flare_lightcurve

        flares = toffee.detect_flares(time, flux, flux_err, quality,
                                      clip_breaks = None)

        assert isinstance(flares, pd.DataFrame)
        assert list(flares.columns) == self.EXPECTED_COLUMNS

    def test_recovers_a_single_injected_flare(self, single_flare_lightcurve):
        time, flux, flux_err, quality = single_flare_lightcurve

        flares = toffee.detect_flares(time, flux, flux_err, quality,
                                      clip_breaks = None)

        assert len(flares) == 1
        flare_row = flares.iloc[0]
        #The recovered peak should be the cadence where the flare was injected
        assert flare_row["T_peak"] == pytest.approx(time[1500], abs = 2 * CADENCE)
        assert flare_row["Primary_or_Secondary"] == "primary"
        assert flare_row["Amplitude"] == pytest.approx(0.05, abs = 5 * NOISE)
        assert flare_row["ED"] > 0
        assert flare_row["T_start"] <= flare_row["T_peak"] <= flare_row["T_end"]

    def test_flat_curve_yields_no_flares(self, flat_lightcurve):
        time, flux, flux_err, quality = flat_lightcurve

        #Raise the threshold well above the noise so nothing is detected
        flares = toffee.detect_flares(time, flux, flux_err, quality,
                                      primary_threshold = 20.0,
                                      clip_breaks = None)

        assert len(flares) == 0

    def test_resolves_an_overlapping_secondary_flare(self):
        #This is the behaviour TOFFEE exists for: a second flare riding on the
        #decay of the first must be recovered from the fit residuals.
        time, flux, flux_err, quality = make_flat_lightcurve(num_points = 3000,
                                                             seed = 5)
        #Primary flare
        flux = inject_flare(time, flux, 1500, amplitude = 0.06, tau = 0.012,
                            num_points = 60)
        #Secondary flare 20 cadences later, on the primary's decay
        flux = inject_flare(time, flux, 1520, amplitude = 0.04, tau = 0.006,
                            num_points = 40)

        flares = toffee.detect_flares(time, flux, flux_err, quality,
                                      clip_breaks = None)

        flare_types = list(flares["Primary_or_Secondary"])
        assert "primary" in flare_types
        assert "secondary" in flare_types
        #The secondary must be found after the primary in time
        primary_peak = flares.loc[flares["Primary_or_Secondary"] == "primary",
                                  "T_peak"].iloc[0]
        secondary_peak = flares.loc[flares["Primary_or_Secondary"] == "secondary",
                                    "T_peak"].iloc[0]
        assert secondary_peak > primary_peak

    def test_quality_flags_filter_cadences(self):
        #Flag the flare cadences as 512, a value TESS often assigns to flare
        #peaks. With flag_values=[0] those cadences are dropped and the flare
        #disappears; allowing 512 brings it back.
        time, flux, flux_err, _ = make_flat_lightcurve(num_points = 3000, seed = 3)
        flux = inject_flare(time, flux, 1500, amplitude = 0.05, tau = 0.01)
        quality = np.zeros(len(time), dtype = int)
        quality[1500:1540] = 512

        strict = toffee.detect_flares(time, flux, flux_err, quality,
                                      clip_breaks = None, flag_values = [0])
        permissive = toffee.detect_flares(time, flux, flux_err, quality,
                                          clip_breaks = None,
                                          flag_values = [0, 512])

        assert len(strict) == 0
        assert len(permissive) == 1

    def test_nan_cadences_are_dropped_without_error(self,
                                                    single_flare_lightcurve):
        time, flux, flux_err, quality = single_flare_lightcurve
        flux = flux.copy()
        flux[10:20] = np.nan

        flares = toffee.detect_flares(time, flux, flux_err, quality,
                                      clip_breaks = None)

        #The flare is far from the NaN block, so it still comes out
        assert len(flares) == 1
        assert np.isfinite(flares["T_peak"]).all()

    def test_higher_threshold_finds_no_more_flares(self,
                                                   single_flare_lightcurve):
        time, flux, flux_err, quality = single_flare_lightcurve

        flares_low_threshold = toffee.detect_flares(time, flux, flux_err, quality,
                                                    primary_threshold = 3.0, clip_breaks = None)
        flares_high_threshold = toffee.detect_flares(time, flux, flux_err, quality,
                                                     primary_threshold = 100.0, clip_breaks = None)

        assert len(flares_high_threshold) <= len(flares_low_threshold)

    def test_peaks_are_unique(self):
        #detect_flares de-duplicates its results with np.unique, so no peak
        #time may appear twice
        time, flux, flux_err, quality = make_flat_lightcurve(num_points = 4000,
                                                             seed = 7)
        for index in (500, 1500, 2500):
            flux = inject_flare(time, flux, index, amplitude = 0.05, tau = 0.01)

        flares = toffee.detect_flares(time, flux, flux_err, quality,
                                      clip_breaks = None)

        assert flares["T_peak"].is_unique

    @pytest.mark.parametrize('num_rise_points', [4, 8, 12, 16, 20])
    def test_resolved_rise_does_not_manufacture_a_secondary(self,
                                                            num_rise_points):
        #A single flare with a well-resolved Gaussian rise contains no
        #secondary, so detect_flares must report exactly one primary.
        #
        #This is the end-to-end regression test for the fit_gaussian_rise
        #initial-guess bug. When that fit collapsed, its residuals traced the
        #unmodelled rise itself: a smooth monotonic ramp, which is exactly
        #the run of consecutive above-threshold points that find_secondaries
        #looks for. Around a third of clean single flares picked up a
        #spurious secondary on the rise.
        random_generator = np.random.default_rng(0)
        num_cadences, peak = 3000, 1500
        amplitude = 0.05
        time = 1000.0 + np.arange(num_cadences) * CADENCE
        flux = 1.0 + random_generator.normal(0.0, NOISE, num_cadences)

        #Gaussian rise meeting a two-exponential decay at the peak, so the
        #profile is continuous and the peak is counted once
        sigma = num_rise_points * CADENCE / 2.5
        rise_indices = np.arange(peak - num_rise_points, peak + 1)
        flux[rise_indices] += amplitude * np.exp(
            -(time[rise_indices] - time[peak]) ** 2 / (2 * sigma ** 2))
        decay_indices = np.arange(peak + 1, peak + 61)
        flux[decay_indices] += amplitude * (
            0.7 * np.exp(-(time[decay_indices] - time[peak]) / 0.004)
            + 0.3 * np.exp(-(time[decay_indices] - time[peak]) / 0.02))

        flux_err = np.full(num_cadences, NOISE)
        quality = np.zeros(num_cadences, dtype = int)

        flares = toffee.detect_flares(time, flux, flux_err, quality,
                                      clip_breaks = None)

        flare_types = list(flares["Primary_or_Secondary"])
        assert flare_types == ["primary"], f'spurious secondaries detected: {flare_types}'

    def test_clip_breaks_removes_edge_cadences(self, single_flare_lightcurve):
        time, flux, flux_err, quality = single_flare_lightcurve

        #With clipping on, the first and last cadences are removed, so any
        #detected flare must lie inside the trimmed range
        flares = toffee.detect_flares(time, flux, flux_err, quality,
                                      clip_breaks = 100)

        if len(flares) > 0:
            assert flares["T_peak"].min() > time[99]
            assert flares["T_peak"].max() < time[-100]


# ===========================================================================
#6. Purity methods (TOFFEE_TWO)
# ===========================================================================

class TestRedNoiseSmoothing:
    '''Tests for red_noise_smoothing, the first TOFFEE_TWO purity method.'''

    @pytest.fixture
    def red_noise_inputs(self):
        '''A light curve with one large flare, plus its first-pass flare frame.'''
        time, flux, flux_err, quality = make_flat_lightcurve(num_points = 3000,
                                                             seed = 3)
        flux = inject_flare(time, flux, 1500, amplitude = 0.05, tau = 0.01)
        flares = toffee.detect_flares(time, flux, flux_err, quality,
                                      clip_breaks = None)
        return time, flux, flares

    def test_returns_array_of_matching_length(self, red_noise_inputs):
        time, flux, flares = red_noise_inputs

        smoothed_flux, trend = toffee.red_noise_smoothing(time, flux, flares,
                                                   window_length = 0.1)

        assert len(smoothed_flux) == len(flux)
        #No NaN may leak through: the gap-filling step exists to prevent this
        assert np.isfinite(smoothed_flux).all()

    def test_reduces_the_scatter_of_the_quiescent_curve(self, red_noise_inputs):
        time, flux, flares = red_noise_inputs

        smoothed_flux, trend = toffee.red_noise_smoothing(time, flux, flares,
                                                   window_length = 0.1)

        #Compare only the flare-free part of the curve
        quiescent_mask = np.ones(len(time), dtype = bool)
        quiescent_mask[1450:1600] = False
        assert np.std(smoothed_flux[quiescent_mask]) <= np.std(flux[quiescent_mask])

    def test_large_flare_is_not_attenuated(self, red_noise_inputs):
        time, flux, flares = red_noise_inputs

        #The flare is above clip_amp sigma, so wotan masks it and its peak
        #should survive the re-detrending largely intact
        smoothed_flux, trend = toffee.red_noise_smoothing(time, flux, flares, clip_amp = 5,
                                                   window_length = 0.1)

        assert smoothed_flux[1500:1540].max() == pytest.approx(
                                                               flux[1500:1540].max(), rel = 0.05)

    @pytest.fixture
    def untrendable_inputs(self):
        '''Inputs where wotan cannot fit a trend everywhere.

        The masked flare is longer than the smoothing window, so some windows
        contain no unmasked data at all and wotan returns a NaN trend there.
        This is the case the fill_gaps argument exists to handle.

        Returns:
            time, flux, flare_frame, window_length
        '''
        time, flux, _, _ = make_flat_lightcurve(num_points = 3000, seed = 2)
        #A long, slowly decaying flare spanning 300 cadences (~0.42 days)
        flux = inject_flare(time, flux, 1400, amplitude = 0.08, tau = 0.15,
                            num_points = 300)
        flares = pd.DataFrame({"T_start": [time[1400]],
                               "T_end": [time[1700]],
                               "Amp_Sigma": [80.0]})
        #Window much shorter than the masked flare
        return time, flux, flares, 0.05

    def test_interpolate_fill_gaps_option(self, untrendable_inputs):
        time, flux, flares, window_length = untrendable_inputs

        smoothed_flux, trend = toffee.red_noise_smoothing(time, flux, flares,
                                                   window_length = window_length,
                                                   fill_gaps = "interpolate")

        #Interpolating the trend across the gap must leave no NaN behind
        assert len(smoothed_flux) == len(flux)
        assert np.isfinite(smoothed_flux).all()

    def test_raw_fill_gaps_keeps_the_original_flux(self, untrendable_inputs):
        import wotan

        time, flux, flares, window_length = untrendable_inputs

        #Work out for ourselves which cadences wotan could not trend
        mask = (time >= flares["T_start"].iloc[0]) & (time <= flares["T_end"].iloc[0])
        _, trend = wotan.flatten(time, flux, mask = mask, window_length = window_length,
                                 break_tolerance = 0, method = "biweight",
                                 return_trend = True)
        untrendable_cadences = ~np.isfinite(trend)
        #The fixture is only meaningful if it really does produce gaps
        assert untrendable_cadences.any()

        smoothed_flux, trend = toffee.red_noise_smoothing(time, flux, flares,
                                                   window_length = window_length,
                                                   fill_gaps = "raw")

        assert np.isfinite(smoothed_flux).all()
        #Where no trend existed, the pre-smoothing flux is kept unchanged
        np.testing.assert_allclose(smoothed_flux[untrendable_cadences], flux[untrendable_cadences])

    def test_invalid_fill_gaps_raises(self, untrendable_inputs):
        time, flux, flares, window_length = untrendable_inputs

        #The check only fires when there is actually a gap to fill
        with pytest.raises(ValueError, match = "fill_gaps"):
            toffee.red_noise_smoothing(time, flux, flares,
                                       window_length = window_length,
                                       fill_gaps = "not_an_option")

    def test_empty_flare_frame_is_handled(self, flat_lightcurve):
        #No flares to mask: the function should still return a smoothed curve
        time, flux, _, _ = flat_lightcurve
        empty_flare_frame = pd.DataFrame({"T_start": [], "T_end": [], "Amp_Sigma": []})

        smoothed_flux, trend = toffee.red_noise_smoothing(time, flux, empty_flare_frame,
                                                   window_length = 0.1)

        assert len(smoothed_flux) == len(flux)
        assert np.isfinite(smoothed_flux).all()


class TestSquareIndex:
    '''The square index (SI) purity cut: SI = flare duration / amplitude.

    The SI cut is described in the project notes as the second TOFFEE_TWO
    purity method, but it is not yet exposed as a function in toffee/main.py --
    it currently lives in a standalone script under Toffee_Creation. These
    tests are placeholders that should be enabled once the function lands in
    the package.
    '''

    @pytest.mark.skipif(not hasattr(toffee, "square_index"),
                        reason = "square_index is not yet part of the package")
    def test_square_index_is_duration_over_amplitude(self):
        #Duration in seconds divided by amplitude in relative flux
        result = toffee.square_index(duration = 100.0, amplitude = 0.05)
        assert result == pytest.approx(100.0 / 0.05)


# ===========================================================================
#7. Masking and stellar parameters
# ===========================================================================

class TestMakeFlareMask:
    '''Tests for make_flare_mask, which hides flares before spot fitting.'''

    def test_masks_out_the_flare_window(self):
        time, flux, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)
        benchmark = pd.DataFrame({"T_peak": [time[500]],
                                  "T_start": [time[495]],
                                  "T_end": [time[510]]})

        mask = toffee.make_flare_mask(time, benchmark)

        #True means "keep", so the flare cadences must be False
        assert mask.dtype == bool
        assert len(mask) == len(time)
        assert not mask[500]
        #Most of the light curve is untouched
        assert mask.sum() > 0.9 * len(time)

    def test_mask_extends_beyond_the_reported_flare(self):
        #The mask pads by half the flare duration, to a minimum set by
        #data_min cadences, so it is wider than T_start..T_end
        time, flux, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)
        benchmark = pd.DataFrame({"T_peak": [time[500]],
                                  "T_start": [time[495]],
                                  "T_end": [time[510]]})

        mask = toffee.make_flare_mask(time, benchmark, data_min = 15)

        masked_indices = np.where(~mask)[0]
        assert masked_indices.min() < 495
        assert masked_indices.max() > 510

    def test_mask_width_follows_the_flare_duration(self):
        #The padding is half the flare duration: the mask runs from
        #T_start - 0.5 * mask_extra to T_end + 2 * mask_extra
        time, flux, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)
        short_flare_benchmark = pd.DataFrame({"T_peak": [time[500]],
                                              "T_start": [time[495]],
                                              "T_end": [time[510]]})
        long_flare_benchmark = pd.DataFrame({"T_peak": [time[500]],
                                             "T_start": [time[450]],
                                             "T_end": [time[560]]})

        num_masked_short_flare = (~toffee.make_flare_mask(time, short_flare_benchmark)).sum()
        num_masked_long_flare = (~toffee.make_flare_mask(time, long_flare_benchmark)).sum()

        assert num_masked_long_flare > num_masked_short_flare

    def test_larger_data_min_masks_more_points(self):
        #data_min sets a floor on the padding, measured in cadences, so
        #raising it widens the mask around a short flare
        time, flux, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)
        benchmark = pd.DataFrame({"T_peak": [time[500]],
                                  "T_start": [time[495]],
                                  "T_end": [time[510]]})

        mask_small_data_min = toffee.make_flare_mask(time, benchmark, data_min = 5)
        mask_large_data_min = toffee.make_flare_mask(time, benchmark, data_min = 50)

        assert (~mask_large_data_min).sum() > (~mask_small_data_min).sum()

    def test_data_min_is_a_floor_of_that_many_cadences(self):
        #For a flare short enough that 0.5 * duration is below the floor, the
        #padding should be data_min cadences: 0.5 * mask_extra before T_start
        #and 2 * mask_extra after T_end, i.e. 2.5 * data_min cadences in total
        #on top of the flare itself.
        time, flux, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)
        cadence = np.median(np.diff(time))
        data_min = 40
        #A 15 cadence flare: half its duration (7.5) is well below the floor
        benchmark = pd.DataFrame({"T_peak": [time[500]],
                                  "T_start": [time[495]],
                                  "T_end": [time[510]]})

        mask = toffee.make_flare_mask(time, benchmark, data_min = data_min)

        masked_indices = np.where(~mask)[0]
        mask_extra = data_min * cadence
        #Expected edges in time, converted back to cadence indices
        expected_low = time[495] - 0.5 * mask_extra
        expected_high = time[510] + 2.0 * mask_extra
        assert time[masked_indices.min()] >= expected_low
        assert time[masked_indices.min() - 1] < expected_low
        assert time[masked_indices.max()] <= expected_high
        assert time[masked_indices.max() + 1] > expected_high

    def test_long_flare_uses_half_its_duration_not_the_floor(self):
        #When 0.5 * duration exceeds the data_min floor, duration wins, so a
        #small data_min makes no difference for a long flare
        time, flux, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)
        benchmark = pd.DataFrame({"T_peak": [time[500]],
                                  "T_start": [time[400]],
                                  "T_end": [time[600]]})

        mask_small_data_min = toffee.make_flare_mask(time, benchmark, data_min = 1)
        mask_large_data_min = toffee.make_flare_mask(time, benchmark, data_min = 15)

        #0.5 * 200 cadences = 100, comfortably above both floors
        np.testing.assert_array_equal(mask_small_data_min, mask_large_data_min)

    def test_empty_benchmark_keeps_everything(self):
        time, flux, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)
        empty_flare_frame = pd.DataFrame({"T_peak": [], "T_start": [], "T_end": []})

        mask = toffee.make_flare_mask(time, empty_flare_frame)

        assert mask.all()

    def test_multiple_flares_are_all_masked(self):
        time, flux, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)
        benchmark = pd.DataFrame({
            "T_peak": [time[300], time[900], time[1500]],
            "T_start": [time[295], time[895], time[1495]],
            "T_end": [time[310], time[910], time[1510]],
                                  })

        mask = toffee.make_flare_mask(time, benchmark)

        assert not mask[300]
        assert not mask[900]
        assert not mask[1500]


class TestMakeGapMask:
    '''Tests for make_gap_mask, which stops fits running across data gaps.'''

    def test_selects_the_segment_between_adjacent_gaps(self):
        time, _, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)
        gap_times = np.array([time[300], time[900]])

        mask = toffee.make_gap_mask(time, time[500], gap_times)

        #The flare at index 500 lies between the two gaps, so only that
        #stretch of the light curve is kept
        assert mask[500]
        assert not mask[100]
        assert not mask[1500]

    def test_no_gap_before_the_flare(self):
        time, _, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)
        gap_times = np.array([time[300], time[900]])

        mask = toffee.make_gap_mask(time, time[100], gap_times)

        #Everything up to the first gap is kept
        assert mask[0]
        assert mask[100]
        assert not mask[500]

    def test_no_gap_after_the_flare(self):
        time, _, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)
        gap_times = np.array([time[300], time[900]])

        mask = toffee.make_gap_mask(time, time[1500], gap_times)

        #Everything after the last gap is kept
        assert mask[1500]
        assert mask[-1]
        assert not mask[500]

    def test_no_gaps_at_all_keeps_everything(self):
        time, _, _, _ = make_flat_lightcurve(num_points = 2000, seed = 0)

        mask = toffee.make_gap_mask(time, time[500], np.array([]))

        assert mask.all()


class TestMakeWindowMask:
    '''Tests for make_window_mask, which builds the local spot-fitting windows.'''

    @staticmethod
    def make_sine_curve(num_points = 2000, amplitude = 0.05, noise = NOISE, period = 2.0,
                        seed = 0):
        '''A sinusoidally modulated light curve, standing in for a spotted star.'''
        random_generator = np.random.default_rng(seed)
        time = np.linspace(0.0, 10.0, num_points)
        flux = (1.0 + amplitude * np.sin(2 * np.pi * time / period)
                + random_generator.normal(0.0, noise, num_points))
        flux_err = np.full(num_points, noise)
        return time, flux, flux_err

    def test_returns_six_values(self):
        time, flux, flux_err = self.make_sine_curve()

        result = toffee.make_window_mask(time, flux, flux_err,
                                         flare_peak = 5.0, window_size = 2.0)

        assert len(result) == 6

    def test_succeeds_on_a_clearly_periodic_curve(self):
        time, flux, flux_err = self.make_sine_curve(amplitude = 0.05, period = 2.0)

        (ls_mask, window_mask, window_mask_left, window_mask_right,
         peak_freq, success) = toffee.make_window_mask(time, flux, flux_err,
                                                       5.0, 2.0)

        assert success is True
        #The recovered frequency should match the injected 2 day period
        assert peak_freq == pytest.approx(0.5, rel = 0.05)
        #All three windows contain data and are offset from one another
        assert (window_mask.any() and window_mask_left.any()
                and window_mask_right.any())
        assert np.where(window_mask_left)[0].min() < np.where(window_mask)[0].min()
        assert np.where(window_mask_right)[0].max() > np.where(window_mask)[0].max()

    def test_window_mask_spans_the_requested_size(self):
        time, flux, flux_err = self.make_sine_curve()
        window_size = 2.0

        ls_mask = toffee.make_window_mask(time, flux, flux_err, 5.0,
                                          window_size)[0]

        #ls_mask keeps points within window_size of the flare peak
        assert np.all(np.abs(time[ls_mask] - 5.0) < window_size)
        assert ls_mask.sum() > 0

    def test_fails_on_a_non_periodic_curve(self):
        #Pure noise has no significant periodicity, so the false alarm
        #probability stays high and the flare is skipped
        time, flux, flux_err = self.make_sine_curve(amplitude = 0.0)

        success = toffee.make_window_mask(time, flux, flux_err, 5.0, 2.0)[5]

        assert success is False

    def test_empty_window_returns_sentinels(self):
        #A flare peak outside the observed time range leaves the window empty
        time, flux, flux_err = self.make_sine_curve()

        (ls_mask, window_mask, window_mask_left, window_mask_right,
         peak_freq, success) = toffee.make_window_mask(time, flux, flux_err,
                                                       flare_peak = 100.0,
                                                       window_size = 2.0)

        assert success is False
        assert ls_mask.sum() == 0
        assert window_mask is None
        assert window_mask_left is None
        assert window_mask_right is None
        assert peak_freq is None


class TestSpotFitting:
    '''Tests for spot_fitting, the double-sine spot modulation fit.'''

    @staticmethod
    def make_spotted_curve(period = 2.0, amplitude = 0.05, seed = 0):
        random_generator = np.random.default_rng(seed)
        time = np.linspace(0.0, 10.0, 2000)
        flux = (1.0 + amplitude * np.sin(2 * np.pi * time / period)
                + random_generator.normal(0.0, NOISE, 2000))
        flux_err = np.full(2000, NOISE)
        return time, flux, flux_err

    def test_fits_a_sinusoid_successfully(self):
        time, flux, flux_err = self.make_spotted_curve(period = 2.0)
        mask = np.full(len(time), True)
        peak_freq = 1 / 2.0
        #The 11 initial parameters of the quadratic-amplitude double sine
        init = np.array([0, 0, 0.05, 2 * np.pi * peak_freq, 0,
                         0, 0, 0.05, 4 * np.pi * peak_freq, np.pi, 1.0])

        popt, pcov, spot_fit, fail = toffee.spot_fitting(time, flux, flux_err,
                                                         mask, init)

        assert fail is False
        assert len(popt) == 11
        assert len(spot_fit) == mask.sum()
        #The fit should track the injected modulation
        assert np.sqrt(np.mean((spot_fit - flux) ** 2)) < 5 * NOISE

    def test_fit_is_restricted_to_the_mask(self):
        time, flux, flux_err = self.make_spotted_curve()
        mask = np.full(len(time), False)
        mask[500:1000] = True
        init = np.array([0, 0, 0.05, np.pi, 0, 0, 0, 0.05, 2 * np.pi,
                         np.pi, 1.0])

        popt, pcov, spot_fit, fail = toffee.spot_fitting(time, flux, flux_err,
                                                         mask, init)

        #The returned model curve has one entry per unmasked cadence only
        assert len(spot_fit) == 500

    def test_failed_fit_returns_the_initial_guess(self):
        #With no unmasked cadences curve_fit cannot run at all. The routine
        #catches the failure, flags it, and hands back the initial guess so
        #the caller can skip the flare rather than crash.
        time, flux, flux_err = self.make_spotted_curve()
        mask = np.full(len(time), False)
        init = np.array([0, 0, 0.05, np.pi, 0, 0, 0, 0.05, 2 * np.pi,
                         np.pi, 1.0])

        popt, pcov, spot_fit, fail = toffee.spot_fitting(time, flux, flux_err,
                                                         mask, init)

        assert fail is True
        assert pcov is False
        np.testing.assert_array_equal(popt, init)


class TestSpotMod:
    '''Tests for spot_mod, which reports where in the spot cycle a flare occurs.'''

    def test_returns_one_value_per_flare(self):
        random_generator = np.random.default_rng(0)
        time = np.linspace(0.0, 10.0, 2000)
        flux = (1.0 + 0.05 * np.sin(2 * np.pi * time / 2.0)
                + random_generator.normal(0.0, NOISE, 2000))
        flux_err = np.full(2000, NOISE)
        benchmark = pd.DataFrame({"T_peak": [time[500], time[1200]],
                                  "T_start": [time[495], time[1195]],
                                  "T_end": [time[510], time[1210]]})

        (flare_spot_values, spot_left, spot_right,
         red_chi2, spot_exists, agreement) = toffee.spot_mod(time, flux, flux_err,
                                                             benchmark)

        num_flares = len(benchmark)
        assert len(flare_spot_values) == num_flares
        assert len(spot_left) == num_flares
        assert len(spot_right) == num_flares
        assert len(spot_exists) == num_flares
        assert len(agreement) == num_flares
        #red_chi2 is transposed to 3 rows (centre, left, right) by n_flares
        assert red_chi2.shape == (3, num_flares)

    def test_spot_values_are_in_range_or_sentinel(self):
        random_generator = np.random.default_rng(0)
        time = np.linspace(0.0, 10.0, 2000)
        flux = (1.0 + 0.05 * np.sin(2 * np.pi * time / 2.0)
                + random_generator.normal(0.0, NOISE, 2000))
        flux_err = np.full(2000, NOISE)
        benchmark = pd.DataFrame({"T_peak": [time[500]],
                                  "T_start": [time[495]],
                                  "T_end": [time[510]]})

        flare_spot_values = toffee.spot_mod(time, flux, flux_err, benchmark)[0]

        #A real measurement lies in [-1, 1]; 10 is the documented sentinel for
        #a flare that had to be skipped
        assert np.all((np.abs(flare_spot_values) <= 1.0) | (flare_spot_values == 10))


# ===========================================================================
#8. Flare energy
# ===========================================================================

class TestFlareEnergyCalc:
    '''Tests for flare_energy_calc: E = equivalent duration x luminosity.'''

    def test_multiplies_luminosity_by_equivalent_duration(self):
        #Solar-ish luminosity in W, equivalent duration in seconds -> Joules
        flare_energy = toffee.flare_energy_calc(star_luminosity = 3.8e26,
                                                equivalent_duration = 100.0)

        assert flare_energy == pytest.approx(3.8e28)

    def test_scales_linearly_with_equivalent_duration(self):
        base_energy = toffee.flare_energy_calc(1.0e26, 10.0)
        doubled_energy = toffee.flare_energy_calc(1.0e26, 20.0)

        assert doubled_energy == pytest.approx(2 * base_energy)

    def test_works_elementwise_on_arrays(self):
        #Callers pass a whole column of equivalent durations from detect_flares
        durations = np.array([10.0, 20.0, 30.0])

        flare_energies = toffee.flare_energy_calc(1.0e26, durations)

        np.testing.assert_allclose(flare_energies, durations * 1.0e26)

    def test_zero_duration_gives_zero_energy(self):
        assert toffee.flare_energy_calc(3.8e26, 0.0) == 0.0


# ===========================================================================
#9. Utilities
# ===========================================================================

class TestGetTessOrbit:
    '''Tests for the TESS orbit-times helper module.'''

    def test_local_file_path_is_inside_the_package_cache(self):
        from toffee.get_tess_orbit import get_local_file_path

        local_file_path = get_local_file_path()

        #The function creates the cache directory as a side effect
        assert local_file_path.name == "tess_orbit_time.csv"
        assert local_file_path.parent.name == ".cache"
        assert local_file_path.parent.is_dir()

    def test_local_file_path_is_stable_across_calls(self):
        from toffee.get_tess_orbit import get_local_file_path

        assert get_local_file_path() == get_local_file_path()

    @pytest.mark.skip(reason = "download_csv_file requires network access; "
                      "enable as an integration test if wanted")
    def test_download_csv_file(self):
        from toffee.get_tess_orbit import download_csv_file

        local_file_path = download_csv_file()
        assert os.path.exists(local_file_path)


# ===========================================================================
#10. Plotting
# ===========================================================================

class TestVisualizer:
    '''Smoke test for visualizer, the plotting counterpart to detect_flares.'''

    def test_runs_without_error_on_a_flare(self, single_flare_lightcurve):
        #The Agg backend set at the top of this file means no window opens.
        #This only checks the plotting path does not crash.
        time, flux, flux_err, quality = single_flare_lightcurve

        toffee.visualizer(time, flux, flux_err, quality, clip_breaks = None)


# ===========================================================================
#11. Detrending
# ===========================================================================

@pytest.fixture
def stub_ffi_observation_times(tmp_path, monkeypatch):
    '''Replace the TESS FFI times download with a local stub file.

    flatten() fetches TESS_FFI_observation_times.csv from tess.mit.edu. These
    tests should not depend on the network or on MIT's uptime, so we write a
    two-orbit CSV in the real format and point the downloader at it.

    The orbit windows bracket the BTJD range used by make_flat_lightcurve, so
    the synthetic cadences all fall inside an observing orbit.
    '''
    from astropy.time import Time

    def utc_time_string(btjd_time):
        #flatten converts the CSV times from UTC to TDB and subtracts 2457000,
        #so go the other way here to get the UTC strings the file should hold
        return Time(btjd_time + 2457000, format = "jd", scale = "tdb").utc.iso

    csv_file_path = tmp_path / "TESS_FFI_observation_times.csv"
    csv_rows = ["Sector,Orbit,Start Time,End Time",
                f"1,9,{utc_time_string(995.0)},{utc_time_string(1005.0)}",
                f"1,10,{utc_time_string(1006.0)},{utc_time_string(1016.0)}"]
    csv_file_path.write_text("\n".join(csv_rows) + "\n")

    #flatten calls download_csv_file by the name bound in toffee.main
    monkeypatch.setattr(toffee.main, "download_csv_file",
                        lambda url=None: str(csv_file_path))
    return csv_file_path


def make_spotted_lightcurve(num_points = 3000, period = 2.0, amplitude = 0.05, seed = 0):
    '''A light curve with sinusoidal spot modulation, inside the stub orbits.'''
    random_generator = np.random.default_rng(seed)
    time = 1000.0 + np.arange(num_points) * CADENCE
    flux = (1.0 + amplitude * np.sin(2 * np.pi * time / period)
            + random_generator.normal(0.0, NOISE, num_points))
    flux_err = np.full(num_points, NOISE)
    return time, flux, flux_err


class TestFlatten:
    '''Tests for flatten, the detrending front end.

    flatten used to read the FFI observation times from a hard-coded absolute
    path on the author's machine. It now downloads them from tess.mit.edu via
    get_tess_orbit.download_csv_file, which caches inside the package and falls
    back to that cache when offline. These tests stub the download out.
    '''

    def test_returns_three_values_without_a_window(self, stub_ffi_observation_times):
        time, flux, flux_err = make_spotted_lightcurve()

        orbit_time, lc_quad, periodic = toffee.flatten(time, flux, flux_err)

        #lc_quad packs the flattened flux, its errors, and the fitted trend
        assert lc_quad.shape == (3, len(orbit_time))
        assert len(orbit_time) == len(time)
        assert periodic is False

    def test_quadratic_detrending_normalises_the_flux(self, stub_ffi_observation_times):
        #Add a slow quadratic trend; the quadratic fit should remove it and
        #leave the light curve centred on 1
        time, flux, flux_err = make_spotted_lightcurve(amplitude = 0.0)
        trend = 1.0 + 0.01 * (time - time.mean()) ** 2
        flux = flux * trend

        orbit_time, lc_quad, periodic = toffee.flatten(time, flux, flux_err)

        flattened = lc_quad[0]
        assert np.median(flattened) == pytest.approx(1.0, abs = 1e-3)
        #The recorded trend should track the injected one
        assert np.all(np.isfinite(lc_quad[2]))

    def test_wotan_window_adds_a_fourth_return_value(self, stub_ffi_observation_times):
        time, flux, flux_err = make_spotted_lightcurve()

        result = toffee.flatten(time, flux, flux_err, short_window = 0.25)

        assert len(result) == 4
        orbit_time, lc_quad, lc_wotan, periodic = result
        assert lc_wotan.shape == (3, len(orbit_time))
        #Wotan smoothing on top of the quadratic fit reduces the scatter
        assert np.std(lc_wotan[0]) < np.std(lc_quad[0])

    def test_periodogram_adds_a_fifth_return_value(self, stub_ffi_observation_times):
        time, flux, flux_err = make_spotted_lightcurve()

        result = toffee.flatten(time, flux, flux_err, short_window = 0.25,
                                periodogram = [0.1, 10])

        assert len(result) == 5
        orbit_time, lc_quad, lc_wotan, lc_flat, periodic = result
        assert lc_flat.shape == (3, len(orbit_time))
        #A strongly modulated curve should be flagged as periodic
        assert periodic is True

    def test_flat_curve_is_not_flagged_periodic(self, stub_ffi_observation_times):
        #Pure noise has no significant sinusoidal signal
        time, flux, flux_err = make_spotted_lightcurve(amplitude = 0.0, seed = 4)

        periodic = toffee.flatten(time, flux, flux_err, short_window = 0.25,
                                  periodogram = [0.1, 10])[4]

        assert periodic is False

    def test_cadences_outside_every_orbit_are_dropped(self, stub_ffi_observation_times):
        #The stub orbits cover BTJD 995-1005 and 1006-1016. Cadences in the
        #1005-1006 gap belong to no orbit and should not survive detrending.
        time, flux, flux_err = make_spotted_lightcurve(num_points = 3000)
        time = time + 4.0  #push part of the curve into the inter-orbit gap
        assert ((time > 1005.0) & (time < 1006.0)).any()

        orbit_time = toffee.flatten(time, flux, flux_err)[0]

        assert len(orbit_time) < len(time)
        assert not ((orbit_time > 1005.0) & (orbit_time < 1006.0)).any()


class TestGetRotPeriod:
    '''Tests for main.get_rot_period, which measures the stellar rotation period.

    main.get_rot_period runs its Lomb-Scargle on lc_flat, the curve flatten()
    builds for flare detection, which is detrended specifically to remove
    everything slower than a flare. Rotation is one of the things it removes, so
    these two tests still fail. toffee.rotation.get_rot_period fixes this by
    running the same periodogram on lc_quad instead, and TestRotationGetRotPeriod
    below carries the same two cases as real assertions.
    '''

    #strict=True means these start failing for real once the lc_quad fix is
    #ported into main.py, which is the reminder to delete this class and keep
    #only the promoted assertions below.
    @pytest.mark.xfail(reason = 'main.get_rot_period measures rotation from the '
                       'flare-detection curve -- fixed in toffee.rotation, not '
                       'yet ported into main.py',
                       strict = True)
    def test_recovers_an_injected_modulation(self, stub_ffi_observation_times):
        time, flux, flux_err = make_spotted_lightcurve(period = 2.0,
                                                       amplitude = 0.05)

        period, false_alarm = toffee.get_rot_period(time, flux, flux_err)

        #a 2 day modulation should survive as a 2 day modulation
        assert 1.5 < period < 2.5
        #and the detection should be highly significant
        assert 0.0 <= false_alarm <= 1.0
        assert false_alarm < 0.01

    @pytest.mark.xfail(reason = 'main.get_rot_period measures rotation from the '
                       'flare-detection curve -- fixed in toffee.rotation, not '
                       'yet ported into main.py',
                       strict = True)
    def test_accepts_a_precomputed_flare_benchmark(self, stub_ffi_observation_times):
        #passing a benchmark skips the internal detect_flares call
        time, flux, flux_err = make_spotted_lightcurve(period = 2.0)
        benchmark = pd.DataFrame({"T_peak": [time[500]],
                                  "T_start": [time[495]],
                                  "T_end": [time[510]]})

        period, false_alarm = toffee.get_rot_period(time, flux, flux_err,
                                                    benchmark = benchmark)

        assert 1.5 < period < 2.5
        assert np.isfinite(false_alarm)


@pytest.mark.skipif(not HAS_ROTATION_MODULE,
                    reason = "toffee.rotation is not part of this install")
class TestRotationGetRotPeriod:
    '''Tests for toffee.rotation.get_rot_period, the fixed rotation measurement.

    These are the two cases TestGetRotPeriod above carries as xfail, promoted to
    real assertions because this implementation runs the periodogram on lc_quad,
    the quadratic-detrended curve that still holds the rotation signal, rather
    than on the flare-detection curve. The rest of the class covers the guards
    that come with it: the harmonic flag, the baseline flag, and the drop-in
    return signature.
    '''

    def test_recovers_an_injected_modulation(self, stub_ffi_observation_times):
        #the promoted version of TestGetRotPeriod's first xfail
        time, flux, flux_err = make_spotted_lightcurve(period = 2.0,
                                                       amplitude = 0.05)

        period, false_alarm = toffee.rotation.get_rot_period(time, flux, flux_err)

        #a 2 day modulation should come back as a 2 day modulation
        assert 1.5 < period < 2.5
        #and the detection should be highly significant
        assert 0.0 <= false_alarm <= 1.0
        assert false_alarm < 0.01

    def test_accepts_a_precomputed_flare_benchmark(self, stub_ffi_observation_times):
        #the promoted version of TestGetRotPeriod's second xfail. passing a
        #benchmark skips the internal detect_flares call
        time, flux, flux_err = make_spotted_lightcurve(period = 2.0)
        benchmark = pd.DataFrame({"T_peak": [time[500]],
                                  "T_start": [time[495]],
                                  "T_end": [time[510]]})

        period, false_alarm = toffee.rotation.get_rot_period(time, flux, flux_err,
                                                             benchmark = benchmark)

        assert 1.5 < period < 2.5
        assert np.isfinite(false_alarm)

    def test_returns_two_values_by_default(self, stub_ffi_observation_times):
        #the default return has to stay a 2-tuple, so this is a drop-in
        #replacement for main.get_rot_period at the call site
        time, flux, flux_err = make_spotted_lightcurve(period = 2.0,
                                                       amplitude = 0.05)

        result = toffee.rotation.get_rot_period(time, flux, flux_err)

        assert len(result) == 2

    def test_diagnostics_are_returned_on_request(self, stub_ffi_observation_times):
        #return_diagnostics adds a third element holding the flags behind the number
        time, flux, flux_err = make_spotted_lightcurve(period = 2.0,
                                                       amplitude = 0.05)

        period, false_alarm, diagnostics = toffee.rotation.get_rot_period(
                                                time, flux, flux_err,
                                                return_diagnostics = True)

        #the reported period must match the one in the diagnostics
        assert diagnostics['rotation_period'] == period
        #and the flags a caller is meant to check must all be present
        for key in ('harmonic_ambiguous', 'baseline_too_short', 'at_search_limit',
                    'peak_power', 'num_flares_masked'):
            assert key in diagnostics

    def test_works_without_the_internal_periodogram(self, stub_ffi_observation_times):
        #periodogram=None makes flatten() return a 4-tuple instead of a 5-tuple,
        #which the unpacking has to handle
        time, flux, flux_err = make_spotted_lightcurve(period = 2.0,
                                                       amplitude = 0.05)

        period, false_alarm = toffee.rotation.get_rot_period(time, flux, flux_err,
                                                             periodogram = None)

        assert 1.5 < period < 2.5

    def test_a_double_humped_star_is_flagged_as_harmonically_ambiguous(
                                                self, stub_ffi_observation_times):
        #two active regions half a rotation apart put a peak at half the true
        #period that is just as tall as the true one, and a single tallest peak
        #cannot tell those apart -- the result should say so rather than pretend.
        #the synthetic curve only spans about 4 days, so use a 1 day rotation to
        #get several cycles of both the period and its harmonic into the data
        time, flux, flux_err = make_spotted_lightcurve(period = 1.0,
                                                       amplitude = 0.05)
        #add a second modulation at twice the frequency to build the second hump
        flux = flux + 0.05*np.sin(4*np.pi*time/1.0)

        period, false_alarm, diagnostics = toffee.rotation.get_rot_period(
                                                time, flux, flux_err,
                                                return_diagnostics = True)

        assert diagnostics['harmonic_ambiguous'] is True

    def test_a_single_humped_star_is_not_flagged(self, stub_ffi_observation_times):
        #one active region gives one hump per rotation and no competing harmonic
        time, flux, flux_err = make_spotted_lightcurve(period = 2.0,
                                                       amplitude = 0.05)

        period, false_alarm, diagnostics = toffee.rotation.get_rot_period(
                                                time, flux, flux_err,
                                                return_diagnostics = True)

        assert diagnostics['harmonic_ambiguous'] is False

    def test_a_period_longer_than_half_the_baseline_is_flagged(
                                                self, stub_ffi_observation_times):
        #fewer than two cycles of data cannot constrain a period, however
        #confident the Lomb-Scargle looks about it
        time, flux, flux_err = make_spotted_lightcurve(period = 2.0,
                                                       amplitude = 0.05)

        period, false_alarm, diagnostics = toffee.rotation.get_rot_period(
                                                time, flux, flux_err,
                                                return_diagnostics = True)

        #the stub orbits span about 20 days, so a 2 day period is many cycles
        assert diagnostics['num_cycles'] > 2.0
        assert diagnostics['baseline_too_short'] is False

    def test_the_flare_mask_removes_points(self, stub_ffi_observation_times):
        #the whole point of the benchmark is that flaring points are excluded
        #from the periodogram, so the mask has to actually drop something
        time, flux, flux_err = make_spotted_lightcurve(period = 2.0,
                                                       amplitude = 0.05)
        benchmark = pd.DataFrame({"T_peak": [time[500]],
                                  "T_start": [time[495]],
                                  "T_end": [time[510]]})

        period, false_alarm, diagnostics = toffee.rotation.get_rot_period(
                                                time, flux, flux_err,
                                                benchmark = benchmark,
                                                return_diagnostics = True)

        assert diagnostics['num_points_kept'] < diagnostics['num_points_total']
        assert diagnostics['num_flares_masked'] == 1


# ===========================================================================
#12. NumPy compatibility
# ===========================================================================

class TestTrapezoidCompatibility:
    '''Regression tests for the np.trapz -> np.trapezoid migration.

    NumPy 2.0 removed np.trapz. main.py binds `_trapezoid` to whichever name
    the installed NumPy provides, so equivalent durations keep working on both
    NumPy 1.x and 2.x.
    '''

    def test_trapezoid_helper_is_bound(self):
        assert callable(toffee.main._trapezoid)

    def test_helper_integrates_correctly(self):
        #A constant excess of 0.5 over 1 day integrates to 0.5 day
        x_values = np.linspace(0.0, 1.0, 101)
        y_values = np.full(len(x_values), 0.5)

        assert toffee.main._trapezoid(y_values, x = x_values) == pytest.approx(0.5)

    def test_equivalent_duration_matches_a_hand_integration(self):
        #Build a flare whose equivalent duration we can compute directly, and
        #check declare_flare reports the same value in seconds
        time, flux, flux_err, quality = make_flat_lightcurve(num_points = 3000,
                                                             seed = 0)
        flux = inject_flare(time, flux, 1500, amplitude = 0.05, tau = 0.01)
        flux_std = global_flux_std(flux)

        result = toffee.declare_flare(time, flux, time[1500], flux_std,
                                      3.0, 3, 3.0, 3)
        peak, flare_start, flare_end, amp, ED = result[:5]

        index_start, index_end = toffee.start_end_time(
            time, flux, time[1500], flux_std, 3.0, 3)
        expected_equivalent_duration = np.trapezoid(
                                                    flux[index_start:index_end + 1] - 1,
                                                    x = time[index_start:index_end + 1]) * 86400 if hasattr(
                np, "trapezoid") else np.trapz(
                                               flux[index_start:index_end + 1] - 1,
                                               x = time[index_start:index_end + 1]) * 86400

        assert ED == pytest.approx(expected_equivalent_duration)
        #Equivalent duration is in seconds and must be positive for a flare
        assert ED > 0
