---
title: '`TOFFEE`: A Threshold-Based Method to Find Close and Overlapping Flares in TESS Lightcurves'
tags:
 - Python
 - stellar activity
 - lightcurve detrending
 - flare detection
authors:
 - name: Jason R. Reeves
   affiliation: 1
 - name: Andrew Zhang
   affiliation: 1
 - name: David V. Martin
   orcid: https://orcid.org/0000-0002-7595-6360
   affiliation: 1
 - name: Veronica G. Pratt
   orcid:
   affiliation: 1
 - name: S. Edelman
   orcid:
   affiliation: 1
affiliations:
 - index: 1
   name: Department of Physics and Astronomy, Tufts University, 574 Boston Ave, Medford, 02155, MA, USA

date: 22 September 2025
bibliography: bibliography.bib
---

# Summary

Stellar flaring is an event wherein a violent magnetic reconnection event on the surface of a star releases plasma through the star's atmosphere. The thermal emission of this lauched plasma temporarily increases the observed brightness of the star from the perspective of a distant observer. In time resolved measurements of stellar flux these correspond to consecutive outliers in the emission. Appearing as spikes with sudden rises and exponential decays in a lightcurve an algorithm can be applied to find these epochs of emission significantly higher than the typical flux after controlling for varaibilities resulting from spot modulation and systematics. Such threshold-based methods are well established and used for their simplicity and efficacy [@2021Ilin, @2016Davenport, @2015Chang]. However, simply isolating epochs of increased emission as singular flaring events obscures the fact that distinct flare events can occur simultaneously and overlap in the lightcurve. These events can be teased out visually but for large scale demographic studies automatic methods are needed in order to have a complete sample of flares with respect to wait times.

# Functionality

`TOFFEE` (TESS Overlapping Flare Finder and Energy Evaluator) is a comprehensive package that detrends and masks lightcurves then detects, models, and calculates the energies of flares. It's build to detect flare events in two minute TESS data. The code hosts endless ways to employ detection and modeling methods with the default being set to be equivalent to those used in Pratt et al, under review. However, the program features a wide set of variables in flare detection for users to alter it to their specific needs.

`TOFFEE` relys on numpy array representations of the lightcurves and involves wotan as one step of the detrending. The detrending method runs a biweight filters following @1977MostellerTukey utilizing wotan's [@2019Hippke] rolling median flattening. The detrending procedure also features a periodogram to remove residual periodicity. After flattening, a mask is applied on the lightcurve to trim cadences on boths side of each break. `TOFFEE` then begins searching for flares. All flux points above a threshold determined by the global spread of the flux points are noted and labeled by the code before being sorted in descending order. The code then goes iteratively through the list of cadences attempts to model a flare around them. If there are enough points with fluxes above the threshold around the given cadence then it's counted as a flare. `TOFFEE` then fits a function for the rise and fall of the flares. `TOFFEE` uses either a quadratic or gaussian [@2014Pitkin] function to model the rise and a double exponential decay [@2014Davenport]. If there are noticable epochs that are brighter than expected from the model then `TOFFEE` determines there is an additional flare in the rise or decay of the flare and notes there is a secondary flare event. The code returns an array of information on the start, end, and peak times of the flare as well as its amplitude in terms of normalized flux and as a ratio of the global spread, the equivalent duration, how many points constitute the flare, how many of those are above the threshold, and a flag telling whether the flare is a primary or secondary flare. `TOFFEE` then features functionality to calculate the energy of the flares in units of erg/s.

# Statement of Need

As either a purely stochastic or sympathetic process it should be expected that flaring events can occur close in time on distinct portions of the stellar surface and thus overlap in the lightcurve. This phenomenon can be teased out using visual verification and further analysis to determine whether a signal is truly a set of multiple flares of a single flare of complex morphology [@2022Ward]. However, in a very large of flares (>100,000) such methods can be replaced with automated ones. `TOFFEE` accomplishes this while using sigma-clipping, capable of distinguishing any number of overlapping flares from a single epoch of increased emission. Given `TOFFEE` is also capable of calculating the equivalent durations of said overlapping flares separately it also provides more accurate estimations of flare energy in the case that flares overlap in a lightcurve.

# Other Software

There is an assortment of public softwares designed to detect and measure flaring events. `Appaloosa` [@2016Davenport] and `AltiaPony` [@2021Ilin, @2016Davenport] are other threshold-based programs  targeting flare detection in Kepler and TESS data, respectively. `stella` [@2020Feinstein] utilizes a convolutional neural network (CNN) approach to find flares based on their shapes in the lightcurve. `allesfitter` [@2021Gunther, @allesfitter] is a multi-faceted program to model various elements of variabiltiy to a lightcurve including complex flaring events.

# Applications

`TOFFEE` has thus far been applied to two projects submitted to publication. The software's core functionality has been used to develop a flare catlog complete to short wait times to analyze the presence of symathetic flaring in a sample of stars (Pratt et al., under review). Additionally, the program has been used to study the correlation between flare occurance and spot presence on a wide sample of stellar targets, targeted to analyzing effects of statistical bias (Zhang et al., submitted).
