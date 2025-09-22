---
title: '\textsc{toffee}: A Threshold-Based Method to Find Close and Overlapping Flares in \textit{TESS}'
tags:
  - Python
  - astronomy
authors:
  - name: Jason R. Reeves
    orcid: 
    affiliation: 1
  - name: Andy Zhang
    orcid: 
    affiliation: 1
  - name: David V. Martin
    orcid: 
    affiliation: 1
  - name: Veronica G. Pratt
    orcid: 
    affiliation: 1
  - name: S. Edelman
    orcid: 
    affiliation: 1
affiliations:
  - name: Department of Physics and Astronomy, Tufts University, 574 Boston Ave, Medford, 02155, MA, USA
    index: 1

date: 22 September 2025
bibliography: paper.bib
---

# Summary

Stellar flaring is an event wherein a violent magnetic reconnection event on the surface of a star releases plasma through the star's atmosphere. The thermal emission of this lauched plasma temporarily increases the observed brightness of the star if launched towards an observer. On occasion such increase in emission is noticably higher than the quiescent state of the star and can be captured from a distance. In time resolved measurements of the fluxes of stars called lightcurves these correspond to outliers in the emission. Appearing as spikes with sudden rises and exponential decays an algorithm can be applied to find epochs of emission significantly higher than the typical flux along the lightcurve after controlling for varaibilities resulting from spot modulation and systematics. Such threshold based methods are well established and used for their simplicity and efficacy. However, simply isolating epochs of increased emission as singular flaring events obscures the fact that distinct flare events can occur simultaneously and overlap in the lightcurve. These events can be teased out visually but for large scale demographic studies automatic methods are needed in order to have a complete sample of flares with respect to wait times.

# Functionality

TOFFEE is a comprehensive package that detrends and masks lightcurves, detects and models flares, and calculates their energies. It's build to detect flare events in two minute TESS data. The code hosts endless ways to employ detection and modeling methods with the default being set to be equivalent to those used in Pratt et al, in prep. However, users are free to employ their own parameterization to suit their science goals.

TOFFEE relys on numpy array representations of the lightcurves and involves wotan as one step of the detrending. The detrending method runs a biweight reduction following **Cite** removing the orbital systematics as a quadratic before running wotan's rolling median flattening. We run a periodogram to remove residual periodicity. After flattening a mask is applied on the lightcurve to trim off points on either side of breaks. Then TOFFEE begins searching for flares. All flux points above a threshold determined by the global spread of the flux points are noted and labeled by the code before being sorted in descending order. The code then goes iteratively through each point and attempts to model a flare around it. If there are enough points with fluxes above the threshold next to the given point then it's counted as a flare. TOFFEE then fits a function for the rise and fall of the flares. If there are noticable epochs that are brighter than expected from the model then TOFFEE determines there is an additional flare in the rise or decay of the flare and notes there is a secondary flare event. The code returns an array of information on the start, end, and peak times of the flare as well as its amplitude in terms of normalized flux and as a ratio of the global spread, the equivalent duration, how many points constitute the flare, how many of those are above the threshold, and a flag telling whether the flare is a primary or secondary flare. TOFFEE then features functionality to calculate the energy of the flares in units of $\text{erg/s}$.



# Acknowledgements
