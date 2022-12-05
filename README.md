# prepdataps

&nbsp;  
This repository provides the `Python` scripts developed to perform corrections required before modeling one-dimensional spectra spectra with `PrepSpec` (a spectral analysis software; see *`PrepSpec` User Manual* in this repository).  Because `PrepSpec` is not designed to handle spikes in data (e.g., due to cosmic rays) and gap regions (e.g., detector gaps in the Mrk 142 example spectra taken with the Gemini North Telescope), we need to remove such features prior to processing the spectra through `PrepSpec`.  

The four scripts in this repository are described below.  

* *correctAffectedPixels.py* :
> Corrects undesirable artefacts in pixel regions specified by the user for each spectrum by either replacing the affected pixels with median values or linearly interpolating in the affected regions and replacing interpolated values with simulated data assuming Gaussian distribution of errors.  This script runs interactively with user inputs required for all spectra.  

* *correctSlitlosses.py* :
> Corrects flux losses in narrow-slit spectra using wide-slit reference spectra taken on the same night.  This script requires the the Image Reduction and Analysis Facility with Python wrapper (`PyRAF`) environment for execution of the `curfit` task non-interactively.  The interactive mode of `curfit` may not work on all systems as is therefore turned off.  

* *recoverCurveShapes.py* :
> Recovers the region of the spectra specified by blue and red wavelength bounds or lower and upper pixel bounds by estimating the shape of reference spectra.  

* *selectSpectralRegion.py* :
> Selects a region of the spectra specified by shorter (blue) and longer (red) wavelength bounds or lower and upper pixel bounds.  

&nbsp;  
### The Configuration File Interface:  
All scripts read their inputs from the configuration file *prepdataps.cfg*, which is a text file containing input parameters organized in separate sections for each of the scripts.  The user can simply modify parameters in the configuration file to execute the scripts for different data sets without having to interact with the code in the scripts.  The default parameter settings in the configuration file are applicable for the example data set of Mrk 142 long-slit spectra taken with the Gemini North Telescope (refer to the *`PrepSpec` User Manual* for further details).  The example spectra will be made available at the time of the publication of the corresponding paper (Khatu et al. 2022, submitted).  Inputs and outputs for all scripts are described in the repository.  
&nbsp;  
### Dependencies:
Install the following dependencies before downloading and using the scripts.  
> `Python` 3.6.5  
`Numpy` 1.19.5  
`Statistics` 1.0.3.5  
`Astropy` 3.0.3  
`Scipy` 1.5.4  
`Matplotlib` 3.3.4  
`PyRAF` 2.1.15  
`Configparser`

&nbsp;  
### Usage:  
To use the scripts with the example spectra, simply download the repository and run the scripts in a `Python` environment except the script for slitloss correction that needs a `PyRAF` environment.  To run the scripts with other data, modify the parameters in the configuration file as required for the new data set.  
&nbsp;  
### Citations:  
Use the following citation for *scripts* and *`PrepSpec` User Manual*.
> Khatu, V. C., Gallagher, S. C., Horne, K., et al. 2022, in prep.  

Use the following citation for *Mrk 142 example data set*.
> Khatu, V. C., Gallagher, S. C., Horne, K., et al. 2022, ApJ, submitted [Manuscript #: AAS43725]

&nbsp;  
### Questions?
Please direct any questions and concerns to *Viraja Khatu* at `vkhatu@uwo.ca`.