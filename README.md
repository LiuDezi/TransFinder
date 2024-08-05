# TransFinder (still under heavy development)

A transient detection pipeline developed for large imaging surveys. This pipeline is currently developed for Mephisto and CSST/MCI surveys, but I am trying to generilize it for other surveys.

## There are three modules in the pipeline: 
### (1) image differencing. 
For this module, we provide several functions: 
* reference image construction using pro-processed single exposure images; 

* new image construction through matching with the reference image; 

* Spatially variaed PSF modeling for both reference and new images; 

* image differencing in Fourier space based on revised ZOGY algorithm ([Zackay et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...830...27Z/abstract)).

### (2) real/bogus classification

### (3) image silumation



Pipeline History:
===================
20240621: create this repository;
