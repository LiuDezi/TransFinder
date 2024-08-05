# TransFinder (still under heavy development)

A transient detection pipeline developed for large imaging surveys. This pipeline is currently developed for Mephisto and CSST/MCI surveys, but I am trying to generilize it for other surveys.

## There are three modules in the pipeline: 
* image differencing. 

For this module, we provide several functions: 1) reference image construction using pro-processed single exposure images; 2) new image construction through matching with the reference image; 3) Spatially variaed PSF modeling for both reference and new images; 4) image differencing in Fourier space based on revised ZOGY algorithm ([Zackay et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...830...27Z/abstract)).

* real/bogus classification
* image silumation



Pipeline History:
===================
20240621: create this repository;
