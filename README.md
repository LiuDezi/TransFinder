# TransFinder (still under heavy development)

A transient detection pipeline developed for large imaging surveys. This pipeline is currently developed for Mephisto and CSST/MCI surveys, but I am trying to generalize it for other surveys.

## There are two modules in the pipeline: 
### (1) image differencing
For this module, we provide several functions: 
* reference image construction using pro-processed single exposure images; 

* new image construction through matching with the reference image; 

* spatially varied PSF modeling for both reference and new images; 

* image differencing in Fourier space based on revised ZOGY algorithm ([Zackay et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...830...27Z/abstract));

* detection of transient candidates on the difference images.

### (2) real/bogus classification
For this module, we implement a ResNet deep learning network based on Pytorch. Users can provide their own data (image stamps of real and bogus transients) generated either by the image differencing module or other codes to train the network. 

Once the network is trained, the parameters can be used as input for transient detections.

Pipeline History:
===================
20240621: create this repository;
20241020: fix two bugs in the pipeline: 1) misuse of x/y indices in constructing PSF normalization map; 2) PSF star regions in the input image are incorrectly relapced in modeling the PSF.
