# TransFinder
Transient detection pipeline developed for Mephisto survey.

There are three parts in the pipeline: 1) image subtraction; 2) real/bogus classification; and 3) transient classification.

Basic steps:
1) construct reference image. The final products include: reference image, reference catalog, star catalog;
2) align new image to the reference image. The final products include: aligned new image, new_image catalog, new_image star catalog;
3) construct spatially varied PSF models for both reference and new images;
4) perform image subtraction based on ZOGY algorithm (Zackay et al., 2016);









# Pipeline History:
20240621: create this repository;
