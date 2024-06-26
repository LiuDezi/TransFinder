# TransFinder
Transient detection pipeline developed for Mephisto survey

There are three parts in this pipeline: 1) image subtraction; 2) real/bogus classification; and 3) transient classification.

Basic steps:
1) construct reference image. The final products include: reference image, reference catalog, star catalog;
2) align new image to the reference image. The final products include: aligned new image, new_image catalog, new_image star catalog; 
