# test target center through interpolation

from astropy.io import fits
import galsim
import numpy as np
import os, sys

#interpor = galsim.Quintic() # this is better
interpor = galsim.Lanczos(7)

star_flux = 1000.0
image_size = 51
pixel_scale = 0.45
dx, dy = -0.2, 0.3

half_size = image_size//2
# generate a star image
star = galsim.Gaussian(fwhm=1.6, flux=star_flux)
star = star.shear(g1=0.04, g2=-0.06)
star_obs = star.drawImage(scale=pixel_scale,nx=image_size,ny=image_size, offset=(dx, dy))
star_obs.addNoise(galsim.GaussianNoise(sigma=0.1))
fits.writeto("psf_star_obs.fits", star_obs.array/star_flux, overwrite=True)

star_true = star.drawImage(scale=pixel_scale,nx=image_size,ny=image_size)
star_true.addNoise(galsim.GaussianNoise(sigma=0.1))
fits.writeto("psf_star_true.fits", star_true.array/star_flux, overwrite=True)

diff = star_obs.array - star_true.array
fits.writeto("psf_star_diff.fits", diff/star_flux, overwrite=True)

# interpolation the PSF
#image_downsample = galsim.ImageF(image_size, image_size, scale=pixel_scale)
star_obj = galsim.Image(star_obs.array, scale=pixel_scale)
stm_moment = star_obj.FindAdaptiveMom()
x_moment = stm_moment.moments_centroid.x - 1.0
y_moment = stm_moment.moments_centroid.y - 1.0
x_offset, y_offset = x_moment-half_size, y_moment-half_size
star_interp = galsim.InterpolatedImage(star_obj, x_interpolant=interpor, normalization='flux')
star_down = star_interp.drawImage(nx=image_size, ny=image_size, scale=pixel_scale, method='no_pixel', offset=(-x_offset, -y_offset))

diff_new = star_down.array - star_true.array
fits.writeto("psf_star_interp.fits", star_down.array/star_flux, overwrite=True)
fits.writeto("psf_star_diff_interp.fits", diff_new/star_flux, overwrite=True)

sys.exit(0)

# the following code shows how to oversample and downsample an image
image_upsample = galsim.ImageF(image_size_pad*oversampling,image_size_pad*oversampling,scale=pixel_scale/oversampling)
image_downsample = galsim.ImageF(image_size, image_size, scale=pixel_scale)

# generate a star image
star = galsim.Gaussian(fwhm=1.6, flux=star_flux)
star = star.shear(g1=0.04, g2=-0.06)
star = star.drawImage(scale=pixel_scale,nx=image_size_pad,ny=image_size_pad, offset=(dx, dy))
star.addNoise(galsim.GaussianNoise(sigma=0.13))
star_matrix = star.array/star_flux
fits.writeto("psf_star_raw.fits", star_matrix, overwrite=True)

# oversample the psf
star_obj = galsim.Image(star_matrix, scale=pixel_scale)
star_interp = galsim.InterpolatedImage(star_obj, x_interpolant=interpor, normalization='flux')
star_up = star_interp.drawImage(image_upsample, method='no_pixel')
fits.writeto("psf_star_upsample.fits", star_up.array, overwrite=True)

# downsample the psf
star_obj = galsim.Image(star_up.array, scale=pixel_scale/oversampling)
star_interp = galsim.InterpolatedImage(star_obj, x_interpolant=interpor, normalization='flux')
star_down = star_interp.drawImage(image_downsample, method='no_pixel', offset=(-5.8, -11.2))
fits.writeto("psf_star_downsample.fits", star_down.array, overwrite=True)
#dresi = star_matrix - star_down.array
#print(np.sum(dresi**2))

#fits.writeto("psf_star_residual.fits", dresi, overwrite=True)
