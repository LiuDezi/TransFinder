# The routine is used to construct the reference image for image differencing
# NOTE: the input image should have already been pre-processed by other pipeline
# The pre-processing includes: bias subtraction, dark subtraction, flat-fielding, astrometric 
# calibration, and photometric calibration. Other reductions are not required: cosmic-ray removal, 
# sky background subtraction, bad pixel mask, etc.

# This routine will resample the image based on the astrometric solution and then subtract the sky 
# background. SWarp will be used for the goal.

# Update history:
# 20240610: /1) enable bad column interpolation;/
#           /2) unify the pixel scale and image size for the resampled image;/
# 20240630: /1) we require that the star catalog should be provided by the user
#           /2) bad column interpolation is optional

import numpy as np
from astropy.io import fits
from astropy.wcs import wcs
from astropy.stats import sigma_clip, sigma_clipped_stats
import mdiff_utils as utl
import mdiff_base as dbase
import os, sys, subprocess
import matplotlib.pyplot as plt

# get the directory of the python file
path_default = os.path.dirname(__file__)
# print(f"^_^ Installed directory is: {path_default}")

def resamp_image(input_image, input_star_catalog, output_image, 
                 survey_mode="pilot",
                 image_center = "00:00:00,+00:00:00",
                 interp_badpixel = True, 
                 ref_catalog  = "reference_image_mephisto.cat"):
    """
    Resample the image based on the astrometric solution in the image header

    Parameters:
    input_image: 
      input FITS image with absolute path
    input_star_catalog:
      pre-selected star catalog with absolute path
    output_image: 
      output resampled FITS image with absolute path
    image_center: 
      celestical center of the resampled image
    """
    # star catalog
    input_star_catalog = input_image[:-5] + "_sexcat_gaia.fits"

    # check the completeness
    base_check = dbase.BaseCheck()
    swarp_comd = base_check.swarp_shell()
    sex_comd = base_check.sextractor_shell()
    base_check.star_catalog_check(input_star_catalog)
    base_check.header_check(input_image)

    # normalize the image by exposure time
    image_matrix, image_header = fits.getdata(input_image, header=True)
    exptime = image_header["EXPOSURE"]
    band    = image_header["FILTER"]
    image_header["GAIN"] = image_header["GAIN"]*exptime
    image_header["SATURATE"] = image_header["SATURATE"]/exptime * 0.85
    image_matrix = image_matrix/exptime
   
    image_meta = dbase.ImageMeta(band, survey_mode=survey_mode)
    if interp_badpixel:
        # bad pixel mask
        xlim_badpix, ylim_badpix = image_meta.bad_column()
        nbadpix = len(xlim_badpix)
        for i in range(nbadpix):
            ix0, ix1 = xlim_badpix[i]
            iy0, iy1 = ylim_badpix[i]
            image_matrix[iy0:iy1+1,ix0:ix1+1] = np.nan
        image_matrix[image_matrix<=0] = np.nan
    
        # fill the background with background noise
        gatelim = image_meta.gates_bound()
        for igate, ibound in gatelim.items():
            #if igate!=0: continue
            ix0, ix1, iy0, iy1 = ibound
            image_matrix_sub = image_matrix[iy0:iy1,ix0:ix1]
            image_matrix_sub_masked = sigma_clip(image_matrix_sub,sigma=3,stdfunc='mad_std',masked=False)
            imedian, imad = np.median(image_matrix_sub_masked), utl.mad(image_matrix_sub_masked)
            idnan = np.isnan(image_matrix_sub)
            #image_matrix_sub[idnan] = imedian
            image_matrix_sub[idnan] = np.random.normal(imedian, imad, size=image_matrix_sub.shape)[idnan]
            image_matrix[iy0:iy1,ix0:ix1] = image_matrix_sub

    # write the normalized (bad-pixel interpolated) image
    fits.writeto(output_image, image_matrix, image_header, overwrite=True)

    # load the default pixel scale and image size for image resampling
    pixel_scale, image_size = image_meta.resample_param()
    image_size = f"{image_size[0]},{image_size[1]}"

    # run swarp
    output_weight = output_image[:-4] + "weight.fits"
    swarp_param_file = os.path.join(path_default, "config/default_config.swarp")
    swarp_run1 = f"{swarp_comd} {output_image} -c {swarp_param_file} "
    swarp_run2 = f"-IMAGEOUT_NAME {output_image} -WEIGHTOUT_NAME {output_weight} "
    swarp_run3 = f"-CENTER {image_center}  -PIXEL_SCALE {pixel_scale} -IMAGE_SIZE {image_size}"
    swarp_run  = swarp_run1 + swarp_run2 + swarp_run3
    #print(f"^_^ {swarp_run}")
    subprocess.run(swarp_run, shell=True)

    # normalize the image by exposure time
    #image_matrix, image_header = fits.getdata(output_image, header=True)
    #exptime = image_header["EXPOSURE"]
    #image_header["GAIN"] = image_header["GAIN"]*exptime
    #image_header["SATURATE"] = image_header["SATURATE"]/exptime * 0.85
    #fits.writeto(output_image, image_matrix/exptime, image_header, overwrite=True)

    # perform photometry
    output_catalog = photometry(output_image, sex_comd)
    output_region = output_catalog[:-4] + "reg"
    photcat = fits.getdata(output_catalog, ext=2)
    ximg, yimg = photcat["XWIN_IMAGE"], photcat["YWIN_IMAGE"]
    utl.wds9reg(ximg, yimg, flag=None, radius=15.0, unit="pixel", color="blue", outfile=output_region)

    # get the star catalog with good photometric quality
    starcat = fits.getdata(input_star_catalog, ext=2)
    ra_star, dec_star = starcat["ra"], starcat["dec"]
    #ra_star, dec_star = starcat["X_WORLD"], starcat["Y_WORLD"]
    ra_pht, dec_pht = photcat["ALPHA_J2000"], photcat["DELTA_J2000"]
    flags, snr = photcat["FLAGS"], photcat["SNR_WIN"]
    sid = (flags==0) & (snr>20) & (snr<1000)
    pid, gid = utl.crossmatch(ra_pht[sid], dec_pht[sid], ra_star, dec_star, aperture=3.0)
    nstar = len(pid)
    if nstar<20: sys.exit(f"!!! At least 20 stars are required (20<SNR<1000). Only {nstar} stars are found")
    #assert nstar>20, f"!!! At least 20 stars are required (20<SNR<1000). Only {nstar} stars are found"
    delta_ra = (ra_star[gid] - ra_pht[sid][pid])*np.cos(dec_star[gid]*np.pi/180.0)*3600.0
    delta_dec = (dec_star[gid] - dec_pht[sid][pid])*3600.0
    mean_fwhm, median_fwhm, std_fwhm = sigma_clipped_stats(photcat["FWHM_IMAGE"][sid][pid], sigma=3.0, maxiters=5.0)
    mean_ra, median_ra, std_ra = sigma_clipped_stats(delta_ra, sigma=3.0, maxiters=5.0)
    mean_dec, median_dec, std_dec = sigma_clipped_stats(delta_dec, sigma=3.0, maxiters=5.0)

    # write the star catalog out
    output_star_catalog = output_image[:-4] + "phot_star.fits"
    output_star_region  = output_image[:-4] + "phot_star.reg"
    fits.writeto(output_star_catalog, photcat[sid][pid], overwrite=True)
    utl.wds9reg(ximg[sid][pid], yimg[sid][pid], radius=20.0, unit="pixel", color="yellow", outfile=output_star_region)

    # update image header
    image_matrix, image_header = fits.getdata(output_image, header=True)
    image_wcs = wcs.WCS(image_header)
    xsize, ysize = image_wcs.pixel_shape
    ximg_center, yimg_center = 0.5*(xsize+1), 0.5*(ysize+1)
    ra_center, dec_center = image_wcs.all_pix2world(ximg_center, yimg_center, 1)
    xra_center, xdec_center  = utl.deg2str(ra_center, dec_center)
    
    image_header["REF_RA"]   = (ra_center.tolist(), "RA of image center [deg]")
    image_header["REF_DEC"]  = (dec_center.tolist(), "DEC of image center [deg]")
    image_header["REF_RAS"]  = (xra_center[0], "RA of image center [hhmmss]")
    image_header["REF_DECS"] = (xdec_center[0], "DEC of image center [ddmmss]")
    image_header["REF_PS"]   = (pixel_scale, "Image pixel scale [arcsec/pixel]")
    image_header["REF_PUN"]  = ("adu/s", "Pixel unit")
    image_header["REF_FWHM"] = (median_fwhm, "Image median FWHM [pixel]")
    image_header["REF_MRA"]  = (median_ra, "Median RA offset of astrometry")
    image_header["REF_MDEC"] = (median_dec, "Median DEC offset of astrometry")
    image_header["REF_SRA"]  = (std_ra, "RA rms of astrometry")
    image_header["REF_SDEC"] = (std_dec, "DEC rms of astrometry")
    image_header["REF_NS"]   = (nstar, "Number of high-quality stars")
    image_header["REF_IMG"]  = (output_image.split("/")[-1], "Reference image name")
    image_header["TF_VERS"]  = (dbase.__version__, "TransFinder version")
    image_header["TF_DATE"]  = (dbase.__version_date__, "TransFinder version date")
    image_header["TF_AUTH"]  = (dbase.__author__, "TransFinder author")

    fits.writeto(output_image, image_matrix, image_header, overwrite=True)
    
    # reference catalog
    output_image_list = output_image.split("/")
    ref_image = output_image_list[-1]
    ref_catalog_path = "/".join(output_image_list[:-1])
    ref_catalog_abs = os.path.join(ref_catalog_path, ref_catalog)
    ref_catalog_fmt = "%30s %12.6f %12.6f %7.3f %7.3f %7.3f %7.3f %3s %7.3f %5d %7.4f %s\n"
    ref_line = ref_catalog_fmt%(ref_image,ra_center.tolist(),dec_center.tolist(),
                                median_ra,median_dec,std_ra,std_dec,band,
                                median_fwhm,nstar,float(image_header["AIRMASS"]),None)
    if not os.path.exists(ref_catalog_abs):
        refcat = open(ref_catalog_abs, "w")
        title  = "#ref_image ra dec mu_ra mu_dec sigma_ra sigma_dec band fwhm nstar airmass ref_path\n"
        refcat.write(title)
    else:
        refcat = open(ref_catalog_abs, "a")
    refcat.write(ref_line)
    refcat.close()

    return

def photometry(input_image, sextractor_shell):
    output_catalog = input_image[:-4] + "phot_all.fits"
    sex_config_file = os.path.join(path_default, "config/default_config.sex")
    sex_param_file = os.path.join(path_default, "config/default_param.sex")
    sex_run1 = f"{sextractor_shell} {input_image} -c {sex_config_file} "
    sex_run2 = f"-CATALOG_NAME {output_catalog} -PARAMETERS_NAME {sex_param_file}"
    sex_run = sex_run1 + sex_run2
    #print(f"^_^ {sex_run}")
    subprocess.run(sex_run, shell=True)

    return output_catalog

#if __name__ == "__main__":
#    import time
#    input_dir = "/Users/dzliu/Workspace/Mephisto/TransFinder/images/kmtnew/sciimg"
#    input_image = "xKMTNt.20180221.003965_sciimg.fits"
#    input_image = os.path.join(input_dir, input_image)
#    input_star_catalog = "combined.gaia_ldac.fits"
#    input_star_catalog = os.path.join(input_dir, input_star_catalog)
#
#    output_dir  = "/Users/dzliu/Workspace/Mephisto/TransFinder/images/kmtnew/refimg"
#    output_image = "KMTNt.20180221.003965_sciimg.ref.fits"
#    output_image = os.path.join(output_dir, output_image)
#
#    image_center = "18:10:45.00,-30:36:20.00"
#    survey_mode = "regular"
#
#    t0 = time.time()
#    image_list = resamp_image(input_image, output_image, 
#                              survey_mode=survey_mode,
#                              image_center=image_center,
#                              interp_badpixel=False,)
#    t1 = time.time()
#    dt = t1 - t0
#    print(f"^_^ Total {dt:.5f} seconds used")
#    # catalog
#    # target ra dec mu_ra mu_dec sigma_ra sigma_dec band fwhm nstar airmass ref_name ref_path
