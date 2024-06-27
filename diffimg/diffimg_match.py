# further reduction about the pre-processed images
# NOTE: the input image for image difference should have already been pre-processed by other pipeline
# The pre-processing includes: bias subtraction, dark subtraction, flat-fielding, astrometric 
# calibration, and photometric calibration. Other reductions are not required: cosmic-ray removal, 
# sky background subtraction, bad pixel mask, etc.

# This routine will resample the image based on the astrometric solution and then subtract the sky 
# background. SWarp will be used for the goal.

import numpy as np
from astropy.io import fits
from astropy.wcs import wcs
from astropy.stats import sigma_clip, sigma_clipped_stats
import diffimg_utils as utl
import diffimg_base as dbase
import os, sys, subprocess
import matplotlib.pyplot as plt

# get the directory of the python file
path_default = os.path.dirname(__file__)
# print(f"^_^ Installed directory is: {path_default}")

def match_image(input_image, output_image, map_image, refine=False, photcal_figure=None):
    """
    Resample the image based on the astrometric solution in the image header

    Parameters:
    input_image: 
      input FITS image with complete absolute path
    output_image: 
      output resampled FITS image with complete absolute path
    map_image: 
      if map_image is specified, the 'image_center', 'image_size' and 'pixel_scale' will be extracted from it
    refine: 
      if map_image is specified, we can further refine the wcs of output_image to map_image
    image_center: 
      celestical center of the resampled image if 'map_image' is None
    pixel_scale: 
      pixel scale of the resampled image if 'map_image' is None
    image_size: 
      image size of the resampled image if 'map_image' is None
    """
    # gaia catalog
    gaia_catalog = input_image[:-5] + "_sexcat_gaia.fits"
    
    # check the completeness
    base_check = dbase.BaseCheck()
    swarp_comd = base_check.swarp_shell()
    sex_comd = base_check.sextractor_shell()
    #base_check.gaia_catalog_check(gaia_catalog)
    #base_check.header_check(input_image)

    # check if the map image header contains wcs keywords
    #base_check.header_check(map_image)
    
    # normalize the image by exposure time
    image_matrix, image_header = fits.getdata(input_image, header=True)
    exptime = image_header["EXPOSURE"]
    band    = image_header["FILTER"]
    image_header["GAIN"] = image_header["GAIN"]*exptime
    image_header["SATURATE"] = image_header["SATURATE"]/exptime * 0.85
    image_matrix = image_matrix/exptime

    # bad pixel mask
    image_meta = dbase.ImageMeta(band)
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

    fits.writeto(output_image, image_matrix, image_header, overwrite=True)

    map_hdr = fits.getheader(map_image)
    ximg, yimg = map_hdr["NAXIS1"], map_hdr["NAXIS1"]
    raCenS, decCenS = map_hdr["REF_RAS"], map_hdr["REF_DECS"]
    pixel_scale = map_hdr["REF_PS"]
    image_size = f"{ximg},{yimg}"
    image_center = f"{raCenS},{decCenS}"

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
    #output_mat, output_hdr = fits.getdata(output_image, header=True)
    #exptime = output_hdr["EXPOSURE"]
    #output_hdr["GAIN"] = output_hdr["GAIN"]*exptime
    #output_hdr["SATURATE"] = output_hdr["SATURATE"]/exptime * 0.85
    #output_mat = output_mat/exptime
    #fits.writeto(output_image, output_mat, output_hdr, overwrite=True)

    # refine the image alignment
    if refine:
        import astroalign as aa
        print("^_^ Refine the image alignment")
        map_mat = fits.getdata(map_image)
        output_mat, output_hdr = fits.getdata(output_image, header=True)
        map_mat = map_mat.byteswap().newbyteorder()
        output_mat = output_mat.byteswap().newbyteorder()
        output_mat_new, __ = aa.register(output_mat, map_mat, max_control_points=100, detection_sigma=10.0, min_area=10.0)
        map_hdr["GAIN"] = output_hdr["GAIN"]
        map_hdr["SATURATE"] = output_hdr["SATURATE"]
        fits.writeto(output_image, output_mat_new, map_hdr, overwrite=True)
    
    # perform first photometry
    output_catalog = photometry(output_image, sex_comd)
    photcat = fits.getdata(output_catalog, ext=2)
    ra_pht, dec_pht = photcat["ALPHA_J2000"], photcat["DELTA_J2000"]
    flags, snr = photcat["FLAGS"], photcat["SNR_WIN"]
    sid = (flags==0) & (snr>20) & (snr<1000)
    ra_pht, dec_pht = ra_pht[sid], dec_pht[sid]

    # perform photometric calibration for the new image
    map_star_catalog = map_image[:-4] + "phot_star.fits"
    map_photcat = fits.getdata(map_star_catalog, ext=1)
    ra_map, dec_map = map_photcat["ALPHA_J2000"], map_photcat["DELTA_J2000"]
    
    id_map, id_pht = utl.crossmatch(ra_map, dec_map, ra_pht, dec_pht, aperture=1.5)
    flux_scale = map_photcat["FLUX_AUTO"][id_map]/photcat["FLUX_AUTO"][sid][id_pht]
    mean_scale, median_scale, std_scale = sigma_clipped_stats(flux_scale, sigma=3.0, maxiters=5.0)

    # update the image
    output_mat, output_hdr = fits.getdata(output_image, header=True)
    output_hdr["GAIN"] = output_hdr["GAIN"]/median_scale
    output_hdr["SATURATE"] = output_hdr["SATURATE"]*median_scale
    output_hdr["FLXSCL"] = median_scale
    output_hdr["FLXSTD"] = std_scale
    fits.writeto(output_image, output_mat*median_scale, output_hdr, overwrite=True)

    if photcal_figure is not None:
        map_mag = map_photcat["MAG_AUTO"][id_map]
        xlim     = [np.min(map_mag)-0.5, np.max(map_mag)+0.5]
        plt.scatter(map_mag, flux_scale, color="black", marker="o", s=6)
        plt.plot(xlim, [median_scale, median_scale], "r-", linewidth=2.0)
        plt.plot(xlim, [median_scale-std_scale, median_scale-std_scale],"r--",linewidth=1.5)
        plt.plot(xlim, [median_scale+std_scale, median_scale+std_scale],"r--",linewidth=1.5)
        plt.xlim(xlim)
        plt.ylim([median_scale-5.0*std_scale, median_scale+5.0*std_scale])
        plt.title(f"flux_scale = {median_scale:8.5f} $\pm$ {std_scale:8.5f} (#{len(id_map)} stars)", fontsize=15)
        plt.savefig(photcal_figure)
        plt.clf()
        plt.close()

    # perform second photometry
    output_catalog = photometry(output_image, sex_comd)
    output_region = output_catalog[:-4] + "reg"
    photcat = fits.getdata(output_catalog, ext=2)
    ximg, yimg = photcat["XWIN_IMAGE"], photcat["YWIN_IMAGE"]
    utl.wds9reg(ximg, yimg, flag=None, radius=15.0, unit="pixel", color="blue", outfile=output_region)

    # get the star catalog with good photometric quality
    gaiacat = fits.getdata(gaia_catalog, ext=2)
    ra_gaia, dec_gaia = gaiacat["ra"], gaiacat["dec"]
    ra_pht, dec_pht = photcat["ALPHA_J2000"], photcat["DELTA_J2000"]
    flags, snr = photcat["FLAGS"], photcat["SNR_WIN"]
    sid = (flags==0) & (snr>20) & (snr<1000)
    pid, gid = utl.crossmatch(ra_pht[sid], dec_pht[sid], ra_gaia, dec_gaia, aperture=3.0)
    nstar = len(pid)
    if nstar<20: sys.exit(f"!!! At least 20 stars are required (20<SNR<1000). Only {nstar} stars are found")
    #assert nstar>20, f"!!! At least 20 stars are required (20<SNR<1000). Only {nstar} stars are found"

    # write the star catalog out
    output_star_catalog = output_image[:-4] + "phot_star.fits"
    output_star_region  = output_image[:-4] + "phot_star.reg"
    fits.writeto(output_star_catalog, photcat[sid][pid], overwrite=True)
    utl.wds9reg(ximg[sid][pid], yimg[sid][pid], radius=20.0, unit="pixel", color="yellow", outfile=output_star_region)

    image_list = [map_image, output_image]

    return image_list

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
#    input_dir = "/Users/dzliu/Workspace/Mephisto/Pipeline/transFinder/images/sci"
#    input_image = "my_sc_tztf20aawpldl_g_20231210132654_106_sciimg.fits"
#    input_image = os.path.join(input_dir, input_image)
#
#    output_dir  = "/Users/dzliu/Workspace/Mephisto/Pipeline/transFinder/images/diff"
#    output_image = "my_sc_tztf20aawpldl_g_20231210132654_106_sciimg.fits"
#    output_image = os.path.join(output_dir, output_image)
#
#    map_dir = "/Users/dzliu/Workspace/Mephisto/Pipeline/transFinder/images/ref"
#    map_image = "my_ztf20aawpldl_g_001.fits"
#    map_image = os.path.join(map_dir, map_image)
#
#    image_list = match_image(input_image, output_image, map_image=map_image, refine=True, photcal_figure="zscale.png")

