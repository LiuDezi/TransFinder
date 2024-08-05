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

# resample and stacking > single refimg, coadded refimg, resampled new
# match: match with reference, photometric calibration
# functions: ["image_resamp", "image_match"]
# image_resamp: resampe a single exposure image. Basically, it is for new image construction.
# image_match: for a specified new image, find the matched reference image. Basically, it is 
#              for reference image construction.

import numpy as np
from astropy import units
from astropy.io import fits
from astropy.table import Table, Column
from astropy.wcs import wcs
from astropy.stats import sigma_clip, sigma_clipped_stats, mad_std
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import matplotlib.pyplot as plt
import os
import sys
import subprocess

from .base import BaseCheck
from ..utils import wds9reg, crossmatch, deg2str, str2deg, sub_regions

def image_resamp(input_image,
                 input_star_crds, 
                 output_image,
                 swarp_config_file, 
                 sex_config_file, 
                 sex_param_file,
                 swarp_exe = "swarp",
                 sex_exe = "sextractor",
                 input_weight = None, 
                 input_flag = None,
                 output_pixel_scale = None,
                 output_image_size = None,
                 output_image_center = None,
                 output_meta = None,
                 output_meta_overwrite = True,
                 interp_badpixel_mode = "interp",
                 interp_badpixel_grid = (30,30),
                 ):
    """
    Resample the image based on the astrometric solution in the image header

    Parameters:
    input_image: str
      input FITS image with absolute path
    input_star_crds: list
      input star coordinates, e.g. input_star_crds=[ra_array, dec_array]
    output_image: str
      output resampled FITS image with absolute path
    swarp_config_file: file
      swarp configuration file
    sex_config_file: file
      sextractor configuration file
    sex_param_file: file
      sextractor output parameter file
    swarp_exe: str
      executable swarp command in terminal
    sex_exe: str
      executable sextractor command in terminal
    input_weight: str
      same format as input_image
    input_flag: str
      same format as input_image, for bad pixels interpolation
    output_pixel_scale: float
      pixel scale of output image, e.g. output_pixel_scale = 0.3
    output_image_size: tuple
      image size of output image, e.g. output_image_size = (1000, 1000)
    output_image_center: tuple
      image celestial center of output image, 
      e.g. output_image_center = (hh:mm:ss.ss, +dd:mm:ss.ss)
    output_meta: str
      meta table to store the parameters of resampled reference image
    output_meta_oberwrite: bool
      whether replace the meta data if it is already in the meta table
    interp_badpixel_mode: str
      see function 'interp_badpixel' for more detail
    interp_badpixel_grid: tuple
      see function 'interp_badpixel' for more detail
    """
    print("^_^ Construct reference image and corresponding catalog")
    base_check = BaseCheck()
    base_check.header_check(input_image)

    # 0) basic setup
    output_catalog = output_image[:-4] + "phot.fits"
    output_region = output_catalog[:-4] + "star.reg"
    output_weight = output_image[:-4] + "weight.fits"

    print(f"    Input image: {os.path.basename(input_image)}")
    print(f"    Resampled reference image: {os.path.basename(output_image)}")
    print(f"    Resampled reference catalog: {os.path.basename(output_catalog)}")
    
    # 1) load input image matrix and header
    image_matrix, image_header = fits.getdata(input_image, header=True)
    image_wcs = wcs.WCS(image_header)
    xsize, ysize = image_wcs.pixel_shape

    # 2) normalize input image by exposure time
    exptime, band = image_header["EXPTIME"], image_header["FILTER"]
    image_header["GAIN"] = image_header["GAIN"]*exptime
    image_header["SATURATE"] = image_header["SATURATE"]/exptime * 0.90
    image_matrix = image_matrix/exptime

    # 3) interpolate bad pixels
    flag_matrix = None
    if input_flag is not None: flag_matrix = fits.getdata(input_flag)
    image_matrix = interp_badpixel(image_matrix, flag_map=flag_matrix,
                                   mode=interp_badpixel_mode,
                                   image_grid=interp_badpixel_grid)

    # 4) save the normalized (and bad-pixel interpolated) image
    fits.writeto(output_image, image_matrix, image_header, overwrite=True)

    # 5) swarp configuration
    swarp_mpar = f"{swarp_exe} {output_image} -c {swarp_config_file} -IMAGEOUT_NAME {output_image} "
    if input_weight is not None:
        swarp_wpar = f"-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE {input_weight} -WEIGHTOUT_NAME {output_weight} "
    else:
        swarp_wpar = f"-WEIGHT_TYPE NONE -WEIGHTOUT_NAME {output_weight} "
    if output_image_center is None:
        ximg_center, yimg_center = 0.5*(xsize+1), 0.5*(ysize+1)
        ra_center, dec_center = image_wcs.all_pix2world(ximg_center, yimg_center, 1)
        ra_center, dec_center = ra_center.tolist(), dec_center.tolist()
        sra_center, sdec_center = deg2str(ra_center, dec_center)
        output_image_center = (sra_center[0], sdec_center[0])
    if output_pixel_scale is None:
        pixel_scale = np.mean([ips.value*3600.0 for ips in image_wcs.proj_plane_pixel_scales()])
        output_pixel_scale = float(f"{pixel_scale:.2f}")
    if output_image_size is None:
        output_image_size = (xsize, ysize)
    
    simage_center = f"{output_image_center[0]},{output_image_center[1]}"
    simage_size = f"{output_image_size[0]},{output_image_size[1]}"
    swarp_rpar = f"-CENTER {simage_center} -PIXEL_SCALE {output_pixel_scale} -IMAGE_SIZE {simage_size} "
    
    print("    Resampled parameters:")
    print(f"   1) image center: {output_image_center}")
    print(f"   2) image size: {output_image_size}")
    print(f"   3) image pixel scale: {output_pixel_scale} arcsec/pixel")

    # 6) run swarp and perform photometry
    swarp_run  = swarp_mpar + swarp_wpar + swarp_rpar
    subprocess.run(swarp_run, shell=True)
    photometry(output_image, output_weight, output_catalog, sex_exe, sex_config_file, sex_param_file)
    os.remove(output_weight)

    # 7) load star catalog with good photometric quality
    photcat = Table.read(output_catalog, format="fits", hdu=2)
    nobj = len(photcat)
    ra, dec = photcat["ALPHA_J2000"], photcat["DELTA_J2000"]
    flags, snr, fwhm = photcat["FLAGS"], photcat["SNR_WIN"], photcat["FWHM_IMAGE"]
    sid = (flags==0) & (snr>20) & (snr<1000) & (fwhm>0.0)
    
    rab, decb = input_star_crds
    pid, gid = crossmatch(ra[sid], dec[sid], rab, decb, aperture=3.0)
    nstar = len(pid)
    if nstar<20: sys.exit(f"!!! At least 20 stars are required (20<SNR<1000). Only {nstar} stars are found")
    sid = np.where(sid==True)[0][pid]

    delta_ra = (rab[gid] - ra[sid])*np.cos(decb[gid]*np.pi/180.0)*3600.0
    delta_dec = (decb[gid] - dec[sid])*3600.0
    mean_ra, median_ra, std_ra = sigma_clipped_stats(delta_ra, sigma=3.0, maxiters=5.0)
    mean_dec, median_dec, std_dec = sigma_clipped_stats(delta_dec, sigma=3.0, maxiters=5.0)

    # 8) update the catalog
    rab_full, decb_full = np.zeros(nobj) - 99.0, np.zeros(nobj) - 99.0
    rab_full[sid], decb_full[sid] = rab[gid], decb[gid]
    photcat.add_columns([rab_full, decb_full], names=["RA_BASE", "DEC_BASE"])
    photcat.write(output_catalog, format="fits", overwrite=True)
    wds9reg(ra[sid], dec[sid], radius=5.0, unit="arcsec", color="red", outfile=output_region)

    # 9) update image header
    image_matrix, image_header = fits.getdata(output_image, header=True)
    std_bkg = mad_std(image_matrix, ignore_nan=True)
    mean_fwhm, median_fwhm, std_fwhm = sigma_clipped_stats(fwhm[sid], sigma=3.0, maxiters=5.0)
    
    image_wcs = wcs.WCS(image_header)
    xsize, ysize = image_wcs.pixel_shape
    ximg_center, yimg_center = 0.5*(xsize+1), 0.5*(ysize+1)
    ra_center, dec_center = image_wcs.all_pix2world(ximg_center, yimg_center, 1)
    ra_min, dec_min = image_wcs.all_pix2world(0.5, 0.5, 1)
    ra_max, dec_max = image_wcs.all_pix2world(xsize+0.5, ysize+0.5, 1)
    ra_center, dec_center = ra_center.tolist(), dec_center.tolist()
    ra_min, dec_min = ra_min.tolist(), dec_min.tolist()
    ra_max, dec_max = ra_max.tolist(), dec_max.tolist()
    if ra_min>ra_max: ra_min, ra_max = ra_max, ra_min
    if dec_min>dec_max: dec_min, dec_max = dec_max, dec_min
    sra_center, sdec_center = output_image_center

    oimage_name = os.path.basename(output_image)
    oimage_path = os.path.dirname(output_image)

    image_header["BKGSIG"]  = (std_bkg, "background sigma")
    image_header["RA0"]     = (ra_center, "RA of image center [deg]")
    image_header["DEC0"]    = (dec_center, "DEC of image center [deg]")
    image_header["SRA0"]    = (sra_center, "RA of image center [hhmmss]")
    image_header["SDEC0"]   = (sdec_center, "DEC of image center [ddmmss]")
    image_header["PSCALE"]  = (output_pixel_scale, "Image pixel scale [arcsec/pixel]")
    image_header["PIXUN"]   = ("adu/s", "Pixel unit")
    image_header["FWHM"]    = (median_fwhm, "Image median FWHM [pixel]")
    image_header["RABIAS"]  = (median_ra, "Median RA offset of astrometry")
    image_header["DECBIAS"] = (median_dec, "Median DEC offset of astrometry")
    image_header["RASTD"]   = (std_ra, "RA rms of astrometry")
    image_header["DECSTD"]  = (std_dec, "DEC rms of astrometry")
    image_header["NSTAR"]   = (nstar, "Number of high-quality stars")
    image_header["REFIMG"]  = (oimage_name, "Image name")
    fits.writeto(output_image, image_matrix.astype(np.float32), image_header, overwrite=True)
    
    # save meta table
    if output_meta is not None:
        meta_table = meta_table_iter(output_meta)
        update_row = [oimage_name, ra_center, dec_center, median_ra, median_dec,
                      std_ra, std_dec, ra_min, ra_max, dec_min, dec_max,
                      band, median_fwhm, nstar, std_bkg, oimage_path]
        if oimage_name in meta_table["image_name"]:
            if output_meta_overwrite:
                idx = np.where(meta_table["image_name"]==oimage_name)[0][0]
                meta_table.remove_row(idx)
                print(f"!!! Image {oimage_name} is already in the meta table: It is overwritten.")
                meta_table.add_row(update_row)
            else:
                print(f"!!! Image {oimage_name} is already in the meta table: Nothing to do.")
        else:
            meta_table.add_row(update_row)

        # save the table
        meta_table.write(output_meta, format="fits", overwrite=True)
    
    return

def build_newimg(input_image, 
                 output_image,
                 swarp_config_file,
                 sex_config_file,
                 sex_param_file,
                 swarp_exe = "swarp",
                 sex_exe = "sextractor",
                 survey_mode="mephisto_pilot",
                 ref_image = None,
                 ref_meta = None,
                 interp_badpixel_mode = None,
                 interp_badpixel_grid = (30,30),
                 interp_badpixel_flag = None,
                 photcal_figure=None,
                 ):
    """
    Resample the image based on the astrometric solution in the image header

    Parameters:
    input_image: str
      input FITS image with absolute path
    output_image: str
      output resampled FITS image with absolute path
    swarp_config_file: file
      swarp configuration file
    sex_config_file: file
      sextractor configuration file
    sex_param_file: file
      sextractor output parameter file
    swarp_exe: str
      executable swarp command in terminal
    sex_exe: str
      executable sextractor command in terminal
    survey_mode: str
      currently, survey_mode should be ["mci", "mephisto", "mephisto_pilot"]
    ref_image: str
      reference image
    ref_meta: str
      reference meta table
    interp_badpixel_mode: str
      see function 'interp_badpixel' for more detail
    interp_badpixel_grid: tuple
      see function 'interp_badpixel' for more detail
    interp_badpixel_flag: array
      see function 'interp_badpixel' for more detail
    photcal_figure: str
      figure of relative photometry
    """
    print("^_^ Generate matched image and corresponding catalog with a specified reference")
    base_check = BaseCheck()
    base_check.header_check(input_image)

    if (ref_image==None) and (ref_meta==None):
        sys.exit(f"!!! You should specify either 'ref_img' or 'ref_meta_table'")
    elif (ref_image!=None) and (ref_meta!=None):
        print(f"    Both 'ref_image' and 'ref_meta' are provided, only 'ref_image' is used")
        ref_meta = None
    else:
        pass

    # basic setup
    output_catalog = output_image[:-4] + "phot.fits"

    image_matrix, image_header = fits.getdata(input_image, header=True)
    band = image_header["FILTER"]

    # find reference image from meta table
    if ref_image is None:
        # estimate the image center
        image_wcs = wcs.WCS(image_header)
        xsize, ysize = image_wcs.pixel_shape
        ximg_center, yimg_center = 0.5*(xsize+1), 0.5*(ysize+1)
        ra_center, dec_center = image_wcs.all_pix2world(ximg_center, yimg_center, 1)
        ra_center, dec_center = ra_center.tolist(), dec_center.tolist()
        hx_arcsec= 0.5 * xsize * image_wcs.proj_plane_pixel_scales()[0].value * 3600.0
        hy_arcsec= 0.5 * ysize * image_wcs.proj_plane_pixel_scales()[1].value * 3600.0
        match_aperture = np.min([hx_arcsec, hy_arcsec])
        
        # find the reference image
        refcat_meta = Table.read(ref_meta, format="fits")
        bid = refcat_meta["band"]==band
        ref_ra, ref_dec = refcat_meta["ra"][bid], refcat_meta["dec"][bid]
        rid, iid = crossmatch(ref_ra,ref_dec,[ra_center],[dec_center], aperture=match_aperture)
        if len(rid)==0: sys.exit(f"!!! No reference image found in {os.path.basename(ref_meta)}")
        ref_image = refcat_meta["ref_image"][bid][rid[0]]
        ref_image_path = refcat_meta["ref_path"][bid][rid[0]]
        ref_image  = os.path.join(ref_image_path, ref_image)
        print(f"    Matched reference image is {os.path.basename(ref_image)}")
    ref_catalog = ref_image[:-4] + "phot.fits"

    print(f"    Input image: {os.path.basename(input_image)}")
    print(f"    Reference image: {os.path.basename(ref_image)}")

    # normalize the image by exposure time
    exptime = image_header["EXPTIME"]
    image_header["GAIN"] = image_header["GAIN"]*exptime
    image_header["SATURATE"] = image_header["SATURATE"]/exptime * 0.85
    image_matrix = image_matrix/exptime
    
    # interpolate bad pixels
    if interp_badpixel_mode is not None:
        image_matrix = interp_badpixel(image_matrix,
                                       flag_map=interp_badpixel_flag,
                                       mode=interp_badpixel_mode,
                                       image_grid=interp_badpixel_grid)
    
    # save the normalized (and bad-pixel interpolated) image
    fits.writeto(output_image, image_matrix, image_header, overwrite=True)

    # matching parameters from reference image
    ref_header = fits.getheader(ref_image)
    ximg, yimg = ref_header["NAXIS1"], ref_header["NAXIS2"]
    raCenS, decCenS = ref_header["SRA0"], ref_header["SDEC0"]
    pixel_scale = ref_header["PSCALE"]
    image_size, image_center = f"{ximg},{yimg}", f"{raCenS},{decCenS}"

    print("    Matched parameters:")
    print(f"   1) image center: {image_center}")
    print(f"   2) image size: {image_size}")
    print(f"   3) image pixel scale: {pixel_scale} arcsec/pixel")

    # run swarp
    output_weight = output_image[:-4] + "weight.fits"
    swarp_run1 = f"{swarp_exe} {output_image} -c {swarp_config_file} "
    swarp_run2 = f"-IMAGEOUT_NAME {output_image} -WEIGHTOUT_NAME {output_weight} "
    swarp_run3 = f"-CENTER {image_center}  -PIXEL_SCALE {pixel_scale} -IMAGE_SIZE {image_size}"
    swarp_run  = swarp_run1 + swarp_run2 + swarp_run3
    subprocess.run(swarp_run, shell=True)
    os.remove(output_weight)

    # perform photometry
    photometry(output_image, output_catalog, sex_exe, sex_config_file, sex_param_file)

    # select high quality objects
    photcat = Table.read(output_catalog, format="fits", hdu=2)
    nobj = len(photcat)
    ra, dec = photcat["ALPHA_J2000"], photcat["DELTA_J2000"]
    aflux, aflux_err = photcat["FLUX_AUTO"], photcat["FLUXERR_AUTO"]
    flags, snr, fwhm = photcat["FLAGS"], photcat["SNR_WIN"], photcat["FWHM_IMAGE"]
    sid = (flags==0) & (snr>20) & (snr<1000) & (fwhm>0)

    # crossmatch with reference stars
    photcatb = Table.read(ref_catalog, format="fits")
    rab, decb = photcatb["RA_BASE"], photcatb["DEC_BASE"]
    afluxb, aflux_errb = photcatb["FLUX_AUTO"], photcatb["FLUXERR_AUTO"]
    sidb = (rab!=-99.0) & (decb!=-99.0)

    pid, gid = crossmatch(ra[sid], dec[sid], rab[sidb], decb[sidb], aperture=3.0)
    nstar = len(pid)
    if nstar<20: 
        sys.exit(f"!!! At least 20 stars are required (20<SNR<1000). Only {nstar} stars are found")

    # dirty indices
    pdid = np.where(sid==True)[0][pid]
    rdid = np.where(sidb==True)[0][gid]

    # perform photometric calibration for the new image
    # 1) a constant flux scaling over entire image
    sr2 = np.sum(afluxb[rdid]*afluxb[rdid])
    sn2 = np.sum(aflux[pdid]*aflux[pdid])
    snr = np.sum(afluxb[rdid]*aflux[pdid])
    ser2 = np.sum(aflux_errb[rdid]*aflux_errb[rdid])
    sen2 = np.sum(aflux_err[pdid]*aflux_err[pdid])
    a, b, c = snr*sen2, sn2*ser2-sr2*sen2, 0.0-snr*ser2
    alpha = (-b + np.sqrt(b*b-4.0*a*c))/(2.0*a)

    beta = alpha**2*snr*sen2 + alpha*(sn2*ser2-sr2*sen2) - snr*ser2
    x1 = (ser2 + alpha**2*sen2)**(-3)
    x2 = -4.0*alpha*sen2*beta
    x3 = (ser2+alpha**2*sen2)*(2.0*alpha*snr*sen2+sn2*ser2-sr2*sen2)
    alpha_err = np.sqrt(1.0/(2.0*x1*(x2+x3)))
    #print(weight_scale, weight_scale_sigma)
    if photcal_figure is not None:
        ref_mag = photcatb["MAG_AUTO"][rdid]
        flux_ratio = afluxb[rdid]/aflux[pdid]
        flux_scale_plot(ref_mag, flux_ratio, alpha, alpha_err, photcal_figure)
    
    # 2) position-dependent flux scaling over entire image
    # to be completed

    #flux_scale = photcatb["FLUX_AUTO"][rdid]/photcat["FLUX_AUTO"][pdid]
    #mean_scale, median_scale, std_scale = sigma_clipped_stats(flux_scale, sigma=3.0, maxiters=5.0)
    #if photcal_figure is not None:
    #    ref_mag = photcatb["MAG_AUTO"][rdid]
    #    flux_scale_plot(ref_mag, flux_scale, median_scale, std_scale, photcal_figure)

    # photometric quality extimate based on the reference star catalog
    delta_ra = (rab[rdid] - ra[pdid])*np.cos(decb[rdid]*np.pi/180.0)*3600.0
    delta_dec = (decb[rdid] - dec[pdid])*3600.0
    mean_fwhm, median_fwhm, std_fwhm = sigma_clipped_stats(fwhm[pdid], sigma=3.0, maxiters=5.0)
    mean_ra, median_ra, std_ra = sigma_clipped_stats(delta_ra, sigma=3.0, maxiters=5.0)
    mean_dec, median_dec, std_dec = sigma_clipped_stats(delta_dec, sigma=3.0, maxiters=5.0)

    # update the phtotmetric catalog
    ra_true_full = np.zeros(nobj) - 99.0
    dec_true_full = np.zeros(nobj) - 99.0
    ra_true_full[pdid] = rab[rdid]
    dec_true_full[pdid] = decb[rdid]
    photcat.add_columns([ra_true_full, dec_true_full], names=["RA_BASE", "DEC_BASE"])
    photcat["FLUX_AUTO"] = alpha * aflux
    photcat["FLUXERR_AUTO"] = alpha * aflux_err
    photcat.write(output_catalog, format="fits", overwrite=True)
    output_star_region  = output_catalog[:-4] + "star.reg"
    wds9reg(ra[pdid], dec[pdid], radius=5.0, unit="arcsec", color="red", outfile=output_star_region)

    # update image header
    image_matrix, image_header = fits.getdata(output_image, header=True)
    sigbkg = alpha * mad_std(image_matrix, ignore_nan=True)

    image_header["BKGSIG"] = (sigbkg, "background sigma")
    #image_header["GAIN"]     = image_header["GAIN"]/median_scale
    #image_header["SATURATE"] = image_header["SATURATE"]*median_scale
    #image_header["FLXSCL"]   = (median_scale, "median flux scale")
    #image_header["FLXSTD"]   = (std_scale, "uncertainty of median flux scale")
    image_header["GAIN"]     = image_header["GAIN"]/alpha
    image_header["SATURATE"] = image_header["SATURATE"]*alpha
    image_header["FLXSCL"] = (alpha, "median flux scale")
    image_header["FLXSTD"] = (alpha_err, "uncertainty of median flux scale")
    image_header["RA0"] = (ref_header["RA0"], "RA of image center [deg]")
    image_header["DEC0"] = (ref_header["DEC0"], "DEC of image center [deg]")
    image_header["SRA0"] = (ref_header["SRA0"], "RA of image center [hhmmss]")
    image_header["SDEC0"] = (ref_header["SDEC0"], "DEC of image center [ddmmss]")
    image_header["PSCALE"] = (ref_header["PSCALE"], "Image pixel scale [arcsec/pixel]")
    image_header["PIXUN"] = ("adu/s", "Pixel unit")
    image_header["FWHM"] = (median_fwhm, "Image median FWHM [pixel]")
    image_header["RABIAS"] = (median_ra, "Median RA offset of astrometry")
    image_header["DECBIAS"] = (median_dec, "Median DEC offset of astrometry")
    image_header["RASTD"] = (std_ra, "RA rms of astrometry")
    image_header["DECSTD"] = (std_dec, "DEC rms of astrometry")
    image_header["NSTAR"] = (nstar, "Number of high-quality stars")
    image_header["NEWIMG"] = (os.path.basename(output_image), "New image")
    image_header["REFIMG"] = (os.path.basename(ref_image), "Reference image")
    #image_header["TF_VERS"]  = (dbase.__version__, "TransFinder version")
    #image_header["TF_DATE"]  = (dbase.__version_date__, "TransFinder version date")
    #image_header["TF_AUTH"]  = (dbase.__author__, "TransFinder author")

    fits.writeto(output_image, alpha*image_matrix.astype(np.float32), image_header, overwrite=True)
    print(f"^_^ Matched image is generated: {os.path.basename(output_image)}")

    return

def meta_table_iter(image_meta):
    """
    build/update image meta information

    image_meta: str
        meta table name
    """
    # open the table
    if not os.path.exists(image_meta):
        meta = {"image_name": ["U80",  None,         "image name"],
                "ra":         ["f8",   units.deg,    "central ra"],
                "dec":        ["f8",   units.deg,    "central dec"],
                "mu_ra":      ["f8",   units.arcsec, "ra astrometric offset"],
                "mu_dec":     ["f8",   units.arcsec, "dec astrometric offset"],
                "std_ra":     ["f8",   units.arcsec, "ra astrometric std"],
                "std_dec":    ["f8",   units.arcsec, "dec astrometric std"],
                "ra_min":     ["f8",   units.deg,    "minimum ra"],
                "ra_max":     ["f8",   units.deg,    "maximum ra"],
                "dec_min":    ["f8",   units.deg,    "minimum dec"],
                "dec_max":    ["f8",   units.deg,    "maximum dec"],
                "band":       ["U5",   None,         "band/filter"],
                "fwhm":       ["f4",   units.pixel,  "median FWHM"],
                "nstar":      ["i8",   None,         "number of high-snr stars"],
                "std_bkg":    ["f4",   None,         "background std"],
                "image_path":   ["U100", None,       "image path"]}
        colList = []
        for ikey, ival in meta.items():
            idtype, iunit, icom = ival
            if iunit==None:
                icol = Column([], name=ikey, description=icom, dtype=idtype,)
            else:
                icol = Column([], name=ikey, unit=iunit, description=icom, dtype=idtype,)
            colList += [icol]
        image_meta_tab = Table(colList)
    else:
        image_meta_tab = Table.read(image_meta, format="fits")
    
    return image_meta_tab

def interp_badpixel(image_matrix, flag_map=None, mode="random", image_grid=(20,20)):
    """
    Interpolate bad pixels in a given image.
    An important note: input image_matrix is assumed to be a pre-processed image 
    but without subtracting sky background. It means that any pixel with value no
    larger than zero will be bad pixels.

    Generally, a flag map should be provided in which bad pixels are flagged 
    as integers larger than zero.

    If no flag map, pixels with nan value in the 'image_matrix' will be 
    interpolated.

    Parameters:
    image_matrix: array
      input image matrix
    flag_map: array
      input flag map
    mode: str
      method to interpolate the bad pixels
      "random": replace the bad pixels with random values by following N(median_local, mad_std_local)
      "interp": replace the bad pixels with local interpolation
    image_grid: tuple
      image grids
      to accelerate the interpolation, the image are divided into grids
    
    Return:
    interpolated image_matrix
    """
    # find the indices of bad pixels
    if flag_map is None:
        flag_map = np.isnan(image_matrix) + (image_matrix<=0.0)
    else:
        flag_map = (flag_map>0) + np.isnan(image_matrix) + (image_matrix<=0.0)
    
    nbad = np.sum(flag_map)
    if nbad==0:
        print("!!! No bad pixels in the image: nothing to do")
        pass
    else:
        xsize, ysize = image_matrix.shape
        crd_grids = sub_regions(xsize, ysize, image_grid=image_grid)
        kernel = Gaussian2DKernel(x_stddev=1)
        for iid, ibound in crd_grids.items():
            ixcen, iycen, ix0, ix1, iy0, iy1 = ibound[0]
            isub_image = image_matrix[ix0:ix1,iy0:iy1].copy()
            isub_flag = flag_map[ix0:ix1,iy0:iy1].copy()
            
            # extra bad pixels
            imean, imed, istd = sigma_clipped_stats(isub_image[~isub_flag], sigma=3.0, maxiters=5.0, stdfunc="mad_std")
            isub_flag = isub_flag + (isub_image<imed-5.0*istd)
            
            if np.sum(isub_flag)==0: continue
            if mode=="random":
                #imean, imed, istd = sigma_clipped_stats(isub_image[~isub_flag], sigma=3.0, maxiters=5.0, stdfunc="mad_std")
                isub_image[isub_flag] = np.random.normal(imed, istd, size=isub_image.shape)[isub_flag]
            else: # mode=="interp"
                isub_image[isub_flag] = np.nan
                isub_image = interpolate_replace_nans(isub_image, kernel)
            image_matrix[ix0:ix1,iy0:iy1] = isub_image
    return image_matrix

def photometry(input_image, input_weight, output_catalog, sex_exe, sex_config_file, sex_param_file):
    sex_run1 = f"{sex_exe} {input_image} -c {sex_config_file} -WEIGHT_IMAGE {input_weight} "
    sex_run2 = f"-CATALOG_NAME {output_catalog} -PARAMETERS_NAME {sex_param_file}"
    sex_run = sex_run1 + sex_run2
    #print(f"^_^ {sex_run}")
    subprocess.run(sex_run, shell=True)
    return

def flux_scale_plot(star_mag, flux_scale, median_scale, std_scale, photcal_figure):
    xlim     = [np.min(star_mag)-0.5, np.max(star_mag)+0.5]
    sig_ratio = mad_std(flux_scale, ignore_nan=True)
    plt.scatter(star_mag, flux_scale, color="black", marker="o", s=6)
    plt.plot(xlim, [median_scale, median_scale], "r-", linewidth=2.0)
    plt.plot(xlim, [median_scale-std_scale, median_scale-std_scale],"r--",linewidth=1.5)
    plt.plot(xlim, [median_scale+std_scale, median_scale+std_scale],"r--",linewidth=1.5)
    plt.xlim(xlim)
    plt.ylim([median_scale-5.0*sig_ratio, median_scale+5.0*sig_ratio])
    plt.title(f"flux_scale = {median_scale:8.5f} $\pm$ {std_scale:8.5f} (#{len(star_mag)} stars)", fontsize=15)
    plt.savefig(photcal_figure)
    plt.clf()
    plt.close()

