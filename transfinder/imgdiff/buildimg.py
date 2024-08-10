# This routine is used to build reference/new images for image differencing
# NOTE: there are several assumptions for all input images:
# 1) images are pre-processed, including bias/dark correction, flat-fielding, 
#    astrometric calibration, and photometric homogenization. 
#    Other reductions are not required but will be helpful if provided, which 
#    are cosmic-ray removal, bad pixel masking, gain correction, etc.
# 2) images are in FITS format.
# 3) A list of star celestial coordinates should be provided because the routine 
#    will not perform star selection seperately. This is crucial for PSF modeling.
#    Stars from Gaia should be sufficient for this goal because this routine 
#    believes that Gaia star catalog will always be available for most modern 
#    surveys. Alternatively, you can construct a star catalog by your own method.

# This routine will resample the image based on the astrometric solution in 
# image header and subtract sky background. SWarp will be used.

import numpy as np
from scipy import optimize
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import wcs
from astropy.stats import sigma_clip, sigma_clipped_stats, mad_std
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import matplotlib.pyplot as plt
import os
import sys
import subprocess

from .base import BaseCheck
from ..utils import wds9reg, crossmatch, deg2str, str2deg, sub_regions, poly_index

__all__ = ["image_resamp", "phot_match", "saturation_mask"]

class BuildImage(object):
    """
    Build resampled reference/new images.

    User should provide two input images: a reference image and a new image. 
    Generally, the two images should have the same observed filter and similar 
    sky coverage. But their exposure time, celestial centers, and other 
    geometric parameters can be different. This routine will resample the two 
    images so that they will have exactly the same sky coverage and image size 
    (either specified by the user or estimated from the new image), and perform 
    relative photometric calibration.

    Reference/new image should be built by following below steps:
    1) build resampled new image
    2) align reference image to the resample new image
    3) match the relative flux scale of new to reference
    4) save the images and corresponding catalog

    NOTE: SExtractor and SWarp should be installed for photometry and image resampling
    
    Parameters:
    swarp_config_file: file
      swarp configuration file
    sextractor_config_file: file
      sextractor configuration file
    sextractor_param_file: file
      sextractor output parameter file
    swarp_exe: str
      executable swarp command in terminal
    sextractor_exe: str
      executable sextractor command in terminal
    resamp_pixel_scale: float
      pixel scale of output image, e.g. output_pixel_scale = 0.3
    resamp_image_size: tuple
      image size of output image, e.g. output_image_size = (1000, 1000)
    resamp_image_center: tuple
      image celestial center of output image, 
      e.g. output_image_center = (hh:mm:ss.ss, +dd:mm:ss.ss)
    interp_grids: tuple
      see function 'interp_badpixel' for more detail
    """
    def __init__(self,
        swarp_config_file, 
        sextractor_config_file, 
        sextractor_param_file, 
        swarp_exe = "swarp",
        sextractor_exe = "source-extractor",
        resamp_pixel_scale = None,
        resamp_image_size = None,
        resamp_image_center = None,
        interp_grids = (30,30),
        ):
        self.swarp_config_file = swarp_config_file
        self.sextractor_config_file = sextractor_config_file
        self.sextractor_param_file = sextractor_param_file
        self.swarp_exe = swarp_exe
        self.sextractor_exe = sextractor_exe
        self.resamp_pixel_scale = resamp_pixel_scale
        self.resamp_image_size = resamp_image_size
        self.resamp_image_center = resamp_image_center
        self.interp_grids = interp_grids

    def resamp_param_update(self, image_header):
        """
        If some resampling parameters are None, update them
        """
        image_wcs = wcs.WCS(image_header)
        xsize, ysize = image_wcs.pixel_shape
        if self.resamp_image_center is None:
            ximg_center, yimg_center = 0.5*(xsize+1), 0.5*(ysize+1)
            ra_center, dec_center = image_wcs.all_pix2world(ximg_center, yimg_center, 1)
            ra_center, dec_center = ra_center.tolist(), dec_center.tolist()
            sra_center, sdec_center = deg2str(ra_center, dec_center)
            self.resamp_image_center = (sra_center[0], sdec_center[0])
        if self.resamp_pixel_scale is None:
            pixel_scale = np.mean([ips.value*3600.0 for ips in image_wcs.proj_plane_pixel_scales()])
            self.resamp_pixel_scale = float(f"{pixel_scale:.2f}")
        if self.resamp_image_size is None:
            self.resamp_image_size = (xsize, ysize)
        return

    def sextractor_runner(self, input_image, input_weight, output_catalog):
        """
        sextractor command. Other parameters are fixed in the configuration file
        """
        sextractor_cmd1 = f"{self.sextractor_exe} {input_image} -c {self.sextractor_config_file} "
        sextractor_cmd2 = f"-CATALOG_NAME {output_catalog} -PARAMETERS_NAME {self.sextractor_param_file} "
        sextractor_cmd3 = f"-WEIGHT_IMAGE {input_weight} "
        sextractor_cmd = sextractor_cmd1 + sextractor_cmd2 + sextractor_cmd3
        return sextractor_cmd

    def swarp_runner(self, input_image, input_weight, output_image, output_weight):
        """
        swarp command. Other parameters are fixed in the configuration file
        """
        resamp_center = f"{self.resamp_image_center[0]},{self.resamp_image_center[1]}"
        resamp_size = f"{self.resamp_image_size[0]},{self.resamp_image_size[1]}"
        swarp_cmd1 = f"{self.swarp_exe} {input_image} -c {self.swarp_config_file} "
        swarp_cmd2 = f"-IMAGEOUT_NAME {output_image} -WEIGHTOUT_NAME {output_weight} "
        swarp_cmd3 = f"-CENTER {resamp_center} -PIXEL_SCALE {self.resamp_pixel_scale} -IMAGE_SIZE {resamp_size} "
        swarp_cmd4 = "-WEIGHT_TYPE NONE "
        if input_weight is not None: swarp_cmd4 = f"-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE {input_weight} "
        swarp_comd = swarp_cmd1 + swarp_cmd2 + swarp_cmd3 + swarp_cmd4
        return swarp_comd

    def image_resamp(self, input_image, input_star_positions, output_image, input_weight = None, input_flag = None,):
        """
        Resample the image based on the astrometric solution in the image header.
        Here 'Input_image' is a pre-processed science image without sky-background
        subtraction.

        New image can be built by this function.

        Parameters:
        input_image: str
          input FITS image with absolute path
        input_star_positions: list
          input star coordinates, e.g. input_star_crds=[ra_array, dec_array]
        output_image: str
          output resampled FITS image with absolute path
        input_weight: str
          same format as input_image
        input_flag: str
          same format as input_image, for bad pixels interpolation
    
        Return:
        resampled image and corresponding photometric catalog
        """
        base_check = BaseCheck()
        base_check.header_check(input_image)

        # 0) basic setup
        output_catalog = output_image[:-4] + "phot.fits"
        output_region = output_catalog[:-4] + "star.reg"
        output_weight = output_image[:-4] + "weight.fits"

        print(f"    Input image: {os.path.basename(input_image)}")
        print(f"    Resampled image: {os.path.basename(output_image)}")
        print(f"    Resampled catalog: {os.path.basename(output_catalog)}")
    
        # 1) load input image matrix and header
        image_matrix, image_header = fits.getdata(input_image, header=True)

        # 2) interpolate bad pixels
        flag_matrix = None
        if input_flag is not None: flag_matrix = fits.getdata(input_flag)
        image_matrix = self.interp_badpixel(image_matrix, flag_map=flag_matrix, grids=self.interp_grids)

        # 3) save the bad-pixel interpolated image
        fits.writeto(output_image, image_matrix, image_header, overwrite=True)

        # 4) update resample parameters
        self.resamp_param_update(image_header)
        print("    Resampled parameters:")
        print(f"    1) image center: {self.resamp_image_center}")
        print(f"    2) image size: {self.resamp_image_size}")
        print(f"    3) image pixel scale: {self.resamp_pixel_scale} arcsec/pixel")

        # 5) run swarp and perform photometry
        swarp_runner  = self.swarp_runner(output_image, input_weight, output_image, output_weight)
        subprocess.run(swarp_runner, shell=True)
        sextractor_runner = self.sextractor_runner(output_image, output_weight, output_catalog)
        subprocess.run(sextractor_runner, shell=True)
        if input_weight is None: os.remove(output_weight)

        # 6) load star catalog with good photometric quality
        phot_matrix = Table.read(output_catalog, format="fits", hdu=2)
        nobj = len(phot_matrix)
        ra, dec = phot_matrix["ALPHA_J2000"], phot_matrix["DELTA_J2000"]
        flags, snr, fwhm = phot_matrix["FLAGS"], phot_matrix["SNR_WIN"], phot_matrix["FWHM_IMAGE"]
        sid = (flags==0) & (snr>20) & (snr<1000) & (fwhm>0.0)
    
        rab, decb = input_star_positions
        pid, gid = crossmatch(ra[sid], dec[sid], rab, decb, aperture=3.0)
        nstar = len(pid)
        if nstar<20: sys.exit(f"!!! At least 20 stars are required (20<SNR<1000). Only {nstar} stars are found")
        sid = np.where(sid==True)[0][pid]

        delta_ra = (rab[gid] - ra[sid])*np.cos(decb[gid]*np.pi/180.0)*3600.0
        delta_dec = (decb[gid] - dec[sid])*3600.0
        mean_ra, median_ra, std_ra = sigma_clipped_stats(delta_ra, sigma=3.0, maxiters=5.0)
        mean_dec, median_dec, std_dec = sigma_clipped_stats(delta_dec, sigma=3.0, maxiters=5.0)

        # 7) update the catalog
        star_flag, rab_full, decb_full = np.zeros(nobj, dtype=int), np.full(nobj, -99.0), np.full(nobj, -99.0)
        star_flag[sid], rab_full[sid], decb_full[sid] = 1, rab[gid], decb[gid]

        phot_matrix.add_columns([star_flag, rab_full, decb_full], names=["FLAG_STAR", "RA_STAR", "DEC_STAR"])
        phot_matrix.write(output_catalog, format="fits", overwrite=True)
        wds9reg(ra[sid], dec[sid], radius=5.0, unit="arcsec", color="red", outfile=output_region)

        # 8) update image header
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
        sra_center, sdec_center = self.resamp_image_center

        oimage_name = os.path.basename(output_image)
        oimage_path = os.path.dirname(output_image)

        image_header["BKGSIG"]  = (std_bkg,            "background sigma")
        image_header["RA0"]     = (ra_center,          "RA of image center [deg]")
        image_header["DEC0"]    = (dec_center,         "DEC of image center [deg]")
        image_header["SRA0"]    = (sra_center,         "RA of image center [hhmmss]")
        image_header["SDEC0"]   = (sdec_center,        "DEC of image center [ddmmss]")
        image_header["RAMIN"]   = (ra_min,             "RA lower limit")
        image_header["RAMAX"]   = (ra_max,             "RA upper limit")
        image_header["DECMIN"]  = (dec_min,            "DEC lower limit")
        image_header["DECMAX"]  = (dec_max,            "DEC upper limit")
        image_header["PSCALE"]  = (self.resamp_pixel_scale, "Image pixel scale [arcsec/pixel]")
        image_header["FWHM"]    = (median_fwhm,        "Image median FWHM [pixel]")
        image_header["RABIAS"]  = (median_ra,          "Median RA offset of astrometry")
        image_header["DECBIAS"] = (median_dec,         "Median DEC offset of astrometry")
        image_header["RASTD"]   = (std_ra,             "RA rms of astrometry")
        image_header["DECSTD"]  = (std_dec,            "DEC rms of astrometry")
        image_header["NSTAR"]   = (nstar,              "Number of high-quality stars")
        image_header["IMGNAME"] = (oimage_name,        "Image name")

        image_matrix = image_matrix.astype(np.float32)
        fits.writeto(output_image, image_matrix, image_header, overwrite=True)
    
        return output_image, output_catalog, image_header, image_matrix, phot_matrix

    def phot_match(self, new_meta, ref_meta, method="median", poly_degree=3, photcal_figure=None):
        """
        photometric calibration

        Parameters:
        new_meta: tuple
          meta data of new image generated by function 'image_resamp'
        ref_meta: tuple
          meta data of ref image generated by function 'image_resamp'
        method: str
          median: sigma-clipped median
          weighted: weighted mean
          fit: polynomial 2D model derived by chi square minimization
        poly_degree: int
          if method=fit, this is the degree of a 2D polynomial.
        photcal_figure: str
          save comparison figure of final calibrated fluxes of new and ref.
        """
        new_image, new_catalog, new_header, new_matrix, new_phot = new_meta
        ref_image, ref_catalog, ref_header, ref_matrix, ref_phot = ref_meta
        
        new_pid = new_phot["FLAG_STAR"]==1
        ref_pid = ref_phot["FLAG_STAR"]==1
        
        # find common stars
        new_ra, new_dec = new_phot["RA_STAR"][new_pid], new_phot["DEC_STAR"][new_pid]
        ref_ra, ref_dec = ref_phot["RA_STAR"][ref_pid], ref_phot["DEC_STAR"][ref_pid]

        tid, rid = crossmatch(new_ra, new_dec, ref_ra, ref_dec, aperture=3.0)
        nstar = len(tid)
        if nstar<200 and method=="fitted":
            print(f"!!! Only {nstar} stars for relative photometric calibration.")
            method = "weighted"
            print(f"!!! 'method={method}' will be used instead.")
         
        new_flux, new_eflux = new_phot["FLUX_AUTO"][new_pid][tid], new_phot["FLUXERR_AUTO"][new_pid][tid]
        ref_flux, ref_eflux = ref_phot["FLUX_AUTO"][ref_pid][rid], ref_phot["FLUXERR_AUTO"][ref_pid][rid]

        s = ref_flux/new_flux
        es = np.sqrt((ref_eflux/new_flux)**2 + (ref_flux*new_eflux/(new_flux**2))**2)

        if method=="median":
            s_masked = sigma_clip(s, sigma=3.0, maxiters=5.0, stdfunc="mad_std", masked=False)
            ns, s_std = len(s_masked), mad_std(s_masked)
            alpha = np.median(s_masked)
            alpha_err = s_std/np.sqrt(ns) * np.sqrt(np.pi*(2*ns+1)/(4*ns))

            # update new image and catalog
            new_matrix = alpha * new_matrix
            new_header["BKGSIG"] = alpha * new_header["BKGSIG"]
            new_phot["FLUX_AUTO"] = alpha * new_phot["FLUX_AUTO"]
            new_phot["FLUXERR_AUTO"] = alpha * new_phot["FLUXERR_AUTO"]
        elif method=="weighted":
            alpha = np.sum(s/es**2)/np.sum(1.0/es**2)
            alpha_err = np.sqrt(0.5/np.sum(1.0/es**2))

            # update new image and catalog
            new_matrix = alpha * new_matrix
            new_header["BKGSIG"] = alpha * new_header["BKGSIG"]
            new_phot["FLUX_AUTO"] = alpha * new_phot["FLUX_AUTO"]
            new_phot["FLUXERR_AUTO"] = alpha * new_phot["FLUXERR_AUTO"]
        elif method=="fitted":
            xsize, ysize = self.resamp_image_size
            xpos = (new_phot["XWIN_IMAGE"]-0.5)/xsize-0.5
            ypos = (new_phot["YWIN_IMAGE"]-0.5)/ysize-0.5
            xpos_star = xpos[new_pid][tid]
            ypos_star = ypos[new_pid][tid]

            # estimate polynomial coefficients
            poly_indices = poly_index(poly_degree)
            poly_ncoeff = len(poly_indices)
            crd_matrix = np.zeros((poly_ncoeff, nstar), dtype=np.float32)
            for i in range(poly_ncoeff):
                ix_index, iy_index = poly_indices[i]
                crd_matrix[i,:] = (xpos_star**ix_index) * (ypos_star**iy_index)
            init_coeffs = np.ones(poly_ncoeff)
            poly_coeffs, pcov = optimize.curve_fit(self.poly_model, crd_matrix, s, p0=init_coeffs, sigma=es)
            poly_coeffs_err = np.sqrt(np.diag(pcov))
            
            # construct flux scaling field
            xgrid, ygrid = self.interp_grid(xsize, ysize)
            xgrid, ygrid = (xgrid+0.5)/xsize-0.5, (ygrid+0.5)/ysize-0.5
            alpha = np.zeros((xsize, ysize), dtype=np.float32)
            alpha_phot = np.zeros(len(new_phot), dtype=np.float32)
            for i in range(poly_ncoeff):
                ix_index, iy_index = poly_indices[i]
                alpha += poly_coeffs[i] * (xgrid**ix_index) * (ygrid**iy_index)
                alpha_phot += poly_coeffs[i] * (xpos**ix_index) * (ypos**iy_index)
            alpha = alpha.T

            # update new image and catalog
            new_matrix = alpha * new_matrix
            new_header["BKGSIG"] = mad_std(new_matrix, ignore_nan=True)
            new_phot["FLUX_AUTO"] = alpha_phot * new_phot["FLUX_AUTO"]
            new_phot["FLUXERR_AUTO"] = alpha_phot * new_phot["FLUXERR_AUTO"]
        else:
            sys.exit(f"!!! Incorrect calibration 'method': method={method}")
        
        # save the matched new image and catalog
        fits.writeto(new_image, new_matrix, new_header, overwrite=True)
        new_phot.write(new_catalog, format="fits", overwrite=True)

        if photcal_figure is not None:
            new_flux = new_phot["FLUX_AUTO"][new_pid][tid]
            ref_flux = ref_phot["FLUX_AUTO"][ref_pid][rid]
            self.flux_scale_plot(ref_flux, new_flux, alpha, photcal_figure)
        
        return new_image, new_catalog, new_header, new_matrix, new_phot

    def saturation_mask(self, photcat, mask_scale=1.5):
        """
        Calculate the circular masking regions of saturated stars

        Parameters:
        photcat: array
          photometric catalog
        mask_scale: float
          scale parameter applied to the original mask circular

        Return:
          Bool matrix with masking pixels assigned as True
        """
        saturate_param = self.select_saturated_object(photcat)
        xsize, ysize = self.resamp_image_size

        mask_matrix = np.full((ysize,xsize),False)
        xstep, ystep = np.ogrid[:xsize, :ysize]
        for sid, ipar in saturate_param.items():
            ixcen, iycen, irad = ipar
            irad = irad * mask_scale

            # extract the subimage to speed up
            irad_int = int(np.ceil(irad))
            ixcen_int, iycen_int = int(round(ixcen-1)), int(round(iycen-1))
            ix0, ix1 = ixcen_int-irad_int, ixcen_int+irad_int
            iy0, iy1 = iycen_int-irad_int, iycen_int+irad_int

            ixstep_sub, iystep_sub = xstep[ix0:ix1+1,:], ystep[:,iy0:iy1+1]
            ixcen_new, iycen_new = np.median(ixstep_sub), np.median(iystep_sub)
            idist = np.sqrt((ixstep_sub-ixcen_new)**2 + (iystep_sub-iycen_new)**2)
            mask_matrix[iy0:iy1+1,ix0:ix1+1] += idist.T <= irad
            
            # classical method is slow in the case of many saturated stars
            #idist = np.sqrt((xstep-ixcen+1.0)**2 + (ystep-iycen+1.0)**2)
            #mask += idist <= irad
        return mask_matrix

    def select_saturated_object(self, photcat):
        """
        Find the saturated stars in a given catalog.
        The catalog is generated by Sextractor in LDAC format, in which 
        the X/Y_Image and FLAGS should provided
        """
        ximg, yimg = photcat["XWIN_IMAGE"], photcat["YWIN_IMAGE"]
        xmin, xmax = photcat["XMIN_IMAGE"], photcat["XMAX_IMAGE"]
        ymin, ymax = photcat["YMIN_IMAGE"], photcat["YMAX_IMAGE"]
        flag = photcat["FLAGS"]
        nobj = len(photcat)

        #find the saturated stars
        nstar   = 0
        saturate_param = {}
        for i in range(nobj):
            iflag = int(flag[i])
            ipow  = self.decomp_flags(iflag)
            if 4 not in ipow: continue
            ixmax = np.max([ximg[i]-xmin[i], xmax[i]-ximg[i]])
            iymax = np.max([yimg[i]-ymin[i], ymax[i]-yimg[i]])
            irad  = np.max([ixmax, iymax])
            nstar += 1
            saturate_param[i] = [ximg[i],yimg[i],irad]

        print(f"    Total {nstar} saturated stars")
        return saturate_param

    def decomp_flags(self, flag):
        """
        Decompose Sextractor flags into single numbers of power 2
    
        Parameter:
        flag: int
              Sextractor FLAGS
        Example: 
            pow = decompFlags(25)
            pow = [1, 8, 16]
        """
        assert type(flag)==int, "!!! flag must be integer."
        powers = []
        spow = 1
        while spow <= flag:
            if spow & flag: powers.append(spow)
            spow <<= 1
        return powers

    def interp_badpixel(self, image_matrix, flag_map=None, grids=(20,20)):
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
        grids: tuple
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
            crd_grids = sub_regions(xsize, ysize, grids=grids)
            kernel = Gaussian2DKernel(x_stddev=1)
            for iid, ibound in crd_grids.items():
                ixcen, iycen, ix0, ix1, iy0, iy1 = ibound[0]
                isub_image = image_matrix[ix0:ix1,iy0:iy1].copy()
                isub_flag = flag_map[ix0:ix1,iy0:iy1].copy()
                
                # extra bad pixels
                imean, imed, istd = sigma_clipped_stats(isub_image[~isub_flag], sigma=3.0, maxiters=5.0, stdfunc="mad_std")
                isub_flag = isub_flag + (isub_image<imed-5.0*istd)
                if np.sum(isub_flag)==0: continue
                isub_image[isub_flag] = np.nan
                isub_image = interpolate_replace_nans(isub_image, kernel)
                image_matrix[ix0:ix1,iy0:iy1] = isub_image
        return image_matrix

    def interp_grid(self, xsize, ysize):
        """ 
        grid of coordinates used for coefficient field construction
        """
        xgrid, ygrid = np.mgrid[:xsize, :ysize]
        return xgrid, ygrid

    def poly_model(self, crd_matrix, *coeffs_list):
        coeff_matrix = np.array(coeffs_list).reshape(1, -1)
        model = np.matmul(coeff_matrix, crd_matrix)
        return model.flatten()

    def flux_scale_plot(self, ref_flux, new_flux, alpha, photcal_figure):
        star_mag = -2.5*np.log10(ref_flux)
        xlim     = [np.min(star_mag)-0.5, np.max(star_mag)+0.5]
        flux_ratio = new_flux/ref_flux
        smean, smed, sstd = sigma_clipped_stats(flux_ratio, sigma=3.0, maxiters=5.0, stdfunc="mad_std")
        line_label = f"flux_ratio = {smed:7.5f} $\pm$ {sstd:7.5f}"
        title_label = f"flux_scale = {np.mean(alpha):8.5f} (#{len(ref_flux)} stars)"
        plt.scatter(star_mag, flux_ratio, color="black", marker="o", s=6)
        plt.plot(xlim, [smed, smed], "r-", linewidth=2.0, label=line_label)
        plt.xlim(xlim)
        plt.ylim([smed-5.0*sstd, smed+5.0*sstd])
        plt.title(title_label, fontsize=15)
        plt.legend()
        plt.savefig(photcal_figure)
        plt.clf()
        plt.close()

