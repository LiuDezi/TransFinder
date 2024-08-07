# This routine is used to build reference/target images for image differencing
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
from ..utils import wds9reg, crossmatch, deg2str, str2deg, sub_regions, poly_index

class BuildImage(object):
    """
    Build reference/target images.

    To do this, the following steps should be used:
    1) build resampled target image
    2) find initial reference image
    3) match initial reference image to resampled target image

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
    interp_grids: tuple
      see function 'interp_badpixel' for more detail
    output_pixel_scale: float
      pixel scale of output image, e.g. output_pixel_scale = 0.3
    output_image_size: tuple
      image size of output image, e.g. output_image_size = (1000, 1000)
    output_image_center: tuple
      image celestial center of output image, 
      e.g. output_image_center = (hh:mm:ss.ss, +dd:mm:ss.ss)
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
        sextractor command. Other parameters are fixed in the configiration file
        """
        sextractor_cmd1 = f"{self.sextractor_exe} {input_image} -c {self.sextractor_config_file} "
        sextractor_cmd2 = f"-CATALOG_NAME {output_catalog} -PARAMETERS_NAME {self.sextractor_param_file} "
        sextractor_cmd3 = f"-WEIGHT_IMAGE {input_weight} "
        sextractor_cmd = sextractor_cmd1 + sextractor_cmd2 + sextractor_cmd3
        return sextractor_cmd

    def swarp_runner(self, input_image, input_weight, output_image, output_weight):
        """
        swarp command. Other parameters are fixed in the configiration file
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
        input_star_crds: list
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
        print("^_^ Build resampled image and corresponding catalog")
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

        # 3) save he bad-pixel interpolated image
        fits.writeto(output_image, image_matrix, image_header, overwrite=True)

        # 4) update resample parameters
        self.resamp_param_update(image_header)
        print("    Resampled parameters:")
        print(f"   1) image center: {self.resamp_image_center}")
        print(f"   2) image size: {self.resamp_image_size}")
        print(f"   3) image pixel scale: {self.resamp_pixel_scale} arcsec/pixel")

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

        # 8) update the catalog
        rab_full, decb_full = np.zeros(nobj) - 99.0, np.zeros(nobj) - 99.0
        rab_full[sid], decb_full[sid] = rab[gid], decb[gid]
        phot_matrix.add_columns([rab_full, decb_full], names=["RA_BASE", "DEC_BASE"])
        phot_matrix.write(output_catalog, format="fits", overwrite=True)
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
    
        return image_matrix, image_header, phot_matrix

    def phot_match(self, tar_meta, ref_meta, method="constant_median", poly_degree=3, photcal_figure=None):
        """
        photometric calibration

        Parameters:
        method: str
          median: sigma-clipped median
          weighted: weighted mean
          fit: polynomial 2D model derived by chi square minimization
        """
        tar_matrix, tar_header, tar_phot = tar_meta
        ref_matrix, ref_header, ref_phot = ref_meta
        
        tar_pid = (tar_phot["RA_BASE"]!=-99.0) & (tar_phot["DEC_BASE"]!=-99.0)
        ref_pid = (ref_phot["RA_BASE"]!=-99.0) & (ref_phot["DEC_BASE"]!=-99.0)
        
        # find common stars
        tar_ra, tar_dec = tar_phot["RA_BASE"][tar_pid], tar_phot["DEC_BASE"][tar_pid]
        ref_ra, ref_dec = ref_phot["RA_BASE"][ref_pid], ref_phot["DEC_BASE"][ref_pid]

        tid, rid = crossmatch(tar_ra, tar_dec, ref_ra, ref_dec, aperture=3.0)
        nstar = len(tid)
        if nstar<200 and method=="fit":
            print(f"!!! Only {nstar} stars for relative photometric calibration.")
            method = "weighted"
            print(f"!!! 'method={method}' will be used instead.")
         
        tar_flux, tar_eflux = tar_phot["FLUX_AUTO"][tar_pid][tid], tar_phot["FLUXERR_AUTO"][tar_pid][tid]
        ref_flux, ref_eflux = ref_phot["FLUX_AUTO"][ref_pid][rid], ref_phot["FLUXERR_AUTO"][ref_pid][rid]

        s = ref_flux/tar_flux
        es = np.sqrt((ref_eflux/tar_flux)**2 + (ref_flux*tar_eflux/(tar_flux**2))**2)

        if method=="median":
            s_masked = sigma_clip(s, sigma=3.0, maxiters=5.0, stdfunc="mad_std", masked=False)
            ns, s_std = len(s_masked), mad_std(s_masked)
            alpha = np.median(s_masked)
            alpha_err = s_std/np.sqrt(ns) * np.sqrt(np.pi*(2*ns+1)/(4*ns))

            # update target image and catalog
            tar_matrix = alpha * tar_matrix
            tar_header["BKGSIG"] = alpha * tar_header["BKGSIG"]
            tar_phot["FLUX_AUTO"] = alpha * tar_phot["FLUX_AUTO"]
            tar_phot["FLUXERR_AUTO"] = alpha * tar_phot["FLUXERR_AUTO"]
        elif method=="weighted":
            alpha = np.sum(s/es**2)/np.sum(1.0/es**2)
            alpha_err = np.sqrt(0.5/np.sum(1.0/es**2))

            # update target image and catalog
            tar_matrix = alpha * tar_matrix
            tar_header["BKGSIG"] = alpha * tar_header["BKGSIG"]
            tar_phot["FLUX_AUTO"] = alpha * tar_phot["FLUX_AUTO"]
            tar_phot["FLUXERR_AUTO"] = alpha * tar_phot["FLUXERR_AUTO"]
        elif method=="fit":
            ysize, xsize = tar_matrix.shape
            xpos = (tar_phot["XWIN_IMAGE"]-0.5)/xsize-0.5
            ypos = (tar_phot["YWIN_IMAGE"]-0.5)/ysize-0.5
            xpos_star = xpos[tar_pid][tid]
            ypos_star = ypos[tar_pid][tid]

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
            alpha_phot = np.zeros(len(tar_phot), dtype=np.float32)
            for i in range(poly_ncoeff):
                ix_index, iy_index = poly_indices[i]
                alpha += poly_coeffs[i] * (xgrid**ix_index) * (ygrid**iy_index)
                alpha_phot += poly_coeffs[i] * (xpos**ix_index) * (ypos**iy_index)
            alpha = alpha.T

            # update target image and catalog
            tar_matrix = alpha * tar_matrix
            tar_header["BKGSIG"] = mad_std(tar_matrix, ignore_nan=True)
            tar_phot["FLUX_AUTO"] = alpha_phot * tar_phot["FLUX_AUTO"]
            tar_phot["FLUXERR_AUTO"] = alpha_phot * tar_phot["FLUXERR_AUTO"]
        else:
            sys.exit(f"!!! Incorrect calibration 'method': method={method}")
        
        if photcal_figure is not None:
            tar_flux = tar_phot["FLUX_AUTO"][tar_pid][tid]
            ref_flux = ref_phot["FLUX_AUTO"][ref_pid][rid]
            self.flux_scale_plot(ref_flux, tar_flux, alpha, photcal_figure)

        return

    def achieve_ref(self, input_meta, band):
        """
        find reference image in a given meta data through matching band and central celestial coordinate
        
        """
        ra_center, dec_center = self.resamp_image_center
        xsize, ysize = self.resamp_image_size
        ra_center, dec_center = str2deg(ra_center, dec_center)
        
        hx_arcsec= 0.5 * xsize * self.resamp_pixel_scale
        hy_arcsec= 0.5 * ysize * self.resamp_pixel_scale
        match_aperture = np.min([hx_arcsec, hy_arcsec])

        # find the reference image
        metatab = Table.read(input_meta, format="fits")
        bid = metatab["band"]==band
        if len(bid)==0: sys.exit(f"!!! No {band}-band image found in {os.path.basename(input_meta)}")
        ra_ref, dec_ref = metatab["ra"][bid], metatab["dec"][bid]
        rid, iid = crossmatch(ra_ref, dec_ref, [ra_center], [dec_center], aperture=match_aperture)
        if len(rid)==0: sys.exit(f"!!! No reference image found in {os.path.basename(input_meta)}")
        
        ref_image_name = metatab["image_name"][bid][rid[0]]
        ref_image_path = metatab["image_path"][bid][rid[0]]
        ref_image  = os.path.join(ref_image_path, ref_image_name)
        return ref_image

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

    def flux_scale_plot(self, ref_flux, tar_flux, alpha, photcal_figure):
        star_mag = -2.5*np.log10(ref_flux)
        xlim     = [np.min(star_mag)-0.5, np.max(star_mag)+0.5]
        flux_ratio = tar_flux/ref_flux
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

