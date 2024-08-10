# detect transient candidates from difference images:
# there are two difference images: one is New-Ref, the other is Ref-New

# general modules
import numpy as np
from astropy import units
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import wcs
from astropy.time import Time
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
import os, sys
import subprocess

# self-defined modules
from ..utils import crossmatch, wds9reg, deg2str
from .metatable import InitMetaTable

class ExtractTrans(object):
    """
    extract transient candidates from the direct/inverse difference image
    
    trans_meta_table: "mephisto_transient_metadata.fits"
    """
    def __init__(self,
        sex_config_file, sex_param_file,
        sex_exe = "sextractor",
        detect_thresh = 2.0,
        analysis_thresh = 2.0,
        detect_minarea = 5,
        trans_stamp_size = 49,
        deblend_aper = None,
        trans_meta_update = False,
        trans_meta_path = None,
        trans_meta_table = "mephisto_transient_metadata.fits",
        ):
        self.sex_config_file = sex_config_file
        self.sex_param_file = sex_param_file
        self.sex_exe = sex_exe
        self.detect_thresh = detect_thresh
        self.analysis_thresh = analysis_thresh
        self.detect_minarea = detect_minarea
        self.trans_stamp_size = trans_stamp_size
        self.deblend_aper = deblend_aper
        self.trans_meta_update = trans_meta_update
        self.trans_meta_path = trans_meta_path

        if trans_meta_path is not None:
            trans_meta_table = os.path.join(trans_meta_path, trans_meta_table)
        self.trans_meta_table = trans_meta_table

    def photometry(self, image_name, catalog_name, check=False):
        """
        photometry by sextractor
        """
        sexcmd1 = f"{self.sex_exe} {image_name} -c {self.sex_config_file} -PARAMETERS_NAME {self.sex_param_file} "
        sexcmd2 = f"-DETECT_THRESH {self.detect_thresh} -ANALYSIS_THRESH {self.analysis_thresh} "
        sexcmd3 = f"-DETECT_MINAREA {self.detect_minarea} "
        sexcmd4 = f"-CATALOG_NAME {catalog_name} -WEIGHT_TYPE NONE -CHECKIMAGE_TYPE NONE "
        
        if check:
            check_name = image_name[:-4] + "check.fits"
            sexcmd4 = f"-CATALOG_NAME {catalog_name} -WEIGHT_TYPE NONE -CHECKIMAGE_TYPE APERTURES -CHECKIMAGE_NAME {check_name} "
            
        sexcmd  = sexcmd1 + sexcmd2 + sexcmd3 + sexcmd4
        subprocess.run(sexcmd, shell=True)
        return

    def extract_trans(self, diff_image, diff_matrix, ref_meta, new_meta, 
                      cutout_write=False, cutout_path=None):
        """
        extract transient stamps from the difference image
        """
        print("^_^ Detect transient candidates from difference image")
        # set direct/inverse difference images
        diff_inv_image = diff_image[:-4] + "inverse.fits"
        diff_mode = {1:diff_image, -1:diff_inv_image}
        
        meta_obj = InitMetaTable(stamp_size=self.trans_stamp_size)
        # initialize meta table
        if self.trans_meta_update:
            main_meta = meta_obj.meta_table_iter(self.trans_meta_table, mode="main")
        
        # estimate typical positional uncertainty for deblending. I know that 
        # astrometry can also lead to positional uncertainty, but it is much 
        # smaller than that induced by fwhm.
        pixel_scale = new_meta.image_header["PSCALE"]
        if self.deblend_aper is None:
            ref_fwhm = ref_meta.image_header["FWHM"]
            new_fwhm = new_meta.image_header["FWHM"]
            upos = int(np.ceil(np.sqrt(ref_fwhm**2+new_fwhm**2)*1.2739827))
        else:
            upos = self.deblend_aper / pixel_scale
        upos_sky = upos * pixel_scale
       
        # check if trans_stamp_size is a good value
        half_stamp_size = self.trans_stamp_size//2
        inner_aper = half_stamp_size-5
        if upos>=inner_aper:
            print("!!! 'trans_stamp_size'={self.trans_staimp_size} is too smaller")
            sys.exit(f"!!! set it to 'trans_stamp_size'={self.trans_staimp_size+12} or larger ???")

        # load some useful information
        band = new_meta.image_header["FILTER"]
        image_wcs = wcs.WCS(new_meta.image_header)
        
        # configuration for aperture photometry
        kernel = Gaussian2DKernel(x_stddev=1)
        phot_pos = (half_stamp_size, half_stamp_size)
        phot_aper = CircularAperture(phot_pos, r=upos)
        inner_aper = np.max([half_stamp_size-5, upos])
        phot_annu = CircularAnnulus(phot_pos, r_in=inner_aper, r_out=half_stamp_size)

        # start transient extraction
        diff_trans_table = diff_image[:-5] + "_trans.fits"
        diff_trans_iter = meta_obj.meta_table_iter(diff_trans_table, mode="trans")
        crds_ra, crds_dec, ntrans = np.array([]), np.array([]), 0
        for idiff_mode, idiff_image in diff_mode.items():
            # first save difference image
            if idiff_mode==1:
                fits.writeto(idiff_image, diff_matrix.T, new_meta.image_header, overwrite=True)
                region_color = "red"
            else:
                fits.writeto(idiff_image, 0.0-diff_matrix.T, new_meta.image_header, overwrite=True)
                region_color = "green"

            # then photometry
            idiff_catalog = idiff_image[:-4] + "ldac"
            self.photometry(idiff_image, idiff_catalog, check=False)
            
            # load the photometric catalog
            idiff_catalog_matrix = Table.read(idiff_catalog, format="fits", hdu=2)
            inobj = len(idiff_catalog_matrix)
            print(f"    Total {inobj} objects detected on difference image: {idiff_mode}")
            
            # save the region file for check
            idiff_regn = idiff_image[:-4] + "reg"
            ira, idec = idiff_catalog_matrix["ALPHA_J2000"], idiff_catalog_matrix["DELTA_J2000"]
            wds9reg(ira,idec,radius=5.0,unit="arcsec",color=region_color,outfile=idiff_regn)
            
            # remove temporary files
            os.remove(idiff_catalog)
            if idiff_mode==-1: os.remove(idiff_image)

            for j in range(inobj):
                jobj_param = idiff_catalog_matrix[j]
                jximg = int(round(jobj_param["XWIN_IMAGE"]-1.0))
                jyimg = int(round(jobj_param["YWIN_IMAGE"]-1.0))
                
                jx0, jx1 = jximg - upos, jximg + upos
                jy0, jy1 = jyimg - upos, jyimg + upos
                
                if jx0<=0 or jy0<=0: continue
                if jx1>new_meta.xsize or jy1>new_meta.ysize: continue

                # get the stamp
                jref_stm = ref_meta.image_matrix[jx0:jx1+1,jy0:jy1+1]
                jnew_stm = new_meta.image_matrix[jx0:jx1+1,jy0:jy1+1]
                jref_stm_conv = convolve(jref_stm, kernel)
                jnew_stm_conv = convolve(jnew_stm, kernel)
                jref_maxid = np.unravel_index(np.argmax(jref_stm_conv, axis=None), jref_stm_conv.shape)
                jnew_maxid = np.unravel_index(np.argmax(jnew_stm_conv, axis=None), jnew_stm_conv.shape) 
                jmaxid = [jref_maxid, jnew_maxid]

                jref_bias, jnew_bias = np.array(jref_maxid)-upos, np.array(jnew_maxid)-upos
                jref_bias, jnew_bias = np.sqrt(sum(jref_bias**2)), np.sqrt(sum(jnew_bias**2))
                jbias = [jref_bias, jnew_bias]

                min_bias, min_id = np.min(jbias), np.argmin(jbias)
                if min_bias>upos:
                    print(f"    Object {j+1} is an alien, it is rejected")
                    continue
                jx_peak, jy_peak = jmaxid[min_id]
                jx_peak = jx_peak - upos + jximg
                jy_peak = jy_peak - upos + jyimg
                jra_peak, jdec_peak = image_wcs.all_pix2world(jx_peak, jy_peak, 0)
                jra_peak, jdec_peak = jra_peak.tolist(), jdec_peak.tolist()

                if len(crds_ra)==0:
                    crds_ra = np.append(crds_ra, jra_peak)
                    crds_dec = np.append(crds_dec, jdec_peak)
                else:
                    pid, gid = crossmatch(crds_ra, crds_dec, [jra_peak], [jdec_peak], aperture=upos_sky)
                    if len(pid)!=0:
                        continue
                    else:
                        crds_ra = np.append(crds_ra, jra_peak)
                        crds_dec = np.append(crds_dec, jdec_peak)
                
                # save the stamp
                jx0, jx1 = jx_peak - half_stamp_size, jx_peak + half_stamp_size
                jy0, jy1 = jy_peak - half_stamp_size, jy_peak + half_stamp_size
                if jx0<=0 or jy0<=0: continue
                if jx1>new_meta.xsize or jy1>new_meta.ysize: continue

                jdiff_matrix = diff_matrix[jx0:jx1+1,jy0:jy1+1]
                jref_matrix = ref_meta.image_matrix[jx0:jx1+1,jy0:jy1+1]
                jnew_matrix = new_meta.image_matrix[jx0:jx1+1,jy0:jy1+1]
                
                # perform aperture photometry
                jdiff_aper = ApertureStats(jdiff_matrix, phot_aper)
                jdiff_annu = ApertureStats(jdiff_matrix, phot_annu)
                jdiff_flux = jdiff_aper.sum - jdiff_annu.median * (np.pi * upos**2)
                jdiff_sig = jdiff_annu.mad_std * np.sqrt(np.pi * upos**2)
                jdiff_snr = abs(jdiff_flux/jdiff_sig)
                jdiff_fwhm = jdiff_aper.fwhm.value
                if np.isnan(jdiff_fwhm): jdiff_fwhm = -99.0

                jra_hms, jdec_dms = deg2str(jra_peak, jdec_peak)
                jtrans_id = f"J{jra_hms[1]}{jdec_dms[2]}"
                
                if cutout_write:
                    jstm_cutout = f"MTC_J{jra_hms[1]}{jdec_dms[3]}_cutout.fits"
                    if cutout_path is not None:
                        jstm_cutout = os.path.join(cutout_path, jstm_cutout)

                    # cutout header
                    jhdr = fits.Header()
                    jhdr["RA"]      = jra_peak
                    jhdr["DEC"]     = jdec_peak
                    jhdr["FILE"]    = jstm_cutout
                    jhdr["XPOS"]    = jx_peak
                    jhdr["YPOS"]    = jx_peak
                    jhdr["TRANSID"] = (jtrans_id, "candidate id")
                    jhdr["FILTER"]  = (band, "filter")
                    jhdr["FLUX"]    = (jdiff_flux, f"aperture flux wirh r={upos} pixels")
                    jhdr["SNR"]     = (jdiff_snr, "snr for aperture flux")
                    jhdr["FWHM"]    = (jdiff_fwhm, "FWHM in pixels")
                    jhdr["DIFFIMG"] = (os.path.basename(diff_image), "diff image")
                    jhdr["DIFFMODE"] = (idiff_mode, "mode: -1=inverse diff; 1=direct diff")

                    jhdr["NEWIMG"]  = (new_meta.image_header["NEWIMG"], "new image")
                    jhdr["NEWDATE"] = (new_meta.image_header["DATE"], "date of new image")
                    jhdr["NEWFWHM"] = (new_meta.image_header["FWHM"], "FWHM of new image in pixels")
                    jhdr["NEWMRA"]  = (new_meta.image_header["RABIAS"], "RA astrometric offset of new image")
                    jhdr["NEWMDEC"] = (new_meta.image_header["DECBIAS"], "DEC astrometric offset of new image")
                    jhdr["NEWSRA"]  = (new_meta.image_header["RASTD"], "RA astrometric rms of new image")
                    jhdr["NEWSDEC"] = (new_meta.image_header["DECSTD"], "DEC astrometric rms of new image")

                    jhdr["REFIMG"]  = (ref_meta.image_header["REFIMG"], "ref image")
                    jhdr["REFDATE"] = (ref_meta.image_header["DATE"], "date of reference image")
                    jhdr["REFFWHM"] = (ref_meta.image_header["FWHM"], "FWHM of ref image in pixels")
                    jhdr["REFMRA"]  = (ref_meta.image_header["RABIAS"], "RA astrometric offset of ref image")
                    jhdr["REFMDEC"] = (ref_meta.image_header["DECBIAS"], "DEC astrometric offset of ref image")
                    jhdr["REFSRA"]  = (ref_meta.image_header["RASTD"], "RA astrometric rms of ref image")
                    jhdr["REFSDEC"] = (ref_meta.image_header["DECSTD"], "DEC astrometric rms of ref image")
                
                    jcut = np.array([jref_matrix, jnew_matrix, jdiff_matrix], dtype=np.float32)
                    fits.writeto(jstm_cutout, jcut, jhdr, overwrite=True)
                else:
                    jstm_cutout = None

                # basic parameters:
                jutc_new = new_meta.image_header["DATE"]
                jutc_ref = ref_meta.image_header["DATE"]
                jmjd_new = Time(jutc_new, scale="utc").mjd
                jmjd_ref = Time(jutc_ref, scale="utc").mjd
                jcrd = SkyCoord(ra=jra_peak*units.degree, dec=jdec_peak*units.degree, frame='icrs')
                jlon, jlat = jcrd.barycentricmeanecliptic.lon.value, jcrd.barycentricmeanecliptic.lat.value
                jl, jb = jcrd.galactic.l.value, jcrd.galactic.b.value
                jtrans_param = [jtrans_id, jutc_new, jmjd_new, jra_peak, jdec_peak, jlon, jlat, jl, jb,
                                band, jdiff_flux, jdiff_snr, jdiff_fwhm, idiff_mode, jutc_ref, jmjd_ref, jstm_cutout,
                                ref_meta.image_header["REFIMG"], new_meta.image_header["NEWIMG"],
                                os.path.basename(diff_image), jref_matrix.T, jnew_matrix.T, jdiff_matrix.T,
                               ]
                diff_trans_iter.add_row(jtrans_param)

                if self.trans_meta_update:
                    # save the transient metadata
                    jtrans_meta_table = f"MTC_J{jra_hms[1]}{jdec_dms[3]}.fits"
                    if self.trans_meta_path is not None:
                        jtrans_meta_table = os.path.join(self.trans_meta_path, jtrans_meta_table)
                    jtrans_meta_iter = meta_obj.meta_table_iter(jtrans_meta_table, mode="trans")
                    jtrans_meta_iter.add_row(jtrans_param)
                    jtrans_meta_iter.write(jtrans_meta_table, format="fits", overwrite=True)
                ntrans += 1
        print(f"^_^ Total {ntrans} transients detected")
        # save the table and region file
        diff_trans_region = diff_trans_table[:-4] + "reg"
        wds9reg(crds_ra, crds_dec, radius=6.0, unit="arcsec", color="cyan", outfile=diff_trans_region)
        diff_trans_iter.write(diff_trans_table, format="fits", overwrite=True)
        return

