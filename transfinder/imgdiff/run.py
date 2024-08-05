# this is the main routine to perform image differencing using TransFinder
# Steps:
# 0) Mask saturated stars
# 1) Select PSF stars
# 2) Construct spatially varied PSF model
# 3) Perform image differencing
# 4) Detect high-probability outliers
# 5) Measure the fluxes of these outliers
# 6) Store the image and photometric info of the outliers

# version history:
# 2021.03.31: v1.0.0 change log 
#  the basic ZOGY image differencing algorithm is finished.
#  In this version, only constant PSF is considered.
# 2021.12.29:
#  save the detected residuals into stamps. Necessary keywords
#  are added to the stamp header.

# general modules
from astropy.io    import fits
from astropy.table import vstack, Table
from astropy.wcs import wcs
import numpy as np
import time, os, sys, subprocess

# self-defined modules
from .base import BaseCheck, LoadMeta, swarp_shell, sextractor_shell
from .buildimg import build_newimg
from .psfmodel import MaskStar, PSFStar, PSFModel
from .diff import DiffImg
from ..utils import crossmatch, find_blends, wds9reg, deg2str

def run(sci_image_name, sci_star_name, sci_image_path, 
        diff_image_path, 
        trans_cand_path, 
        refcat_meta_path,
        swarp_config_file, 
        sex_config_file, 
        sex_param_file,
        refcat_meta="reference_image_mephisto.cat",
        survey_mode="pilot",
        interp_badpixel=False,
        trans_stamp_size=49,
        decorr_grid=(10,10)):
    """
    Perform image differencing for a new science image
    """
    print(f"^_^ Input science image is {sci_image_name}")
    sci_image_abs = os.path.join(sci_image_path, sci_image_name)
    sci_star_abs = os.path.join(sci_image_path, sci_star_name)
    refcat_meta_abs = os.path.join(refcat_meta_path, refcat_meta)

    # check the completeness for the entire environment
    base_check = BaseCheck()
    base_check.file_check(sci_image_abs)
    base_check.file_check(sci_star_abs)
    base_check.file_check(refcat_meta_abs)
    base_check.header_check(sci_image_abs)
    
    # swarp and sextractor executable
    swarp_exe = swarp_shell()
    sex_exe = sextractor_shell()

    # estimate the image center
    sci_image_header = fits.getheader(sci_image_abs)
    sci_image_wcs = wcs.WCS(sci_image_header)
    sci_xsize, sci_ysize = sci_image_wcs.pixel_shape
    sci_ximg_center, sci_yimg_center = 0.5*(sci_xsize+1), 0.5*(sci_ysize+1)
    sci_ra_center, sci_dec_center = sci_image_wcs.all_pix2world(sci_ximg_center, sci_yimg_center, 1)
    sci_ra_center, sci_dec_center = sci_ra_center.tolist(), sci_dec_center.tolist()
    sci_image_band = sci_image_header["FILTER"]
    
    hx_arcsec= 0.5 * sci_xsize * sci_image_wcs.proj_plane_pixel_scales()[0].value * 3600.0
    hy_arcsec= 0.5 * sci_ysize * sci_image_wcs.proj_plane_pixel_scales()[1].value * 3600.0
    match_aperture = np.min([hx_arcsec, hy_arcsec])

    # find the reference image
    refcat_meta_table = Table.read(refcat_meta_abs, format="ascii")
    bid = refcat_meta_table["band"]==sci_image_band
    ref_ra, ref_dec = refcat_meta_table["ra"][bid],refcat_meta_table["dec"][bid]
    ref_id, sci_id = crossmatch(ref_ra,ref_dec,[sci_ra_center],[sci_dec_center], aperture=match_aperture)
    if len(ref_id)==0: sys.exit("!!! No reference found")
    ref_image_name = refcat_meta_table["ref_image"][bid][ref_id[0]]
    ref_image_path = refcat_meta_table["ref_path"][bid][ref_id[0]]
    if ref_image_path=="None": ref_image_path = refcat_meta_path
    ref_image_abs  = os.path.join(ref_image_path, ref_image_name)
    base_check.file_check(ref_image_abs)
    print(f"^_^ Matched reference image is {ref_image_name}")

    # define the matched new image and difference image
    new_image_name = sci_image_name[:-4] + "new.fits"
    new_image_abs = os.path.join(diff_image_path, new_image_name)
    diff_image_name = sci_image_name[:-4] + "diff.fits"
    diff_image_abs = os.path.join(diff_image_path, diff_image_name)
    diff_psf_name = diff_image_name[:-4] + "psf.fits"
    diff_psf_abs = os.path.join(diff_image_path, diff_psf_name)

    # start image differencing
    t0 = time.time()
    print("^_^ Match reference and new images")
    photcal_figure = new_image_abs[:-4] + "fluxcal.png"
    imgout = build_newimg(sci_image_abs,sci_star_abs, 
                          new_image_abs,ref_image_abs,
                          swarp_config_file, sex_config_file, sex_param_file,
                          survey_mode=survey_mode,
                          interp_badpixel=interp_badpixel,
                          refine=False,
                          photcal_figure=photcal_figure,
                          swarp_exe=swarp_exe, sex_exe=sex_exe)
    
    t1 = time.time()
    dt1 = t1 - t0
    print(f"^_^ Images are aligned, {dt1:7.3f} seconds used")
    print(f"^_^ Reference image is: {ref_image_name}")
    print(f"    New image is: {new_image_name}")

    # prepare meta data of reference and new image
    print(f"^_^ Load the meta data for both")
    ref_meta = LoadMeta(ref_image_abs)
    ref_mask = MaskStar(ref_meta,scale=1.5)
    #ref_meta.image_matrix[ref_mask.mask] = np.nan
    #fits.writeto("zRef.fits",ref_meta.image_matrix.T,ref_meta.image_header,overwrite=True)
    new_meta = LoadMeta(new_image_abs)
    new_mask = MaskStar(new_meta,scale=1.5)
    #new_meta.image_matrix[new_mask.mask] = np.nan
    #fits.writeto("zNew.fits", new_meta.image_matrix.T, new_meta.image_header, overwrite=True)
    
    # construct PSF models for reference and new
    print("^_^ Construct PSF models for both")
    psf_size = np.max([ref_meta.psf_size, new_meta.psf_size])
    print(f"    Input PSF size: (x_size, y_size)=({psf_size}, {psf_size})")
    
    print("    1) PSF modeling for reference")
    ref_psf_star_meta = PSFStar(ref_meta, psf_size=psf_size, nstar_max=500)
    ref_psf_model = PSFModel(ref_psf_star_meta, ref_meta, info_frac=0.95, nbasis_max=3, poly_degree=3)
    ref_psf_model.psf_model_diagnosis(ref_psf_star_meta, output_path=diff_image_path, output_prefix=ref_image_name[:-5])
    
    print("    2) PSF modeling for new")
    new_psf_star_meta = PSFStar(new_meta, psf_size=psf_size, nstar_max=500)
    new_psf_model = PSFModel(new_psf_star_meta, new_meta, info_frac=0.95, nbasis_max=3, poly_degree=3)
    new_psf_model.psf_model_diagnosis(new_psf_star_meta, output_path=diff_image_path, output_prefix=new_image_name[:-5])
    t2 = time.time()
    dt2 = t2 - t1
    print(f"    PSF models are ready, {dt2:7.3f} seconds used")
    
    # perform image differencing
    print("^_^ Do image differencing ...")
    diffObj = DiffImg(degrid=decorr_grid, nthreads=-1)
    diff_matrix = diffObj.diff(ref_meta, new_meta, ref_psf_model, new_psf_model,
                               ref_mask=ref_mask.mask, new_mask=new_mask.mask)
    #fits.writeto(diff_image_abs, diff_matrix.T.astype(np.float32), header=new_meta.image_header, overwrite=True)
    t3  = time.time()
    dt3 = t3 - t2
    print(f"^_^ Image differencing is done, {dt3:7.3f} seconds used")

    # source detection on both the difference and inverse-difference images
    print("^_^ Detect objects on the difference image")
    diff_mcat = []
    for idetmode in [1, -1]:
        if idetmode==1:
            idet_key = "direct"
            idiff_image_abs = diff_image_abs
            idiff_chkn = idiff_image_abs[:-4] + f"{idet_key}.check.fits"
            idiff_catn = idiff_image_abs[:-4] + f"{idet_key}.ldac"
            idiff_regn = idiff_image_abs[:-4] + f"{idet_key}.reg"
            fits.writeto(idiff_image_abs, diff_matrix.T, header=new_meta.image_header, overwrite=True)
        else: # idetmode==-1
            idet_key = "inverse"
            idiff_image_abs = diff_image_abs[:-4] + f"{idet_key}.fits"
            idiff_chkn = idiff_image_abs[:-4] + "check.fits"
            idiff_catn = idiff_image_abs[:-4] + "ldac"
            idiff_regn = idiff_image_abs[:-4] + "reg"
            fits.writeto(idiff_image_abs, 0.0-diff_matrix.T, header=new_meta.image_header, overwrite=True)

        sexComd1 = f"{sex_exe} {idiff_image_abs} -c {sex_config_file} -PARAMETERS_NAME {sex_param_file} "
        sexComd2 = f"-CATALOG_NAME {idiff_catn} -WEIGHT_TYPE NONE -CHECKIMAGE_TYPE APERTURES -CHECKIMAGE_NAME {idiff_chkn} "
        sexComd3 = "-DETECT_THRESH 2.0 -ANALYSIS_THRESH 2.0 -DETECT_MINAREA 5 "
        sexComd  = sexComd1 + sexComd2 + sexComd3
        subprocess.run(sexComd, shell=True)

        idiff_cat = Table.read(idiff_catn, format="fits", hdu=2)
        inobj = len(idiff_cat)
        print(f"    Total {inobj} objects detected on the {idet_key} difference image")
        ira     = idiff_cat["ALPHA_J2000"]
        idec    = idiff_cat["DELTA_J2000"]
        wds9reg(ira,idec,radius=5.0,unit="arcsec",color="green",outfile=idiff_regn)
        
        diff_mcat += [idiff_cat]

    # stack the photometric catalogs
    diff_mcat = vstack(diff_mcat)
    nobj   = len(diff_mcat)
    ra     = diff_mcat["ALPHA_J2000"]
    dec    = diff_mcat["DELTA_J2000"]
    ximg   = diff_mcat["XWIN_IMAGE"]
    yimg   = diff_mcat["YWIN_IMAGE"]
    mag    = diff_mcat["MAG_AUTO"]
    magErr = diff_mcat["MAGERR_AUTO"]
    fwhm   = diff_mcat["FWHM_IMAGE"]
    flag   = diff_mcat["FLAGS"]
    snr    = diff_mcat["SNR_WIN"]
    
    # find blending objects
    print("^_^ Clean the detected objects")
    ref_fwhm = ref_meta.image_header["REF_FWHM"]
    new_fwhm = new_meta.image_header["NEW_FWHM"]
    pixel_scale = ref_meta.image_header["REF_PS"]
    mblend_aperture = np.sqrt(ref_fwhm**2+new_fwhm**2) * pixel_scale
    blend_ids, isolate_ids = find_blends(ra, dec, aperture=mblend_aperture)

    diff_catn = diff_image_abs[:-4] + "ldac"
    diff_regn = diff_image_abs[:-4] + "reg"
    diff_mcat.write(diff_catn, format="fits", overwrite=True)
    wds9reg(ra,dec,radius=5.0,unit="arcsec",color="green",outfile=diff_regn)
    if len(blend_ids)!=0:
        diff_bregn = diff_image_abs[:-4] + "blends.reg"
        wds9reg(ra[blend_ids],dec[blend_ids],radius=5.0,unit="arcsec",color="red",outfile=diff_bregn)

    print("^_^ Save the stamps of the transient candidates")
    # extract image stamps
    #diffImgMat = fits.getdata(diff_image_abs)
    #alertCatn  = altdir + newimg[:-4] + "alert.cat"
    #alertCat   = open(alertCatn, "w")
    #alertCat.write("#id name ra dec ximg yimg mag magerr fwhm flags atEdge SNR\n")
    #afmt = "%5d %30s %12.6f %12.6f %9.3f %9.3f %8.3f %8.3f %8.3f %2d %2d %9.3f\n"
    #stmX = stmSize//2
    # transient image name: MTC_JHHMMSS.SSPDDMMSS.SS.fits
    # transient ID: JHHMMSS.SS±DDMMSS.SS
    for iobj in range(nobj):
        if fwhm[iobj]<=1.0 or snr[iobj]<=0.0: continue
        if iobj in blend_ids: continue

        iximg, iyimg = int(round(ximg[iobj])), int(round(yimg[iobj]))
        ix0, ix1 = iximg - trans_stamp_size//2, iximg + trans_stamp_size//2
        iy0, iy1 = iyimg - trans_stamp_size//2, iyimg + trans_stamp_size//2
        if ix0<=0    or iy0<=0: continue
        if ix1>sci_xsize or iy1>sci_xsize: continue

        idiff_marix = diff_matrix[ix0-1:ix1,iy0-1:iy1]
        iref_matrix = ref_meta.image_matrix[ix0-1:ix1,iy0-1:iy1]
        inew_matrix = new_meta.image_matrix[ix0-1:ix1,iy0-1:iy1]
        icut    = np.array([iref_matrix, inew_matrix, idiff_marix], dtype=float)

        ira_hms, idec_dms = deg2str(ra[iobj], dec[iobj])
        itrans_id = f"J{ira_hms[1]}{idec_dms[2]}"
        istm_name = f"MTC_J{ira_hms[1]}{idec_dms[3]}.fits"
        istm_name_abs = os.path.join(trans_cand_path, istm_name)
        ihdr = fits.Header()
        ihdr["RA"]      = ra[iobj]
        ihdr["DEC"]     = dec[iobj]
        ihdr["FILE"]    = istm_name
        ihdr["XPOS"]    = ximg[iobj]
        ihdr["YPOS"]    = yimg[iobj]
        ihdr["TRANSID"] = (itrans_id, "candidate id")
        ihdr["MAG"]     = (mag[iobj], "auto magnitude")
        ihdr["MAGERR"]  = (magErr[iobj], "auto magnitude error")
        ihdr["FILTER"]  = (sci_image_band, "filter")
        ihdr["FWHM"]    = (fwhm[iobj], "FWHM in pixels")
        ihdr["SNR"]     = (snr[iobj], "signal-to-noise ratio")
        ihdr["DETMODE"] = (idetmode, "detection mode: -1=inverse diff; 1=direct diff")
        ihdr["IMGDIFF"] = (diff_image_name, "diff image")

        ihdr["IMGNEW"]  = (new_image_name, "new image")
        ihdr["DATENEW"] = (new_meta.image_header["DATE"], "date of new image")
        ihdr["FWHMNEW"] = (new_meta.image_header["NEW_FWHM"], "FWHM of new image in pixels")
        ihdr["MRANEW"]  = (new_meta.image_header["NEW_MRA"], "RA astrometric offset of new image")
        ihdr["MDECNEW"] = (new_meta.image_header["NEW_MDEC"], "DEC astrometric offset of new image")
        ihdr["SRANEW"]  = (new_meta.image_header["NEW_SRA"], "RA astrometric rms of new image")
        ihdr["SDECNEW"] = (new_meta.image_header["NEW_SDEC"], "DEC astrometric rms of new image")

        ihdr["IMGREF"]  = (ref_image_name, "ref image")
        ihdr["DATEREF"] = (ref_meta.image_header["DATE"], "date of reference image")
        ihdr["FWHMREF"] = (ref_meta.image_header["REF_FWHM"], "FWHM of ref image in pixels")
        ihdr["MRAREF"]  = (ref_meta.image_header["REF_MRA"], "RA astrometric offset of ref image")
        ihdr["MDECREF"] = (ref_meta.image_header["REF_MDEC"], "DEC astrometric offset of ref image")
        ihdr["SRAREF"]  = (ref_meta.image_header["REF_SRA"], "RA astrometric rms of ref image")
        ihdr["SDECREF"] = (ref_meta.image_header["REF_SDEC"], "DEC astrometric rms of ref image")

        #ihdr["TF_VERS"] = (dbase.__version__, "TransFinder version")
        #ihdr["TF_DATE"] = (dbase.__version_date__, "TransFinder version date")
        #ihdr["TF_AUTH"] = (dbase.__author__, "TransFinder author")

        fits.writeto(istm_name_abs,icut,ihdr,overwrite=True)

    t4  = time.time()
    dt4 = t4 - t3
    print(f"    Detection is done, {dt4:.3f} seconds used")
    dt  = t4 - t0
    print(f"^_^ All Done with {dt:.3f} seconds.")
    return

