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
import mdiff_base as dbase
import mdiff_match as dmatch
import mdiff_psf as dpsf
import mdiff_diff as ddiff
import mdiff_utils as dutl

# get the directory of the python file
path_default = os.path.dirname(__file__)
# print(f"^_^ Installed directory is: {path_default}")

def diffimg_run(sci_image_name, sci_image_path, 
                diff_image_path, 
                trans_cand_path, 
                refcat_meta_path,
                refcat_meta="reference_image_mephisto.cat",
                survey_mode="pilot",
                interp_badpixel=False,
                trans_stamp_size=49):
    """
    Perform image differencing for a new science image
    """
    print(f"^_^ Input science image is {sci_image_name}")
    sci_image_abs = os.path.join(sci_image_path, sci_image_name)
    sci_gaia_abs  = sci_image_abs[:-5] + "_sexcat_gaia.fits"

    # check the completeness for the entire environment
    base_check = dbase.BaseCheck()
    swarp_comd = base_check.swarp_shell()
    sex_comd   = base_check.sextractor_shell()
    base_check.star_catalog_check(sci_gaia_abs)
    base_check.header_check(sci_image_abs)

    # estimate the image center
    sci_image_header = fits.getheader(sci_image_abs)
    sci_image_wcs = wcs.WCS(sci_image_header)
    sci_xsize, sci_ysize = sci_image_wcs.pixel_shape
    sci_ximg_center, sci_yimg_center = 0.5*(sci_xsize+1), 0.5*(sci_ysize+1)
    sci_ra_center, sci_dec_center = sci_image_wcs.all_pix2world(sci_ximg_center, sci_yimg_center, 1)
    sci_ra_center, sci_dec_center = sci_ra_center.tolist(), sci_dec_center.tolist()
    sci_image_band = sci_image_header["FILTER"]
    
    # find the reference image
    refcat_meta_abs = os.path.join(refcat_meta_path, refcat_meta)
    refcat_meta_table = Table.read(refcat_meta_abs, format="ascii")
    bid = refcat_meta_table["band"]==sci_image_band
    ref_ra, ref_dec = refcat_meta_table["ra"][bid],refcat_meta_table["dec"][bid]
    match_aperture = 20.0*60.0
    ref_id, sci_id = dutl.crossmatch(ref_ra,ref_dec,[sci_ra_center],[sci_dec_center], aperture=match_aperture)
    if len(ref_id)==0: sys.exit("!!! No reference found")
    ref_image_name = refcat_meta_table["ref_image"][bid][ref_id[0]]
    ref_image_path = refcat_meta_table["ref_path"][bid][ref_id[0]]
    if ref_image_path=="None": ref_image_path = refcat_meta_path
    ref_image_abs  = os.path.join(ref_image_path, ref_image_name)
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
    imgout = dmatch.match_image(sci_image_abs,new_image_abs,ref_image_abs,
                                survey_mode=survey_mode,
                                interp_badpixel=interp_badpixel,
                                refine=False,
                                photcal_figure=photcal_figure)
    t1 = time.time()
    dt1 = t1 - t0
    print(f"^_^ Images are aligned, {dt1:7.3f} seconds used")

    # prepare the reference image
    print(f"^_^ Prepare the reference image: {ref_image_name}")
    ref_meta = dbase.LoadMeta(ref_image_abs)
    ref_mask = dpsf.MaskStar(ref_meta,scale=1.5)
    #ref_meta.image_matrix[ref_mask.mask] = np.nan
    #fits.writeto("zRef.fits",ref_meta.image_matrix.T,ref_meta.image_header,overwrite=True)
    ref_psf_star = dpsf.PSFStar(ref_meta)
    ref_psf_model = dpsf.PSFModel(ref_psf_star, image_size=(ref_meta.xsize,ref_meta.ysize), info_frac=0.95, poly_degree=3)
    t2 = time.time()
    dt2 = t2 - t1
    print(f"^_^ Reference image is ready, {dt2:7.3f} seconds used")

    # prepare the new image
    print(f"^_^ Prepare the new image: {new_image_name}")
    new_meta = dbase.LoadMeta(new_image_abs)
    new_mask = dpsf.MaskStar(new_meta,scale=1.5)
    #new_meta.image_matrix[new_mask.mask] = np.nan
    #fits.writeto("zNew.fits", new_meta.image_matrix.T, new_meta.image_header, overwrite=True)
    new_psf_star = dpsf.PSFStar(new_meta)
    new_psf_model = dpsf.PSFModel(new_psf_star, image_size=(new_meta.xsize,new_meta.ysize), info_frac=0.95, poly_degree=3)
    t3 = time.time()
    dt3 = t3 - t2
    print(f"^_^ New image is ready, {dt3:7.3f} seconds used")

    # perform image differencing
    print("^_^ Do image differencing ...")
    diffObj = ddiff.DiffImg(ref_meta, new_meta, ref_psf_model, new_psf_model, ref_mask=ref_mask.mask, new_mask=new_mask.mask)
    fits.writeto(diff_psf_abs, diffObj.Dpsf.T, overwrite=True)
    t4  = time.time()
    dt4 = t4 - t3
    print(f"^_^ Image differencing is done, {dt4:7.3f} seconds used")
    
    # source detection on both the difference and inverse-difference images
    print("^_^ Detect objects on the difference image")
    sex_config_file = os.path.join(path_default, "config/default_config.sex")
    sex_param_file = os.path.join(path_default, "config/default_param.sex")
    diff_mcat = []
    for idetmode in [1, -1]:
        if idetmode==1:
            idet_key = "direct"
            idiff_image_abs = diff_image_abs
            idiff_chkn = idiff_image_abs[:-4] + f"{idet_key}.check.fits"
            idiff_catn = idiff_image_abs[:-4] + f"{idet_key}.ldac"
            idiff_regn = idiff_image_abs[:-4] + f"{idet_key}.reg"
            fits.writeto(idiff_image_abs, diffObj.Dimg.T, header=new_meta.image_header, overwrite=True)
        else: # idetmode==-1
            idet_key = "inverse"
            idiff_image_abs = diff_image_abs[:-4] + f"{idet_key}.fits"
            idiff_chkn = idiff_image_abs[:-4] + "check.fits"
            idiff_catn = idiff_image_abs[:-4] + "ldac"
            idiff_regn = idiff_image_abs[:-4] + "reg"
            fits.writeto(idiff_image_abs, 0.0-diffObj.Dimg.T, header=new_meta.image_header, overwrite=True)

        sexComd1 = f"{sex_comd} {idiff_image_abs} -c {sex_config_file} -PARAMETERS_NAME {sex_param_file} "
        sexComd2 = f"-CATALOG_NAME {idiff_catn} -WEIGHT_TYPE NONE -CHECKIMAGE_TYPE APERTURES -CHECKIMAGE_NAME {idiff_chkn} "
        sexComd3 = "-DETECT_THRESH 1.5 -ANALYSIS_THRESH 1.5 -DETECT_MINAREA 5 "
        sexComd  = sexComd1 + sexComd2 + sexComd3
        subprocess.run(sexComd, shell=True)

        idiff_cat = Table.read(idiff_catn, format="fits", hdu=2)
        inobj = len(idiff_cat)
        print(f"    Total {inobj} objects detected on the {idet_key} difference image")
        ira     = idiff_cat["ALPHA_J2000"]
        idec    = idiff_cat["DELTA_J2000"]
        dutl.wds9reg(ira,idec,radius=5.0,unit="arcsec",color="green",outfile=idiff_regn)
        
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
    match_aperture = np.sqrt(ref_fwhm**2+new_fwhm**2) * pixel_scale
    blend_ids, isolate_ids = dutl.find_blends(ra, dec, aperture=match_aperture)
    
    diff_catn = diff_image_abs[:-4] + "ldac"
    diff_regn = diff_image_abs[:-4] + "reg"
    diff_bregn = diff_image_abs[:-4] + "blends.reg"
    diff_mcat.write(diff_catn, format="fits", overwrite=True)
    dutl.wds9reg(ra,dec,radius=5.0,unit="arcsec",color="green",outfile=diff_regn)
    dutl.wds9reg(ra[blend_ids],dec[blend_ids],radius=5.0,unit="arcsec",color="red",outfile=diff_bregn)

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

        idiff_marix = diffObj.Dimg[ix0-1:ix1,iy0-1:iy1]
        iref_matrix = ref_meta.image_matrix[ix0-1:ix1,iy0-1:iy1]
        inew_matrix = new_meta.image_matrix[ix0-1:ix1,iy0-1:iy1]
        icut    = np.array([iref_matrix, inew_matrix, idiff_marix], dtype=float)

        ira_hms, idec_dms = dutl.deg2str(ra[iobj], dec[iobj])
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

        ihdr["TF_VERS"] = (dbase.__version__, "TransFinder version")
        ihdr["TF_DATE"] = (dbase.__version_date__, "TransFinder version date")
        ihdr["TF_AUTH"] = (dbase.__author__, "TransFinder author")

        fits.writeto(istm_name_abs,icut,ihdr,overwrite=True)

    t5  = time.time()
    dt5 = t5 - t4
    print(f"    Detection is done, {dt5:.3f} seconds used")
    dt  = t5 - t0
    print(f"^_^ All Done with {dt:.3f} seconds.")
    return

if __name__ == "__main__":
    sci_image_name = "xKMTNk.20180221.003965_sciimg.fits"
    sci_image_path = "/Users/dzliu/Workspace/Mephisto/TransFinder/images/kmtnew/sciimg"
    diff_image_path = "/Users/dzliu/Workspace/Mephisto/TransFinder/images/kmtnew/diffimg"
    trans_cand_path = "/Users/dzliu/Workspace/Mephisto/TransFinder/images/kmtnew/diffimg/trans_candy"
    refcat_meta_path = "/Users/dzliu/Workspace/Mephisto/TransFinder/images/kmtnew/refimg"
    
    survey_mode = "regular"

    diffimg_run(sci_image_name, sci_image_path, diff_image_path, trans_cand_path, refcat_meta_path, 
                survey_mode=survey_mode,
                interp_badpixel=False)
