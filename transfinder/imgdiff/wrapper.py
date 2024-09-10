# module to perform image differencing in one loop
import time
from astropy.table import Table
import os, shutil

from .base import swarp_shell, sextractor_shell
from .buildimg import BuildImage
from .psfmodel import PSFModel
from .diff import DiffImg
from .transdet import ExtractTrans

def run(new_sciimg,
        ref_sciimg,
        new_sciimg_path,
        ref_sciimg_path,
        diff_image_path, 
        config_path,
        swarp_config = "default_config.swarp",
        sextractor_config = "default_config.sex",
        sextractor_param = "default_param.sex",
        psf_size = 25, 
        npsf_star_max = 500,
        trans_stamp_size = 49,
        ):
    """
    run the image difference code for Mephisto pilot survey
    """
    # swarp and sextractor
    swarp_config = os.path.join(config_path, swarp_config)
    sextractor_config = os.path.join(config_path, sextractor_config)
    sextractor_param = os.path.join(config_path, sextractor_param)
    swarp_executor = swarp_shell()
    sextractor_executor = sextractor_shell()
    
    # input new image and reference image
    new_sciimg_abs = os.path.join(new_sciimg_path, new_sciimg)
    ref_sciimg_abs = os.path.join(ref_sciimg_path, ref_sciimg)

    new_star_abs = new_sciimg_abs[:-5] + "_sexcat_gaia.fits"
    new_star_matrix = Table.read(new_star_abs, format="fits", hdu=2)
    new_star_pos = [new_star_matrix["ra"], new_star_matrix["dec"]]

    ref_star_abs = ref_sciimg_abs[:-5] + "_sexcat_gaia.fits"
    ref_star_matrix = Table.read(ref_star_abs, format="fits", hdu=2)
    ref_star_pos = [ref_star_matrix["ra"], ref_star_matrix["dec"]]

    # output setup
    new_image = new_sciimg[:-5] + "_new.fits"
    ref_image = ref_sciimg[:-5] + "_ref.fits"
    diff_image = new_sciimg[:-5] + "_diff.fits"
    photcal_figure = new_sciimg[:-5] + "_photcal.png"

    new_image_abs = os.path.join(diff_image_path, new_image)
    ref_image_abs = os.path.join(diff_image_path, ref_image)
    diff_image_abs = os.path.join(diff_image_path, diff_image)
    photcal_figure_abs = os.path.join(diff_image_path, photcal_figure)
   
    new_psf_model_prefix = new_image_abs[:-4] + "psf_model"
    ref_psf_model_prefix = ref_image_abs[:-4] + "psf_model"

    trans_stamp_path = os.path.join(diff_image_path, "transient_candidates")
    if os.path.exists(trans_stamp_path): shutil.rmtree(trans_stamp_path)
    os.mkdir(trans_stamp_path)

    # main code
    t0 = time.time()
    # 1) match new and reference image
    buildimg_obj = BuildImage(swarp_config, sextractor_config, sextractor_param, 
                              swarp_exe = swarp_executor, 
                              sextractor_exe = sextractor_executor,)
    new_meta = buildimg_obj.image_resamp(new_sciimg_abs, new_star_pos, new_image_abs,)
    ref_meta = buildimg_obj.image_resamp(ref_sciimg_abs, ref_star_pos, ref_image_abs,)
    new_meta = buildimg_obj.phot_match(new_meta, ref_meta, method="fitted", photcal_figure=photcal_figure_abs)
    new_mask = buildimg_obj.saturation_mask(new_meta[-1], mask_scale=1.2)
    ref_mask = buildimg_obj.saturation_mask(ref_meta[-1], mask_scale=1.2)

    # 2) psf modeling
    ref_matrix, new_matrix = ref_meta[-2], new_meta[-2]
    ref_star_matrix, new_star_matrix = ref_meta[-1], new_meta[-1]
    ref_sid, new_sid = ref_star_matrix["FLAG_STAR"]==1, new_star_matrix["FLAG_STAR"]==1
    ref_star_matrix, new_star_matrix = ref_star_matrix[ref_sid], new_star_matrix[new_sid]

    psfmodel_obj = PSFModel(psf_size=psf_size, nstar_max=npsf_star_max, info_frac=0.98, nbasis_max=3)
    ref_psfmodel = psfmodel_obj.run(ref_matrix, ref_star_matrix, output_prefix=ref_psf_model_prefix)
    new_psfmodel = psfmodel_obj.run(new_matrix, new_star_matrix, output_prefix=new_psf_model_prefix)

    # 3) image differencing
    diff_obj = DiffImg(degrid=(16,16), nthreads=-1)
    diff_matrix = diff_obj.diff(ref_matrix, new_matrix, ref_psfmodel, new_psfmodel, ref_mask=ref_mask, new_mask=new_mask)

    # 4) transient detection
    ref_header, new_header = ref_meta[-3], new_meta[-3]
    transdet_obj = ExtractTrans(sextractor_config, sextractor_param, 
                                sextractor_exe = sextractor_executor,
                                trans_stamp_size = trans_stamp_size, )
    transdet_obj.extract_trans(diff_image_abs, diff_matrix,
                               ref_header, ref_matrix,
                               new_header, new_matrix,
                               cutout_write=True, cutout_path=trans_stamp_path)

    t1 = time.time()
    dt = t1 - t0
    print(f"^_^ Total {dt:.5f} seconds used")
    return

