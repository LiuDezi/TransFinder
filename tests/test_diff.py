# this is the main routine to perform image differencing using TransFinder

# general modules
from astropy.io    import fits
from astropy.table import Table
from astropy.wcs import wcs
import numpy as np
import time, os, sys

from transfinder import imgdiff

survey_mode = "pilot"
sci_image_name = "my_sc_tat2024ggi_r_20240523132824_062_sciimg.fits"
sci_star_name = "my_sc_tat2024ggi_r_20240523132824_062_sciimg_sexcat_gaia.fits"
sci_image_path = "/Users/dzliu/Workspace/Mephisto/TransFinder/images/tarimg/at2024ggi"
diff_image_path = "/Users/dzliu/Workspace/TransFinder/tests/output"
trans_cand_path = "/Users/dzliu/Workspace/TransFinder/tests/output"
refcat_meta_path = "/Users/dzliu/Workspace/Mephisto/TransFinder/images/refimg"

config_path = "/Users/dzliu/Workspace/TransFinder/config"
swarp_config = os.path.join(config_path, "default_config.swarp")
sex_config = os.path.join(config_path, "default_config.sex")
sex_param = os.path.join(config_path, "default_param.sex")

imgdiff.run(sci_image_name, sci_star_name, sci_image_path,
            diff_image_path,
            trans_cand_path,
            refcat_meta_path,
            swarp_config,
            sex_config,
            sex_param,
            refcat_meta="reference_image_mephisto.cat",
            survey_mode="pilot",
            interp_badpixel=False,
            trans_stamp_size=49)

