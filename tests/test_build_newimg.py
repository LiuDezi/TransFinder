# test buildref.py

import time
import os
import sys
from transfinder import imgdiff

input_path = "/Users/dzliu/Workspace/Mephisto/TransFinder/images/tarimg/at2024ggi"
input_image = "my_sc_tat2024ggi_g_20240415143146_086_sciimg.fits"
input_star_catalog = "my_sc_tat2024ggi_g_20240415143146_086_sciimg_sexcat_gaia.fits"
input_image = os.path.join(input_path, input_image)
input_star_catalog = os.path.join(input_path, input_star_catalog)

output_path  = "/Users/dzliu/Workspace/TransFinder/tests/output"
output_image = "my_sc_tat2024ggi_g_20240415143146_086_sciimg.new.fits"
output_image = os.path.join(output_path, output_image)

map_path = "/Users/dzliu/Workspace/TransFinder/tests/output"
map_image = "my_sc_tAT2024ggi_g_20240411152856_074_sciimg.ref.fits"
map_image = os.path.join(map_path, map_image)

survey_mode = "pilot"

config_path = "/Users/dzliu/Workspace/TransFinder/config"
swarp_config = os.path.join(config_path, "default_config.swarp")
sex_config = os.path.join(config_path, "default_config.sex")
sex_param = os.path.join(config_path, "default_param.sex")

# basic check
base_check = imgdiff.BaseCheck()
base_check.file_check(input_image)
base_check.file_check(map_image)
base_check.file_check(input_star_catalog)
base_check.header_check(input_image)

swarp_exe = imgdiff.swarp_shell()
sex_exe = imgdiff.sextractor_shell()

t0 = time.time()
image_list = imgdiff.build_newimg(input_image, input_star_catalog, 
                                  output_image,
                                  map_image,
                                  swarp_config, sex_config, sex_param,
                                  interp_badpixel=False,
                                  refine=False,
                                  survey_mode=survey_mode,
                                  photcal_figure="phot_scale.png",
                                  swarp_exe=swarp_exe, sex_exe=sex_exe)
t1 = time.time()
dt = t1 - t0
print(f"^_^ Total {dt:.5f} seconds used")


