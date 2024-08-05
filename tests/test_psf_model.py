# test PSF modelling

# general modules
import time, os, sys
from transfinder import imgdiff

sci_image_name = "my_sc_tsn2024jz_g_20240128232058_186_sciimg.ref.fits"
sci_image_path = "/Users/dzliu/Workspace/Mephisto/TransFinder/images/refimg"
output_path = "/Users/dzliu/Workspace/TransFinder/tests/output"

print(f"^_^ Input science image is {sci_image_name}")
sci_image_abs = os.path.join(sci_image_path, sci_image_name)
    
t1 = time.time()
print(f"^_^ Model PSF: {sci_image_name}")
sci_meta = imgdiff.LoadMeta(sci_image_abs)
sci_mask = imgdiff.MaskStar(sci_meta,scale=1.5)
sci_psf_star_meta = imgdiff.PSFStar(sci_meta, nstar_max=500)
sci_psf_model = imgdiff.PSFModel(sci_psf_star_meta, sci_meta, info_frac=0.95, nbasis_max=3, poly_degree=3)
sci_psf_model.psf_model_diagnosis(sci_psf_star_meta, output_path=output_path)

t2 = time.time()
dt2 = t2 - t1
print(f"^_^ Science image is ready, {dt2:7.3f} seconds used")


