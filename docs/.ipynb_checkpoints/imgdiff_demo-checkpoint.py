{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d51e0607-75d4-4058-b157-b6d6d58c3b01",
   "metadata": {},
   "source": [
    "## This tutorial presents a simple usage of image differencing module in TransFinder (Mephisto as example)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f19e2c62-1ef6-417d-a994-b6f926756fd9",
   "metadata": {},
   "source": [
    "___________ \n",
    "## Install TransFinder with terminal: \n",
    "* Download package: git clone https://github.com/LiuDezi/TransFinder.git\n",
    "* Change directory: cd TransFinder\n",
    "* Install: python -m pip install .\n",
    "___________"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5874c994-552d-4148-bd59-380461a195cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import necessary modules\n",
    "from transfinder import imgdiff\n",
    "from astropy.table import Table\n",
    "from astropy.io import fits\n",
    "import numpy as np\n",
    "import os, time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8cebfe87-2edb-42c8-9f2f-7433b5e58d61",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1) define the inputs\n",
    "raw_path = \"/Users/dzliu/Workspace/Mephisto/TransFinder/images/tarimg/at2024ggi\"\n",
    "raw_ref_image = \"my_sc_tAT2024ggi_g_20240411152856_074_sciimg.fits\"\n",
    "raw_ref_star = \"my_sc_tAT2024ggi_g_20240411152856_074_sciimg_sexcat_gaia.fits\"\n",
    "raw_sci_image = \"my_sc_tat2024ggi_g_20240415143146_086_sciimg.fits\"\n",
    "\n",
    "raw_ref_image_abs = os.path.join(raw_path, raw_ref_image)\n",
    "raw_ref_star_abs = os.path.join(raw_path, raw_ref_star)\n",
    "raw_sci_image_abs = os.path.join(raw_path, raw_sci_image)\n",
    "\n",
    "# load the coordinates of stars\n",
    "raw_ref_star_matrix = Table.read(raw_ref_star_abs, format=\"fits\", hdu=2)\n",
    "ra_base, dec_base = raw_ref_star_matrix[\"ra\"], raw_ref_star_matrix[\"dec\"]\n",
    "\n",
    "# sextractor and swarp configuration\n",
    "config_path = \"/Users/dzliu/Workspace/TransFinder/config\"\n",
    "swarp_config_file = os.path.join(config_path, \"default_config.swarp\")\n",
    "sex_config_file = os.path.join(config_path, \"default_config.sex\")\n",
    "sex_param_file = os.path.join(config_path, \"default_param.sex\")\n",
    "swarp_exe = imgdiff.swarp_shell()\n",
    "sex_exe = imgdiff.sextractor_shell()\n",
    "survey_mode = \"mephisto_pilot\"\n",
    "\n",
    "# 2) define the ouputs\n",
    "output_path = \"/Users/dzliu/Workspace/TransFinder/tests/output\"\n",
    "ref_image_abs = os.path.join(output_path, raw_ref_image[:-4] + \"ref.fits\")\n",
    "new_image_abs = os.path.join(output_path, raw_sci_image[:-4] + \"new.fits\")\n",
    "diff_image_abs = new_image_abs[:-4] + \"diff.fits\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0e4273ab-1577-42d3-b3f7-a101997c6b87",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "^_^ Construct reference image and corresponding catalog\n",
      "    Input image: my_sc_tAT2024ggi_g_20240411152856_074_sciimg.fits\n",
      "    Resampled reference image: my_sc_tAT2024ggi_g_20240411152856_074_sciimg.ref.fits\n",
      "    Resampled reference catalog: my_sc_tAT2024ggi_g_20240411152856_074_sciimg.ref.phot.fits\n",
      "    Resampled parameters:\n",
      "   1) image center: 11:18:26.14,-32:59:16.97\n",
      "   2) image size: 5802,5818\n",
      "   3) image pixel scale: 0.45 arcsec/pixel\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "> WARNING: /Users/dzliu/Workspace/TransFinder/tests/output/my_sc_tAT2024ggi_g_20240411152856_074_sciimg.ref.fits has flux scale = 0: I will take 1 instead\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "^_^ Total 8.69634 seconds used\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: VerifyWarning: Card is too long, comment will be truncated. [astropy.io.fits.card]\n"
     ]
    }
   ],
   "source": [
    "# build reference image\n",
    "t0 = time.time()\n",
    "imgdiff.build_refimg(raw_ref_image_abs, \n",
    "                     ref_image_abs,\n",
    "                     swarp_config_file, \n",
    "                     sex_config_file, \n",
    "                     sex_param_file, \n",
    "                     swarp_exe = swarp_exe,\n",
    "                     sex_exe = sex_exe,\n",
    "                     star_crds = [ra_base, dec_base],\n",
    "                     survey_mode = survey_mode\n",
    "                     )\n",
    "t1 = time.time()\n",
    "dt = t1 - t0\n",
    "print(f\"^_^ Total {dt:.5f} seconds used\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ebb40328-3447-467a-8bf7-56a65eb76a46",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "^_^ Generate matched image corresponding catalog with a specified reference\n",
      "    Input image: my_sc_tat2024ggi_g_20240415143146_086_sciimg.fits\n",
      "    Reference image: my_sc_tAT2024ggi_g_20240411152856_074_sciimg.ref.fits\n",
      "    Matched parameters:\n",
      "   1) image center: 11:18:26.14,-32:59:16.97\n",
      "   2) image size: 5802,5818\n",
      "   3) image pixel scale: 0.45 arcsec/pixel\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "> WARNING: /Users/dzliu/Workspace/TransFinder/tests/output/my_sc_tat2024ggi_g_20240415143146_086_sciimg.new.fits has flux scale = 0: I will take 1 instead\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "^_^ Total 6.08098 seconds used\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: VerifyWarning: Card is too long, comment will be truncated. [astropy.io.fits.card]\n"
     ]
    }
   ],
   "source": [
    "# match a science image with the reference image\n",
    "t0 = time.time()\n",
    "photcal_fig = os.path.join(output_path, \"photcal.png\")\n",
    "imgdiff.build_newimg(raw_sci_image_abs,\n",
    "                     new_image_abs,\n",
    "                     swarp_config_file,\n",
    "                     sex_config_file,\n",
    "                     sex_param_file,\n",
    "                     swarp_exe = swarp_exe,\n",
    "                     sex_exe = sex_exe,\n",
    "                     ref_image = ref_image_abs,\n",
    "                     survey_mode=survey_mode,\n",
    "                     photcal_figure = photcal_fig,\n",
    "                     )\n",
    "t1 = time.time()\n",
    "dt = t1 - t0\n",
    "print(f\"^_^ Total {dt:.5f} seconds used\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f1815b94-28d9-4bdc-9a21-ac24035762f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Total 35 saturated stars\n",
      "    Total 9 saturated stars\n",
      "    Total 500 PSF stars are selected\n",
      "    Estimate PSF basis functions\n",
      "    Total 1 PSF basis functions, accounting 0.962297761307319 info\n",
      "    Estimate the spatially varied PSF fields\n",
      "    Calculate the PSF normalization map\n",
      "    Total 372 PSF stars are selected\n",
      "    Estimate PSF basis functions\n",
      "    Total 153 PSF basis functions, accounting 0.9502318541331234 info\n",
      "!!! Number of PSF basis functions larger than nbasis_max=3\n",
      "!!! Only 3 PSF basis functions are kept, accounting 0.622687873240409 info\n",
      "    Estimate the spatially varied PSF fields\n",
      "    Calculate the PSF normalization map\n",
      "^_^ Science image is ready, 32.79637 seconds used\n"
     ]
    }
   ],
   "source": [
    "# construct psf models for both reference and new images\n",
    "t0 = time.time()\n",
    "ref_meta = imgdiff.LoadMeta(ref_image_abs)\n",
    "ref_mask = imgdiff.MaskStar(ref_meta, scale=1.5)\n",
    "\n",
    "new_meta = imgdiff.LoadMeta(new_image_abs)\n",
    "new_mask = imgdiff.MaskStar(new_meta, scale=1.5)\n",
    "\n",
    "psf_size = np.max([ref_meta.psf_size, new_meta.psf_size])\n",
    "\n",
    "ref_psf_star_meta = imgdiff.PSFStar(ref_meta, psf_size=psf_size, nstar_max=500)\n",
    "ref_psf_model = imgdiff.PSFModel(ref_psf_star_meta, ref_meta, info_frac=0.95, nbasis_max=3, poly_degree=3)\n",
    "ref_psf_model.psf_model_diagnosis(ref_psf_star_meta, output_path=output_path)\n",
    "\n",
    "new_psf_star_meta = imgdiff.PSFStar(new_meta, psf_size=psf_size, nstar_max=500)\n",
    "new_psf_model = imgdiff.PSFModel(new_psf_star_meta, new_meta, info_frac=0.95, nbasis_max=3, poly_degree=3)\n",
    "new_psf_model.psf_model_diagnosis(new_psf_star_meta, output_path=output_path)\n",
    "\n",
    "t1 = time.time()\n",
    "dt = t1 - t0\n",
    "print(f\"^_^ Science image is ready, {dt:.5f} seconds used\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "67e2431b-3da9-4d08-a4e0-9fabf5126613",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Total 865 objects detected on direct image\n",
      "    Total 792 objects detected on inverse image\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "0",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[0;31mSystemExit\u001b[0m\u001b[0;31m:\u001b[0m 0\n"
     ]
    }
   ],
   "source": [
    "# perform image difference and source detection\n",
    "diffObj = imgdiff.DiffImg(degrid=(12,12), nthreads=-1)\n",
    "diff_matrix = diffObj.diff(ref_meta, new_meta, ref_psf_model, new_psf_model,\n",
    "                               ref_mask=ref_mask.mask, new_mask=new_mask.mask)\n",
    "\n",
    "detObj = imgdiff.ExtractTrans(diff_image_abs, \n",
    "                              output_path, \n",
    "                              sex_config_file,\n",
    "                              sex_param_file,\n",
    "                              sex_exe=sex_exe,)\n",
    "detObj.extract_trans(diff_matrix, ref_meta, new_meta)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5fa93334-6d26-48c0-b3d4-1e30b354016a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "79b8cf4b-f581-4f33-b65d-17ca841d3655",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
