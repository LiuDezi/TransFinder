# The routine provides basic functions for image differencing
# Functions and classes:
# 1) BaseCheck
# 2) LoadMeta

# History:
# 1) /20240530/ Create this routine

from astropy.io import fits
import numpy as np
import os, sys, subprocess
import diffimg_utils as dutl

# basic informatiom
__version__ = "v1.0.1"
__version_date__ = "20240610"
__author__ = "Dezi Liu"

#__version__ = "v1.0.0"
#__version_date__ = "20240530"
#__author__ = "Dezi Liu"

class BaseCheck(object):
    """
    Check if SWarp and SExtractor are installed.
    If no, stop the routine
    If yes, return the shell command

    Check if the image and catalog are complete
    """
    def __init__(self):
        pass
    
    def swarp_shell(self):
        """
        Check if swarp is installed. 
        If no,  stop the routine.
        If yes, return the shell commond
        """
        swarp_list = ["swarp", "SWarp"]
        eid = [iswarp for iswarp in swarp_list if subprocess.getstatusoutput(iswarp)[0]==0]
        if len(eid)==0: sys.exit("!!! No SWarp installed")
        #assert len(eid)!=0, "!!! No swarp installed"
        return eid[0]

    def sextractor_shell(self):
        """
        Check if source extractor is installed
        If no, stop the routine.
        If yes, return the shell commond
        """
        sextractor_list = ["sex", "sextractor", "source-extractor"]
        eid = [isex for isex in sextractor_list if subprocess.getstatusoutput(isex)[0]==0]
        if len(eid)==0: sys.exit("!!! No SExtractor installed")
        #assert len(eid)!=0, "!!! No SExtractor installed"
        return eid[0]
    
    def header_check(self, image):
        """
        check if the image header includes the wcs parameters
        """
        image_header = fits.getheader(image)
        imagge_keys  = list(image_header.keys())
        wcs_keys     = self.__wcs_header()
        mres = set(wcs_keys).issubset(set(imagge_keys))
        if not mres: sys.exit("!!! Input image does not contain complete wcs keywords")
        #assert mres, "!!! Input image does not contain complete wcs keywords"
        return

    def gaia_catalog_check(self, catalog):
        """
        check if the gaia star catalog presents
        """
        if not os.path.exists(catalog): sys.exit("!!! No GAIA catalog provided")
        #assert os.path.exists(catalog), "!!! No GAIA catalog provided"
        return

    def __wcs_header(self):
        wcs_keys = ["RADESYS", "EQUINOX",
                    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
                    "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
                    "CD1_1", "CD1_2", "CD2_1", "CD2_2"]
        return wcs_keys


class ImageMeta(object):
    """
    Basic parameters for the resampled image
    """
    def __init__(self, band):
        self.band = band
    
    def resample_param(self):
        """
        Raw image parameters:
        Blue/Yellow channels: pixel_scale=0.426, image_size=(6144,6160)
        Red channel:          pixel_scale=0.286, image_size=(9216,9232)
        """
        pixel_scale = 0.45
        image_size = (5815, 5815)
        return pixel_scale, image_size
    
    def image_size(self):
        if self.band in ["u", "v", "g", "r"]:
            size = (6144, 6160)
        else:
            size = (9216, 9232)
        return size

    def gates_bound(self):
        xsize, ysize = self.image_size()
        gates_list = {}
        if self.band in ["u", "v", "g", "r"]:
            xmed, ymed = xsize//2, ysize//2
            gates_list[0] = [0,    xmed,     0, ymed]
            gates_list[1] = [xmed, xsize,    0, ymed]
            gates_list[2] = [xmed, xsize, ymed, ysize]
            gates_list[3] = [0,    xmed,  ymed, ysize]
        else:
            ngates = 16
            xmed, ymed = xsize//8, ysize//2
            for i in range(16):
                ii = i//8
                iy0, iy1 = 0+ymed*ii, ymed*(ii+1)
                if i<=7:
                    ix0, ix1 = 0+xmed*i, xmed*(i+1)
                else:
                    jj = 16-1-i
                    ix0, ix1 = 0+xmed*jj, xmed*(jj+1)
                gates_list[i] = [ix0, ix1, iy0, iy1]
        return gates_list

    def bad_column(self):
        """
        index in Pythonic style
        """
        if self.band in ["u", "v"]:
            xlim = [(0,0)]
            ylim = [(0,0)]
        elif self.band in ["g", "r"]:
            xlim = [(2148,2154)]
            ylim = [(697, 3079)]
        else: # self.band in ["i", "z"]
            xlim = [(4179,4183), (5693, 5697)]
            ylim = [(4615,7809), ( 480, 4616)]
        return xlim, ylim

class LoadMeta(object):
    """
    Load the image and catalog from a given image type (ref/new)

    Parameters:
    image: str
        name of input image with absolute path
    """
    def __init__(self, image):
        self.image = image
        self.catalog_full = image[:-4] + "phot_all.fits"
        self.catalog_star = image[:-4] + "phot_star.fits"

        self.image_meta()
        self.catalog_full_meta()
        self.catalog_star_meta()

    def image_meta(self,hdu=0):
        image_matrix, image_header = fits.getdata(self.image, header=True, ext=hdu)
        self.image_matrix = image_matrix.T
        self.image_header = image_header
        self.xsize     = image_header["NAXIS1"]
        self.ysize     = image_header["NAXIS2"]
        self.image_sigma = dutl.mad(image_matrix)
        return

    def catalog_full_meta(self, hdu=2):
        catalog_full_matrix = fits.getdata(self.catalog_full, ext=2)
        self.nobj = len(catalog_full_matrix)
        self.obj_matrix = catalog_full_matrix
        return

    def catalog_star_meta(self, hdu=1):
        catalog_star_matrix = fits.getdata(self.catalog_star, ext=1)
        self.nstar = len(catalog_star_matrix)
        self.star_matrix = catalog_star_matrix
        self.star_fwhm = np.median(catalog_star_matrix["FWHM_IMAGE"])
        return
    

