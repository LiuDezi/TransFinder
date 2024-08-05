# The routine provides basic functions for image differencing
# Functions and classes:
# 1) BaseCheck
# 2) LoadMeta

import numpy as np
from astropy.table import Table
from astropy.io import fits
import os
import sys
import subprocess

class BaseCheck(object):
    """
    Check if SWarp and SExtractor are installed.
    If no, stop the routine
    If yes, return the shell command

    Check if the image and catalog are complete
    """
    def __init__(self):
        pass
    
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

    def file_check(self, filename):
        """
        check if input filename presents
        """
        if not os.path.exists(filename): sys.exit(f"!!! {filename} does not exist")
        return

    def __wcs_header(self):
        wcs_keys = ["RADESYS", "EQUINOX",
                    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
                    "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
                    "CD1_1", "CD1_2", "CD2_1", "CD2_2"]
        return wcs_keys

class LoadMeta(object):
    """
    Load the image and catalog from a given image type (ref/new)

    Parameters:
    image: str
        name of input image with absolute path
    """
    def __init__(self, image, catalog=None):
        self.image = image
        self.catalog = self.image[:-4] + "phot.fits"
        
        if catalog is not None: self.catalog = catalog

        self.image_meta()
        self.catalog_meta()

    def image_meta(self,hdu=0):
        image_matrix, image_header = fits.getdata(self.image, header=True, ext=hdu)
        self.image_matrix = image_matrix.T
        self.image_header = image_header
        self.pixel_scale = image_header["PSCALE"]
        self.xsize = image_header["NAXIS1"]
        self.ysize = image_header["NAXIS2"]
        self.image_sigma = image_header["BKGSIG"]
        self.star_fwhm = image_header["FWHM"]
        self.nstar = image_header["NSTAR"]

        psf_size = int(7.0 * self.star_fwhm)
        if np.mod(psf_size,2)==0: psf_size += 1
        self.psf_size = np.max([31, psf_size])
        return

    def catalog_meta(self):
        catalog_matrix = Table.read(self.catalog, format="fits")
        self.nobj = len(catalog_matrix)
        self.obj_matrix = catalog_matrix
        
        gid = (catalog_matrix["RA_BASE"]!=-99.0) & (catalog_matrix["DEC_BASE"]!=-99.0)
        self.star_matrix = catalog_matrix[gid]
        return

def swarp_shell():
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

def sextractor_shell():
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

