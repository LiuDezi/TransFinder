# The routine provides basic functions for image differencing
# Functions and classes:
# 1) BaseCheck
# 2) swarp_shell
# 3) sextractor_shell

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
