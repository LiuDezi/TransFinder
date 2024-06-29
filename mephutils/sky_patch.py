# this function is used to define the sky regions for transient storage

import numpy as np
import os, sys
from astropy.io import fits

def regions(dec_limit, ra_interval=10.0, dec_interval=10.0):
    dec_lower, dec_upper = dec_limit
    ra   = np.arange(0.0, 360.0, ra_interval)
    dec  = np.arange(dec_lower, dec_upper, dec_interval)
    print(ra)

if __name__ == "__main__":
    regions([-10, 80])
