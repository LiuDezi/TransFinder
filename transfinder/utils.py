# some functions to do dirty things

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from collections import Counter
from scipy.spatial import KDTree
import os

def deg2str(ra, dec):
    """
    convert ra&dec in (degree, degree) to (hhmmss, ddmmss),

    Parameters:
    ra, dec: 
       if (ra, dec) is in (deg, deg), float
    
    Return:
       ra  = {0:HH:MM:SS.SS, 
              1:HHMMSS.SS}
       dec = {0:(+/-)DD:MM:SS.SS,
              1:(P/N)DD:MM:SS.SS,
              2:(+/-)DDMMSS.SS,
              3:(P/N)DDMMSS.SS}
    Example:
    deg2str(1.0, 1.0, output_fmt=[0,1])
    """
    rah = ra/15.0
    ram = (rah - int(rah))*60.0
    ras = (ram - int(ram))*60.0

    ra_list = [f"0{int(rah)}"[-2:], f"0{int(ram)}"[-2:], f"0{float(ras):.2f}"[-5:]]
    sra = {0: ":".join(ra_list),
           1: "".join(ra_list)}

    decabs = abs(dec)
    decd   = int(decabs)
    decm   = (decabs - decd)*60.0
    decs   = (decm - int(decm))*60.0
    dec_list = [f"0{int(decd)}"[-2:], f"0{int(decm)}"[-2:], f"0{float(decs):.2f}"[-5:]]
    if dec < 0.0:
        sdec = {0: "-" + ":".join(dec_list),
                1: "N" + ":".join(dec_list),
                2: "-" + "".join(dec_list),
                3: "N" + "".join(dec_list)}
    else:
        sdec = {0: "+" + ":".join(dec_list),
                1: "P" + ":".join(dec_list),
                2: "+" + "".join(dec_list),
                3: "P" + "".join(dec_list)}

    return sra, sdec

def str2deg(ra, dec):
    """
    convert ra&dec in (hhmmss, ddmmss) to (degree, degree)

    Parameters:
    ra, dec: string in (hms, dms)

    Example:
    str2deg("00:00:00.00", "+00:00:00.00")
    str2deg("00:00:00.00", "N00:00:00.00")
    """
    decSign = dec[0]
    
    sra = np.array(ra.split(":"), dtype=float)
    sra = ((sra[-1]/60.0+sra[1])/60.0 + sra[0])*15.0
    
    sdec = np.array(dec[1:].split(":"), dtype=float)
    if decSign in ["-", "N"]:
        sdec = -((sdec[-1]/60.0+sdec[1])/60.0 + abs(sdec[0]))
    elif decSign in ["+", "P"]:
        sdec = (sdec[-1]/60.0+sdec[1])/60.0 + sdec[0]
    else:
        raise ValueError("!!! Give a right dec value")
    
    return sra, sdec


def crossmatch(ra1, dec1, ra2, dec2, aperture=1.0):
    """
    Match two sets of on-sky coordinates to each other.
    I.e., find nearest neighbor of one that's in the other.
    """
    """
    Finds matches in one catalog to another.
    
    Parameters
    ra1 : array-like
          Right Ascension in degrees of the first catalog
    dec1 : array-like
          Declination in degrees of the first catalog (shape of array must match `ra1`)
    ra2 : array-like
          Right Ascension in degrees of the second catalog
    dec2 : array-like
          Declination in degrees of the second catalog (shape of array must match `ra2`)
    aperture : cross-matching aperture, float, default 1.0"
                How close (in arcseconds) a match has to be to count as a match.
    Returns
    -------
    idx1 : int array
           Indecies into the first catalog of the matches. Will never be
           larger than `ra1`/`dec1`.
    idx2 : int array
           Indecies into the second catalog of the matches. Will never be
           larger than `ra1`/`dec1`.
    """
    ra1  = np.array(ra1, copy=False)
    dec1 = np.array(dec1, copy=False)
    ra2  = np.array(ra2, copy=False)
    dec2 = np.array(dec2, copy=False)

    if ra1.shape != dec1.shape:
        raise ValueError('!! ra1 and dec1 do not match!')
    if ra2.shape != dec2.shape:
        raise ValueError('!! ra2 and dec2 do not match!')

    nobj1, nobj2 = len(ra1), len(ra2)
    if nobj1 > nobj2:
        ra1, ra2 = ra2, ra1
        dec1, dec2 = dec2, dec1

    x1, y1, z1 = _spherical_to_cartesian(ra1.ravel(), dec1.ravel())
    coords1 = np.empty((x1.size, 3))
    coords1[:, 0], coords1[:, 1], coords1[:, 2] = x1, y1, z1

    x2, y2, z2 = _spherical_to_cartesian(ra2.ravel(), dec2.ravel())
    coords2 = np.empty((x2.size, 3))
    coords2[:, 0], coords2[:, 1], coords2[:, 2] = x2, y2, z2

    kdt   = KDTree(coords2)
    idxs2 = kdt.query(coords1)[1]

    ds    = _great_circle_distance(ra1, dec1, ra2[idxs2], dec2[idxs2]) # in arcsecond
    idxs1 = np.arange(ra1.size)

    msk = ds < aperture
    idxs1, idxs2 = idxs1[msk], idxs2[msk]
    ds = ds[msk]
    # Usually, there is duplicate ID in idxs2, here we only keep the one with smaller distance
    dupid = [xx for xx, yy in Counter(idxs2).items() if yy > 1]
    if len(dupid) > 0:
        badid = np.array([])
        for k in dupid:
            kid   = np.where(idxs2==k)[0]
            nkid  = np.delete(kid,ds[kid].argmin())
            badid = np.append(badid,nkid)
        badid = badid.astype(int)
        idxs1, idxs2 = np.delete(idxs1,badid), np.delete(idxs2,badid)

    if nobj1 > nobj2:
        newid = np.argsort(idxs2)
        idxs1, idxs2 = idxs2[newid], idxs1[newid]

    return idxs1, idxs2

def _spherical_to_cartesian(ra, dec):
    """
    (Private internal function)
    Inputs in degrees. Outputs x,y,z
    """
    rar  = np.radians(ra)
    decr = np.radians(dec)

    x    = np.cos(rar) * np.cos(decr)
    y    = np.sin(rar) * np.cos(decr)
    z    = np.sin(decr)

    return x, y, z

def _great_circle_distance(ra1, dec1, ra2, dec2):
    """
    (Private internal function)
    Returns great ciircle distance. Inputs in degrees.

    Uses vicenty distance formula - a bit slower than others, but
    numerically stable.
    """
    lambs = np.radians(ra1)
    phis  = np.radians(dec1)
    lambf = np.radians(ra2)
    phif  = np.radians(dec2)

    dlamb = lambf - lambs

    numera = np.cos(phif) * np.sin(dlamb)
    numerb = np.cos(phis)*np.sin(phif) - np.sin(phis)*np.cos(phif)*np.cos(dlamb)
    numer  = np.hypot(numera, numerb)
    denom  = np.sin(phis)*np.sin(phif) + np.cos(phis)*np.cos(phif)*np.cos(dlamb)
    return np.degrees(np.arctan2(numer, denom))*3600.0 # convert to arcsecond

def wds9reg(x,y,flag=None,radius=5.0,unit="arcsec",color="green",outfile="out.reg"):
    """
    Write ds9 region file.

    Parameters:
    coordinate: 2D array
       coordinate to be written.  It could be image coordinates or RA/Dec.  
       Former, set unit="pixel"
       Later, set unit="arcsec"
    radius: float
       in unit of pixels or arcsec (when 'unit' keyword is set)
    unit: string
        pixel: write region file in unit of pixels
        arcsec (default): write region file in unit of RA and Dec
    color: string
       to specify which color to use.  Default is green
    outfile: string
       name of output region file
    
    Return:
       "outfile": can be read by ds9

    Example:
        pos = [100, 200]
        wds9reg(pos,outfile="pos.reg")
    """
    if not unit in ["arcsec","pixel"]:
        raise ValueError("!! Please set 'unit' as 'arcsec' or 'pixel'")

    fileobj = open(outfile, "w")
    note0 = "# Region file for DS9\n"
    global_pro1 = "global color=%s font='helvetica 10 normal' "%color
    global_pro2 = "select=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n"
    fileobj.write(note0)
    fileobj.write(global_pro1+global_pro2)

    if unit == "arcsec":
        fileobj.write("fk5\n")
        fmt = 'circle(%10.6f,%10.6f,%5.2f")\n'
        if flag is not None: fmt='circle(%10.6f,%10.6f,%5.2f") # text={%d}\n'
    if unit == "pixel":
        fileobj.write("image\n")
        fmt = 'circle(%10.6f,%10.6f,%5.2f)\n'
        if flag is not None: fmt='circle(%10.6f,%10.6f,%5.2f) # text={%d}\n'

    for i in range(len(x)):
        if flag is not None:
            ds9row = fmt%(x[i],y[i],radius,flag[i])
        else:
            ds9row = fmt%(x[i], y[i], radius)
        fileobj.write(ds9row)
    fileobj.close()

    return

def mad(data):
    """
    Median absolute deviation, which is defined as
    MAD = median(abs(data-median(data)))

    If data follows normal distributon, then the relationship between 
    the standard deviation (std) of the data and MAD is
    std = 1.4826*MAD

    Return: Normal like-MAD
    """
    mm = np.median(data)
    mx = np.median(abs(data-mm))
    return 1.4826*mx

def poly_index(ndeg):
    """
    index list for a given 2D polynomial with degree of n
    """
    index_list = []
    for p in range(ndeg+1):
        for i in range(p+1):
            j = p-i
            index_list.append((i,j))
    return index_list

def sub_regions(xsize, ysize, image_grid=(20, 20), extend=(0, 0)):
    """
    divide entire image into sub-regions
    """
    xgrid, ygrid = image_grid
    dx, dy = xsize/xgrid, ysize/ygrid
    ex, ey = extend
    crd_grids, iid = {}, 0
    for i in range(xgrid):
        ix0, ix1 = int(round(i*dx)), round((i+1)*dx)
        iex0, iex1 = max([ix0-ex, 0]), min([ix1+ex, xsize])
        icx, iecx = (ix0+ix1-1)//2, (iex0+iex1-1)//2
        for j in range(ygrid):
            jy0, jy1 = int(round(j*dy)), int(round((j+1)*dy))
            jey0, jey1 = max([jy0-ey, 0]), min([jy1+ey, ysize])
            jcy,  jecy = (jy0+jy1-1)//2, (jey0+jey1-1)//2
            
            oregion = [icx,  jcy,  ix0,  ix1,  jy0,  jy1]
            eregion = [iecx, jecy, iex0, iex1, jey0, jey1]
            crd_grids[iid] = [oregion, eregion]
            iid += 1
    return crd_grids

