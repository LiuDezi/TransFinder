# Functions to process the input image and corresponding catalog
#
# Author: Dezi Liu at SWIFAR; E-mail: adzliu@ynu.edu.cn
#
# Input image format: FITS
# Input catalog format: SExtractor LDAC
# 
# Several classes are included in this module:
# 1) class imgCatInfo()
#    Obtain the basic information on the image and catalog
#
# 2) class PSFStar()
#    Construct the PSF star catalog by magnitude-size diagram or external 
#    catalog (e.g. GAIA; to be implemented)
#
# 3) class MaskStar()
#    mask saturated stars
#
# Version history:
# 0) Start time is lost

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.modeling import fitting, models
import numpy as np
import os, sys

# Masking saturated stars
class MaskStar(object):
    """
    Class to mask saturated stars. Saturated stars are flagged 
    in Sextractor as FLAGS=4. Therefore, here we simply select 
    these stars and mask them with circular regions. The radius 
    of the circle is determined by the area of segmentation, 
    and a scaling factor can be applied to enlarge the radius.

    Parameters:
    catalog: 2D array
        photometric catalog of a given image
    imgSize: tuple
        pixel size of the given image: (xsize, ysize)
    scale: float
        scale parameter applied to the original mask circle
    """
    def __init__(self, data_meta, scale=1.5):
        self.__catalog = data_meta.obj_matrix
        self.__xsize   = data_meta.xsize
        self.__ysize   = data_meta.ysize
        self.scale     = scale
        
        self.mask      = self.circular_mask()

    def circular_mask(self):
        """
        Calculate the circular masking regions of saturated stars

        Parameters:
        xsize: int
            image size in x-axis
        ysize: int
            image size in y-axis
        satPar: dict
            parameters of saturated stars, with format of
            satPar = {ID:[xcen, ycen, mask_radius], ...}
        scale: float
            scale parameter applied to the original mask circular

        Return:
            Bool matrix with masking pixels assigned as True
        """
        saturate_param = self.select_saturated_stars()
        xsize, ysize = self.__xsize, self.__ysize
        assert type(saturate_param)==dict, "!!! Parameter 'saturate_param' must be a dictionary."
        mask = np.full((xsize,ysize),False)

        xstep, ystep = np.ogrid[:xsize, :ysize]
        for sid, ipar in saturate_param.items():
            ixcen, iycen, irad = ipar
            irad = irad * self.scale

            # extract the subimage to speed up
            irad_int = int(np.ceil(irad))
            ixcen_int, iycen_int = int(round(ixcen-1)), int(round(iycen-1))
            ix0, ix1 = ixcen_int-irad_int, ixcen_int+irad_int
            iy0, iy1 = iycen_int-irad_int, iycen_int+irad_int
            
            ixstep_sub, iystep_sub = xstep[ix0:ix1+1,:], ystep[:,iy0:iy1+1]
            ixcen_new, iycen_new = np.median(ixstep_sub), np.median(iystep_sub)
            idist = np.sqrt((ixstep_sub-ixcen_new)**2 + (iystep_sub-iycen_new)**2)
            mask[ix0:ix1+1,iy0:iy1+1] += idist <= irad
            
            # classical method
            #idist = np.sqrt((xstep-ixcen+1.0)**2 + (ystep-iycen+1.0)**2)
            #mask += idist <= irad
        self.mask = mask
        return self.mask

    def select_saturated_stars(self):
        """
        Find the saturated stars in a given catalog.
        The catalog is generated by Sextractor in LDAC format, in which 
        the X/Y_Image and FLAGS should provided
        """
        # print("^_^ Total %d objects"%nobj)
        ximg = self.__catalog["X_IMAGE"]
        yimg = self.__catalog["Y_IMAGE"]
        flag = self.__catalog["FLAGS"]
        xmin = self.__catalog["XMIN_IMAGE"]
        xmax = self.__catalog["XMAX_IMAGE"]
        ymin = self.__catalog["YMIN_IMAGE"]
        ymax = self.__catalog["YMAX_IMAGE"]

        #find the saturated stars
        nstar   = 0
        saturate_param = {}
        nobj = len(self.__catalog)
        for i in range(nobj):
            iflag = int(flag[i])
            ipow  = self.decomp_flags(iflag)
            if 4 not in ipow: continue
            ixmax = np.max([ximg[i]-xmin[i], xmax[i]-ximg[i]])
            iymax = np.max([yimg[i]-ymin[i], ymax[i]-yimg[i]])
            irad  = np.max([ixmax, iymax])
            nstar += 1
            saturate_param[i] = [ximg[i],yimg[i],irad]
    
        print("    Total %s saturated stars"%nstar)
        
        self.saturate_param = saturate_param
        return self.saturate_param

    def decomp_flags(self, flag):
        """
        Decompose Sextractor flags into single numbers of power 2
    
        Parameter:
        flag: int
              Sextractor FLAGS
        Example: 
            pow = decompFlags(25)
            pow = [1, 8, 16]
        """
        assert type(flag)==int, "!!! flag must be integer."
        powers = []
        spow = 1
        while spow <= flag:
            if spow & flag: powers.append(spow)
            spow <<= 1
        return powers

# select PSF stars
class PSFStar(object):
    """
    select stars from a given catalog for PSF modelling

    Parameters:
    catalog: array
        Catalog in SEXtractor LDAC_FITS format
    image: array
        Image array
    snrLim: float
        minimum SNR of the selected PSF point sources
    stmSize: int
        pixel size of the image stamps of PSF stars
    regout: name of the output region file
    figout: name of the output PNG figure

    Return:
    IDs of the selected objects for each table
    """
    def __init__(self, data_meta):
        self.__image   = data_meta.image_matrix
        self.__catalog = data_meta.star_matrix
        self.__fwhm    = data_meta.star_fwhm
        self.__nstar   = data_meta.nstar

        self.select_star()

    def select_star(self):
        """
        select stars from a given catalog for modelling PSF

        Parameters:
        catn: string
            Catalog name in SEXtractor LDAC_FITS format
        snr_lim: float
            minimum SNR of the selected PSF point sources
        regout: name of the output region file
        figout: name of the output PNG figure

        Return:
        IDs of the selected objects for each table
        """
        # catalog analysis
        #psf_xpos = self.__catalog["XWIN_IMAGE"]
        #psf_ypos = self.__catalog["YWIN_IMAGE"]
        
        sid_bright = np.argsort(self.__catalog["SNR_WIN"])[::-1]
        psf_xpos = self.__catalog["XWIN_IMAGE"][sid_bright]
        psf_ypos = self.__catalog["YWIN_IMAGE"][sid_bright]

        psf_size = int(7.0 * self.__fwhm)
        if psf_size%2==0: psf_size += 1
        psf_size = np.max([51, psf_size])
        print(f"    PSF stamp size is {psf_size}*{psf_size}")
        
        half_size = int((psf_size-1)/2)
        kernel_interp = Gaussian2DKernel(x_stddev=5)
        nstar_psf, sid = 0, []
        for i in range(self.__nstar):
            if nstar_psf>=500: break
            ix, iy = psf_xpos[i]-1.0, psf_ypos[i]-1.0
            ix, iy = int(round(ix)), int(round(iy))
            ix0, ix1 = ix-half_size, ix+half_size
            iy0, iy1 = iy-half_size, iy+half_size
            istm = self.__image[ix0:ix1+1,iy0:iy1+1]
            if istm.shape != (psf_size, psf_size): continue
            if np.sum(np.isnan(istm))>0: continue
            istm[istm<=0.0] = np.nan
            istm = interpolate_replace_nans(istm, kernel_interp)
            istm = istm/np.sum(istm)
            nstar_psf += 1
            sid += [i]
            if nstar_psf==1:
                psf_matrix = istm
            else:
                psf_matrix = np.dstack((psf_matrix,istm))

        self.psf_size = psf_size
        self.psf_matrix = psf_matrix
        self.psf_nstar = nstar_psf
        self.psf_xpos = psf_xpos[sid]
        self.psf_ypos = psf_ypos[sid]

        return

class PSFModel(object):
    """
    Two tasks to perform:
    1) spatially varied PSF modelling
    2) image differencing in Fouries space

    Parameters:
    psf_star: class
        parameters of PSF stars to model the image PSF
    imgCatPar: class
        parameters related to input image and catalog
    info_frac: float
        fraction of effective infomation used for PCA analysis
    poly_degree: int
        degree to model the PSF variation over the entire image
    """
    def __init__(self, psf_star=None, image_size=(6000,6000), info_frac=0.8, poly_degree=3):
        assert 0.0<info_frac<1.0, "!!! Parameter 'info_frac' should be within (0,1)"
        assert type(poly_degree)==int, "!!! Parameter 'poly_degree' should be integer"

        self.__psf_star = psf_star
        self.image_size = image_size
        self.info_frac  = info_frac
        self.poly_degree = poly_degree

        self.psf_basis()
        self.basis_coeff()

    def __psf_covmat(self):
        """
        Calculate the covariance matrix of PSF stars
        """
        print("    Total %d PSF stars"%self.__psf_star.psf_nstar)
        size_flat = self.__psf_star.psf_size * self.__psf_star.psf_size
        psf_mat_flat = np.zeros((size_flat, self.__psf_star.psf_nstar))
        for i in range(self.__psf_star.psf_nstar):
            psf_mat_flat[:,i] = self.__psf_star.psf_matrix[:,:,i].flatten()
        self.cov_matrix = np.cov(psf_mat_flat, rowvar=False)
        return self.cov_matrix

    def __eigen_decom(self):
        """
        calculate the eigenvalues and eigenvectors of covariance matrix
        using the scipy module

        Return eigenvalues in descendent order
        """
        #evalues, evectors = scipy.linalg.eigh(self.cov_matrix)
        evalues, evectors = np.linalg.eigh(self.cov_matrix)
        self.eigval = evalues[::-1]
        self.eigvec = evectors[:,::-1]
        return self.eigval, self.eigvec

    def psf_basis(self):
        """
        derive the PSF basis functions
        """
        cov_matrix = self.__psf_covmat()
        valm, vecm = self.__eigen_decom()
        ptot = abs(valm)/sum(abs(valm))
        nbasis = 1
        pfrac  = ptot[0]
        while pfrac<self.info_frac:
            pfrac  += ptot[nbasis]
            nbasis += 1
        print("    Total %d PSF basis, accounting %6.4f info."%(nbasis,pfrac))

        vecm = vecm[:,:nbasis]
        psfBasis = np.zeros((self.__psf_star.psf_size,self.__psf_star.psf_size,nbasis))
        for i in range(nbasis):
            base = np.zeros((self.__psf_star.psf_size,self.__psf_star.psf_size))
            for j in range(self.__psf_star.psf_nstar):
                jpsf  = self.__psf_star.psf_matrix[:,:,j]
                base += vecm[j, i] * jpsf
            base  = base / np.sum(base)
            base  = base - np.abs(np.min(base))
            base  = base / np.sum(base)
            psfBasis[:,:,i] = base

        self.psf_base = psfBasis
        self.psf_nbase   = nbasis
        return self.psf_base

    def basis_coeff(self):
        """
        construct the coefficient fields based on the PSF base and star positions.
        """
        if self.psf_nbase == 1:
            self.psf_base_coeff = [None]
            return
        ximg = self.__psf_star.psf_xpos
        yimg = self.__psf_star.psf_ypos

        coeffBasis = []
        eigvecEff  = np.flip(self.eigvec[:,:self.psf_nbase].T,0)
        for i in range(self.psf_nbase):
            polyModel  = models.Polynomial2D(degree=self.poly_degree)
            fitter     = fitting.LinearLSQFitter()
            coeffModel = fitter(polyModel,ximg,yimg,eigvecEff[i, :])
            coeffBasis.append(coeffModel)
        self.psf_base_coeff = coeffBasis
        return self.psf_base_coeff

    def coeff_field(self):
        """
        grid of coordinates used for coefficient field construction
        """
        xgrid, ygrid = np.mgrid[:self.image_size[0], :self.image_size[1]]
        return xgrid, ygrid

