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

import galsim
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from scipy import signal, linalg, fft, optimize
from photutils.segmentation import SourceFinder
from photutils.segmentation import make_2dgaussian_kernel
import warnings
import os
import sys

from ..utils import wds9reg, poly_index

#some potential useful modules
#from photutils.segmentation import SourceFinder, SourceCatalog
#from photutils.background import Background2D, MedianBackground
#from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
#from scipy.fft import fft2, ifft2
#from astropy.convolution import convolve_fft
#from scipy.signal import fftconvolve

# Masking saturated stars
class MaskStar(object):
    """
    Class to mask saturated stars. Saturated stars are flagged 
    in Sextractor as FLAGS=4. Therefore, here we simply select 
    these stars and mask them with circular regions. The radius 
    of the circle is determined by the area of segmentation, 
    and a scaling factor can be applied to enlarge the radius.

    Parameters:
    scale: float
        scale parameter applied to the original mask circle
    """
    def __init__(self, data_meta, scale=1.5):
        self.scale = scale
        self.mask = self.circular_mask(data_meta)

    def circular_mask(self, data_meta):
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
        saturate_param = self.select_saturated_stars(data_meta)
        xsize, ysize = data_meta.xsize, data_meta.ysize
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
            
            # classical method is slow in the case of many saturated stars
            #idist = np.sqrt((xstep-ixcen+1.0)**2 + (ystep-iycen+1.0)**2)
            #mask += idist <= irad
        return mask

    def select_saturated_stars(self, data_meta):
        """
        Find the saturated stars in a given catalog.
        The catalog is generated by Sextractor in LDAC format, in which 
        the X/Y_Image and FLAGS should provided
        """
        # print("^_^ Total %d objects"%nobj)
        ximg = data_meta.obj_matrix["XWIN_IMAGE"]
        yimg = data_meta.obj_matrix["YWIN_IMAGE"]
        flag = data_meta.obj_matrix["FLAGS"]
        xmin = data_meta.obj_matrix["XMIN_IMAGE"]
        xmax = data_meta.obj_matrix["XMAX_IMAGE"]
        ymin = data_meta.obj_matrix["YMIN_IMAGE"]
        ymax = data_meta.obj_matrix["YMAX_IMAGE"]

        #find the saturated stars
        nstar   = 0
        saturate_param = {}
        for i in range(data_meta.nobj):
            iflag = int(flag[i])
            ipow  = self.decomp_flags(iflag)
            if 4 not in ipow: continue
            ixmax = np.max([ximg[i]-xmin[i], xmax[i]-ximg[i]])
            iymax = np.max([yimg[i]-ymin[i], ymax[i]-yimg[i]])
            irad  = np.max([ixmax, iymax])
            nstar += 1
            saturate_param[i] = [ximg[i],yimg[i],irad]
    
        print(f"    Total {nstar} saturated stars")
        return saturate_param

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
    def __init__(self, data_meta, psf_size=None, nstar_max=None):
        if psf_size is None:
            psf_size = data_meta.psf_size
        if nstar_max is None: 
            nstar_max = 500

        self.psf_size = psf_size
        self.nstar_max = nstar_max

        self.select_star(data_meta)

    def select_star(self, data_meta):
        """
        Select stars from a given catalog for modelling PSF
        If the number of stars with good quality is larger than 500,
        only the first 500 brightest stars are used (to speed up).

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
        warnings.filterwarnings("ignore")
        # catalog analysis
        sid_bright = np.argsort(data_meta.star_matrix["SNR_WIN"])[::-1]
        psf_xpos = data_meta.star_matrix["XWIN_IMAGE"][sid_bright] - 1.0
        psf_ypos = data_meta.star_matrix["YWIN_IMAGE"][sid_bright] - 1.0
        psf_flux = data_meta.star_matrix["FLUX_AUTO"][sid_bright]

        psf_sigma = data_meta.image_sigma * self.psf_size
        half_size = self.psf_size//2
        # Method 1: the following commented codes are not accurate enough
        #kernel_interp = Gaussian2DKernel(x_stddev=5)
        #nstar_psf, sid = 0, []
        #for i in range(data_meta.nstar):
        #    if nstar_psf>=nstar_max: break
        #    ix, iy = psf_xpos[i], psf_ypos[i]
        #    iflux  = psf_flux[i]
        #    ix_int, iy_int = int(round(ix)), int(round(iy))
        #    ix_offset, iy_offset = ix - ix_int, iy - iy_int
        #    ix0, ix1 = ix_int-half_size, ix_int+half_size
        #    iy0, iy1 = iy_int-half_size, iy_int+half_size
        #    istm = self.data_meta.image_matrix[ix0:ix1+1,iy0:iy1+1]
        #    if istm.shape != (self.psf_size, self.psf_size): continue   # locate at image edge
        #    if np.sum(istm==0.0)>0: continue
        #    if np.sum(np.isnan(istm))>0: continue             # mask pixels
        #
        #    # normalize the star stamp by the Kron flux (correct for the scale flux)
        #    istm = istm/iflux
        #    istm_sigma = np.sqrt(psf_sigma**2 + iflux)/iflux
        #    #print(np.std(istm)*self.psf_size, istm_sigma, np.sum(istm))
        #    if abs(np.sum(istm)-1.0)>2.0*istm_sigma: continue  # reject close neighbors
        #    
        #    # refine the center of PSF star
        #    istm_obj = galsim.Image(istm, scale=data_meta.pixel_scale)
        #    istm_interp = galsim.InterpolatedImage(istm_obj,x_interpolant=galsim.Lanczos(7),normalization='flux')
        #    interp_obj = galsim.ImageF(self.psf_size, self.psf_size, scale=data_meta.pixel_scale)
        #    istm_interp = istm_interp.drawImage(interp_obj, method='no_pixel', offset=(-ix_offset, -iy_offset))
        #    istm = istm_interp.array
        #
        #    nstar_psf += 1
        #    sid += [i]
        #    if nstar_psf==1:
        #        psf_matrix = istm.reshape(-1,1)
        #    else:
        #        psf_matrix = np.hstack((psf_matrix, istm.reshape(-1,1)))
        
        # Method 2: this method is more accurate than Method 1. We first use photutils to
        #           generate segmentation map to exclude close neighbours. And then run 
        #           galsim to refine the center of a star to eliminate dipole effect.
        nstar_psf, sid = 0, []
        psf_xpos_new, psf_ypos_new = [],  []
        for i in range(data_meta.nstar):
            if nstar_psf>=self.nstar_max: break
            ix, iy = psf_xpos[i], psf_ypos[i]
            iflux  = psf_flux[i]
            ix_int, iy_int = int(round(ix)), int(round(iy))
            ix0, ix1 = ix_int-half_size, ix_int+half_size
            iy0, iy1 = iy_int-half_size, iy_int+half_size
            istm = data_meta.image_matrix[ix0:ix1+1,iy0:iy1+1]
            if istm.shape != (self.psf_size, self.psf_size): continue   # locate at image edge
            if np.sum(istm==0.0)>0: continue
            if np.sum(np.isnan(istm))>0: continue             # mask pixels
            
            iex0, iex1 = ix_int-self.psf_size, ix_int+self.psf_size
            iey0, iey1 = iy_int-self.psf_size, iy_int+self.psf_size
            istm_mad = mad_std(data_meta.image_matrix[iex0:iex1,iey0:iey1])
            # normalize the star stamp by the Kron flux (correct for the scale flux)
            istm = istm/iflux
            istm_sigma = np.sqrt((istm_mad*self.psf_size)**2 + iflux)/iflux
            #print(np.std(istm)*self.psf_size, istm_sigma, np.sum(istm))
            if abs(np.sum(istm)-1.0)>2.0*istm_sigma: continue  # reject if close and bright neighbors present

            # refine the stamp
            iker_sig = np.max([3.0, data_meta.star_fwhm/2.35])
            iker_size = int(3.0*iker_sig)
            ikernel = make_2dgaussian_kernel(iker_sig, size=iker_size)
            istm_conv = signal.oaconvolve(istm, ikernel, mode="same")
            ifinder = SourceFinder(npixels=10, progress_bar=False)
            iseg = ifinder(istm_conv, 1.5*istm_mad/iflux)
            if iseg is None: continue
            tid = iseg.data[half_size, half_size]
            if tid==0: continue
            
            ibadpix = (iseg.data!=tid) & (iseg.data!=0)
            istm[ibadpix] = np.random.normal(0.0, istm_mad/iflux, size=(self.psf_size,self.psf_size))[ibadpix]

            # refine the center of PSF star
            istm_obj = galsim.Image(istm, scale=data_meta.pixel_scale)
            ibadpix_obj = galsim.Image(ibadpix.astype(int), scale=data_meta.pixel_scale)
            try:
                istm_moment = istm_obj.FindAdaptiveMom(badpix=ibadpix_obj)
            except:
                continue
            ix_moment = istm_moment.moments_centroid.x - 1.0
            iy_moment = istm_moment.moments_centroid.y - 1.0
            ix_offset, iy_offset = ix_moment-half_size, iy_moment-half_size
            #ixy_offset = np.sqrt(ix_offset**2+iy_offset**2)
            istm_interp = galsim.InterpolatedImage(istm_obj,x_interpolant=galsim.Lanczos(7),normalization='flux')
            istm_interp = istm_interp.drawImage(nx=self.psf_size, ny=self.psf_size, scale=data_meta.pixel_scale, 
                                                method='no_pixel', offset=(-ix_offset, -iy_offset))
            istm = istm_interp.array
            #fits.writeto(f"zstar{i}.fits", istm, overwrite=True)

            nstar_psf += 1
            sid += [i]
            psf_xpos_new.append(ix_offset + ix_int)
            psf_ypos_new.append(iy_offset + iy_int)
            if nstar_psf==1:
                psf_matrix = istm.reshape(-1,1)
            else:
                psf_matrix = np.hstack((psf_matrix, istm.reshape(-1,1)))
            
        print(f"    Total {nstar_psf} PSF stars are selected")
        self.psf_matrix = psf_matrix
        self.psf_nstar = nstar_psf
        self.psf_xpos = np.array(psf_xpos_new)
        self.psf_ypos = np.array(psf_ypos_new)
        #self.psf_xpos = psf_xpos[sid]
        #self.psf_ypos = psf_ypos[sid]
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
    def __init__(self, psf_star_meta, data_meta, info_frac=0.8, nbasis_max=3, poly_degree=3):
        assert 0.0<info_frac<1.0, "!!! Parameter 'info_frac' should be within (0,1)"
        assert type(poly_degree)==int, "!!! Parameter 'poly_degree' should be integer"

        self.xsize = data_meta.xsize
        self.ysize = data_meta.ysize
        self.psf_size = psf_star_meta.psf_size
        self.info_frac  = info_frac
        self.nbasis_max = nbasis_max
        self.poly_degree = poly_degree
        self.poly_ncoeff = (poly_degree+1)*(poly_degree+2)//2
        #self.poly_indices = self.poly_index()
        self.poly_indices = poly_index(poly_degree)

        self.psf_basis_estimator(psf_star_meta)
        self.psf_field_estimator(psf_star_meta)
        self.psf_norm_estimator()

    def __interp_grid(self):
        """
        grid of coordinates used for coefficient field construction
        """
        xgrid, ygrid = np.mgrid[:self.xsize, :self.ysize]
        return xgrid, ygrid

    #def poly_index(self):
    #    index_list = []
    #    for p in range(self.poly_degree+1):
    #        for i in range(p+1):
    #            j = p-i
    #            index_list.append((i,j))
    #    return index_list

    def psf_covmat(self, psf_star_meta):
        """
        Calculate the covariance matrix of PSF stars
        """
        cov_matrix = np.cov(psf_star_meta.psf_matrix, rowvar=False)
        return cov_matrix

    def eigen_decom(self, cov_matrix):
        """
        calculate the eigenvalues and eigenvectors of covariance matrix
        using the scipy module

        Return eigenvalues in descendent order
        """
        evalues, evectors = np.linalg.eigh(cov_matrix)
        #evalues, evectors = linalg.eigh(cov_matrix)
        # note that: the sign of eigenvertors returned by numpy and scipy are different
        eigval = evalues[::-1]
        eigvec = evectors[:,::-1]
        return eigval, eigvec

    def psf_basis_estimator(self, psf_star_meta):
        """
        derive the PSF basis functions. At most three PSF basis functions are kept
        """
        print("    Estimate PSF basis functions")
        cov_matrix = self.psf_covmat(psf_star_meta)
        valm, vecm = self.eigen_decom(cov_matrix)
        ptot = abs(valm)/sum(abs(valm))
        nbasis = 1
        pfrac, pfrac_max  = ptot[0], ptot[0]
        while pfrac<self.info_frac:
            pfrac  += ptot[nbasis]
            nbasis += 1
            if nbasis==self.nbasis_max: pfrac_max = pfrac
        print(f"    Total {nbasis} PSF basis functions, accounting {pfrac} info")
        
        if nbasis>self.nbasis_max:
            print(f"!!! Number of PSF basis functions larger than nbasis_max={self.nbasis_max}")
            print(f"!!! Only {self.nbasis_max} PSF basis functions are kept, accounting {pfrac_max} info")
            nbasis = self.nbasis_max

        vecm = vecm[:,:nbasis]
        psf_basis = np.zeros((self.psf_size**2, nbasis))
        for i in range(nbasis):
            for j in range(psf_star_meta.psf_nstar):
                jpsf = psf_star_meta.psf_matrix[:,j]
                psf_basis[:,i] += vecm[j, i] * jpsf
            
            # normalize the PSF basis
            psf_basis[:,i]  = psf_basis[:,i] / np.sum(psf_basis[:,i])
            
        self.psf_basis = psf_basis
        self.psf_nbasis = nbasis
        return

    def psf_field_estimator(self, psf_star_meta):
        """
        Construct the coefficient fields based on the PSF basis functions and corresponding polynomial fields.
        """
        print("    Estimate the spatially varied PSF fields")
        if self.psf_nbasis == 1:
            self.psf_field = [None]
            return
        
        poly_coeffs, poly_coeffs_unc = self.poly_fit(psf_star_meta)
        xgrid, ygrid = self.__interp_grid()
        xgrid = (xgrid+0.5)/self.xsize - 0.5
        ygrid = (ygrid+0.5)/self.ysize - 0.5
        
        field_list = []
        for i in range(self.psf_nbasis):
            ifield = np.zeros((self.xsize, self.ysize), dtype=np.float32)
            for j in range(self.poly_ncoeff):
                jx_index, jy_index = self.poly_indices[j]
                ifield += poly_coeffs[i,j] * (xgrid**jx_index) * (ygrid**jy_index)
            field_list.append(ifield.astype(np.float32))

        self.psf_field = field_list
        return

    def psf_norm_estimator(self):
        """
        Calculate the PSF normalization image given in Lauer 2002.

        It uses the psf-basis elements and coefficients.
        """
        print("    Calculate the PSF normalization map")
        if self.psf_nbasis == 1:
            self.psf_norm = None
            # method 1: convolution with astropy convolve_fft
            #self.psf_norm = convolve_fft(a,self.psf_basis[0],fftn=fft2,ifftn=ifft2,allow_huge=True,normalize_kernel=False)
            # method 2: convolution with scipy fftconvolve
            #self.psf_norm = fftconvolve(a, self.psf_basis[0], mode="same")
            # method 3: convolution with scipy oaconvolve [fastest]
            #norm_matrix = signal.oaconvolve(a, psf_basis, mode="same")
        else:
            x_size = self.psf_size + self.xsize
            y_size = self.psf_size + self.ysize
            elim, remd1, remd2 = self.psf_size//2, 1, 1
            if np.mod(x_size,2)==1: x_size, remd1 = x_size+1, 2
            if np.mod(y_size,2)==1: y_size, remd2 = y_size+1, 2

            norm_matrix = np.zeros((self.xsize, self.ysize), dtype=np.float32)
            for i in range(self.psf_nbasis):
                ipsf_basis = self.psf_basis[:,i].reshape(self.psf_size, self.psf_size)
                ipsf_field = self.psf_field[i]
                #norm_matrix += signal.oaconvolve(ipsf_field, ipsf_basis, mode="same")
                
                # fft-based convolution. The running time is shorter than oaconvolve
                ipsf_field_new = fft.rfft2(ipsf_field, s=(x_size, y_size), workers=-1)
                ipsf_basis_new = fft.rfft2(ipsf_basis, s=(x_size, y_size), workers=-1)
                norm_matrix_new = fft.irfft2(ipsf_field_new*ipsf_basis_new, workers=-1)
                norm_matrix += norm_matrix_new[elim:x_size-elim-remd1,elim:y_size-elim-remd2]

            self.psf_norm = norm_matrix.astype(np.float32)
        return

    def psf_at_position(self, ximg, yimg):
        """
        Get the PSF model at a specified image position.
        """
        ximg, yimg = int(round(ximg)), int(round(yimg))
        if self.psf_nbasis == 1:
            psf_model = self.psf_basis[:,0]
        else:
            polyvals = np.array([ifield[ximg, yimg] for ifield in self.psf_field])
            psf_model = np.matmul(self.psf_basis, polyvals.reshape(-1,1))
            psf_model = psf_model/self.psf_norm[ximg, yimg]
        psf_model = psf_model.reshape(self.psf_size, self.psf_size)
        return psf_model

    def poly_model(self, crd_matrix, *coeffs_list):
        """
        The chi square is defined in matrix format to speed up, which is 
            chi^2 =||I - A@C@Z||^2, where
        
        I: matrix of observed flux
        A: matrix of PSF basis
        C: matrix of polynomial coefficients
        Z: matrix of star positions
        """
        coeff_matrix = np.array(coeffs_list).reshape((self.psf_nbasis,self.poly_ncoeff))
        model = np.matmul(self.psf_basis, coeff_matrix)
        model = np.matmul(model, crd_matrix)
        return model.flatten()

    def poly_fit(self, psf_star_meta):
        """
        The scipy 'curve_fit' is used to find the best-fitting polynomial coefficients.
        """
        xpos = (psf_star_meta.psf_xpos+0.5)/self.xsize - 0.5
        ypos = (psf_star_meta.psf_ypos+0.5)/self.ysize - 0.5

        crd_matrix = np.zeros((self.poly_ncoeff, psf_star_meta.psf_nstar))
        for i in range(self.poly_ncoeff):
            ix_index, iy_index = self.poly_indices[i]
            crd_matrix[i,:] = (xpos**ix_index) * (ypos**iy_index)
        
        flux = psf_star_meta.psf_matrix.flatten()
        
        init_coeff = np.ones(self.psf_nbasis * self.poly_ncoeff)
        popt, pcov = optimize.curve_fit(self.poly_model, crd_matrix, flux, p0=init_coeff)
        poly_coeffs = popt.reshape(self.psf_nbasis, self.poly_ncoeff)
        poly_coeffs_unc = np.sqrt(np.diag(pcov)).reshape(self.psf_nbasis, self.poly_ncoeff)
        return poly_coeffs, poly_coeffs_unc

    def psf_model_diagnosis(self, psf_star_meta, output_path=None, output_prefix=None):
        """
        save the Karhunen-Loeve PSF decomposition
        """
        psf_basis_name = "psf_basis.fits"
        psf_field_name = "psf_field.fits"
        psf_resi_name = "psf_resi.fits"
        psf_region_name = "psf_star.reg"
        if output_prefix is not None:
            psf_basis_name = ".".join([output_prefix, psf_basis_name])
            psf_field_name = ".".join([output_prefix, psf_field_name])
            psf_resi_name = ".".join([output_prefix, psf_resi_name])
            psf_region_name = ".".join([output_prefix, psf_region_name])
        if output_path is not None:
            psf_basis_name = os.path.join(output_path, psf_basis_name)
            psf_field_name = os.path.join(output_path, psf_field_name)
            psf_resi_name = os.path.join(output_path, psf_resi_name)
            psf_region_name = os.path.join(output_path, psf_region_name)

        # save region file
        wds9reg(psf_star_meta.psf_xpos+1.0, psf_star_meta.psf_ypos+1.0, 
                    radius=15.0,unit="pixel",color="green",outfile=psf_region_name)

        # save the psf basis functions
        for i in range(self.psf_nbasis):
            ipsf_basis = self.psf_basis[:,i].reshape(self.psf_size, self.psf_size)
            if i==0:
                psf_basis_matrix = ipsf_basis
            else:
                psf_basis_matrix = np.hstack((psf_basis_matrix, ipsf_basis), dtype=np.float32)
        fits.writeto(psf_basis_name, psf_basis_matrix, overwrite=True)
        
        # save the psf fields
        if self.psf_nbasis==1:
            pass
        else:
            fits.writeto(psf_field_name, np.array(self.psf_field + [self.psf_norm]), overwrite=True)

        # save the residuals of psf model
        psf_resi_list = []
        for i in range(psf_star_meta.psf_nstar):
            ixpos = psf_star_meta.psf_xpos[i]
            iypos = psf_star_meta.psf_ypos[i]
            ipsf_model = self.psf_at_position(ixpos,iypos)
            ipsf_real = psf_star_meta.psf_matrix[:,i].reshape(self.psf_size, self.psf_size)
            ipsf_resi = ipsf_real - ipsf_model
            ipsf_resi_model = np.hstack((ipsf_real, ipsf_model, ipsf_resi), dtype=np.float32)
            psf_resi_list.append(ipsf_resi_model)
        fits.writeto(psf_resi_name, np.array(psf_resi_list), overwrite=True)

        return

