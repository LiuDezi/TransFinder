# Module to model PSF over the entier image frame
# For this module to work, you need provide an image matrix and pre-selected
# star catalog. And the star catalog must contain the following parameters:
# XWIN_IMAGE, YWIN_IMAGE, FLUX_AUTO, FWHM_IMAGE, SNR_WIN
# 
# This module will
# 1) extract the star stamps from the given image matrix
# 2) perform Karhunen-Loeve PSF decomposition to get PSF basis functions
# 3) calculate PSF amplitude fields using 2D polynomial fitting
# 4) calculate PSF normalization map

import galsim
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip, sigma_clipped_stats
from scipy import signal, fft, optimize
from photutils.segmentation import SourceFinder
from photutils.segmentation import make_2dgaussian_kernel
import matplotlib.pyplot as plt
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

class PSFModel(object):
    """
    Two tasks to perform:
    1) spatially varied PSF modelling
    2) image differencing in Fouries space

    Parameters:
    psf_size: int
        pixel size of output PSF model, much be odd
    info_frac: float
        fraction of effective infomation used for PCA analysis
    nstar_max: int
        maximum number of stars for PSF modeling
    nbasis_max: int
        maximum number of PSF basis functions
    poly_degree: int
        degree to model the PSF variation over the entire image
    """
    def __init__(self, 
        psf_size = 31, 
        info_frac = 0.8, 
        nstar_max = 500, 
        nbasis_max = 3, 
        poly_degree = 3):

        self.psf_size = psf_size
        self.info_frac  = info_frac
        self.nstar_max = nstar_max
        self.nbasis_max = nbasis_max
        self.poly_degree = poly_degree
        self.poly_ncoeff = (poly_degree+1)*(poly_degree+2)//2
        self.poly_indices = poly_index(poly_degree)
    
    def run(self, image_matrix, star_matrix, output_prefix=None):
        """
        put everything together

        Parameters:
        image_matrix: array
            image data loaded by getdata
        star_matrix: array
            star catalog loaded by Table
        """
        star_pos, star_stm = self.select_stars(image_matrix, star_matrix)
        psf_basis = self.psf_basis_estimator(star_stm)
        psf_field = self.psf_field_estimator(star_pos, star_stm, psf_basis)
        psf_norm = self.psf_norm_estimator(psf_basis, psf_field)
        
        # save everything
        if output_prefix is not None:
            self.model_diagnosis(star_pos, star_stm, (psf_basis, psf_field, psf_norm), output_prefix=output_prefix)
        
        return psf_basis, psf_field, psf_norm

    def select_stars(self, image_matrix, star_matrix):
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
        nstar = len(star_matrix)
        sid_bright = np.argsort(star_matrix["SNR_WIN"])[::-1]
        xpos = star_matrix["XWIN_IMAGE"][sid_bright] - 1.0
        ypos = star_matrix["YWIN_IMAGE"][sid_bright] - 1.0
        flux = star_matrix["FLUX_AUTO"][sid_bright]
        fwhm = star_matrix["FWHM_IMAGE"][sid_bright]
        bstd = mad_std(image_matrix)

        # some pixel size definition
        half_size= self.psf_size//2
        dsize = self.psf_size - half_size

        # Method description: 
        # We first use photutils to generate segmentation map to exclude close neighbours. 
        # And then run galsim to refine the center of a star to eliminate dipole effect.
        xpos_new, ypos_new, star_size = np.array([]),  np.array([]), np.array([])
        nstar_new = 0
        for i in range(nstar):
            if nstar_new>=self.nstar_max: break
            ix, iy, iflux, ifwhm = xpos[i], ypos[i], flux[i], fwhm[i]
            ix_int, iy_int = int(round(ix)), int(round(iy))
            #ix0, ix1 = ix_int-half_size, ix_int+half_size
            #iy0, iy1 = iy_int-half_size, iy_int+half_size
            #istm = image_matrix[iy0:iy1+1,ix0:ix1+1]
            #if istm.shape != (self.psf_size, self.psf_size): continue
            #if np.sum(istm==0.0)>0: continue

            # an extended stamp
            ix0, ix1 = ix_int-self.psf_size, ix_int+self.psf_size
            iy0, iy1 = iy_int-self.psf_size, iy_int+self.psf_size
            istm = image_matrix[iy0:iy1+1,ix0:ix1+1]

            # remove stars locating at image edge
            if istm.shape != (2*self.psf_size+1, 2*self.psf_size+1): continue
            if np.sum(istm==0.0)>0: continue

            # detect neighbors
            ikernel_sigma = ifwhm/2.35
            ikernel_size = int(3.0*ikernel_sigma)
            if np.mod(ikernel_size,2)==0: ikernel_size += 1
            ikernel = make_2dgaussian_kernel(ikernel_sigma, size=ikernel_size)
            ifinder = SourceFinder(npixels=5, progress_bar=False)
            istm_conv = signal.oaconvolve(istm, ikernel, mode="same")
            iseg = ifinder(istm_conv, 1.5*bstd)
            if iseg is None: continue
            tid = iseg.data[self.psf_size, self.psf_size]
            if tid==0: continue
            
            # estimate background
            ibkg = istm[iseg.data==0]
            ibmean, ibmed, ibstd = sigma_clipped_stats(ibkg, sigma=3.0, maxiters=5.0, stdfunc="mad_std")
            
            # mask beighbors with random numbers
            ibadpix = (iseg.data!=tid) & (iseg.data!=0)
            istm[ibadpix] = np.random.normal(ibmed, ibstd, size=istm.shape)[ibadpix]
            
            # finally, we get the target stamp
            istm = (istm[dsize:-dsize,dsize:-dsize] - ibmed) / iflux
            ibadpix = ibadpix[dsize:-dsize,dsize:-dsize]

            # refine the center of PSF star
            istm_obj = galsim.Image(istm, scale=1.0)
            ibadpix_obj = galsim.Image(ibadpix.astype(int), scale=1.0)
            try:
                istm_moment = istm_obj.FindAdaptiveMom(badpix=ibadpix_obj)
            except:
                continue
            ix_moment = istm_moment.moments_centroid.x - 1.0
            iy_moment = istm_moment.moments_centroid.y - 1.0
            ix_offset, iy_offset = ix_moment-half_size, iy_moment-half_size
            #ixy_offset = np.sqrt(ix_offset**2+iy_offset**2)
            istm_interp = galsim.InterpolatedImage(istm_obj,x_interpolant=galsim.Lanczos(7),normalization='flux')
            istm_interp = istm_interp.drawImage(nx=self.psf_size, ny=self.psf_size, scale=1.0,
                                                method='no_pixel', offset=(-ix_offset, -iy_offset))
            istm = istm_interp.array
            #print(ix_offset, iy_offset)
            #fits.writeto(f"zstar{i}_r.fits", istm, overwrite=True)
            
            nstar_new += 1
            xpos_new = np.append(xpos_new, ix_offset + ix_int)
            ypos_new = np.append(ypos_new, iy_offset + iy_int)
            star_size = np.append(star_size, istm_moment.moments_sigma)
            if nstar_new==1:
                star_stamp = istm.reshape(-1,1)
            else:
                star_stamp = np.hstack((star_stamp, istm.reshape(-1,1)))
        
        # remove outliers based on star size
        mask_size = sigma_clip(star_size, sigma=3.0, maxiters=5.0, stdfunc="mad_std", masked=True)
        gid = np.invert(mask_size.mask)

        # repeat the loop to find more good stars
        if np.sum(gid)!=nstar_new:
            print(f"    Find {nstar_new-np.sum(gid)} outliers, remove them and find more")
            nstar_new = np.sum(gid)
            xpos_new, ypos_new, star_stamp = xpos_new[gid], ypos_new[gid], star_stamp[:, gid]
            star_size = star_size[gid]
            sigma_size, median_size = mad_std(star_size), np.median(star_size)
            if i<nstar-1:
                for j in range(i+1, nstar):
                    if nstar_new>=self.nstar_max: break
                    ix, iy, iflux, ifwhm = xpos[j], ypos[j], flux[j], fwhm[j]
                    ix_int, iy_int = int(round(ix)), int(round(iy))

                    # an extended stamp
                    ix0, ix1 = ix_int-self.psf_size, ix_int+self.psf_size
                    iy0, iy1 = iy_int-self.psf_size, iy_int+self.psf_size
                    istm = image_matrix[iy0:iy1+1,ix0:ix1+1]

                    # remove stars locating at image edge
                    if istm.shape != (2*self.psf_size+1, 2*self.psf_size+1): continue
                    if np.sum(istm==0.0)>0: continue
                    
                    # detect neighbors
                    ikernel_sigma = ifwhm/2.35
                    ikernel_size = int(3.0*ikernel_sigma)
                    if np.mod(ikernel_size,2)==0: ikernel_size += 1
                    ikernel = make_2dgaussian_kernel(ikernel_sigma, size=ikernel_size)
                    ifinder = SourceFinder(npixels=5, progress_bar=False)
                    istm_conv = signal.oaconvolve(istm, ikernel, mode="same")
                    iseg = ifinder(istm_conv, 1.5*bstd)
                    if iseg is None: continue
                    tid = iseg.data[self.psf_size, self.psf_size]
                    if tid==0: continue

                    # estimate background
                    ibkg = istm[iseg.data==0]
                    ibmean, ibmed, ibstd = sigma_clipped_stats(ibkg, sigma=3.0, maxiters=5.0, stdfunc="mad_std")

                    # mask beighbors with random numbers
                    ibadpix = (iseg.data!=tid) & (iseg.data!=0)
                    istm[ibadpix] = np.random.normal(ibmed, ibstd, size=istm.shape)[ibadpix]

                    # finally, we get the target stamp
                    istm = (istm[dsize:-dsize,dsize:-dsize] - ibmed) / iflux
                    ibadpix = ibadpix[dsize:-dsize,dsize:-dsize]

                    # refine the center of PSF star
                    istm_obj = galsim.Image(istm, scale=1.0)
                    ibadpix_obj = galsim.Image(ibadpix.astype(int), scale=1.0)
                    try:
                        istm_moment = istm_obj.FindAdaptiveMom(badpix=ibadpix_obj)
                    except:
                        continue
                    if abs(istm_moment.moments_sigma-median_size)>3.0*sigma_size: continue

                    ix_moment = istm_moment.moments_centroid.x - 1.0
                    iy_moment = istm_moment.moments_centroid.y - 1.0
                    ix_offset, iy_offset = ix_moment-half_size, iy_moment-half_size
                    #ixy_offset = np.sqrt(ix_offset**2+iy_offset**2)
                    istm_interp = galsim.InterpolatedImage(istm_obj,x_interpolant=galsim.Lanczos(7),normalization='flux')
                    istm_interp = istm_interp.drawImage(nx=self.psf_size, ny=self.psf_size, scale=1.0,
                                                        method='no_pixel', offset=(-ix_offset, -iy_offset))
                    istm = istm_interp.array
                    #print(ix_offset, iy_offset)
                    #fits.writeto(f"zstar{i}_r.fits", istm, overwrite=True)

                    nstar_new += 1
                    xpos_new = np.append(xpos_new, ix_offset + ix_int)
                    ypos_new = np.append(ypos_new, iy_offset + iy_int)
                    star_stamp = np.hstack((star_stamp, istm.reshape(-1,1)))
                    star_size = np.append(star_size, istm_moment.moments_sigma)
                    sigma_size, median_size = mad_std(star_size), np.median(star_size)
            else:
                pass
        else:
            pass
        print(f"    Total {nstar_new} PSF stars are selected")
        
        # save region file
        # wds9reg(xpos_new+1.0, ypos_new+1.0, radius=15.0,unit="pixel",color="green",outfile=psf_region_name)

        self.nstar = nstar_new
        self.ysize, self.xsize = image_matrix.shape
        star_position = np.vstack((xpos_new, ypos_new)).T
        return star_position, star_stamp

    def psf_covmat(self, star_stamp):
        """
        Calculate the covariance matrix of PSF stars
        """
        cov_matrix = np.cov(star_stamp, rowvar=False)
        return cov_matrix

    def eigen_decom(self, cov_matrix):
        """
        calculate the eigenvalues and eigenvectors of covariance matrix
        using the scipy module

        Return eigenvalues in descendent order
        """
        evalues, evectors = np.linalg.eigh(cov_matrix)
        #evalues, evectors = scipy.linalg.eigh(cov_matrix)
        # note that: the sign of eigenvertors returned by numpy and scipy are different
        eigval = evalues[::-1]
        eigvec = evectors[:,::-1]
        return eigval, eigvec

    def psf_basis_estimator(self, star_stamp):
        """
        derive the PSF basis functions. At most three PSF basis functions are kept
        """
        print("    Estimate PSF basis functions")
        cov_matrix = self.psf_covmat(star_stamp)
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
            for j in range(self.nstar):
                jpsf = star_stamp[:,j]
                psf_basis[:,i] += vecm[j, i] * jpsf
            
            # normalize the PSF basis by the sum of first component
            if i==0: psf_norm = np.sum(psf_basis[:,0])
            psf_basis[:,i]  = psf_basis[:,i] / psf_norm
        
        self.nbasis = nbasis
        return psf_basis

    def psf_field_estimator(self, star_position, star_stamp, psf_basis):
        """
        Construct the coefficient fields based on the PSF basis functions and corresponding polynomial fields.
        """
        print("    Estimate the spatially varied PSF fields")
        if self.nbasis == 1: 
            field_list = [None]
        else:
            poly_coeffs, poly_coeffs_unc = self.poly_fit(star_position, star_stamp, psf_basis)
            
            xgrid, ygrid = np.ogrid[:self.xsize, :self.ysize]
            xgrid = (xgrid+0.5)/self.xsize - 0.5
            ygrid = (ygrid+0.5)/self.ysize - 0.5
            field_list = np.zeros((self.nbasis, self.xsize, self.ysize), dtype=np.float32)
            for j in range(self.poly_ncoeff):
                jx_index, jy_index = self.poly_indices[j]
                jpos = np.matmul(xgrid**jx_index, ygrid**jy_index)
                for i in range(self.nbasis):
                    field_list[i] += poly_coeffs[i,j] * jpos
            field_list = np.transpose(field_list, axes=(0,2,1))

        return list(field_list)

    def psf_norm_estimator(self, psf_basis, psf_field):
        """
        Calculate the PSF normalization image given in Lauer 2002.

        It uses the psf-basis elements and coefficients.
        """
        print("    Calculate the PSF normalization map")
        if self.nbasis == 1:
            norm_matrix = None
        else:
            x_size = self.psf_size + self.xsize
            y_size = self.psf_size + self.ysize
            elim, remd1, remd2 = self.psf_size//2, 1, 1
            if np.mod(x_size,2)==1: x_size, remd1 = x_size+1, 2
            if np.mod(y_size,2)==1: y_size, remd2 = y_size+1, 2

            norm_matrix = np.zeros((self.ysize, self.xsize), dtype=np.float32)
            for i in range(self.nbasis):
                ipsf_basis = psf_basis[:,i].reshape(self.psf_size, self.psf_size)
                ipsf_field = psf_field[i]
                
                # fft-based convolution. The running time is shorter than oaconvolve
                #inorm_matrix = signal.oaconvolve(ipsf_field, ipsf_basis, mode="same")
                #norm_matrix += inorm_matrix
                ipsf_field_fft = fft.rfft2(ipsf_field, s=(y_size, x_size), workers=-1)
                ipsf_basis_fft = fft.rfft2(ipsf_basis, s=(y_size, x_size), workers=-1)
                inorm_matrix= fft.irfft2(ipsf_field_fft*ipsf_basis_fft, workers=-1)
                norm_matrix += inorm_matrix[elim:y_size-elim-remd2,elim:x_size-elim-remd1]

        return norm_matrix

    def poly_model(self, input_data, *coeffs_list):
        """
        The chi square is defined in matrix format to speed up, which is 
            chi^2 =||I - A@C@Z||^2, where
        
        I: matrix of observed flux
        A: matrix of PSF basis
        C: matrix of polynomial coefficients
        Z: matrix of star positions
        """
        ncut = self.nbasis * self.psf_size * self.psf_size
        psf_basis = input_data[:ncut].reshape(-1, self.nbasis)
        crd_matrix = input_data[ncut:].reshape(self.poly_ncoeff, self.nstar)
        coeff_matrix = np.array(coeffs_list).reshape((self.nbasis,self.poly_ncoeff))
        model = np.matmul(psf_basis, coeff_matrix)
        model = np.matmul(model, crd_matrix)
        return model.flatten()

    def poly_fit(self, star_position, star_stamp, psf_basis):
        """
        The scipy 'curve_fit' is used to find the best-fitting polynomial coefficients.
        """
        xpos = (star_position[:,0] + 0.5)/self.xsize - 0.5
        ypos = (star_position[:,1] + 0.5)/self.ysize - 0.5

        crd_matrix = np.zeros((self.poly_ncoeff, self.nstar))
        for i in range(self.poly_ncoeff):
            ix_index, iy_index = self.poly_indices[i]
            crd_matrix[i,:] = (xpos**ix_index) * (ypos**iy_index)
        
        flux = star_stamp.flatten()

        init_coeff = np.ones(self.nbasis * self.poly_ncoeff)
        popt, pcov = optimize.curve_fit(self.poly_model, np.append(psf_basis,crd_matrix), flux, p0=init_coeff)
        poly_coeffs = popt.reshape(self.nbasis, self.poly_ncoeff)
        poly_coeffs_unc = np.sqrt(np.diag(pcov)).reshape(self.nbasis, self.poly_ncoeff)
        return poly_coeffs, poly_coeffs_unc

    def psf_at_position(self, ximg, yimg, psf_basis, psf_field, psf_norm):
        """
        Get the PSF model at a specified image position.
        """
        ximg, yimg = int(round(ximg)), int(round(yimg))
        if self.nbasis == 1:
            psf_model = psf_basis[:,0]
        else:
            polyvals = np.array([ifield[ximg, yimg] for ifield in psf_field])
            psf_model = np.matmul(psf_basis, polyvals.reshape(-1,1))
            psf_model = psf_model/psf_norm[ximg, yimg]
        psf_model = psf_model.reshape(self.psf_size, self.psf_size)
        return psf_model

    def model_diagnosis(self, star_position, star_stamp, psf_model, output_prefix="model_diagnosis"):
        """
        save the model residual stamps and figure
        """
        # define output filenames
        output_basis = output_prefix + ".psf_basis.fits"
        output_field = output_prefix + ".psf_field.fits"
        output_resi = output_prefix + ".psf_resi.fits"
        
        psf_basis, psf_field, psf_norm = psf_model
        # psf basis
        for i in range(self.nbasis):
            ibasis = psf_basis[:,i].reshape(self.psf_size, self.psf_size)
            if i==0:
                obasis = ibasis
            else:
                obasis = np.hstack((obasis, ibasis), dtype=np.float32)
        fits.writeto(output_basis, obasis, overwrite=True)
        
        # psf fields and normalization map
        if self.nbasis>1: fits.writeto(output_field, np.array(psf_field + [psf_norm]), overwrite=True)

        # save the residuals of psf model
        xpos, ypos = star_position[:,0], star_position[:,1]
        psf_resi_list = []
        psf_resi_size = np.zeros((self.nstar,2), dtype=np.float32)
        for i in range(self.nstar):
            ixpos, iypos = xpos[i], ypos[i]
            ipsf_model = self.psf_at_position(ixpos, iypos, psf_basis, psf_field, psf_norm)
            ipsf_real = star_stamp[:,i].reshape(self.psf_size, self.psf_size)
            ipsf_resi = ipsf_real - ipsf_model
            ipsf_resi_model = np.hstack((ipsf_real, ipsf_model, ipsf_resi), dtype=np.float32)
            psf_resi_list.append(ipsf_resi_model)

            # calculate the size residuals
            ireal_obj = galsim.Image(ipsf_real, scale=1.0)
            ireal_moment = ireal_obj.FindAdaptiveMom()
            ireal_sigma = ireal_moment.moments_sigma
            imodel_obj = galsim.Image(ipsf_model, scale=1.0)
            imodel_moment = imodel_obj.FindAdaptiveMom()
            imodel_sigma = imodel_moment.moments_sigma
            iresi_size = (ireal_sigma - imodel_sigma) / ireal_sigma
            psf_resi_size[i, :] = [ireal_sigma, iresi_size]
        fits.writeto(output_resi, np.array(psf_resi_list), overwrite=True)
        
        plt.plot(psf_resi_size[:,0], psf_resi_size[:,1], "o", color="black", ms=2.5)
        plt.xlabel("$\sigma_{star}$ [pixels]", fontsize=12)
        plt.ylabel("$(\sigma_{star} - \sigma_{model})/\sigma_{star}$", fontsize=12)
        plt.savefig(output_resi[:-4] + "pdf")
        plt.clf()
        plt.close()

        return

