# Functions to perform image differencing using revised ZOGY (Zackay et al. 2018).

import numpy as np
from scipy import fft
import sys

from ..utils import sub_regions

#some potential useful modules
#from astropy.io import fits

class DiffImg(object):
    """
    Perform image differencing based on revised ZOGY algorithm (Zackay et al. 2016)

    In the original ZOGY algorithm, the spatial variation of PSF is not considered. 
    In our pipeline, we take this variation into account. In addition, to whiten 
    the correlation noise (Liu et al. 2024), we divide the entier image frame into 
    speficied grids. In each grid, the correlation noise is whitened by a constant 
    PSF.

    Parameters:
    degrid: tuple, defalut is None
        number of grids, e.g. degrid=(5,5)
        if degrid is not None, decorrelation will be performed in a 
        given grid using constant PSF
    nthreads: integer, default is -1
        number of threads for FFT/iFFT. In our code, the FFT/iFFT is 
        calculated by 'scipy.fft'. Here 'nthreads' is equivalent to 
        'workers' in 'scipy.fft'. Please see 'scipy.fft' for more detail.
    """
    def __init__(self, degrid=None, nthreads=-1):
        self.degrid = degrid
        self.nthreads = nthreads

    def zero_mask(self, ref_meta, new_meta):
        zero_pixels = (ref_meta.image_matrix==0.0) + (new_meta.image_matrix==0.0)
        return zero_pixels

    def nan_mask(self, ref_meta, new_meta, ref_mask, new_mask):
        """
        combine the mask matrices of reference and new images
        """
        ref_mask_tiny = np.isnan(ref_meta.image_matrix)
        new_mask_tiny = np.isnan(new_meta.image_matrix)

        nmask = ref_mask + new_mask + ref_mask_tiny + new_mask_tiny
        return nmask

    def image_masked(self, image_meta, nmask, zmask):
        """
        Fill the masked pixels with Gaussian noise N(0.0, sig^2), where
        'sig' is the standard deviation of input image
        """
        image_filled = image_meta.image_matrix.copy()
        if np.sum(nmask)!=0:
            gauss_img = np.random.normal(loc=0.0,scale=image_meta.image_sigma,size=image_filled.shape)
            image_filled[nmask] = gauss_img[nmask]
        
        image_filled[zmask] = 0.0
        return image_filled

    def fft_basis(self, xsize, ysize, psize):
        """
        define the effective region of the FFT/iFFT
        """
        x_size = psize + xsize
        y_size = psize + ysize
        elim, remd1, remd2 = psize//2, 1, 1
        if np.mod(x_size,2)==1: x_size, remd1 = x_size+1, 2
        if np.mod(y_size,2)==1: y_size, remd2 = y_size+1, 2

        x_lower, x_upper = elim, x_size-elim-remd1
        y_lower, y_upper = elim, y_size-elim-remd2
        return x_size, y_size, x_lower, x_upper, y_lower, y_upper

    def diff(self, ref_meta, new_meta, ref_psf_model, new_psf_model, ref_mask=None, new_mask=None):
        """
        Perform image differencing between reference and new image
        
        Parameters:
        ref_meta: python object
          meta data of reference image
        new_meta: python object
          meta data of new/target image
        ref_psf_meta: python object
          meta data of reference PSF
        new_psf_meta: python object
          meta data of new/target PSF
        ref_mask: bool array
          mask map of reference image, same size as reference
        new_mask: bool array
          mask map of new/target image, same size as new/target
        """
        # very basis check
        if ref_meta.xsize!=new_meta.xsize or ref_meta.ysize!=new_meta.ysize:
            sys.exit(f"!!! Image sizes of reference and new are not matched (unusual error)")
        if ref_psf_model.psf_size!=new_psf_model.psf_size:
            sys.exit(f"!!! PSF sizes are not matched (very unusual error)")
        
        xsize, ysize, psize = ref_meta.xsize, ref_meta.ysize, ref_psf_model.psf_size
        mean_sigma = np.sqrt(ref_meta.image_sigma**2+new_meta.image_sigma**2)

        # prepare maksed images
        if ref_mask is None: ref_mask = np.zeros((xsize, ysize), dtype=bool)
        if new_mask is None: new_mask = np.zeros((xsize, ysize), dtype=bool)
        nmask = self.nan_mask(ref_meta, new_meta, ref_mask, new_mask)
        zmask = self.zero_mask(ref_meta, new_meta)
        ref_matrix = self.image_masked(ref_meta, nmask, zmask)
        new_matrix = self.image_masked(new_meta, nmask, zmask)
        
        if self.degrid is None: 
            # Note: I know that degrid=None is basically (not completely) equivalent 
            #       to degrid=(1,1), but for completeness, I still keep this condition here
            
            # define some dirty numbers
            xs, ys, x0, x1, y0, y1 = self.fft_basis(xsize, ysize, psize)
            
            # FFT for reference image
            if new_psf_model.psf_nbasis==1:
                new_psf = new_psf_model.psf_basis[:,0].reshape(psize, psize)
                new_psf_hat = fft.rfft2(new_psf, s=(xs,ys), norm="ortho", workers=self.nthreads)
                ref_matrix_hat = fft.rfft2(ref_matrix, s=(xs,ys), norm="ortho", workers=self.nthreads)
                ref_matrix_hat = ref_matrix_hat*new_psf_hat
            else:
                ref_matrix_hat = np.zeros((xs, ys//2+1), dtype=np.complex64)
                for i in range(new_psf_model.psf_nbasis):
                    inew_psf_basis = new_psf_model.psf_basis[:,i].reshape(psize, psize)
                    inew_psf_basis_hat = fft.rfft2(inew_psf_basis, s=(xs,ys), norm="ortho", workers=self.nthreads)
                    if i==0: new_psf_hat = inew_psf_basis_hat.copy()
                    
                    inew_psf_field = new_psf_model.psf_field[i]/new_psf_model.psf_norm
                    iref_matrix = ref_matrix * inew_psf_field
                    iref_matrix_hat = fft.rfft2(iref_matrix, s=(xs,ys), norm="ortho", workers=self.nthreads)
                    ref_matrix_hat += iref_matrix_hat*inew_psf_basis_hat

            # FFT for new image
            if ref_psf_model.psf_nbasis==1:
                ref_psf = ref_psf_model.psf_basis[:,0].reshape(psize, psize)
                ref_psf_hat = fft.rfft2(ref_psf, s=(xs,ys), norm="ortho", workers=self.nthreads)
                new_matrix_hat = fft.rfft2(new_matrix, s=(xs,ys), norm="ortho", workers=self.nthreads)
                new_matrix_hat = new_matrix_hat*ref_psf_hat
            else:
                new_matrix_hat = np.zeros((xs, ys//2+1), dtype=np.complex64)
                for i in range(ref_psf_model.psf_nbasis):
                    iref_psf_basis = ref_psf_model.psf_basis[:,i].reshape(psize, psize)
                    iref_psf_basis_hat = fft.rfft2(iref_psf_basis, s=(xs,ys), norm="ortho", workers=self.nthreads)
                    if i==0: ref_psf_hat = iref_psf_basis_hat.copy()

                    iref_psf_field = ref_psf_model.psf_field[i]/ref_psf_model.psf_norm
                    inew_matrix = new_matrix * iref_psf_field
                    inew_matrix_hat = fft.rfft2(inew_matrix, s=(xs,ys), norm="ortho", workers=self.nthreads)
                    new_matrix_hat += inew_matrix_hat*iref_psf_basis_hat

            diff_matrix_hat_b1 = new_meta.image_sigma**2 * ref_psf_hat * ref_psf_hat.conj()
            diff_matrix_hat_b2 = ref_meta.image_sigma**2 * new_psf_hat * new_psf_hat.conj()
            diff_matrix_norm = np.sqrt(diff_matrix_hat_b1 + diff_matrix_hat_b2)

            diff_matrix_hat = (new_matrix_hat - ref_matrix_hat) / diff_matrix_norm
            diff_matrix = fft.irfft2(diff_matrix_hat, norm="ortho", workers=self.nthreads)
            diff_matrix = diff_matrix[x0:x1, y0:y1] * mean_sigma
        else:
            extend_size = psize//2
            crd_grids = sub_regions(xsize, ysize, image_grid=self.degrid, extend=(extend_size,extend_size))
            ngrid = len(crd_grids)
            
            diff_matrix = np.zeros((xsize, ysize))
            for j in range(ngrid):
                # reference and new sub-images
                jogrid, jegrid = crd_grids[j]
                jx, jy, jx0, jx1, jy0, jy1 = jogrid
                jex, jey, jex0, jex1, jey0, jey1 = jegrid
                
                jref_matrix = ref_matrix[jex0:jex1,jey0:jey1]
                jnew_matrix = new_matrix[jex0:jex1,jey0:jey1]

                # FFT numbers
                jxsize, jysize = jex1 - jex0, jey1 - jey0
                jfxs, jfys, jfx0, jfx1, jfy0, jfy1 = self.fft_basis(jxsize, jysize, psize)

                # calculate effective reference psf and new image
                if ref_psf_model.psf_nbasis==1:
                    jref_psf = ref_psf_model.psf_basis[:,0].reshape(psize, psize)
                    jref_psf_hat = fft.rfft2(jref_psf, s=(jfxs,jfys), norm="ortho", workers=self.nthreads)

                    jnew_matrix_hat = fft.rfft2(jnew_matrix, s=(jfxs,jfys), norm="ortho", workers=self.nthreads)
                    jnew_matrix_hat = jnew_matrix_hat*jref_psf_hat
                else:
                    jref_polyvals = np.array([kfield[jex, jey] for kfield in ref_psf_model.psf_field])
                    jref_psf = np.matmul(ref_psf_model.psf_basis, jref_polyvals.reshape(-1,1))
                    jref_psf = jref_psf.reshape(psize, psize)
                    jref_psf = jref_psf/ref_psf_model.psf_norm[jex, jey]
                    jref_psf_hat = fft.rfft2(jref_psf, s=(jfxs,jfys), norm="ortho", workers=self.nthreads)
                    
                    jnew_matrix_hat = np.zeros((jfxs, jfys//2+1), dtype=np.complex64)
                    for i in range(ref_psf_model.psf_nbasis):
                        iref_psf_basis = ref_psf_model.psf_basis[:,i].reshape(psize, psize)
                        iref_psf_basis_hat = fft.rfft2(iref_psf_basis, s=(jfxs,jfys), norm="ortho", workers=self.nthreads)

                        iref_psf_field = ref_psf_model.psf_field[i][jex0:jex1,jey0:jey1]/ref_psf_model.psf_norm[jex0:jex1,jey0:jey1]
                        inew_matrix = jnew_matrix * iref_psf_field
                        inew_matrix_hat = fft.rfft2(inew_matrix, s=(jfxs,jfys), norm="ortho", workers=self.nthreads)
                        jnew_matrix_hat += inew_matrix_hat*iref_psf_basis_hat

                # calculate effective new psf and reference image
                if new_psf_model.psf_nbasis==1:
                    jnew_psf = new_psf_model.psf_basis[:,0].reshape(psize, psize)
                    jnew_psf_hat = fft.rfft2(jnew_psf, s=(jfxs,jfys), norm="ortho", workers=self.nthreads)

                    jref_matrix_hat = fft.rfft2(jref_matrix, s=(jfxs,jfys), norm="ortho", workers=self.nthreads)
                    jref_matrix_hat = jref_matrix_hat*jnew_psf_hat
                else:
                    jnew_polyvals = np.array([kfield[jex, jey] for kfield in new_psf_model.psf_field])
                    jnew_psf = np.matmul(new_psf_model.psf_basis, jnew_polyvals.reshape(-1,1))
                    jnew_psf = jnew_psf.reshape(psize, psize)
                    jnew_psf = jnew_psf/new_psf_model.psf_norm[jex, jey]
                    jnew_psf_hat = fft.rfft2(jnew_psf, s=(jfxs,jfys), norm="ortho", workers=self.nthreads)
                    
                    jref_matrix_hat = np.zeros((jfxs, jfys//2+1), dtype=np.complex64)
                    for i in range(new_psf_model.psf_nbasis):
                        inew_psf_basis = new_psf_model.psf_basis[:,i].reshape(psize, psize)
                        inew_psf_basis_hat = fft.rfft2(inew_psf_basis, s=(jfxs,jfys), norm="ortho", workers=self.nthreads)

                        inew_psf_field = new_psf_model.psf_field[i][jex0:jex1,jey0:jey1]/new_psf_model.psf_norm[jex0:jex1,jey0:jey1]
                        iref_matrix = jref_matrix * inew_psf_field
                        iref_matrix_hat = fft.rfft2(iref_matrix, s=(jfxs,jfys), norm="ortho", workers=self.nthreads)
                        jref_matrix_hat += iref_matrix_hat*inew_psf_basis_hat
                
                # normalization
                jdiff_matrix_hat_b1 = new_meta.image_sigma**2 * jref_psf_hat * jref_psf_hat.conj()
                jdiff_matrix_hat_b2 = ref_meta.image_sigma**2 * jnew_psf_hat * jnew_psf_hat.conj()
                jdiff_matrix_norm = np.sqrt(jdiff_matrix_hat_b1 + jdiff_matrix_hat_b2)

                # difference image
                jdiff_matrix_hat = (jnew_matrix_hat - jref_matrix_hat) / jdiff_matrix_norm
                jdiff_matrix = fft.irfft2(jdiff_matrix_hat, norm="ortho", workers=self.nthreads)
                
                # correct boundary effect
                jfx0, jfx1 = jfx0 + extend_size, jfx1 - extend_size
                jfy0, jfy1 = jfy0 + extend_size, jfy1 - extend_size
                if jx0==0:     jfx0 = jfx0 - extend_size
                if jx1==xsize: jfx1 = jfx1 + extend_size
                if jy0==0:     jfy0 = jfy0 - extend_size
                if jy1==ysize: jfy1 = jfy1 + extend_size
                diff_matrix[jx0:jx1,jy0:jy1] = jdiff_matrix[jfx0:jfx1,jfy0:jfy1] * mean_sigma

        diff_matrix[zmask] = 0.0
        return diff_matrix.astype(np.float32)

