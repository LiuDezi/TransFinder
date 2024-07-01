# Class of TransFinder

from astropy.convolution import convolve_fft
from scipy.ndimage import convolve, center_of_mass
from scipy.ndimage.fourier import fourier_shift
import numpy as np
import scipy
import pyfftw
import os, sys

# define the fourier transform
_fftwn = pyfftw.interfaces.numpy_fft.fft2
_ifftwn = pyfftw.interfaces.numpy_fft.ifft2
eps = np.finfo(np.float64).eps

class DiffImg(object):
    """
    Perform image differencing
    """
    def __init__(self, ref_meta, new_meta, ref_psf_model, new_psf_model, ref_mask=None, new_mask=None):
        self.__ref_meta = ref_meta
        self.__new_meta = new_meta
        self.__ref_psf_model  = ref_psf_model
        self.__new_psf_model  = new_psf_model

         # construct the master mask matrix
        if ref_mask is None: ref_mask = np.zeros((self.__ref_meta.xsize, self.__ref_meta.ysize), dtype=bool)
        if new_mask is None: new_mask = np.zeros((self.__new_meta.xsize, self.__new_meta.ysize), dtype=bool)
        self.__ref_mask = ref_mask
        self.__new_mask = new_mask
        self.masterMask()

        # mask filled image
        self.imgREF_Filled = self.__imgNanFill(self.__ref_meta.image_matrix, self.__ref_meta.image_sigma, self.masterMask)
        self.imgNEW_Filled = self.__imgNanFill(self.__new_meta.image_matrix, self.__new_meta.image_sigma, self.masterMask)

        self.diff()

    def zeros_pixel(self):
        zeros_pixel_mask = (self.__ref_meta.image_matrix==0.0) + (self.__new_meta.image_matrix==0.0)
        return zeros_pixel_mask

    def masterMask(self):
        """
        combine the mask matrices of reference and new images
        """
        maskTinyREF = np.isnan(self.__ref_meta.image_matrix)
        maskTinyNEW = np.isnan(self.__new_meta.image_matrix)

        mMask = self.__ref_mask + self.__new_mask + maskTinyREF + maskTinyNEW
        self.masterMask = mMask
        return self.masterMask

    def __imgNanFill(self, image, sigma, mask):
        """
        Fill the masked pixels with Gaussian noise N(0.0, sig^2), where
        'sig' is estimated from 

        Parameters:
        image: 2D array
               input image
        sigma: float
               background rms of the input image
        mask:  2D boolean array
               input mask array

        Return:
            image with mask pixels filled with Gaussian noise
        """
        xsize, ysize = image.shape
        if np.sum(mask)!=0:
            gaussImg = np.random.normal(loc=0.0,scale=sigma,size=(xsize,ysize))
            image[mask] = gaussImg[mask]
            imageFilled = image
        else:
            imageFilled = image

        zeros_pixel_mask = self.zeros_pixel()
        imageFilled[zeros_pixel_mask] = 0.0
        return imageFilled


    def diff(self):
        """
        Perform image differencing between reference and new image
        """

        psfRef_real = self.__ref_psf_model.psf_base[:,:,0]
        imgRef_real = self.imgREF_Filled
        sigRef      = self.__ref_meta.image_sigma
        
        psfNew_real = self.__new_psf_model.psf_base[:,:,0]
        imgNew_real = self.imgNEW_Filled
        sigNew      = self.__new_meta.image_sigma


        # FFT of PSF model
        psfRef_hat = _fftwn(psfRef_real, s=imgRef_real.shape, norm="ortho")
        psfNew_hat = _fftwn(psfNew_real, s=imgNew_real.shape, norm="ortho")
        psfRef_hat[np.where(psfRef_hat.real == 0)] = eps
        psfNew_hat[np.where(psfNew_hat.real == 0)] = eps
        psfRef_hat_conj = psfRef_hat.conj()
        psfNew_hat_conj = psfNew_hat.conj()
        
        dxRef, dyRef = center_of_mass(psfRef_real)
        dxNew, dyNew = center_of_mass(psfNew_real)
        
        # FFT of input images
        imgRef_hat = _fftwn(imgRef_real, norm="ortho")
        imgNew_hat = _fftwn(imgNew_real, norm="ortho")
        
        # differene image in Fourier space
        DRef_hat   = fourier_shift(psfNew_hat * imgRef_hat, (-dxNew, -dyNew))
        DNew_hat   = fourier_shift(psfRef_hat * imgNew_hat, (-dxRef, -dyRef))
        
        normNew   = sigRef ** 2 * psfNew_hat * psfNew_hat_conj
        normRef   = sigNew ** 2 * psfRef_hat * psfRef_hat_conj
        norm      = np.sqrt(normNew + normRef)
        Dimg_hat  = (DNew_hat - DRef_hat) / norm
        Dimg      = _ifftwn(Dimg_hat, norm="ortho").real

        # PSF of the difference image
        dzp      = 1.0 / np.sqrt(sigRef ** 2 + sigNew ** 2)
        Dpsf_hat = (psfRef_hat * psfNew_hat) / (norm * dzp)
        
        # get the PSF of the difference image
        Dpsf     = _ifftwn(Dpsf_hat, norm="ortho").real
        xc, yc = np.where(Dpsf==Dpsf.max())
        xc, yc = np.round(xc[0]), np.round(yc[0])
        xxx, yyy = psfRef_real.shape
        xxx, yyy = int(0.5*(xxx-1)), int(0.5*(yyy-1))
        Dpsf = Dpsf[xc-xxx:xc+xxx+1, yc-yyy:yc+yyy+1]

        # re-set the mask pixels to zero
        Dimg[self.masterMask] = 0.0

        self.Dimg = Dimg
        self.Dpsf = Dpsf
        return

class imgFourier(object):
    """
    Perform Fourier transform on a given image and PSF model
    
    Parameters:
    imgCatPar: class
        parameters related to input image and catalog
    psfModel: class
        parameters related to input PSF models
    imgMask: bool
        mask array of input image

    """
    def __init__(self, imgCatPar=None, psfModel=None, imgMask=None):
        self.__imgCatPar = imgCatPar
        self.__psfModel  = psfModel
        self.__imgMask   = imgMask
        
        self.imgS()

    def __imgNanFill(self):
        """
        Fill the masked pixels with Gaussian noise N(0.0, sig^2), where
        'sig' is estimated from 
        """
        xsize = self.__imgCatPar.xsize
        ysize = self.__imgCatPar.ysize
        sigma = self.__imgCatPar.sigImage
        img = self.__imgCatPar.image

        mask = np.isnan(img) + self.__imgMask
        if np.sum(mask)==0: 
            self.imageFilled = img
            return self.imageFilled

        gaussImg = np.random.normal(loc=0.0,scale=sigma,size=(xsize,ysize))
        img[mask] = gaussImg[mask]
        self.imageFilled = img
        self.sigImage    = sigma
        return self.imageFilled
     
    def __imgFFT(self):
        """
        Fourier transform of the input image
        """
        self.__imgNanFill()
        self.imageFFT = _fftwn(self.imageFilled, norm="ortho")
        return self.imageFFT

    def imgSFFT(self):
        """
        calculate Fourier transform of the optimal statistic image S derived 
        by Zackay et al, 2016
        """
        psfBase   = self.__psfModel.psfBase
        coeffBase = self.__psfModel.coeffBase
        xsize     = self.__imgCatPar.xsize
        ysize     = self.__imgCatPar.ysize

        sigma     = self.__imgCatPar.sigImage
        imgNorm   = self.__imgNorm()
        imgFFT    = self.__imgFFT()
        dx, dy    = center_of_mass(psfBase[0])

        if coeffBase[0] is None:
            imageSFFT = imgFFT * _fftwn(psfBase[0],s=(xsize,ysize),norm="ortho").conj()
            imageSFFT = fourier_shift(imageSFFT, (+dx, +dy))
        else:
            imageSFFT    = np.zeros((xsize,ysize), dtype=np.complex128)
            xgrid, ygrid = self.__psfModel.coeffField()
            for icoeff, ipsf in zip(coeffBase, psfBase):
                conv = _fftwn(self.imageFilled*icoeff(xgrid, ygrid)/imgNorm,norm="ortho") \
                     * _fftwn(ipsf,s=(xsize,ysize),norm="ortho").conj()
                conv = fourier_shift(conv, (+dx, +dy))
                np.add(conv, imageSFFT, out=imageSFFT)
        self.imageSFFT = imageSFFT/(sigma ** 2)
        return self.imageSFFT

    def imgS(self):
        self.imgSFFT()
        self.imageS = _ifftwn(self.imageSFFT, norm="ortho").real
        return self.imageS

    def __imgNorm(self):
        """
        normalization of PSF model
        """
        psfBase   = self.__psfModel.psfBase
        coeffBase = self.__psfModel.coeffBase
        xsize     = self.__imgCatPar.xsize
        ysize     = self.__imgCatPar.ysize

        if coeffBase[0] is None:
            coeffField     = np.ones((xsize,ysize))
            self.imageNorm = convolve(coeffField, psfBase[0])
        else:
            xgrid, ygrid = self.__psfModel.coeffField()
            conv = np.zeros((xsize,ysize))
            for icoeff, ipsf in zip(coeffBase, psfBase):
                conv += convolve(icoeff(xgrid, ygrid), ipsf)
                #conv += convolve_fft(icoeff(xgrid, ygrid), ipsf) # slow
                self.imageNorm = conv
        return self.imageNorm


