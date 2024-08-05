# test the performance of different convolution algorithm

import numpy as np
from scipy import fft
from scipy import signal
import pyfftw
import time
import os, sys
from astropy.convolution import convolve_fft, convolve

a = np.random.uniform(-0.1,5,size=(51, 51))
b = np.random.uniform(-0.5,5,size=(9215, 9233))

# c1
#t1 = time.time()
#c = convolve(b,a)
#t2 = time.time()
#print(f"^_^ Total {t2-t1} seconds used")

# c1
t1 = time.time()
#c1 = signal.fftconvolve(b, a/np.sum(a), mode="same")
t2 = time.time()
print(f"^_^ Total {t2-t1} seconds used")

# c2: current the fastest method
t1 = time.time()
c2 = signal.oaconvolve(b, a/np.sum(a), mode="same")
t2 = time.time()
print(f"^_^ Total {t2-t1} seconds used")

# FFT based convolution
t1 = time.time()
x_size = a.shape[0] + b.shape[0]
y_size = a.shape[1] + b.shape[1]
elim, remd1, remd2 = a.shape[0]//2, 1, 1
if np.mod(x_size,2)==1: x_size, remd1 = x_size+1, 2
if np.mod(y_size,2)==1: y_size, remd2 = y_size+1, 2

#a_new = np.pad(a/np.sum(a), ((0,x_size-a.shape[0]),(0,y_size-a.shape[1])))
#b_new = np.pad(b, ((0,x_size-b.shape[0]),(0,y_size-b.shape[1])))
b_hat = fft.rfft2(b, s=(x_size, y_size), workers=-1)
a_hat = fft.rfft2(a/np.sum(a), s=(x_size, y_size), workers=-1)
c3 = fft.irfft2(a_hat*b_hat, workers=-1)
c3 = c3[elim:x_size-elim-remd1,elim:y_size-elim-remd2]
t2 = time.time()
print(f"^_^ Total {t2-t1} seconds used")

# c2
#t1 = time.time()
#c = convolve_fft(b,a,allow_huge=True)
#t2 = time.time()
#print(f"^_^ Total {t2-t1} seconds used")

# c3
#t1 = time.time()
#c3 = convolve_fft(b,a, fftn=fft.fft2, ifftn=fft.ifft2, allow_huge=True, normalize_kernel=True)
#t2 = time.time()
#print(f"^_^ Total {t2-t1} seconds used")

# c4
#t1 = time.time()
#c = convolve_fft(b,a, fftn=np.fft.fft2, ifftn=np.fft.ifft2, allow_huge=True)
#t2 = time.time()
#print(f"^_^ Total {t2-t1} seconds used")

# c5
#t1 = time.time()
#c = convolve_fft(b,a, fftn=pyfftw.interfaces.numpy_fft.fft2, ifftn=pyfftw.interfaces.numpy_fft.ifft2, allow_huge=True)
#t2 = time.time()
#print(f"^_^ Total {t2-t1} seconds used")


# c6
#t1 = time.time()
#c = convolve_fft(b,a, fftn=pyfftw.interfaces.scipy_fft.fft2, ifftn=pyfftw.interfaces.scipy_fft.ifft2, allow_huge=True)
#t2 = time.time()
#print(f"^_^ Total {t2-t1} seconds used")




