
from astropy.table import Table
import pylab as pl
import numpy as np

tdata = Table.read("fft_regions.dat", format="ascii")
td1 = tdata[:50]
td2 = tdata[50:]

pl.errorbar(td1["ngrids"], td1["fft_time"], td1["fft_time_err"], fmt="-", marker="o", ms=4, color="black", label="PSF size: 21$\\times$21")
pl.errorbar(td2["ngrids"], td2["fft_time"], td2["fft_time_err"], fmt="-", marker="s", ms=4, color="red", label="PSF size: 31$\\times$31")
pl.xlabel("number of grids", fontsize=12)
pl.ylabel("running time (seconds)", fontsize=12)
pl.legend(fontsize=12)
pl.show()

