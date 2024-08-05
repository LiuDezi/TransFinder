
import numpy as np

xsize = 100
ysize = 150
grids = (7, 1)

xgrid, ygrid = grids
dx, dy = xsize/xgrid, ysize/ygrid
crd_grids, iid = {}, 0
for i in range(xgrid):
    ix = 0.5*dx + i*dx
    ix0, ix1 = int(np.ceil(ix-0.5*dx)), int(np.ceil(ix+0.5*dx))
    for j in range(ygrid):
        iy = 0.5*dy + j*dy
        iy0, iy1 = int(np.ceil(iy-0.5*dy)), int(np.ceil(iy+0.5*dy))
        #print(f"({i:2f}, {j:2d}) grid: {ix:3d}, {iy:3d}, {ix0:3d}, {ix1:3d}, {iy0:3d}, {iy1:3d}")
        print(iid, ix1-ix0, iy1-iy0)
        crd_grids[iid] = [int(round(ix-0.5)), int(round(iy-0.5)), ix0, ix1, iy0, iy1]
        iid += 1
