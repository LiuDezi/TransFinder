# This routine is used to build reference images for Mephisto surveys.
# The reference images are pre-processed single exposure images. We select 
# images as reference if they have high-quality astrometry and image quality. 
# All the selected reference images are in their own original directories, 
# but relevant parameters will be saved into a table, denoting as meta table. 
# When a new image comes, we will read this meta table to find the best matched 
# reference images.

# To be completed.

from astropy import units
from astropy.table import Table, Column
import os

def achieve_ref(self, input_meta, band):
    """
    find reference image in a given meta data through matching band and central celestial coordinate
    """
    ra_center, dec_center = self.resamp_image_center
    xsize, ysize = self.resamp_image_size
    ra_center, dec_center = str2deg(ra_center, dec_center)

    hx_arcsec= 0.5 * xsize * self.resamp_pixel_scale
    hy_arcsec= 0.5 * ysize * self.resamp_pixel_scale
    match_aperture = np.min([hx_arcsec, hy_arcsec])

    # find the reference image
    metatab = Table.read(input_meta, format="fits")
    bid = metatab["band"]==band
    if len(bid)==0: sys.exit(f"!!! No {band}-band image found in {os.path.basename(input_meta)}")
    ra_ref, dec_ref = metatab["ra"][bid], metatab["dec"][bid]
    rid, iid = crossmatch(ra_ref, dec_ref, [ra_center], [dec_center], aperture=match_aperture)
    if len(rid)==0: sys.exit(f"!!! No reference image found in {os.path.basename(input_meta)}")

    ref_image_name = metatab["image_name"][bid][rid[0]]
    ref_image_path = metatab["image_path"][bid][rid[0]]
    ref_image  = os.path.join(ref_image_path, ref_image_name)
    return ref_image


def meta_table_iter(image_meta):
    """
    build/update reference image meta information

    image_meta: str
        meta table name
    """
    # open the table
    if not os.path.exists(image_meta):
        meta = {"image_name": ["U80",  None,         "image name"],
                "ra":         ["f8",   units.deg,    "central ra"],
                "dec":        ["f8",   units.deg,    "central dec"],
                "mu_ra":      ["f8",   units.arcsec, "ra astrometric offset"],
                "mu_dec":     ["f8",   units.arcsec, "dec astrometric offset"],
                "std_ra":     ["f8",   units.arcsec, "ra astrometric std"],
                "std_dec":    ["f8",   units.arcsec, "dec astrometric std"],
                "ra_min":     ["f8",   units.deg,    "minimum ra"],
                "ra_max":     ["f8",   units.deg,    "maximum ra"],
                "dec_min":    ["f8",   units.deg,    "minimum dec"],
                "dec_max":    ["f8",   units.deg,    "maximum dec"],
                "band":       ["U5",   None,         "band/filter"],
                "fwhm":       ["f4",   units.pixel,  "median FWHM"],
                "nstar":      ["i8",   None,         "number of high-snr stars"],
                "std_bkg":    ["f4",   None,         "background std"],
                "image_path":   ["U100", None,       "image path"]}
        colList = []
        for ikey, ival in meta.items():
            idtype, iunit, icom = ival
            if iunit==None:
                icol = Column([], name=ikey, description=icom, dtype=idtype,)
            else:
                icol = Column([], name=ikey, unit=iunit, description=icom, dtype=idtype,)
            colList += [icol]
        image_meta_tab = Table(colList)
    else:
        image_meta_tab = Table.read(image_meta, format="fits")

    return image_meta_tab

