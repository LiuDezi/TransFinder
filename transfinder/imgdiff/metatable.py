# meta table class

from astropy import units
from astropy.table import Table, Column
import os

class InitMetaTable(object):
    """
    Initialize meta table for transient storage
    """
    def __init__(self, stamp_size=None):
        if stamp_size is None: 
            stamp_size = 49
        self.stamp_size = stamp_size
    
    def meta_table_iter(self, meta_table_name, mode="main"):
        # open the table
        if os.path.exists(meta_table_name):
            meta_table = Table.read(meta_table_name, format="fits")
            #ntrans = len(meta_table)
            #meta_table.remove_rows(slice(0,ntrans,1))
        else:
            if mode=="main":
                meta = self.main_meta()
            else: # mode="trans"
                meta = self.trans_meta()

            colList = []
            for ikey, ival in meta.items():
                idtype, iunit, icom = ival
                if iunit==None:
                    icol = Column(name=ikey, description=icom, dtype=idtype,)
                else:
                    icol = Column(name=ikey, unit=iunit, description=icom, dtype=idtype,)
                colList += [icol]
            meta_table = Table(colList)

        # save the table
        #meta_table.add_row(main_param)
        #meta_table.write(main_meta_table, format="fits", overwrite=True)
        return meta_table

    def main_meta(self):
        """
        main meta table that includes basic parameters for all transients
        """
        meta = {"trans_id":      ["U20",  None,         "transient id"],
                "ra":            ["f8",   units.deg,    "central right ascension"],
                "dec":           ["f8",   units.deg,    "central declination"],
                "dmode":         ["i4",   None,         "detection mode: -1=inverse diff; 1=direct diff"],
                "trans_file":    ["U30",  None,         "transient filename"],
                "trans_path":    ["U80",  None,         "path of transient filename"],
                }
        return meta

    def trans_meta(self):
        """
        meta table for individual transient. Basically,this table provides the time series data
        for a specified transient.
        """
        mat_dtype = f"({self.stamp_size},{self.stamp_size})f4"
        meta = {"trans_id":      ["U20",  None,         "transient id"],
                "date":          ["U22",  None,         "observed UTC date"],
                "mjd":           ["f8",   None,         "modified Julian date of new image"],
                "ra":            ["f8",   units.deg,    "central ra"],
                "dec":           ["f8",   units.deg,    "central dec"],
                "lon":           ["f8",   units.deg,    "central ecliptic longitude"],
                "lat":           ["f8",   units.deg,    "central ecliptic latitude"],
                "l":             ["f8",   units.deg,    "central galactic longitude"],
                "b":             ["f8",   units.deg,    "central galactic latitude"],
                "band":          ["U5",   None,         "band/filter"],
                "flux":          ["f8",   None,         "aperture flux"],
                "snr":           ["f4",   None,         "aperture snr"],
                "fwhm":          ["f4",   units.pixel,  "fwhm in pixels"],
                "dmode":         ["i4",   None,         "mode: -1=inverse diff; 1=direct diff"],
                "date_ref":      ["U22",  None,         "observed UTC date of reference"],
                "mjd_ref":       ["f8",   None,         "modified Julian date of reference image"],
                "trans_name":    ["U30",  None,         "transient filename"],
                "ref_name":      ["U80",  None,         "reference image name"],
                "new_name":      ["U80",  None,         "new image name"],
                "diff_name":     ["U80",  None,         "difference image name"],
                "ref_cutout":    [mat_dtype,  None,     "reference image name"],
                "new_cutout":    [mat_dtype,  None,     "new image name"],
                "diff_cutout":   [mat_dtype,  None,     "difference image name"],
                }
        return meta
