"""
Identifier:     transfinder/instparam.py
Name:           instparam.py
Description:    configuration parameters for a specified survey
Author:         Dezi Liu
Created:        2024-07-31
Modified-History:
    2024-07-31, Dezi Liu, create this function
"""

import sys

class ResampleParam(object):
    """
    resample parameters for different surveys
    """
    def __init__(self, survey="mephisto"):
        self.survey = survey

    def params(self):
        """
        parameters for resampled image
        """
        if self.survey=="mci":
            pixel_scale = 0.05
            image_size = (9216, 9232)
        elif self.survey=="mephisto_pilot":
            pixel_scale = 0.45
            image_size = (5802, 5818)
        elif self.survey=="mephisto":
            pixel_scale = 0.30
            image_size = (8786, 8800)
        else:
            sys.exit("!!! survey parameters are not provided")
        return pixel_scale, image_size
