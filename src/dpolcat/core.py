"""
dpolcat/core - polarimetric categorization of dual-pol. synthetic aperture radar (SAR) data

Author: Luke McQuade
"""

from numba import njit
import numpy as np
import xarray


def linear_to_db(x):
    """Convert linear backscatter to power ratio (decibels)."""
    return 10 * np.log10(x)


# This used to do a different transform, keep this as a passthrough until the rest of the code is updated.
def scale_nice(x):
    return linear_to_db(x)


def xr_scale_nice(x):
    """XArray, Dask-oriented, version of scaling function."""
    return xarray.apply_ufunc(scale_nice, x, dask="parallelized", output_dtypes=[float])


# Remember to keep this updated.
NUM_CATEGORIES = 20
"""Total number of categories."""


@njit
def categorize(vv_scaled, vh_scaled):
    """
    Our dual-polarimetric categorization decision tree algorithm.
    Inputs should be in numpy or xarray-compatible form, scaled using our custom scaling function.
    """
    vv = vv_scaled
    vh = vh_scaled

    i = 0
    if vv is None or vh is None:
        # NoData/invalid
        i = 0
    # Low volume scatter, ie., predominantly surface scatterers
    elif -np.inf <= vh < -20.0:
        if -np.inf <= vv < -18.0:
            i = 1
        elif -18.0 <= vv < -16.0:
            i = 2
        elif -16.0 <= vv < -14.0:
            i = 3
        elif -14.0 <= vv < -6.0:
            i = 4
        elif -6.0 <= vv < 2.0:
            i = 5
        elif vv >= 2.0:
            # Higher values, likely to be double-bounce (buildings) or terrain effects
            i = 13  # [merged A]
    elif vv < -18.0:
        if vh < -14.0:
            # Low specular with some cross-pol, e.g., moist soil
            i = 6
        elif vh >= -14.0:
            # Predominantly polarizing surfaces (unrealistic)
            i = 19  # [merged C]
    # Combinations of scatter, e.g, natural volume scatterers
    elif -18.0 <= vv < -14.0 and -22.0 <= vh < -14.0:
        i = 7
    elif -14.0 <= vv < -2.0 and -14.0 <= vh < -10.0:
        i = 12

    # Further mid-range categories
    elif -18.0 <= vv < -14.0 and -14.0 <= vh < -10.0:
        i = 19  # [merged C]
    elif -18.0 <= vv < -6.0 and -10.0 <= vh < 2.0:
        # Maybe double-bounce
        i = 14
    elif -14.0 <= vv < -6.0 and -22.0 <= vh < -17.0:
        i = 8
    elif -14.0 <= vv < -6.0 and -17.0 <= vh < -15.6:
        i = 9
    elif -14.0 <= vv < -6.0 and -15.6 <= vh < -14.0:
        i = 10
    elif -6.0 <= vv < 2.0 and -22.0 <= vh < -14.0:
        i = 11
    elif -2.0 <= vv < 2.0 and -14.0 <= vh < -10.0:
        i = 15  # [merged B] #

    # Higher values, likely to be double-bounce (buildings) or terrain effects.
    elif -6.0 <= vv < 2.0 and -10.0 <= vh < 2.0:
        i = 15  # [merged B]
    elif vv >= 2.0 and -22.0 <= vh < -10.0:
        i = 13  # [merged A]
    elif vv >= 2.0 and -10.0 <= vh < 2.0:
        i = 16
    elif vh >= 2.0 and -22.0 < vv < 2.0:
        i = 17
    elif vv >= 2.0 and vh >= 2.0:
        i = 18

    return i


categorize_np = np.vectorize(categorize, otypes=[np.uint8])
"""Numpy vectorized version of the categorizer."""


def xr_categorize(vv_scaled, vh_scaled):
    """XArray, Dask-oriented, version of the categorizer."""
    return xarray.apply_ufunc(
        categorize_np,
        vv_scaled,
        vh_scaled,
        dask="parallelized",
        output_dtypes=[np.uint8],
    ).rename("dpolcat")


color_list = np.array(
    [
        [0, 0, 0],  # 0
        [129, 244, 255],  # 1
        [199, 180, 215],  # 2
        [166, 121, 215],  # 3
        [255, 200, 128],  # 4
        [203, 73, 114],  # 5
        [222, 123, 214],  # 6
        [251, 210, 212],  # 7
        [231, 251, 187],  # 8
        [200, 187, 66],  # 9
        [157, 225, 90],  # 10
        [114, 73, 114],  # 11
        [105, 69, 42],  # 12
        [200, 10, 30],  # 13
        [255, 255, 0],  # 14
        [255, 204, 0],  # 15
        [158, 34, 108],  # 16
        [217, 0, 255],  # 17
        [252, 74, 0],  # 18
        [70, 70, 70],  # 19
    ]
)
"""Suggested category colours, of the form [R,G,B (0 to 255)]"""
