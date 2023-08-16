"""
dpolcat - polarimetric categorization of dual-pol. synthetic aperture radar (SAR) data

Author: Luke McQuade
"""

from numba import njit
import numpy as np
import xarray


def scale_nice(x):
    """Our custom scale function to transform linear radar data into a more convenient form with 0 to 1+ range."""
    return np.sqrt(np.log(x + 1))


# Remember to keep this updated.
NUM_CATEGORIES = 22
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
    if vv is None or vh is None or vv < 0 or vh < 0:
        # NoData/invalid
        i = 0
    elif 0 <= vh < 0.2: # Low volume scatter, ie., predominantly surface scatterers
        if 0 <= vv < 0.2:
            i = 1
        elif 0.2 <= vv < 0.3:
            i = 2
        elif 0.3 <= vv < 0.4:
            i = 3
        elif 0.4 <= vv < 0.6:
            i = 4
        # Higher values, likely to be double-bounce (buildings) or terrain effects
        elif 0.6 <= vv < 0.8:
            i = 5
        elif vv >= 0.8:
            i = 6
    elif vv < 0.2:
        if vh < 0.4:
            # Low specular with some cross-pol, e.g., moist soil 
            i = 7
        elif vh >= 0.4:
            # Predominantly polarizing surfaces (rare)
            i = 8
    # Combinations of scatter, e.g, natural volume scatterers
    elif 0.2 <= vv < 0.4 and 0.2 <= vh < 0.4:
        i = 9
    elif 0.4 <= vv < 0.6 and 0.4 <= vh < 0.6:
        i = 10
    
    # Further mid-range categories [experimental]
    elif 0.2 <= vv < 0.4 and 0.4 <= vh < 0.6:
        i = 16
    elif 0.2 <= vv < 0.4 and 0.6 <= vh < 0.8:
        i = 17
    elif 0.4 <= vv < 0.6 and 0.6 <= vh < 0.8:
        i = 18
    elif 0.4 <= vv < 0.6 and 0.2 <= vh < 0.4:
        i = 19
    elif 0.6 <= vv < 0.8 and 0.2 <= vh < 0.4:
        i = 20
    elif 0.6 <= vv < 0.8 and 0.4 <= vh < 0.6:
        i = 21
    
    # Higher values, likely to be double-bounce (buildings) or terrain effects.
    elif 0.6 <= vv < 0.8 and 0.6 <= vh < 0.8:
        i = 11
    elif vv >= 0.8 and 0.2 <= vh < 0.5:
        i = 12
    elif vv >= 0.8 and 0.5 <= vh < 0.8:
        i = 13
    elif vh >= 0.8 and 0.2 < vv < 0.8:
        i = 14
    elif vv >= 0.8 and vh >= 0.8:
        i = 15
        
    return i


categorize_np = np.vectorize(categorize, otypes=[np.uint8])
"""Numpy vectorized version of the categorizer."""


def categorize_xa(vv_scaled, vh_scaled):
    """xarray version of the categorizer."""
    return xarray.apply_ufunc(categorize_np, vv_scaled, vh_scaled)


color_list = np.array([
    [0,0,0],       # 0
    [129,244,255], # 1
    [199,180,215], # 2 
    [166,121,215], # 3
    [200,187,66],  # 4
    [203,73,114],  # 5
    [200,10,30],   # 6
    [222,123,214], # 7
    [255, 0, 63],  # 8
    [100,226,113], # 9
    [200,255,0],  # 10
    [188, 65, 14],   # 11
    [157,225,90],  # 12
    [255,255,217],   # 13
    [217,0,255],   # 14
    [255,204,0],   # 15
    [255,255,127], # 16
    [255,255,0],   # 17
    [255,255,0], # 18 # Intentionally same as above, as these will likely be merged.
    [0,255,0],  # 19
    [114,73,114],  # 20
    [0, 127, 255], # 21
])
"""Suggested category colours, of the form [R,G,B (0 to 255)]"""
