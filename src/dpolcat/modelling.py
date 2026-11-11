"""
dpolcat/modelling - helper classes for semantic modelling

Author: Luke McQuade
"""

from enum import IntEnum

import dpolcat

# Semantic API
# ------------
# Here we aim to present the polarimetric categories in a way
# that they can be used to create understandable semantic models.


class Surface(IntEnum):
    """
    Predominantly surface-scattering mechanics.
    - `Smooth` (1): Very smooth surface scattering.
    - `SlightlyRough` (2): Slightly rough surface scattering.
    - `MediumRough` (3): Medium-rough surface scattering.
    - `Rough` (4): Rough surface scattering.
    - `VeryRough` (5): Very rough surface scattering.
    """

    Smooth = 1
    """Very smooth surface scattering."""

    SlightlyRough = 2
    """Slightly rough surface scattering."""

    MediumRough = 3
    """Medium-rough surface scattering."""

    Rough = 4
    """Rough surface scattering."""

    VeryRough = 5
    """Very rough surface scattering."""


class Volume(IntEnum):
    """
    Volume-scattering mechanics.
    - `LowCo` (6): Volume scatterer with low co-polar response.
    - `MediumCo` (7): Volume scatterer with medium co-polar response.
    - `Low` (8): Weakly volume-scattering object.
    - `MediumLow` (9): Medium-weakly volume-scattering object.
    - `Medium` (10): Medium-strength volume-scattering object.
    - `HighCo` (11): Volume scatterer with high co-polar response.
    - `High` (12): Highly volume-scattering object.
    """

    LowCo = 6
    """Volume scatterer with low co-polar response."""

    MediumCo = 7
    """Volume scatterer with medium co-polar response."""

    Low = 8
    """Weakly volume-scattering object."""

    MediumLow = 9
    """Medium-weakly volume scattering object."""

    Medium = 10
    """Medium-strength volume scattering object."""

    HighCo = 11
    """Volume scatterer with high co-polar response."""

    High = 12
    """Highly volume-scattering object."""


class MultiBounce(IntEnum):
    """
    Double/triple-bounce mechanics.
    - `HighCo` (13): High-strength co-polar double/triple-bounce, with low cross-polar response.  
    - `MedHighCross` (14): Medium-strength cross-polar double/triple-bounce, with low co-polar response.  
    - `MediumHigh` (15): Medium-high co- and cross-polar double/triple-bounce response.  
    - `HighMixedCo` (16): High-strength double/triple-bounce with a mostly co-polar response.  
    - `HighMixedCross` (17): High-strength double/triple-bounce with a mostly cross-polar response.  
    - `VeryHigh` (18): Very strong co- and cross-polar double/triple-bounce response.  
    """

    HighCo = 13
    """High strength co-polar double/triple-bounce, with low cross-polar, response."""

    MedHighCross = 14
    """Medium strength cross-polar double/triple-bounce, with low co-polar, response."""

    MediumHigh = 15
    """Medium-high co- and cross-polar double/triple-bounce response."""

    HighMixedCo = 16
    """High-strength double/triple-bounce with a mostly co-polar response."""

    HighMixedCross = 17
    """High-strength double/triple-bounce with a mostly cross-polar response."""

    VeryHigh = 18
    """Very strong co- and cross-polar double/triple-bounce response."""


class Other(IntEnum):
    """
    Other categories.
    - `NoData` (0): No data.  
    - `Unknown` (19): Physically unrealistic/unknown mechanism.  
    """

    NoData = 0
    """No data."""

    Unknown = 19
    """Physically unrealistic/unknown mechanism."""


# A quick sanity check - do the definitions above line up with our expected number of categories?
assert set(range(dpolcat.NUM_CATEGORIES)) == set(Surface) | set(Volume) | set(
    MultiBounce
) | set(Other), "Category definitions do not match the expected number."
