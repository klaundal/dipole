"""
Functions for calculating subsolar point


MIT License

Copyright (c) 2017 Karl M. Laundal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pandas as pd
from .utils import sph_to_car

d2r = np.pi/180
r2d = 180/np.pi


""" function for computing subsolar point """
def subsol(datetimes):
    """ 
    calculate subsolar point at given datetime(s)

    Parameters
    ----------
    datetimes : datetime or list of datetimes
        datetime or list (or other iterable) of datetimes

    Returns
    -------
    subsol_lat : ndarray
        latitude(s) of the subsolar point
    subsol_lon : ndarray
        longiutde(s) of the subsolar point

    Note
    ----
    The code is vectorized, so it should be fast.

    After Fortran code by: 961026 A. D. Richmond, NCAR

    Documentation from original code:
    Find subsolar geographic latitude and longitude from date and time.
    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994).
    Usable for years 1601-2100, inclusive.  According to the Almanac, 
    results are good to at least 0.01 degree latitude and 0.025 degree 
    longitude between years 1950 and 2050.  Accuracy for other years 
    has not been tested.  Every day is assumed to have exactly
    86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored:  their effect is below the accuracy threshold of
    the algorithm.
    """

    # use pandas DatetimeIndex for fast access to year, month day etc...
    if hasattr(datetimes, '__iter__'): 
        datetimes = pd.DatetimeIndex(datetimes)
    else:
        datetimes = pd.DatetimeIndex([datetimes])

    year = np.float64(datetimes.year)
    # day of year:
    doy  = np.float64(datetimes.dayofyear)
    # seconds since start of day:
    ut   = np.float64(datetimes.hour * 60.**2 + datetimes.minute*60. + datetimes.second )
 
    yr = year - 2000

    if year.max() >= 2100 or year.min() <= 1600:
        raise ValueError('subsol.py: subsol invalid after 2100 and before 1600')

    nleap = np.floor((year-1601)/4.)
    nleap = np.array(nleap) - 99

    # exception for years <= 1900:
    ncent = np.floor((year-1601)/100.)
    ncent = 3 - ncent
    nleap[year <= 1900] = nleap[year <= 1900] + ncent[year <= 1900]

    l0 = -79.549 + (-.238699*(yr-4*nleap) + 3.08514e-2*nleap)

    g0 = -2.472 + (-.2558905*(yr-4*nleap) - 3.79617e-2*nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut/86400. - 1.5) + doy

    # Addition to Mean longitude of Sun since January 1 of IYR:
    lf = .9856474*df

    # Addition to Mean anomaly since January 1 of IYR:
    gf = .9856003*df

    # Mean longitude of Sun:
    l = l0 + lf

    # Mean anomaly:
    g = g0 + gf
    grad = g*np.pi/180.

    # Ecliptic longitude:
    lmbda = l + 1.915*np.sin(grad) + .020*np.sin(2.*grad)
    lmrad = lmbda*np.pi/180.
    sinlm = np.sin(lmrad)

    # Days (including fraction) since 12 UT on January 1 of 2000:
    n = df + 365.*yr + nleap

    # Obliquity of ecliptic:
    epsilon = 23.439 - 4.e-7*n
    epsrad  = epsilon*np.pi/180.

    # Right ascension:
    alpha = np.arctan2(np.cos(epsrad)*sinlm, np.cos(lmrad)) * 180./np.pi

    # Declination:
    delta = np.arcsin(np.sin(epsrad)*sinlm) * 180./np.pi

    # Subsolar latitude:
    sbsllat = delta

    # Equation of time (degrees):
    etdeg = l - alpha
    nrot = np.round(etdeg/360.)
    etdeg = etdeg - 360.*nrot

    # Apparent time (degrees):
    aptime = ut/240. + etdeg    # Earth rotates one degree every 240 s.

    # Subsolar longitude:
    sbsllon = 180. - aptime
    nrot = np.round(sbsllon/360.)
    sbsllon = sbsllon - 360.*nrot

    return sbsllat, sbsllon



def is_leapyear(year):
    """ Return True if leapyear else False

        Handles arrays and preserves shape
    
        Parameters
        ----------
        year : array_like
            array of years

        Returns
        -------
        is_leapyear : ndarray of bools
            True where input is leapyear, False elsewhere
    """

    # if array:
    if type(year) is np.ndarray:
        out = np.full_like(year, False, dtype = bool)

        out[ year % 4   == 0] = True
        out[ year % 100 == 0] = False
        out[ year % 400 == 0] = True

        return out

    # if scalar:
    if year % 400 == 0:
        return True

    if year % 100 == 0:
        return False

    if year % 4 == 0:
        return True

    else:
        return False

def sza(glat, glon, time):
    """ return solar zenith angle - the angle of a vertical axis with the Sun Earth line - 
        at given geographic latitudes, longitudes, and times

        input and output in degrees
    """

    # calculate subsolar points
    sslat, sslon = subsol(time)

    ssr = sph_to_car(np.vstack((np.ones_like(sslat), 90 - sslat, 90 - sslon)), deg = True)
    gcr = sph_to_car(np.array([[1.], [90 - glat ], [glon ]]), deg = True)

    # the angle is arccos of the dot product of these two vectors
    return np.arccos(np.sum(ssr*gcr, axis = 0))*180/np.pi

