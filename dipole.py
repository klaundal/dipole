"""
Functions for calculating dipole axis and dipole poles at given epoch(s)
using IGRF Gauss coefficients. 

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
from .utils import car_to_sph

d2r = np.pi/180
r2d = 180/np.pi

# first make arrays of IGRF dipole coefficients. This is used to make rotation matrix from geo to cd coords
# these values are from https://www.ngdc.noaa.gov/IAGA/vmod/igrf12coeffs.txt
time =[1900.0, 1905.0, 1910.0, 1915.0, 1920.0, 1925.0, 1930.0, 1935.0, 1940.0, 1945.0, 1950.0, 1955.0, 1960.0, 1965.0, 1970.0, 1975.0, 1980.0, 1985.0, 1990.0, 1995.0,   2000.0,    2005.0,    2010.0,   2015.0, 2020.0]
g10 = [-31543, -31464, -31354, -31212, -31060, -30926, -30805, -30715, -30654, -30594, -30554, -30500, -30421, -30334, -30220, -30100, -29992, -29873, -29775, -29692, -29619.4, -29554.63, -29496.57, -29442.0]
g11 = [ -2298,  -2298,  -2297,  -2306,  -2317,  -2318,  -2316,  -2306,  -2292,  -2285,  -2250,  -2215,  -2169,  -2119,  -2068,  -2013,  -1956,  -1905,  -1848,  -1784,  -1728.2,  -1669.05,  -1586.42,  -1501.0]
h11 = [  5922,   5909,   5898,   5875,   5845,   5817,   5808,   5812,   5821,   5810,   5815,   5820,   5791,   5776,   5737,   5675,   5604,   5500,   5406,   5306,   5186.1,   5077.99,   4944.26,   4797.1]
g10sv =  10.3 # secular variation coefficients
g11sv =  18.1
h11sv = -26.6
g10.append(g10[-1] + g10sv * 5) # append 2020 values using secular variation coefficients
g11.append(g11[-1] + g11sv * 5)
h11.append(h11[-1] + h11sv * 5)
igrf_dipole = {'g10':np.array(g10), 'g11':np.array(g11), 'h11':np.array(h11), 'index':time}
igrf_dipole['B0'] = np.sqrt(igrf_dipole['g10']**2 + igrf_dipole['g11']**2 + igrf_dipole['h11']**2)



def dipole_axis(epoch):
    """ calculate dipole axis in geocentric ECEF coordinates for given epoch(s)

    Calculations are based on IGRF coefficients, and linear interpolation is used 
    in between IGRF models (defined every 5 years). Secular variation coefficients
    are used for the five years after the latest model. 

    Parameters
    ----------
    epoch : float or array of floats
        year (with fraction) for which the dipole axis will be calculated. Multiple
        epochs can be given, as an array of N floats, resulting in a N x 3-dimensional
        return value

    Returns
    -------
    axes : array
        N x 3-dimensional array, where N is the number of inputs (epochs), and the
        columns contain the x, y, and z components of the corresponding dipole axes

    """

    epoch = np.squeeze(np.array(epoch)).flatten() # turn input into array in case it isn't already

    # interpolate Gauss coefficients to the input times:
    params = {key: np.interp(epoch, igrf_dipole['index'], igrf_dipole[key], left = np.nan, right = np.nan) 
              for key in ['g10', 'g11', 'h11', 'B0']}

    Z_cd = -np.vstack((params['g11'], params['h11'], params['g10']))/params['B0']

    return Z_cd.T






def dipole_poles(epoch):
    """ calculate dipole pole positions at given epoch(s)

    Parameters
    ----------
    epoch : float or array of floats
        year (with fraction) for which the dipole axis will be calculated. Multiple
        epochs can be given, as an array of N floats, resulting in a N x 3-dimensional
        return value
    
    Returns
    -------
    north_colat : array
        colatitude of the dipole pole in the northern hemisphere, same number of
        values as input
    north_longitude: array
        longitude of the dipole pole in the northern hemisphere, same number of
        values as input
    south_colat : array
        colatitude of the dipole pole in the southern hemisphere, same number of
        values as input
    south_longitude: array
        longitude of the dipole pole in the southern hemisphere, same number of
        values as input

       


    """

    north_colat, north_longitude = car_to_sph( dipole_axis(epoch).T, deg = True)[1:]
    south_colat, south_longitude = car_to_sph(-dipole_axis(epoch).T, deg = True)[1:]
    
    return north_colat, north_longitude, south_colat, south_longitude

