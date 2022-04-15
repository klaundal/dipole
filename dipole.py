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
import pandas as pd
from .utils import car_to_sph, sph_to_car, enu_to_ecef, ecef_to_enu
from .utils import car_to_sph

RE = 6371.2 # reference radius in km

d2r = np.pi/180
r2d = 180/np.pi

# first make arrays of IGRF dipole coefficients. This is used to make rotation matrix from geo to cd coords
# these values are from https://www.ngdc.noaa.gov/IAGA/vmod/igrf12coeffs.txt
time =[1900.0, 1905.0, 1910.0, 1915.0, 1920.0, 1925.0, 1930.0, 1935.0, 1940.0, 1945.0, 1950.0, 1955.0, 1960.0, 1965.0, 1970.0, 1975.0, 1980.0, 1985.0, 1990.0, 1995.0,   2000.0,    2005.0,    2010.0,   2015.0, 2020.0, 2025.0]
g10 = [-31543, -31464, -31354, -31212, -31060, -30926, -30805, -30715, -30654, -30594, -30554, -30500, -30421, -30334, -30220, -30100, -29992, -29873, -29775, -29692, -29619.4, -29554.63, -29496.57, -29441.46,  -29404.8]
g11 = [ -2298,  -2298,  -2297,  -2306,  -2317,  -2318,  -2316,  -2306,  -2292,  -2285,  -2250,  -2215,  -2169,  -2119,  -2068,  -2013,  -1956,  -1905,  -1848,  -1784,  -1728.2,  -1669.05,  -1586.42,  -1501.77,   -1450.9]
h11 = [  5922,   5909,   5898,   5875,   5845,   5817,   5808,   5812,   5821,   5810,   5815,   5820,   5791,   5776,   5737,   5675,   5604,   5500,   5406,   5306,   5186.1,   5077.99,   4944.26,   4795.99,    4652.5]
g10sv =  5.7 # secular variations
g11sv =  7.4
h11sv = -25.9
g10.append(g10[-1] + g10sv * 5) # append 2025 values using secular variation coefficients
g11.append(g11[-1] + g11sv * 5)
h11.append(h11[-1] + h11sv * 5)
igrf_dipole = pd.DataFrame({'g10':np.array(g10), 'g11':np.array(g11), 'h11':np.array(h11)}, index = time)
igrf_dipole['B0'] = np.sqrt(igrf_dipole['g10']**2 + igrf_dipole['g11']**2 + igrf_dipole['h11']**2)



def dipole_field(mlat, r, epoch = 2020):
    """ calculate components of the dipole field in dipole coordinates 

    The dipole moment will be calculated from IGRF coefficients at given epoch

    Parameters
    ----------
    mlat : array
        magnetic latitude (latitude in a system with dipole pole at pole)
    r : array
        radius in km
    epoch : float, optional
        The dipole moment will be calculated from IGRF coefficients at given epoch
        default epoch is 2020
    
    Returns
    -------
    Bn : array
        dipole field in northward direction, in nT. Same shape as mlat/r
    Br : array
        dipole field in radial direction, in nT. Same shape as mlat/r
    """

    shape = np.broadcast(mlat, r).shape
    colat = (90 - (mlat * np.ones_like(r)).flatten()) * d2r
    r    = (np.ones_like(mlat) * r).flatten()


    # Find IGRF parameters for given epoch:
    dipole = igrf_dipole.reindex(list(igrf_dipole.index) + [epoch]).sort_index().interpolate().drop_duplicates() 
    dipole = dipole.loc[epoch, :]

    B0 = dipole['B0']

    Bn = B0 * (RE / r) ** 3 * np.sin( colat )
    Br = -2 * B0 * (RE / r) ** 3 * np.cos( colat )

    return Bn.reshape(shape), Br.reshape(shape)




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

    epoch = np.asarray(epoch).flatten() # turn input into array in case it isn't already

    # interpolate Gauss coefficients to the input times:
    dipole = igrf_dipole.reindex(list(igrf_dipole.index) + list(epoch)).sort_index().interpolate().drop_duplicates() 

    params = {key: dipole.loc[epoch, key].values for key in ['g10', 'g11', 'h11', 'B0']}

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
    print(dipole_axis(epoch))
    north_colat, north_longitude = car_to_sph( dipole_axis(epoch).T, deg = True)[1:]
    south_colat, south_longitude = car_to_sph(-dipole_axis(epoch).T, deg = True)[1:]
    
    return north_colat, north_longitude, south_colat, south_longitude


def geo2mag(glat, glon, Ae = None, An = None, epoch = 2020, deg = True, inverse = False):
    """ Convert geographic (geocentric) to centered dipole coordinates

    The conversion uses IGRF coefficients directly, interpolated
    to the provided epoch. The construction of the rotation matrix
    follows Laundal & Richmond (2017) [4]_ . 

    Preserves shape. glat, glon, Ae, and An should have matching shapes

    Parameters
    ----------
    glat : array_like
        array of geographic latitudes
    glon : array_like
        array of geographic longitudes
    Ae   : array-like, optional
        array of eastward vector components to be converted. Default
        is 'none', and no converted vector components will be returned
    An   : array-like, optional
        array of northtward vector components to be converted. Default
        is 'none', and no converted vector components will be returned
    epoch : float, optional
        epoch (year) for the dipole used in the conversion, default 2020
    deg : bool, optional
        True if input is in degrees, False otherwise
    inverse: bool, optional
        set to True to convert from magnetic to geographic. 
        Default is False

    Returns
    -------
    cdlat : ndarray
        array of centered dipole latitudes [degrees]
    cdlon : ndarray
        array of centered dipole longitudes [degrees]
    Ae_cd : ndarray
        array of eastward vector components in dipole coords
        (if Ae != None and An != None)
    An_cd : ndarray
        array of northward vector components in dipole coords
        (if Ae != None and An != None)

    """

    shape = np.asarray(glat).shape
    glat, glon = np.asarray(glat).flatten(), np.asarray(glon).flatten()

    # Find IGRF parameters for given epoch:
    dipole = igrf_dipole.reindex(list(igrf_dipole.index) + [epoch]).sort_index().interpolate().drop_duplicates() 
    dipole = dipole.loc[epoch, :]

    # make rotation matrix from geo to cd
    Zcd = -np.array([dipole.g11, dipole.h11, dipole.g10])/dipole.B0
    Zgeo_x_Zcd = np.cross(np.array([0, 0, 1]), Zcd)
    Ycd = Zgeo_x_Zcd / np.linalg.norm(Zgeo_x_Zcd)
    Xcd = np.cross(Ycd, Zcd)

    Rgeo_to_cd = np.vstack((Xcd, Ycd, Zcd))

    if inverse: # transpose rotation matrix to get inverse operation
        Rgeo_to_cd = Rgeo_to_cd.T

    # convert input to ECEF:
    colat = 90 - glat.flatten() if deg else np.pi/2 - glat.flatten()
    glon  = glon.flatten()
    r_geo = sph_to_car(np.vstack((np.ones_like(colat), colat, glon)), deg = deg)

    # rotate:
    r_cd = Rgeo_to_cd.dot(r_geo)

    # convert result back to spherical:
    _, colat_cd, lon_cd = car_to_sph(r_cd, deg = True)

    # return coords if vector components are not to be calculated
    if any([Ae is None, An is None]):
        return 90 - colat_cd.reshape(shape), lon_cd.reshape(shape)

    Ae, An = np.asarray(Ae).flatten(), np.asarray(An).flatten()
    A_geo_enu  = np.vstack((Ae, An, np.zeros(Ae.size)))
    A = np.sqrt(Ae**2 + An**2)
    A_geo_ecef = enu_to_ecef((A_geo_enu / A).T, glon, glat ) # rotate normalized vectors to ecef
    A_cd_ecef = Rgeo_to_cd.dot(A_geo_ecef.T)
    A_cd_enu  = ecef_to_enu(A_cd_ecef.T, lon_cd, 90 - colat_cd).T * A 

    # return coords and vector components:
    return 90 - colat_cd.reshape(shape), lon_cd.reshape(shape), A_cd_enu[0].reshape(shape), A_cd_enu[1].reshape(shape)



def generic_dipole_field(r, m, r0 = np.zeros((1, 3))):
    """ 
    Evaluate the magnetic field of a dipole with magnetic moment m
    at coordinates r. 

    Note that we use a normalization common in geomagnetism (Schmidt semi-normalization),
    so that the magnitude of the magnetic moment vector used here is RE/4pi times the 
    magnetic moment on Wikipedia*, where RE = 6371200 m. This means that you can pass an
    m that consists of IGRF coefficients g11, h11, and g10, and get the dipole field.

    *https://en.wikipedia.org/wiki/Magnetic_dipole 


    Parameters:
    -----------
    r: N x 3 array
        N coordinates (x, y, z) where field will be evaluated. 
        Should have units [km]
    m: 3-element array
        dipole moment vector
    r0: 3-element array, optional
        position of the dipole moment (x, y, z). By default, the
        dipole is assumed to be located in the origin
    
    Returns:
    --------
    B: N x 3 array
        Magnetic field - N Cartesian components
        
    """

    g11, h11, g10 = m # get Gauss coefficients

    # convert input to spherical coordinates and reshape to column vectors:
    radius, theta, phi = map(lambda x: x.reshape((-1, 1)), car_to_sph((r - r0).T))
    theta, phi = theta * d2r, phi * d2r


    # calculate the magnetic field spherical components:
    rr = (RE / radius) ** 3
    Br     = 2 * rr * np.cos(theta) * ( g11 * np.cos(phi) + h11 * np.sin(phi) + g10)
    Btheta =     rr * np.sin(theta) * ( g11 * np.cos(phi) + h11 * np.sin(phi) + g10)
    Bphi   =    -rr * np.cos(theta) * (-g11 * np.sin(phi) + h11 * np.cos(phi)      ) / np.sin(theta)

    # convert to Cartesian:
    lon, lat = phi.flatten() / d2r, 90 - theta.flatten() / d2r
    Bx, By, Bz = enu_to_ecef(np.hstack((Bphi, -Btheta, Br)), lon, lat).T

    return Bx, By, Bz



if __name__ == '__main__':

    # A test that generic_dipole_field gives results that are consistent with dipole_field
    ######################################################################################
    r = (np.random.random((4, 3)) - 1) * 15000
    m = igrf_dipole.loc[2020, ["g11", "h11", "g10"]].values # tilted dipole
    m = np.array([0, 0, -np.linalg.norm(m)]) # aligned dipole

    Bx, By, Bz = generic_dipole_field(r, m)
    radius = np.linalg.norm(r, axis = 1)
    lat = np.arcsin(r[:, 2] / radius) / d2r
    lon = np.arctan2(r[:, 1], r[:, 0]) / d2r

    Be_, Bn_, Bu_ = ecef_to_enu(np.vstack((Bx, By, Bz)).T, lon, lat).T

    # this should be the same as what I get with the dipole_field function:
    Bn, Br = dipole_field(lat, radius * 1e-3, epoch = 2020)

    assert np.allclose(Bn, Bn_) & np.allclose(Br, Bu_)






