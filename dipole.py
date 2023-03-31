"""
Dipole class - Calculate parameters that involve the Earth's magnetic dipole

The Dipole class is initialized with an epoch (decimal year) that is used to
get the relevant dipole Gauss coefficients from the IGRF coefficients. After initialization,
the following methods are available (all are vectorized):
 * B(lat, r)                                 - calculate dipole magnetic field values
 * tilt(times)                               - calculate dipole tilt angle for one or more times
 * geo2mag(lat, lon)                         - convert from geocentric to centered dipole coords and components 
 * mag2geo(lat, lon)                         - convert from centered dipole to geocentric coords and components
 * mlt2mlon(mlt , time)                      - convert magnetic local time to magnetic longitude
 * mlon2mlt(mlon, time)                      - convert magnetic longitude to magnetic local time
 * get_apex_base_vectors(lat, r, R = 6371.2) - get apex basis vectors appropriate for dipole field **not using full IGRF**

and the following parameters:
 * north_pole - dipole pole position in northern hemisphere
 * south_pole - dipole pole position in southern hemisphere
 * axis       - ECEF dipole axis unit vector (pointing to north)
 * B0         - reference magnetic field 

For definitions see Section 3 in 
"Magnetic Coordinate Systems" by Laundal & Richmond (2017), DOI 10.1007/s11214-016-0275-y


In addition to the Dipole class, this script includes the following helper functions:
 * sph_to_car(sph, deg = True) - convert from spherical to cartesian coordinates
 * car_to_sph(car, deg = True) - convert from cartesian to spherical coordinates
 * enu_to_ecef(v, lon, lat)    - convert from ENU to ECEF components
 * ecef_to_enu(v, lon, lat)    - convert from ECEF to ENU components
 * subsol(datetimes)           - calculate location(s) of subsolar point

"""

import numpy as np
import pandas as pd
from ppigrf.ppigrf import read_shc, yearfrac_to_datetime
d2r = np.pi/180
r2d = 1/d2r

RE = 6371.2 # reference radius in km
MU0 = 4 * np.pi * 1e-7

# HELPER FUNCTIONS - SPHERICAL COORDINATES
##########################################
def sph_to_car(sph, deg = True):
    """ Convert from spherical to cartesian coordinates

    Parameters
    ----------
    sph : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        radius, colatitude, and longitude
    deg : bool, optional
        set to True if input is given in degrees. False if radians

    Returns
    -------
    car : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        x, y, z, in ECEF coordinates
    """

    r, theta, phi = sph

    if deg == False:
        conv = 1.
    else:
        conv = d2r

    return np.vstack((r * np.sin(theta * conv) * np.cos(phi * conv), 
                      r * np.sin(theta * conv) * np.sin(phi * conv), 
                      r * np.cos(theta * conv)))


def car_to_sph(car, deg = True):
    """ Convert from spherical to cartesian coordinates

    Parameters
    ----------
    car : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        x, y, z, in ECEF coordinates

    Returns
    -------
    sph : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        radius, colatitude, and longitude
    deg : bool, optional
        set to True if output is wanted in degrees. False if radians
    """

    x, y, z = car

    if deg == False:
        conv = 1.
    else:
        conv = r2d

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)*conv
    phi = ((np.arctan2(y, x)*180/np.pi) % 360)/180*np.pi * conv

    return np.vstack((r, theta, phi))


def enu_to_ecef(v, lon, lat, reverse = False):
    """ convert vector(s) v from ENU to ECEF (or opposite)

    Parameters
    ----------
    v: array
        N x 3 array of east, north, up components
    lat: array
        N array of latitudes (degrees)
    lon: array
        N array of longitudes (degrees)
    reverse: bool (optional)
        perform the reverse operation (ecef -> enu). Default False

    Returns
    -------
    v_ecef: array
        N x 3 array of x, y, z components

    """

    # construct unit vectors in east, north, up directions:
    ph = lon * d2r
    th = (90 - lat) * d2r

    e = np.vstack((-np.sin(ph)             ,               np.cos(ph), np.zeros_like(ph))).T # (N, 3)
    n = np.vstack((-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th)       )).T # (N, 3)
    u = np.vstack(( np.sin(th) * np.cos(ph),  np.sin(th) * np.sin(ph), np.cos(th)       )).T # (N, 3)

    # rotation matrices (enu in columns if reverse, in rows otherwise):
    R_EN_2_ECEF = np.stack((e, n, u), axis = 1 if reverse else 2) # (N, 3, 3)

    # perform the rotations:
    return np.einsum('nij, nj -> ni', R_EN_2_ECEF, v)


def ecef_to_enu(v, lon, lat):
    """ convert vector(s) v from ECEF to ENU

    Parameters
    ----------
    v: array
        N x 3 array of x, y, z components
    lat: array
        N array of latitudes (degrees)
    lon: array
        N array of longitudes (degrees)

    Returns
    -------
    v_ecef: array
        N x 3 array of east, north, up components

    See enu_to_ecef for implementation details
    """
    return enu_to_ecef(v, lon, lat, reverse = True)



# HELPER FUNCTIONS - SUNLIGHT
#############################
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



# DIPOLE CLASS
##############
class Dipole(object):
    def __init__(self, epoch = 2020.):
        """ Initialize Dipole object

        Parameters
        ----------
        epoch: float, optional
            epoch for IGRF dipole Gauss coefficients in decimal year. 
            Must be scalar. Default is 2020.
        """

        if np.array(epoch).size != 1:
            raise Exception('epoch must be scalar')

        # read coefficient file:
        g, h = read_shc()

        date = yearfrac_to_datetime(np.array([epoch]))

        if (date > g.index[-1]) or (date < g.index[0]):
            print('Warning: You provided date(s) not covered by coefficient file \n({} to {})'.format(
                  g.index[0].date(), g.index[-1].date()))

        # Interpolate IGRF coefficients to the given epoch:
        index = g.index.union(date)
        g = g.reindex(index).groupby(index).first() # reindex and skip duplicates
        h = h.reindex(index).groupby(index).first() # reindex and skip duplicates
        g = g.interpolate(method = 'time').loc[date, :]
        h = h.interpolate(method = 'time').loc[date, :]

        self.date = date
        self.epoch = epoch

        # Select dipole coefficicents
        self.g10, self.g11, self.h11 = g[(1, 0)].values[0], g[(1, 1)].values[0], h[(1, 1)].values[0]

        # Reference magnetic field:
        self.B0 = np.sqrt(self.g10**2 + self.g11**2 + self.h11**2)

        # Unit vector pointing to dipole pole in the north (negative dipole axis in physics convention) 
        self.axis = -np.hstack((self.g11, self.h11, self.g10))/self.B0

        # Pole locations:
        colat, longitude = car_to_sph( self.axis.reshape((-1, 1)), deg = True)[1:]
        self.north_pole = np.array((  90 - colat, longitude) ).flatten()
        self.south_pole = np.array((- 90 + colat, (longitude + 180) % 360)).flatten()


    def __str__(self):
        return 'Dipole object for epoch {}'.format(str(self.date[0].date()))

    def __repr__(self):
        return str(self)

    def set_epoch(self, epoch):
        """ Initialize the Dipole object again with new epoch """
        self.__init__(epoch)

    def B(self, lat, r):
        """ 
        Calculate components of the dipole field in dipole coordinates 

        Parameters
        ----------
        lat : array
            latitude in centered dipole coordinates - in degrees
        r : array
            radius in km
        
        Returns
        -------
        Bn : array
            dipole field in northward direction, in nT. Shape implied by lat and r
            using numpy broadcasting rules
        Br : array
            dipole field in radial direction, in nT. Shape implied by lat and r
            using numpy broadcasting rules
        """

        shape = np.broadcast(lat, r).shape
        colat = np.deg2rad(90 - (lat * np.ones_like(r)).flatten())
        r    = (np.ones_like(lat) * r).flatten()

        Bn = self.B0 * (RE / r) ** 3 * np.sin( colat )
        Br = -2 * self.B0 * (RE / r) ** 3 * np.cos( colat )

        return Bn.reshape(shape), Br.reshape(shape)
        

    def tilt(self, times):
        """
        Calculate dipole tilt angle for selected time(s)

        Parameters
        ----------
        times: array
            array of datetimes for which dipole tilt angle shall be calculated

        Returns
        -------
        tilt: array
            array of dipole tilt angles, in degrees, with same shape as input

        """

        shape = np.array(times).shape
        times = np.squeeze(np.array(times)).flatten()

        # get subsolar point coordinates
        sslat, sslon = subsol(times)
        
        s = np.vstack(( np.cos(sslat * d2r) * np.cos(sslon * d2r),
                        np.cos(sslat * d2r) * np.sin(sslon * d2r),
                        np.sin(sslat * d2r))).T
        m = self.axis

        # calculate tilt angle:
        return np.arcsin( np.sum( s * m , axis = 1 )).reshape(shape) * r2d


    def geo2mag(self, lat, lon, Ae = None, An = None, inverse = False):
        """ Convert geographic (geocentric) to centered dipole coordinates

        The conversion uses IGRF coefficients directly, interpolated
        to the provided epoch. The construction of the rotation matrix
        follows Laundal & Richmond (2017) [4]_ . 

        Preserves shape. glat, glon, Ae, and An should have matching shapes

        Parameters
        ----------
        lat : array_like
            array of geographic latitudes
        lon : array_like
            array of geographic longitudes
        Ae  : array-like, optional
            array of eastward vector components to be converted. Default
            is None, and no converted vector components will be returned
        An  : array-like, optional
            array of northtward vector components to be converted. Default
            is None, and no converted vector components will be returned
        inverse: bool, optional
            set to True to convert from magnetic to geographic. 
            Default is False

        Returns
        -------
        cdlat : array
            array of centered dipole latitudes [degrees]
        cdlon : array
            array of centered dipole longitudes [degrees]
        Ae_cd : array
            array of eastward vector components in dipole coords
            (if Ae != None and An != None)
        An_cd : ndarray
            array of northward vector components in dipole coords
            (if Ae != None and An != None)

        """

        try:
            shape = np.broadcast(lat, lon, Ae, An).shape
        except:
            raise Exception('Input have inconsistent shapes')

        if any([Ae is None, An is None]):
            Ae, An = 1, 1
            return_components = False
        else:
            return_components = True

        lat = np.broadcast_to(lat, shape).flatten()
        lon = np.broadcast_to(lon, shape).flatten()

        # make rotation matrix from geo to cd
        Zcd = self.axis
        Zgeo_x_Zcd = np.cross(np.array([0, 0, 1]), Zcd)
        Ycd = Zgeo_x_Zcd / np.linalg.norm(Zgeo_x_Zcd)
        Xcd = np.cross(Ycd, Zcd)

        Rgeo_to_cd = np.vstack((Xcd, Ycd, Zcd))

        if inverse: # transpose rotation matrix to get inverse operation
            Rgeo_to_cd = Rgeo_to_cd.T

        # convert input to ECEF:
        colat = 90 - lat
        r_geo = sph_to_car(np.vstack((np.ones_like(colat), colat, lon)), deg = True)

        # rotate:
        r_cd = Rgeo_to_cd.dot(r_geo)

        # convert result back to spherical:
        _, colat_cd, lon_cd = car_to_sph(r_cd, deg = True)

        # return coords if vector components are not to be calculated
        if return_components == False:
            return 90 - colat_cd.reshape(shape), lon_cd.reshape(shape)

        # convert components:
        Ae = (np.ones_like(lon * lat * An) * Ae).flatten()
        An = (np.ones_like(lon * lat * Ae) * An).flatten()

        A_geo_enu  = np.vstack((Ae, An, np.zeros(Ae.size)))
        A = np.sqrt(Ae**2 + An**2)
        A_geo_ecef = enu_to_ecef((A_geo_enu / A).T, lon, lat ) # rotate normalized vectors to ecef
        A_cd_ecef = Rgeo_to_cd.dot(A_geo_ecef.T)
        A_cd_enu  = ecef_to_enu(A_cd_ecef.T, lon_cd, 90 - colat_cd).T * A 

        # return coords and vector components:
        return 90 - colat_cd.reshape(shape), lon_cd.reshape(shape), A_cd_enu[0].reshape(shape), A_cd_enu[1].reshape(shape)


    def mag2geo(self, lat, lon, Ae = None, An = None):
        """ 
        Convert centered dipole coordinates to geocentric coordinates

        The conversion uses IGRF coefficients directly, interpolated
        to the provided epoch. The construction of the rotation matrix
        follows Laundal & Richmond (2017) [4]_ . 

        Preserves shape. glat, glon, Ae, and An should have matching shapes

        Parameters
        ----------
        lat : array_like
            array of centered dipole latitudes
        lon : array_like
            array of centered dipole longitudes
        Ae  : array-like, optional
            array of eastward vector components to be converted. Default
            is None, and no converted vector components will be returned
        An  : array-like, optional
            array of northtward vector components to be converted. Default
            is None, and no converted vector components will be returned

        Returns
        -------
        cdlat : array
            array of geocentric latitudes [degrees]
        cdlon : array
            array of geocentric longitudes [degrees]
        Ae_cd : array
            array of eastward vector components in geocentric
            (if Ae != None and An != None)
        An_cd : ndarray
            array of northward vector components in geocentric
            (if Ae != None and An != None)

        """        
        return self.geo2mag(lat, lon, Ae = Ae, An = An, inverse = True)


    def mlon2mlt(self, mlon, times):
        """ 
        Convert magnetic longitude to magnetic local time using equation (93) 
        in Laundal & Richmond (2017) [4]_. This equation is valid for longitudes
        given in several different coordinate systems, including Apex coordinates, 
        AACGM, eccentric dipole coordinates, and of course dipole coordinates. 

        Calculations are vectorized, and shapes are preserved, using numpy
        broadcasting rules

        Parameters
        ----------
        mlon: array
            array of magnetic longitudes [degrees]
        times: array
            array of datetimes

        Returns
        -------
        mlt: array
            array of magnetic local times [hours], with shape implied by mlon and times 

        """

        shape = np.broadcast(mlon, times).shape
        mlon  = np.broadcast_to(mlon , shape).flatten()
        times = np.broadcast_to(times, shape).flatten()

        ssglat, ssglon = subsol(times)
        sqlat, ssqlon = self.geo2mag(ssglat, ssglon)

        londiff = mlon - ssqlon
        londiff = (londiff + 180) % 360 - 180 # signed difference in longitude

        mlt = (180. + londiff)/15. # convert to mlt with ssqlon at noon

        return mlt.reshape(shape)


    def mlt2mlon(self, mlt, times):
        """
        Convert magnetic local time to magnetic longitude, using the inverse
        operation of mlon2mlt

        Parameters
        ----------
        mlt: array
            array of magnetic local times [hours]
        times: array
            array of datetimes

        Returns
        -------
        mlon: array
            array of magnetic longitudes [degrees], hape implied by input

        """
        
        shape = np.broadcast(mlt, times).shape
        mlt   = np.broadcast_to(mlt  , shape).flatten()
        times = np.broadcast_to(times, shape).flatten()

        ssglat, ssglon = map(np.array, subsol(times))
        sqlat, ssqlon = self.geo2mag(ssglat, ssglon)

        mlon = (15 * mlt - 180 + ssqlon + 360) % 360

        return mlon.reshape(shape)


    def get_apex_base_vectors(self, lat, r, R = 6371.2):
        """ Calculate apex coordinate base vectors d_i and e_i (i = 1, 2, 3)

        The base vectors are defined in Richmond (1995). They can be calculated analytically
        for a dipole magnetic field and spherical Earth. 

        The output vectors will have shape (3, N) where N is the combined size of the input,
        after numpy broadcasting. The three rows correspond to east, north and radial components
        for a spherical Earth

        Note
        ----
        This function only calculates Modified Apex base vectors. QD base vectors f_i and g_i are 
        just eastward, northward, and radial unit vectors for a dipole field (and f_i = g_i)
            
        Parameters
        ----------
        r : array-like
            radii of the points where the base vectors shall be calculated, in same unit as R
        lat : array-like
            centered dipole latitude [deg] of the points where the base vectors shall be calculated
        R : float, optional
            Reference radius used in modified apex coordinates. Default is 6371.2 km

        Returns
        -------
        d1 : array-like
            modified apex base vector d1 for a dipole magnetic field, shape (3, N)
        d2 : array-like
            modified apex base vector d2 for a dipole magnetic field, shape (3, N)
        d3 : array-like
            modified apex base vector d3 for a dipole magnetic field, shape (3, N)
        e1 : array-like
            modified apex base vector e1 for a dipole magnetic field, shape (3, N)
        e2 : array-like
            modified apex base vector e2 for a dipole magnetic field, shape (3, N)
        e3 : array-like
            modified apex base vector e3 for a dipole magnetic field, shape (3, N)
        """
        try:
            r, la = map(np.ravel, np.broadcast_arrays(r, lat))
        except:
            raise ValueError('get_apex_base_vectors: Input not broadcastable')

        la = np.deg2rad(la)
        if np.any(r / np.cos(la)**2 < R):
            raise ValueError('get_apex_base_vectors: Some points have apex height < R. Apex height is r/cos(lat)**2.')

        N = r.size

        R2r = R / r 
        C   = np.sqrt(4 - 3 * R2r * np.cos(la)**2)
        
        d1 =  R2r ** (3./2) * np.vstack((np.ones(N), np.zeros(N), np.zeros(N)))
        d2 = -np.sign(la) * R2r ** (3./2) / C * np.vstack((np.zeros(N), 2 * np.sin(la), np.cos(la)))
        d3 =  R2r ** (-3) * C / (4 - 3 * np.cos(la)**2) * np.vstack((np.zeros(N), np.cos(la), -2*np.sin(la)))

        e1 = np.cross(d2.T, d3.T).T
        e2 = np.cross(d3.T, d1.T).T
        e3 = np.cross(d1.T, d2.T).T

        return (d1, d2, d3, e1, e2, e3)


    def get_flux(self, lon, lat, dlon = 1., dlat = 0.1, R = 6371.2):
        """ Calculate magnetic flux poleward of a closed curve described by lon, lat

        The magnetic flux calculation is performed by first interpolating the given boundary points to a constant
        step size, and then applying Richmond95's Equation 4.15. 

        Parameters
        ----------
        lon : array
            longitudes [deg] describing the closed contour poleward of which we calculate magnetic flux
        lat : array
            latitudes [deg] describing the closed contour poleward of which we calculate magnetic flux
        dlon : float, optional
            longitude resolution to use in the integral. Default is 1 degree
        dlat: float, optional
            latitude resolution to use in the integral. Default is 0.1 degree
        R : float, optional
            radius [km] at which to calcualte flux. Default is 6371.2 km

        Returns
        -------
        flux : float
            Magnetic flux poleward of the boundary, in Weber
        """

        assert np.all(lat >= 0) # only use positive latitudes
        lon, lat = map(np.ravel, np.broadcast_arrays(lon, lat)) 

        N = np.int32(360 / dlon) + 1 # number of points in longitude direction
        lonxx = np.linspace(0, 360, N) # longitude coordinate
        boundary = np.interp(lonxx, lon, lat, period = 360) # interpolate input to constant step size

        minlat = lat.min()
        latxx = np.r_[90 - dlat/2:minlat:-dlat][::-1] # latitude integration steps

        d1, d2, d3, e1, e2, e3 = self.get_apex_base_vectors(latxx, R, R = R) # get Apex base vectors
        Bn, Br = self.B(latxx, R)
        Be3 = d3[1] * Bn + d3[2] * Br 
        Be3 = Be3 * 1e-9 # nT -> T

        # in the equations below, I use expressions appropriate for apex coordinates, which is ok since we
        # use apex reference radius equal to evaluation radius
        sinIm = 2 * np.sin(np.deg2rad(latxx)) / np.sqrt(4 - 3 * np.cos(np.deg2rad(latxx))**2)
        dF = (R * 1e3)**2 * np.cos(np.deg2rad(latxx)) * np.abs(sinIm) * Be3 * np.deg2rad(dlon) * np.deg2rad(dlat) # flux per lon and lat according to Richmond95 Eq 4.15
        dF = np.tile(dF, (lonxx.size, 1))
        dF[latxx.reshape((1, -1)) < boundary.reshape((-1, 1))] = 0 # mask elements equatorward of boundary

        return( np.sum(dF) )




if __name__ == '__main__':

    print('Running tests on apex base vectors')
    N = 1000 # number of points in random cloud
    R = 6371.2+200 # apex reference height
    x, y, z = np.random.random(N), np.random.random(N), np.random.random(N)
    iii = x**2 + y**2 + z**2 <= 1
    r  = np.sqrt(x**2 + y**2 + z**2)[iii]
    la = np.rad2deg(np.arcsin(z[iii] / r))
    r = R * (1 + r)

    iii = r / np.cos(np.deg2rad(la))**2 >= R
    r = r[iii]
    la = la[iii]


    d = Dipole(2010)
    d1, d2, d3, e1, e2, e3 = d.get_apex_base_vectors(la, r, R = R)
    Bn, Br = d.B(la, r)
    BB = np.vstack((np.zeros_like(Bn), Bn, Br))

    # test orthogonality properties
    assert np.allclose(np.abs(np.sum(d1*d2, axis = 0)), 0)
    assert np.allclose(np.abs(np.sum(d1*d3, axis = 0)), 0)
    assert np.allclose(np.abs(np.sum(d2*d3, axis = 0)), 0)
    assert np.allclose(np.abs(np.sum(e1*e2, axis = 0)), 0)
    assert np.allclose(np.abs(np.sum(e1*e3, axis = 0)), 0)
    assert np.allclose(np.abs(np.sum(e2*e3, axis = 0)), 0)
    assert np.allclose(np.abs(np.sum(d1*e2, axis = 0)), 0)
    assert np.allclose(np.abs(np.sum(d1*e3, axis = 0)), 0)
    assert np.allclose(np.abs(np.sum(d2*e1, axis = 0)), 0)
    assert np.allclose(np.abs(np.sum(d2*e3, axis = 0)), 0)
    assert np.allclose(np.abs(np.sum(d3*e1, axis = 0)), 0)
    assert np.allclose(np.abs(np.sum(d3*e2, axis = 0)), 0)
    assert np.allclose(np.linalg.norm(np.cross(e3.T, d3.T), axis = 1), 0) # e3 perpendicular to B
    assert np.allclose(np.linalg.norm(np.cross(e3.T, BB.T), axis = 1), 0) # d3 perpendicular to B
    assert np.all(np.sum(d3 * BB, axis = 0) > 0) # e3 along B
    assert np.all(np.sum(e3 * BB, axis = 0) > 0) # d3 along B

    # test scaling properties
    la = -np.linspace(1e-3, .7 * np.pi/2, 100)
    req = 10*R
    r = req * np.cos(la)**2
    d1, d2, d3, e1, e2, e3 = d.get_apex_base_vectors(np.rad2deg(la), r, R = R)
    Bn, Br = d.B(np.rad2deg(la), r)
    BB = np.vstack((np.zeros_like(Bn), Bn, Br))
    Be3 = d3[1] * Bn + d3[2] * Br # should be constant since all d3 are on same field line
    D = np.linalg.norm(np.cross(d1.T, d2.T), axis = 1) # B / D should be equal to Be3
    B = np.sqrt(Bn**2 + Br**2)

    assert np.allclose(Be3 - B/D, 0)
    assert np.allclose(Be3 - Be3[0], 0)

    print ('testing flux calculation')
    lon = np.linspace(0, 360, 100)
    # test with constant latitude
    lat0 = 80
    lat = np.zeros_like(lon) + lat0
    flux = d.get_flux(lon, lat, dlon = 0.1, dlat = 0.01, R = R)
    # compare to the analytical expression, obtained from integrating dipole Br:
    analytical = 2 * np.pi * d.B0 * 1e-9 * (RE*1e3)** 3 / (R * 1e3) * np.cos(np.deg2rad(lat0))**2

    assert np.isclose(flux, analytical, rtol = 1e-3)

