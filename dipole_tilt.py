"""
Module for calculating the dipole tilt angle, using Equation 15 of

Laundal, K.M. & Richmond, A.D. Space Sci Rev (2017) 206: 27. 
https://doi.org/10.1007/s11214-016-0275-y






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
from dipole import dipole_axis
from .dipole import dipole_axis
from .sun import subsol
r2d = 180/np.pi
d2r = 1./r2d

def dipole_tilt(times, epoch = 2015.):
    """ Calculate dipole tilt angle for given set of times, at given epoch(s)

    Parameters
    ----------
    times : datetime or array/list of datetimes
        Times for which the dipole tilt angle should be calculated
    epoch : float or array/list of floats, optional
        Year (with fraction) for calculation of dipole axis. This should either be
        a scalar, or contain as man elements as times. Default epoch is 2015.0

    Return
    ------
    tilt_angle : array
        Array of dipole tilt angles in degrees

    Example
    -------
    >>> from datetime import datetime
    
    >>> print dipole_tilt(datetime(1927, 6, 10, 12, 00), epoch = 1927)
    [ 26.79107107]

    >>> # several times can be given. If they are close in time, one epoch should be fine 
    >>> print dipole_tilt([datetime(1927, 6, 10, 12, 00), datetime(1927, 6, 10, 10, 00)], epoch = 1927)
    [ 26.79107107  20.89550663]
    
    >>> # if the times are far apart, the epoch for each can be specified to take field changes into account
    >>> # this will be a bit slower if many times are given
    >>> print dipole_tilt([datetime(1927, 6, 10, 12, 00), datetime(2015, 6, 10, 10, 00)], epoch = (1927, 2015))
    [ 26.79107107  20.59137527]

    """

    epoch = np.squeeze(np.array(epoch)).flatten()
    times = np.squeeze(np.array(times)).flatten()

    if not (epoch.shape == times.shape or epoch.shape == (1,)):
        raise ValueError('epoch should either be scalar or have as many elements as times')

    # get subsolar point coordinates
    sslat, sslon = subsol(times)
    
    s = np.vstack(( np.cos(sslat * d2r) * np.cos(sslon * d2r),
                    np.cos(sslat * d2r) * np.sin(sslon * d2r),
                    np.sin(sslat * d2r))).T
    m = dipole_axis(epoch)

    # calculate tilt angle:
    return np.arcsin( np.sum( s * m , axis = 1 )) * r2d
