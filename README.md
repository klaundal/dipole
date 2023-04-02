# Dipole - calculations involving dipole model of Earth's magnetic field
Python code to calculate 

- dipole magnetic field
- dipole tilt angle
- centered dipole coordinates and vector components
- magnetic local time
- dipole pole locations
- magnetic flux poleward of some boundary
- apex base vectors for a dipole magnetic field

The calculations use time-dependendent IGRF coefficients (see https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html for details and https://doi.org/10.1186/s40623-020-01163-9 for even more details) accessed through the ppigrf Python module (https://github.com/klaundal/ppigrf).

For a definition of centered dipole coordinates, see Section 3.1 in https://doi.org/10.1007/s11214-016-0275-y

The code is vectorized, so all calculations should be pretty fast.  

## Examples

Here is an example calculation of dipole tilt angle. In this example, a Dipole object is initialized for epoch 2022.5, and dipole tilt angle is calculated for two different times

```python
import dipole
from datetime import datetime

dates = [datetime(2022, 7, 13, 10, 0, 0), datetime(2022, 7, 13, 22, 0, 0)]
tilts = dipole.Dipole(2022.5).tilt(dates)
```

Here is an example converting northward-pointing unit vectors from geocentric to centered dipole coordinates. This example also illustrates how numpy broadcasting rules are used for input with different shapes

```python
import dipole
import numpy as np

lat = 70
lon = np.r_[0:360:15]
north = 1
east = 0

d = dipole.Dipole(2010.) # IGRF epoch 2010.
cdlat, cdlon, cd_east, cd_north = d.geo2mag(lat, lon, Ae = east, An = north)

```



## Contact
If you find errors, please let me know! 

You don't need permission to copy or use this code.

karl.laundal at uib.no
