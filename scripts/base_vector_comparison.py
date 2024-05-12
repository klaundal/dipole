""" compare base vector components with dipole field and IGRF """

import numpy as np
import matplotlib.pyplot as plt
import datetime
import apexpy
import dipole

def longitude_difference(lon1, lon2):
    difference = (lon2 - lon1 + 180) % 360 - 180
    return np.where(difference > 180, difference - 360, difference)


RE = 6371.2e3

date = datetime.datetime(2020, 1, 1)
apx = apexpy.Apex(date.year)
dpl = dipole.Dipole(date.year)


lat_, lon_ = np.linspace(-89, 89, 50), np.linspace(-180, 180, 100)
lat, lon = np.meshgrid(lat_, lon_)
la, lo = np.ravel(lat), np.ravel(lon)

# calculate base vectors in apex coordinates:
f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = apx.basevectors_apex(la, lo, 0)

# calculate dipole coordinates:
lat_dp, lon_dp = dpl.geo2mag(lat, lon)

# calculate dipole base vectors in dipole coordinates:
dpl_basevectors = dpl.get_apex_base_vectors(lat_dp, r = RE, R = RE)

# convert the components to geographic:
dpl_basevectors_geo = []
for basevector in dpl_basevectors:
    lat_, lon_, east, north = dpl.mag2geo(lat_dp.flatten(), lon_dp.flatten(), basevector[0], basevector[1])
    basevector_geo = np.vstack((east, north, basevector[2]))

    assert np.all(np.isclose(lat_ - lat.flatten(), 0))
    assert np.all(np.isclose(longitude_difference(lon_, lon.flatten()), 0))

    dpl_basevectors_geo.append(basevector_geo)



# plot components vs each other:
fig, axes = plt.subplots(ncols = 3, nrows = 6, figsize = (8, 20))
axes[0, 0].scatter(dpl_basevectors_geo[0][0], d1[0], label = '$d_{1e}$')
axes[0, 1].scatter(dpl_basevectors_geo[0][1], d1[1], label = '$d_{1n}$')
axes[0, 2].scatter(dpl_basevectors_geo[0][2], d1[2], label = '$d_{1u}$')

axes[1, 0].scatter(dpl_basevectors_geo[1][0], d2[0], label = '$d_{2e}$')
axes[1, 1].scatter(dpl_basevectors_geo[1][1], d2[1], label = '$d_{2n}$')
axes[1, 2].scatter(dpl_basevectors_geo[1][2], d2[2], label = '$d_{2u}$')

axes[2, 0].scatter(dpl_basevectors_geo[2][0], d3[0], label = '$d_{3e}$')
axes[2, 1].scatter(dpl_basevectors_geo[2][1], d3[1], label = '$d_{3n}$')
axes[2, 2].scatter(dpl_basevectors_geo[2][2], d3[2], label = '$d_{3u}$')

axes[3, 0].scatter(dpl_basevectors_geo[0][0], e1[0], label = '$e_{1e}$')
axes[3, 1].scatter(dpl_basevectors_geo[0][1], e1[1], label = '$e_{1n}$')
axes[3, 2].scatter(dpl_basevectors_geo[0][2], e1[2], label = '$e_{1u}$')

axes[4, 0].scatter(dpl_basevectors_geo[1][0], e2[0], label = '$e_{2e}$')
axes[4, 1].scatter(dpl_basevectors_geo[1][1], e2[1], label = '$e_{2n}$')
axes[4, 2].scatter(dpl_basevectors_geo[1][2], e2[2], label = '$e_{2u}$')

axes[5, 0].scatter(dpl_basevectors_geo[2][0], e3[0], label = '$e_{3e}$')
axes[5, 1].scatter(dpl_basevectors_geo[2][1], e3[1], label = '$e_{3n}$')
axes[5, 2].scatter(dpl_basevectors_geo[2][2], e3[2], label = '$e_{3u}$')

axes[0, 0].set_title('east')
axes[0, 1].set_title('north')
axes[0, 2].set_title('up')

for ax in axes.flatten():
    ax.set_xlabel('dipole')
    ax.set_ylabel('IGRF / apex')
    ax.legend(frameon = False)



plt.show()