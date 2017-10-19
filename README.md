# dipole
Calculations of dipole axis, dipole poles, and dipole tilt angle


Example of how to calculate dipole tilt angle: 
```python
>>> print dipole_tilt(datetime(1927, 6, 10, 12, 00), epoch = 1927)
[ 26.79107107]
```

One can also give several times as input, and the tilt angle will be calculated for each of them. Since the calculations are entirely based on array calculations, this is quite fast.
```python
>>> print dipole_tilt([datetime(1927, 6, 10, 12, 00), datetime(1927, 6, 10, 10, 00)], epoch = 1927)
[ 26.79107107  20.89550663]
```

The epoch keyword is the year (with fraction) for which the dipole axis is calculated. This is done from IGRF coefficients, using linear interpolation between models. Setting the epoch is optional, and the default value is 2015. One can provide either one value for the epoch, or one value for each input. That may be quite slow, and is usually overkill unless the dates are very far apart:
```python
>>> print dipole_tilt([datetime(1927, 6, 10, 12, 00), datetime(2015, 6, 10, 10, 00)], epoch = (1927, 2015))
[ 26.79107107  20.59137527]
```



Calculation of dipole axis, dipole poles, dipole tilt angle, and subsolar point is described in

Laundal, K.M. & Richmond, A.D. Space Sci Rev (2017) 206: 27. https://doi.org/10.1007/s11214-016-0275-y


# dependencies
numpy, pandas

