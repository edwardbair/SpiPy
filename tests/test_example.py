import numpy
import numpy as np
import spires

# R is 7 band spectra for the 2 pixels
R = np.array([[0.8203, 0.6796, 0.8076, 0.8361, 0.1879, 0.0321, 0.0144],
              [0.4773, 0.4482, 0.4474, 0.4823, 0.1815, 0.1019, 0.0748]])

# R0 is the 7 band background spectra
R0 = np.array([[0.2219, 0.2681, 0.1016, 0.1787, 0.3097, 0.2997, 0.2970],
               [0.1377, 0.2185, 0.0807, 0.1127, 0.2588, 0.2696, 0.1822]])

# modis central wavelengths, for plotting
wavelengths = np.array([0.6450, 0.8585, 0.4690, 0.5550, 1.2400, 1.6400, 2.1300])

# need to sort those as the MODIS bands don't go in increasing order
idx = wavelengths.argsort(axis=0)

# solar zenith angle for both days
solarZ = np.array([24.0, 24.71])

# ideal shade endmember
shade = np.zeros(len(R[0]))


def F(pts):
    # Mockup lookup table
    band = int(pts[0])
    l = [0.94016741, 0.919606, 0.92147043, 0.93222527, 0.63493878, 0.18821908, 0.11910248]
    val = l[band-1]
    return val


def test_lookup():
    i_pixel = 0
    mode = 4
    res, model_refl = spires.speedy_invert(F, R[i_pixel], R0[i_pixel], solarZ[i_pixel], shade, mode)
    rmse = res.fun
    fsca = res.x[0]
    fshade = res.x[1]
    rg = res.x[2]
    dust = res.x[3]

    expected = numpy.array([0.94842048,   0.70156266,  30.        , 250.        ])
    numpy.testing.assert_array_almost_equal(res.x, expected)




test_lookup()