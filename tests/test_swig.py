import numpy as np
import spires.core
import spires
import pytest

## Testing the .core functions

interpolator = spires.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat', )

spectrum_target = np.array([0.3424, 0.366, 0.3624, 0.38932347, 0.41624767, 0.39567757, 0.07043362, 0.06267947, 0.3792])
spectrum_background = np.array(
    [0.0182, 0.0265, 0.0283, 0.05606749, 0.09543234, 0.12036866, 0.12491679, 0.07888655, 0.1406])
spectrum_shade = np.zeros_like(spectrum_target)
solar_angle = 55.73733298

dust_concentration = 491
grain_size = 550
x0 = [0.5, 0.05, 10, 250]


def test_interpolate_all():
    ret = interpolator.interpolate_all(solar_angle=solar_angle, dust_concentration=dust_concentration,
                                       grain_size=grain_size)
    expected = np.array(
        [0.69418118, 0.72305336, 0.75899187, 0.76630307, 0.76921281, 0.75832135, 0.01766575, 0.02501143, 0.73101483])
    np.testing.assert_allclose(ret, expected, rtol=1e-5)


def test_interpolate_all_array():
    # this guy returns an array rather than a tuple
    ret = spires.core.interpolate_all_array(lut=interpolator.reflectances,
                                            bands=interpolator.bands,
                                            solar_angles=interpolator.solar_angles,
                                            dust_concentrations=interpolator.dust_concentrations,
                                            grain_sizes=interpolator.grain_sizes,
                                            solar_angle=solar_angle,
                                            dust_concentration=dust_concentration,
                                            grain_size=grain_size)
    expected = np.array(
        [0.69418118, 0.72305336, 0.75899187, 0.76630307, 0.76921281, 0.75832135, 0.01766575, 0.02501143, 0.73101483])
    np.testing.assert_allclose(ret, expected, rtol=1e-5)


def test_spectrum_difference():
    x = [0.5, 0.01, dust_concentration, grain_size]
    ret = spires.core.spectrum_difference(x=x,
                                          spectrum_background=spectrum_background,
                                          spectrum_target=spectrum_target,
                                          spectrum_shade=spectrum_shade,
                                          solar_angle=solar_angle,
                                          bands=interpolator.bands,
                                          solar_angles=interpolator.solar_angles,
                                          dust_concentrations=interpolator.dust_concentrations,
                                          grain_sizes=interpolator.grain_sizes,
                                          lut=interpolator.reflectances)

    assert pytest.approx(ret, rel=1e-2) == 0.08295740267234748


def test_invert():
    x = spires.core.invert(spectrum_background=spectrum_background,
                           spectrum_target=spectrum_target,
                           spectrum_shade=spectrum_shade,
                           solar_angle=solar_angle,
                           bands=interpolator.bands,
                           solar_angles=interpolator.solar_angles,
                           dust_concentrations=interpolator.dust_concentrations,
                           grain_sizes=interpolator.grain_sizes,
                           lut=interpolator.reflectances,
                           max_eval=100,
                           x0=x0,
                           algorithm=1)

    expected = np.array([4.089303e-01, 1.552017e-01, 1.387936e+02, 3.645840e+02])
    np.testing.assert_allclose(x, expected, rtol=1e-5)


def test_invert_array():
    n = 3
    results = np.empty((n, 4), dtype=np.double)
    spectra_backgrounds = np.tile(spectrum_background, (n, 1))
    spectra_targets = np.tile(spectrum_target, (n, 1))
    obs_solar_angles = np.repeat(solar_angle, n)

    spires.core.invert_array1d(spectra_backgrounds=spectra_backgrounds,
                               spectra_targets=spectra_targets,
                               spectrum_shade=spectrum_shade,
                               obs_solar_angles=obs_solar_angles,
                               bands=interpolator.bands,
                               solar_angles=interpolator.solar_angles,
                               dust_concentrations=interpolator.dust_concentrations,
                               grain_sizes=interpolator.grain_sizes,
                               lut=interpolator.reflectances,
                               results=results,
                               max_eval=100,
                               x0=x0,
                               algorithm=1)

    expected = np.array([[4.089303e-01, 1.552017e-01, 1.387936e+02, 3.645840e+02],
                         [4.089303e-01, 1.552017e-01, 1.387936e+02, 3.645840e+02],
                         [4.089303e-01, 1.552017e-01, 1.387936e+02, 3.645840e+02]])
    np.testing.assert_allclose(results, expected, rtol=1e-5)
