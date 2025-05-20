import spires
import numpy as np

interpolator = spires.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
interpolator.make_scipy_interpolator_legacy()
spectrum_target = np.array([0.3424, 0.366, 0.3624, 0.38932347, 0.41624767, 0.39567757, 0.07043362, 0.06267947, 0.3792])
spectrum_background = np.array([0.0182, 0.0265, 0.0283, 0.056067, 0.095432, 0.12036866, 0.12491679, 0.07888655, 0.1406])
solar_angle = 24.0

expected_scipy3 = np.array([4.36429085e-01, 5.63570915e-01, 9.91000000e+02, 4.12331162e+01])
expected_scipy4 = np.array([4.01536561e-01, 5.40285882e-02, 1.17707939e+02, 2.46971074e+02])


def test_scipy_mode3():
    res, model_refl = spires.speedy_invert_scipy(interpolator=interpolator,
                                                 spectrum_target=spectrum_target,
                                                 spectrum_background=spectrum_background,
                                                 solar_angle=solar_angle,
                                                 mode=3, method='SLSQP')

    np.testing.assert_allclose(res.x, expected_scipy3, rtol=1e-5)
    print('New syntax, mode 3:', res.x)


def test_scipy_mode4():
    res, model_refl = spires.speedy_invert_scipy(interpolator=interpolator,
                                                 spectrum_target=spectrum_target,
                                                 spectrum_background=spectrum_background,
                                                 solar_angle=solar_angle,
                                                 mode=4, method='SLSQP')

    np.testing.assert_allclose(res.x, expected_scipy4, rtol=1e-5)
    print('New syntax, mode 4:', res.x)


def test_legacy_mode3():
    res, model_refl = spires.legacy.speedy_invert(f=interpolator.interpolator_scipy,
                                                  spectrum_target=spectrum_target,
                                                  spectrum_background=spectrum_background,
                                                  shade=np.zeros_like(spectrum_target),
                                                  solar_angle=solar_angle,
                                                  mode=3,
                                                  method='SLSQP')
    np.testing.assert_allclose(res.x, expected_scipy3, rtol=1e-5, atol=1e-3)
    print('Legacy syntax, mode 3:', res.x)


def test_legacy_mode4():
    res, model_refl = spires.legacy.speedy_invert(f=interpolator.interpolator_scipy,
                                                  spectrum_target=spectrum_target,
                                                  spectrum_background=spectrum_background,
                                                  shade=np.zeros_like(spectrum_target),
                                                  solar_angle=solar_angle,
                                                  mode=4,
                                                  method='SLSQP')
    np.testing.assert_allclose(res.x, expected_scipy4, rtol=1e-3, atol=1e-3)
    print('Legacy syntax, mode 4:', res.x)


def test_nlop_cobyla():
    res = spires.speedy_invert(interpolator=interpolator,
                               spectrum_target=spectrum_target,
                               spectrum_background=spectrum_background,
                               solar_angle=solar_angle,
                               algorithm=1)

    expected = (4.228997e-01, 8.201566e-02, 1.520627e+02, 3.418598e+02)
    np.testing.assert_allclose(res, expected, rtol=1e-4)


def test_nlop_neldermead():
    res = spires.speedy_invert(interpolator=interpolator,
                               spectrum_target=spectrum_target,
                               spectrum_background=spectrum_background,
                               solar_angle=solar_angle,
                               algorithm=2)

    expected = (4.375782e-01, 7.412245e-03, 1.513136e+02, 5.179564e+02)
    np.testing.assert_allclose(res, expected, rtol=1e-4)
