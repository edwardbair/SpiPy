import spires.interpolator
import spires.core
import numpy as np
import scipy.interpolate

algorithm_dict = {'COBYLA': 1,
                  'NELDER_MEAD': 2,
                  'SLSQP': 3}


def speedy_invert(spectrum_target, spectrum_background, solar_angle, spectrum_shade=None,
                  bands=None, solar_angles=None, dust_concentrations=None, grain_sizes=None, reflectances=None,
                  interpolator=None, lut_dataarray=None, max_eval=100, x0=np.array([0.5, 0.05, 10, 250]), algorithm=2):
    """
    Inverts the snow reflectance spectrum. Optimization is performed using Nlopt's LN_COBYLA algorithm.

    Parameters
    ----------

    spectrum_background: numpy.ndarray
        the background (R_0) spectrum
    spectrum_target: numpy.ndarray
        the mixed spectrum to invert. Has to be same length as `spectrum_background`
    spectrum_shade: numpy.ndarray
        the idea; shaded spectrum. Has to be same length as `spectrum_target`
    solar_angle: float
        the solar angle of the spectrum target
    bands: numpy.ndarray
        band coordinates of reflectances
    solar_angles: numpy.ndarray
        solar angles of reflectances
    dust_concentrations: numpy.ndarray
        dust concentrations of reflectances
    grain_sizes: numpy.ndarray
        grain sizes of reflectances.
    reflectances: numpy.ndarray
        4D Snow reflectance lookup table with coordinates bands, solar_angles, dust_concentrations and grain_sizes.
    interpolator: spires.interpolator.LutInterpolator
        specify the interpolator instead of bands, solar_angles, dust_concentrations and grain_sizes.
    lut_dataarray: xarray.DataArray
        specify the lut_dataarray instead of bands, solar_angles, dust_concentrations and grain_sizes.
    algorithm: int
        Algorithm to use for inverting the snow reflectance spectrum.
        1: LN_COBYLA
        2: LN_NELDERMEAD
        3: LD_SLSQP
    x0: array-like
        Initial guess. x0[0]: fsca, x0[1]: fshade, x[2]: dust_conc, x[3]: grain_size
    max_eval: int
        maximum number of iterations

    Returns
    --------
    x: numpy.ndarray
        x[0]: f_sca
        x[1]: f_shade
        x[2]: dust_concentration
        x[3]: grain_size

    Examples
    ---------
    >>> import spires
    >>> spectrum_target = np.array([0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.07043362,0.06267947, 0.3792])
    >>> spectrum_background = np.array([0.0182,0.0265,0.0283,0.056067,0.095432,0.12036866,0.12491679,0.07888655,0.1406])
    >>> solar_angle = 55.73733298
    >>> interpolator = spires.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> spires.speedy_invert(spectrum_target=spectrum_target, spectrum_background=spectrum_background,
    ...                      solar_angle=solar_angle, interpolator=interpolator, algorithm=1)
    (0.4089303296055291, 0.155201675059351, 138.79357872804923, 364.58404302094834)
    """

    if spectrum_shade is None:
        spectrum_shade = np.zeros_like(spectrum_target)

    if interpolator is not None:
        bands = interpolator.bands
        solar_angles = interpolator.solar_angles
        dust_concentrations = interpolator.dust_concentrations
        grain_sizes = interpolator.grain_sizes
        reflectances = interpolator.reflectances

    return spires.core.invert(spectrum_background=spectrum_background, spectrum_target=spectrum_target,
                              spectrum_shade=spectrum_shade,
                              solar_angle=solar_angle, bands=bands, solar_angles=solar_angles,
                              dust_concentrations=dust_concentrations, grain_sizes=grain_sizes, lut=reflectances,
                              max_eval=max_eval, x0=x0, algorithm=algorithm)


def speedy_invert_array1d(spectra_targets, spectra_backgrounds, obs_solar_angles, spectrum_shade=None,
                          bands=None, solar_angles=None, dust_concentrations=None, grain_sizes=None, reflectances=None,
                          interpolator=None, lut_dataarray=None, max_eval=100,
                          x0=np.array([0.5, 0.05, 10, 250]), algorithm=2):
    """
    Inverts the snow reflectance spectrum for an array of spectrum backgrounds, spectrum targets, and solar angles

    Parameters
    ----------
    max_eval: int
        maximum number of iterations
    algorithm: int
        Algorithm to use for inverting the snow reflectance spectrum.
        1: LN_COBYLA
        2: LN_NELDERMEAD
        3: LD_SLSQP
    x0: array-like
        Initial guess. x0[0]: fsca, x0[1]: fshade, x0[2]: dust_conc, x0[3]: grain_size
    spectra_targets: numpy.ndarray
        a 2d array holding the mixed spectrum to invert
            - dim1: locations/observations (e.g. flattend space). Must be same length as `spectra_background`
            - dim2: bands
    spectra_backgrounds: numpy.ndarray
        a 2D array holding the background (R_0) spectra.
            - dim1: locations/observations (e.g. flattend space). Must be same length as `spectra_target`
            - dim2: bands
    spectrum_shade: numpy.ndarray
        an array holding the shaded spectrum
    obs_solar_angles: numpy.ndarray
        the solar angles of spectrum targets. Must be of same length as spectra_background.
    bands: numpy.ndarray
        band coordinates of reflectances
    solar_angles: numpy.ndarray
        solar angle coordinates of reflectances
    dust_concentrations: numpy.ndarray
        dust concentration coordinates of reflectances
    grain_sizes: numpy.ndarray
        grain size coordinates of reflectances.
    reflectances: numpy.ndarray
        4D Snow reflectance lookup table with coordinates bands, solar_angles, dust_concentrations and grain_sizes.
    interpolator: spires.LutInterpolator
        specify the interpolator instead of bands, solar_angles, dust_concentrations and grain_sizes.
    lut_dataarray: xarray.DataArray
        specify the lut_dataarray instead of bands, solar_angles, dust_concentrations and grain_sizes.

    Returns
    -------
    results: numpy.ndarray
        2D array with
            - dim1: locations/observations and
            - dim2: 4.
                - results[:, 0] = f_sca
                - results[:, 1] = f_shade
                - results[:, 2] = dust_concentration
                - results[:, 3] = grain_size

    Examples
    ----------
    >>> import spires
    >>> spectra_targets = np.array([[0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.0704336,0.06267947,0.3792],
    ...                            [0.2866,0.3046,0.324,0.34468558,0.35373732,0.35651454,0.1807259,0.16601688,0.3488]])
    >>> spectra_backgrounds = np.array([[0.0182,0.0265,0.0283,0.0560674,0.0954323,0.1203686,0.1249167,0.0788865,0.1406],
    ...                                [0.1002,0.1492,0.2088,0.2179780,0.2314920,0.2514020,0.3103066,0.2875081,0.2546]])
    >>> obs_solar_angles = np.array([55.73733298, 55.83733298])
    >>> interpolator = spires.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> spires.speedy_invert_array1d(spectra_targets=spectra_targets, spectra_backgrounds=spectra_backgrounds,
    ...                            obs_solar_angles=obs_solar_angles, interpolator=interpolator, algorithm=1)
    array([[4.06627881e-01, 1.45134251e-01, 1.37503982e+02, 3.61158500e+02],
           [2.63873228e-01, 1.83226478e-01, 1.94343159e+02, 3.80170927e+02]])

    """
    if spectrum_shade is None:
        spectrum_shade = np.zeros_like(spectra_targets[0])

    if interpolator is not None:
        bands = interpolator.bands
        solar_angles = interpolator.solar_angles
        dust_concentrations = interpolator.dust_concentrations
        grain_sizes = interpolator.grain_sizes
        reflectances = interpolator.reflectances

    n = spectra_targets.shape[0]
    results = np.empty((n, 4), dtype=np.double)

    spires.core.invert_array1d(spectra_targets=spectra_targets, spectra_backgrounds=spectra_backgrounds,
                               spectrum_shade=spectrum_shade,
                               obs_solar_angles=obs_solar_angles, bands=bands, solar_angles=solar_angles,
                               dust_concentrations=dust_concentrations,
                               grain_sizes=grain_sizes, lut=reflectances, results=results,
                               max_eval=max_eval, x0=x0, algorithm=algorithm)
    return results


def speedy_invert_array2d(spectra_targets, spectra_backgrounds, obs_solar_angles, spectrum_shade=None,
                          bands=None, solar_angles=None, dust_concentrations=None, grain_sizes=None, reflectances=None,
                          interpolator=None, lut_dataarray=None, max_eval=100,
                          x0=np.array([0.5, 0.05, 10, 250]), algorithm=2):

    spectra_targets = spectra_targets.transpose('y', 'x', 'band')
    spectra_backgrounds = spectra_backgrounds.transpose('y', 'x', 'band')
    obs_solar_angles = obs_solar_angles.transpose('y', 'x')

    if spectrum_shade is None:
        spectrum_shade = np.zeros(spectra_targets.band.size, dtype=np.double)

    if interpolator is not None:
        bands = interpolator.bands
        solar_angles = interpolator.solar_angles
        dust_concentrations = interpolator.dust_concentrations
        grain_sizes = interpolator.grain_sizes
        reflectances = interpolator.reflectances

    results = np.empty((spectra_targets.y.size, spectra_targets.x.size, 4), dtype=np.double)

    spires.core.invert_array2d(spectra_backgrounds=spectra_backgrounds,
                               spectra_targets=spectra_targets,
                               spectrum_shade=spectrum_shade,
                               obs_solar_angles=obs_solar_angles,
                               bands=bands, solar_angles=solar_angles, dust_concentrations=dust_concentrations,
                               grain_sizes=grain_sizes, lut=reflectances,
                               results=results,
                               max_eval=max_eval,
                               x0=x0,
                               algorithm=algorithm)
    return results


def snow_diff_4(x, spectrum_target, spectrum_background, solar_angle, interpolator, shade):
    """
    Calculates the difference between a spectrum modelled subject to x (x=[f_sca, f_shade, dust_concentration, grain_size)
    and the target spectrum.

    model_reflectances = pur_snow_reflectance(solar_angle, dust_concentration, grain size) * f_sca
                        + shade * f_shade
                        + spectrum_background * (1 - f_sca - f_shade)

    Note:
    If f_sca is within 2 pct, use 3 variable solution. Error will be higher, but 4 parameter solution likely overfits.

    Parameters
    ----------
    x: array_like
        x[0]: f_sca
        x[1]: f_shade
        x[2]: dust_concentration
        x[3]: grain_size
    spectrum_target: numpy.ndarray
        The mixed observed spectrum to invert
    spectrum_background: numpy.ndarray
        The (snow free) background spectrum R_0
    solar_angle: float
        The solar angle of the spectrum target
    interpolator: spires.interpolator.LutInterpolator
        a callable object that returns a modelled spectrum subject to solar_angle, dust_concentration, and grain_size
    shade: numpy.ndarray
        Ideal shade endmember

    Returns
    -------
    distance: double
        The distance between the modelled spectrum and the target spectrum

    Examples
    -----------
    >>> import spires
    >>> import numpy as np
    >>> interpolator = spires.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> f_sca = 0.482
    >>> f_shade = 0.065
    >>> dust_concentration = 1000 # ppm
    >>> grain_size = 220
    >>> solar_angle = 55.73733298
    >>> x = [f_sca, f_shade, dust_concentration, grain_size]
    >>> spectrum_target = np.array([0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.07043362,0.06267947, 0.3792])
    >>> spectrum_background = np.array([0.0182,0.0265,0.0283,0.056067,0.095432,0.12036866,0.12491679,0.07888655,0.1406])
    >>> shade = np.array([0,0,0,0,0,0,0,0,0])
    >>> diff = spires.snow_diff_4(x=x, spectrum_target=spectrum_target, spectrum_background=spectrum_background,
    ...                    solar_angle=solar_angle, interpolator=interpolator, shade=shade)
    >>> diff
    0.08870043573321955
    """

    model_reflectances = interpolator.interpolate_all(solar_angle=solar_angle,
                                                      dust_concentration=x[2],
                                                      grain_size=x[3])
    model_reflectances = model_reflectances * x[0] + shade * x[1] + spectrum_background * (1 - x[0] - x[1])
    distance = np.linalg.norm(spectrum_target - model_reflectances)
    return distance


def snow_diff_3(x, spectrum_target, solar_angle, interpolator, shade):
    """
    Calculates the difference between a spectrum modelled subject to x (x=[f_sca, f_shade, dust_concentration, grain_size)
    and the target spectrum.

    model_reflectances = pur_snow_reflectance(solar_angle, dust_concentration, grain size) * f_sca
                        + shade * (1-f_sca)

    Note:
    If f_sca is within 2 pct, use 3 variable solution. Error will be higher, but 4 parameter solution likely overfits.

    Parameters
    ----------
    x: array_like
        x[0]: f_sca
        x[1]: f_shade
        x[2]: dust_concentration
        x[3]: grain_size
    spectrum_target: numpy.ndarray
        The mixed observed spectrum to invert
    solar_angle: float
        The solar angle of the spectrum target
    interpolator: spires.interpolator.LutInterpolator
        a callable object that returns a modelled spectrum subject to solar_angle, dust_concentration, and grain_size
    shade: numpy.ndarray
        Ideal shade endmember

    Returns
    -------
    distance: double
        The distance between the modelled spectrum and the target spectrum

    Examples
    ---------
    >>> import spires
    >>> import numpy as np
    >>> interpolator = spires.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> f_sca = 0.482
    >>> dust_concentration = 1000 # ppm
    >>> grain_size = 220
    >>> solar_angle = 55.73733298
    >>> x = [f_sca, dust_concentration, grain_size]
    >>> spectrum_target = np.array([0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.07043362,0.06267947, 0.3792])
    >>> shade = np.array([0,0,0,0,0,0,0,0,0])
    >>> spires.snow_diff_3(x=x, spectrum_target=spectrum_target,
    ...                    solar_angle=solar_angle, interpolator=interpolator, shade=shade)
    0.06984199561833446
    """

    model_reflectances = interpolator.interpolate_all(solar_angle=solar_angle,
                                                      dust_concentration=x[1],
                                                      grain_size=x[2])

    model_reflectances = model_reflectances * x[0] + shade * (1 - x[0])
    distance = np.linalg.norm(spectrum_target - model_reflectances)
    return distance


def speedy_invert_scipy(interpolator: spires.interpolator.LutInterpolator, spectrum_target, spectrum_background,
                        solar_angle, shade=None,
                        scipy_options=None, mode=3, method='SLSQP'):
    """
    Invert snow spectra using scipy.optimize.minimize (rather than NLOPT).

    This method should functionally be equivalent to spires.legacy.speedy_invert() and only differs in style.

    Parameters
    ------------

    interpolator: spires.interpolator.LutInterpolator
        an object that
        a) has attributes `bands`, `solar_angle`, `dust_concentration`, `grain_size` that hold the coordinates
        b) has a method interpolate_all(solar_angle, dust_concentration, and grain_size) that returns a spectrum
    spectrum_target: numpy.ndarray
        target spectrum to be inverted. Must be same shape as spectrum_background
    spectrum_background: numpy.ndarray
        snow-free background spectrum. Must be same shape as spectrum_target
    solar_angle: double
        solar zenith angle of spectrum_target in same unites as interpolator coordinates
    shade: numpy.ndarray (optional)
        ideal shade endmember. Must be same shape as spectrum_target. If none, reflectance of 0 on all bands is used.
    mode: int
        3 or 4 variable inversion.
        Note: If f_sca is within 2 pct, use 3 variable solution. Error will be higher, but 4 parameter solution likely overfits.
    scipy_options: dict
        the scipy solver options.
        Default:  `scipy_options = {'disp': False, 'iprint': 100, 'maxiter': 1000, 'ftol': 1e-9}`
    method: str
        solver method.

    Returns
    ---------
    res: scipy.optimize.OptimizeResult
        see [scipy.optimize.OptimizeResult](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult)
        res.x contains the solution of the optimization problem, with
        - x[0]: f_sca
        - x[1]: f_shade
        - x[2]: dust_concentration
        - x[3]: grain_size
    model_ref: numpy.ndarray
        the optimized modelled reflectance

    Examples
    --------
    >>> import spires
    >>> import numpy as np
    >>> interpolator = spires.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> interpolator.make_scipy_interpolator_legacy()
    >>> spectrum_target = np.array([0.3424,0.366,0.3624,0.38932347,0.41624767,0.39567757,0.07043362,0.06267947, 0.3792])
    >>> spectrum_background = np.array([0.0182,0.0265,0.0283,0.056067,0.095432,0.12036866,0.12491679,0.07888655,0.1406])
    >>> solar_angle = 24.0
    >>> res, model_refl = spires.speedy_invert_scipy(interpolator=interpolator,
    ...                                              spectrum_target=spectrum_target,
    ...                                              spectrum_background=spectrum_background,
    ...                                              solar_angle=solar_angle,
    ...                                              mode=3, method='SLSQP')
    >>> res.x
    array([4.36429085e-01, 5.63570915e-01, 9.91000000e+02, 4.12331162e+01])
    """

    bounds_fsca = [0, 1]
    bounds_fshade = [0, 1]
    bounds_dust = [interpolator.dust_concentrations.min(), interpolator.dust_concentrations.max()]
    bounds_grain = [interpolator.grain_sizes.min(), interpolator.grain_sizes.max()]

    if scipy_options is None:
        scipy_options = {'disp': False, 'iprint': 100, 'maxiter': 1000, 'ftol': 1e-9}

    if shade is None:
        shade = np.zeros_like(spectrum_target)

    if mode == 4:
        bounds = np.array([bounds_fsca, bounds_fshade, bounds_dust, bounds_grain])

        # inequality: constraint is => 0
        constraints = {"type": "ineq", "fun": lambda x: 1 - x[0] + x[1]}

        # initial guesses for f_sca, f_shade, dust, & grain size
        x0 = np.array([0.5, 0.05, 10, 250])

        res = scipy.optimize.minimize(snow_diff_4,
                                      x0,
                                      options=scipy_options,
                                      bounds=bounds,
                                      method=method,
                                      constraints=constraints,
                                      args=(spectrum_target, spectrum_background, solar_angle, interpolator, shade))
    elif mode == 3:
        bounds = np.array([bounds_fsca, bounds_dust, bounds_grain])

        # initial guesses for f_sca, dust, & grain size
        x0 = np.array([0.5, 10, 250])

        res = scipy.optimize.minimize(snow_diff_3,
                                      x0,
                                      options=scipy_options,
                                      bounds=bounds,
                                      method=method,
                                      args=(spectrum_target, solar_angle, interpolator, shade)
                                      )
        # insert f_shade (x[1] as 1-f_sca
        res.x = np.insert(res.x, 1, 1 - res.x[0])
    else:
        raise ValueError('mode must be either 4 or 3')

    # Lookup modelled reflectances
    model_refl = interpolator.interpolate_all(solar_angle=solar_angle, dust_concentration=res.x[2], grain_size=res.x[3])

    return res, model_refl


def index_to_value(index, coords):
    """
    Converts an index value to a coordinate value

    Parameters
    ----------
    index:
    coords: numpy.array

    Returns
    -------

    """
    idx = index * coords.size
    l_idx = int(idx)
    r_idx = l_idx + 1
    diff = coords[r_idx] - coords[l_idx]
    dist = idx - l_idx
    return coords[l_idx] + dist * diff


def speedy_invert_scipy_normalized(interpolator: spires.interpolator.LutInterpolator,
                                   spectrum_target, spectrum_background, solar_angle, spectrum_shade=None,
                                   method='COBYLA'):
    """
    Perform speedy invert with COBYLA solver. The scipy COBYLA solver does not allow us to specify
    initial steps (rhobeg) for each parameter separately. We therefore need to scale the problem,
    i.e. for the solver, all parameters are on a scale of 0-1.

    Parameters
    ----------
    method: str
        the scipy solver method
    spectrum_shade: np.ndarray
        the shade spectrum
    interpolator: spires.interpolator.LutInterpolator
        A lut interpolator
    spectrum_target: np.ndarray
        the spectrum target
    spectrum_background: np.ndarray
        the background spectrum
    solar_angle: float
        the solar angle of the spectrum target

    Returns
    -------

    """
    if spectrum_shade is None:
        spectrum_shade = np.zeros_like(spectrum_target)

    scipy_options = {'disp': False, 'rhobeg': 0.05, 'maxiter': 100, 'tol': 1e-4}

    bounds_fsca = [0, 1]
    bounds_fshade = [0, 1]
    bounds_dust = [0, 1]
    bounds_grain = [0, 1]
    bounds = np.array([bounds_fsca, bounds_fshade, bounds_dust, bounds_grain], dtype=float)
    x0 = np.array([0.5, 0.05, 0.01, 0.1])

    res = scipy.optimize.minimize(spires.core.spectrum_difference_scaled,
                                  x0,
                                  method=method,
                                  options=scipy_options,
                                  bounds=bounds,
                                  args=(spectrum_background,
                                        spectrum_target,
                                        spectrum_shade,
                                        solar_angle,
                                        interpolator.bands,
                                        interpolator.solar_angles,
                                        interpolator.dust_concentrations,
                                        interpolator.grain_sizes,
                                        interpolator.reflectances)
                                  )

    res.x[2] = index_to_value(res.x[2], interpolator.dust_concentrations)
    res.x[3] = index_to_value(res.x[3], interpolator.grain_sizes)

    model_refl = interpolator.interpolate_all(solar_angle=solar_angle, dust_concentration=res.x[2], grain_size=res.x[3])
    return res, model_refl
