import numpy as np
import scipy
import h5py


scipy_options = {'disp': False, 'iprint': 100, 'maxiter': 1000, 'ftol': 1e-9}


def load_lut(lut_file):
    with h5py.File(lut_file, 'r') as lut:
        d = {}
        for k in lut.keys():
            d[k] = np.squeeze(np.array(lut.get(k)))
    f = scipy.interpolate.RegularGridInterpolator(points=[d['X4'], d['X3'], d['X2'], d['X1']], values=d['X'])
    return f


def speedy_invert(f, spectrum_target, spectrum_background,
                 solar_angle, shade, mode=3, scipy_options=scipy_options, method='SLSQP'):
    """
    Inverts a spectrum and determines f_sca, f_shade, dust_concentration and grain_size.

    Parameters
    ----------
    f: scipy.interpolate.RegularGridInterpolator
        A gridded RT/Mie interpolator with bands, solar angles, dust, and grain sizes.
    spectrum_target: np.ndarray
        The target spectrum (i.e. the observation) to be inverted
    spectrum_background: np.ndarray
        The snow-free background ("R_0") spectrum
    solar_angle: float
        The solar zenith angle of the target spectrum.
        Must be in same unit as the solar zenith angle of the LUT (e.g. degrees)
    shade: np.ndarray
        ideal shade endmember
    mode: int
        3 or 4 variable inversion. 4 = full model (1-fsca, 2-fshade,
        fother(1-fsca-fshade), 3-grain radius, 4-dust).
        3 = simplified model (1-fsca, fshade (1-fsca), 2-grain radius, 3-dust).
    scipy_options: dict
        Dictionary of options to pass to scipy.optimize.minimize()
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    method: str
        Method to pass to scipy.optimize.minimize(). E.g. `SLSQP`.

    Returns
    -------
    res: scipy.optimize.OptimizeResult
        see scipy.optimize.OptimizeResult documentation.
        res.x contains the solution of the optimization problem with
        x[0]=f_sca, x[1]=f_shade, x[2]=dust_concentration, x[3]=grain_size.
    model_ref: numpy.array
        the optimized modelled reflectance

    Examples
    --------
    >>> import spires
    >>> import numpy as np
    >>> f = spires.legacy.load_lut('tests/data/LUT_MODIS.mat')
    >>> R = np.array([0.8203,0.6796,0.8076,0.8361,0.1879,0.0321,0.0144])
    >>> R0 = np.array([0.2219,0.2681,0.1016,0.1787,0.3097,0.2997,0.2970])
    >>> solar_z = 24.0
    >>> shade = np.zeros(len(R))
    >>> res, model_refl = spires.legacy.speedy_invert(f, R, R0, solar_z, shade, mode=4)
    >>> np.allclose(res.x, np.array([0.8848, 0.0485, 430.2819, 18.2311]), rtol=1e-2)
    True
    """

    # bounds: fsca, fshade, grain size, dust (note: order for grain radius and dust is switched from F)
    bounds_fsca = [0, 1]
    bounds_fshade = [0, 1]
    bounds_dust = [f.grid[2].min(), f.grid[2].max()]
    bounds_grain = [f.grid[3].min(), f.grid[3].max()]
    bounds = np.array([bounds_fsca, bounds_fshade, bounds_grain, bounds_dust])

    # initial guesses for fsca, fshade,dust, & grain size
    x0 = [0.5, 0.05, 10, 250]

    # model reflectance pre-allocation
    model_reflectances = np.zeros(len(spectrum_target))

    # objective function
    def snow_diff(x):
        """
        
        Parameters
        ----------
        x: np.array
            - x[0] f_sca
            - x[1] f_shade
            - x[2] dust_concentration
            - x[3] grain_radius

        Returns
        -------
        diff_r: float:
            Euclidian norm of the difference between the modelled spectrum and the target spectrum.

        """
        # nonlocal vars from parent function
        nonlocal model_reflectances, mode

        # mode - 4 variable solution (1-fsca,2-fshade,fother(1-fsca-fshade),3-grain radius,4-dust)
        # mode - 3 variable solution (1-fsca,fshade (1-fsca),2-grain radius,3-dust)

        # fill in model_reflectances for each band for snow properties ie if pixel were pure snow (no fshade, no fother)
        if mode == 4:
            for i in range(0, len(spectrum_target)):
                # x[2] and x[3] are grain radius and dust
                pts = [i + 1, solar_angle, x[3], x[2]]
                interpolated_reflectance = f(pts)
                if isinstance(interpolated_reflectance, np.ndarray):
                    # The interpolator seems to occasionally return a float, instead of an array. This makes numpy gag.
                    interpolated_reflectance = interpolated_reflectance[0]
                model_reflectances[i] = interpolated_reflectance
            # now adjust model reflectance for a mixed pixel, with x[0] and x[1]
            # as fsca, fshade, and 1-x[0]-x[1] as fother
            model_reflectances = model_reflectances * x[0] + shade * x[1] + spectrum_background * (1 - x[0] - x[1])

        if mode == 3:
            for i in range(0, len(spectrum_target)):
                # x[1] and x[2] are grain radius and dust
                pts = np.array([i + 1, solar_angle, x[2], x[1]])
                model_reflectances[i] = f(pts)[0]

            model_reflectances = model_reflectances * x[0] + shade * (1 - x[0])

        # Euclidean norm of measured - modeled reflectance
        diff_r = np.linalg.norm(spectrum_target - model_reflectances)
        return diff_r

    # construct the bounds in the form of constraints
    # inequality: constraint is => 0
    constraints = [{"type": "ineq", "fun": lambda x: 1 - x[0] + x[1]}]
    #  1-(x[0]+x[1]) >= 0 <-> 1 >= x[0]+x[1]
    #  mixed pixel contraint: 1 >= fsca+fshade <-> 1 = fsca+fshade+(1-fsca)

    # rranges=(slice(0,1,0.1),slice(0,1,0.1),slice(30,1200,30),slice(0,1000,50))
    # resBrute = brute(SnowDiff, rranges, full_output=False, disp=False, finish=None)
    # x0=resBrute
    if mode == 4:
        # run minimization w/ 4 variables to solve
        res = scipy.optimize.minimize(snow_diff, x0, method=method, options=scipy_options, bounds=bounds, constraints=constraints)
    elif mode == 3:
        # run minimization w/ 3 variables to solve, no constraint needed
        # delete fshade(index 1) guess and bounds
        x0 = np.delete(x0, 1)
        bounds = np.delete(bounds, 1, axis=0)
        res = scipy.optimize.minimize(snow_diff, x0, method=method, options=scipy_options, bounds=bounds)
        # insert a zero in for consistency for x[2] (fother)
        res.x = np.insert(res.x, 1, 1 - res.x[0])

    # store modeled refl
    model_refl = np.copy(model_reflectances)

    # if fsca is within 2 pct, use 3 variable solution
    # error will be higher, but 4 parameter solution likely overfits
    # if (abs(res1.x[0]-res2.x[0]) < 0.02):
    #     choice=3
    #     res=res2
    #     model_reflectances=model_refl2
    # else:
    #     choice=4
    #     res=res1
    #     model_reflectances=model_refl1

    return res, model_refl