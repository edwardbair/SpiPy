import spires.interpolator
import spires.core
import numpy as np
import scipy


def speedy_invert(spectrum_target, spectrum_background, solar_angle, spectrum_shade=None,
                  bands=None, solar_angles=None, dust_concentrations=None, grain_sizes=None, reflectances=None,
                  interpolator=None, lut_dataarray=None, max_eval=100, x0=np.array([0.5, 0.05, 10, 250]), algorithm=2):
    """
    Inverts the snow reflectance spectrum using nonlinear optimization.

    Parameters
    ----------
    spectrum_target : numpy.ndarray
        The mixed spectrum to invert. Must be same length as `spectrum_background`.
        Must have same band order as `spectrum_background` and `bands`.
    spectrum_background : numpy.ndarray
        The background (snow-free, R_0) spectrum.
    solar_angle : float
        The solar zenith angle of the spectrum target (degrees).
    spectrum_shade : numpy.ndarray, optional
        The ideal shaded spectrum. Must be same length as `spectrum_target`.
        If None, uses zeros (default: None).
    bands : numpy.ndarray, optional
        Band wavelength coordinates of reflectances. Required if interpolator not provided.
    solar_angles : numpy.ndarray, optional
        Solar angle coordinates of reflectances. Required if interpolator not provided.
    dust_concentrations : numpy.ndarray, optional
        Dust concentration coordinates of reflectances (ppm). Required if interpolator not provided.
    grain_sizes : numpy.ndarray, optional
        Grain size coordinates of reflectances (μm). Required if interpolator not provided.
    reflectances : numpy.ndarray, optional
        4D snow reflectance lookup table with dimensions (bands, solar_angles,
        dust_concentrations, grain_sizes). Required if interpolator not provided.
    interpolator : spires.interpolator.LutInterpolator, optional
        Pre-configured interpolator. If provided, overrides individual LUT parameters.
    lut_dataarray : xarray.DataArray, optional
        Not currently used. Reserved for future xarray support.
    max_eval : int, optional
        Maximum number of optimization iterations. Default is 100.
    x0 : array-like, optional
        Initial guess for [fsca, fshade, dust_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).
        1 = LN_COBYLA (Constrained Optimization BY Linear Approximations),
        2 = LN_NELDERMEAD (Nelder-Mead simplex),
        3 = LD_SLSQP (Sequential Least Squares Programming, not working in C++).

    Returns
    -------
    tuple
        Optimization results as (fsca, fshade, dust_concentration, grain_size) where:

        - fsca : float - Fractional snow-covered area (0-1)
        - fshade : float - Fractional shaded area (0-1)
        - dust_concentration : float - Dust concentration in snow (ppm)
        - grain_size : float - Effective snow grain radius (μm)

    Examples
    --------
    >>> import spires
    >>> import numpy as np
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
    Batch inversion of snow reflectance spectra for 1D arrays of observations.

    Efficiently processes multiple pixels/observations sequentially using optimized
    C++ implementations for improved performance.

    Parameters
    ----------
    spectra_targets : numpy.ndarray
        2D array of mixed spectra to invert with shape (n_observations, n_bands).
        Must have same length as `spectra_backgrounds` along first dimension.
    spectra_backgrounds : numpy.ndarray
        2D array of background (snow-free, R_0) spectra with shape (n_observations, n_bands).
        Must have same length as `spectra_targets` along first dimension.
    obs_solar_angles : numpy.ndarray
        1D array of solar zenith angles (degrees) for each observation.
        Must have same length as first dimension of `spectra_targets`.
    spectrum_shade : numpy.ndarray, optional
        1D array representing the ideal shaded spectrum for all observations.
        Must have same length as number of bands. If None, uses zeros (default: None).
    bands : numpy.ndarray, optional
        Band wavelength coordinates of reflectances. Required if interpolator not provided.
    solar_angles : numpy.ndarray, optional
        Solar angle coordinates of reflectances. Required if interpolator not provided.
    dust_concentrations : numpy.ndarray, optional
        Dust concentration coordinates of reflectances (ppm). Required if interpolator not provided.
    grain_sizes : numpy.ndarray, optional
        Grain size coordinates of reflectances (μm). Required if interpolator not provided.
    reflectances : numpy.ndarray, optional
        4D snow reflectance lookup table with dimensions (bands, solar_angles,
        dust_concentrations, grain_sizes). Required if interpolator not provided.
    interpolator : spires.interpolator.LutInterpolator, optional
        Pre-configured interpolator. If provided, overrides individual LUT parameters.
    lut_dataarray : xarray.DataArray, optional
        Not currently used. Reserved for future xarray support.
    max_eval : int, optional
        Maximum number of optimization iterations per observation (default: 100).
    x0 : array-like, optional
        Initial guess for [fsca, fshade, dust_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).
        1 = LN_COBYLA (Constrained Optimization BY Linear Approximations),
        2 = LN_NELDERMEAD (Nelder-Mead simplex),
        3 = LD_SLSQP (Sequential Least Squares Programming, not working in C++).

    Returns
    -------
    numpy.ndarray
        2D array of shape (n_observations, 4) containing inversion results:
        - results[:, 0] : Fractional snow-covered area (0-1)
        - results[:, 1] : Fractional shaded area (0-1)
        - results[:, 2] : Dust concentration in snow (ppm)
        - results[:, 3] : Effective snow grain radius (μm)

    Examples
    --------
    >>> import spires
    >>> import numpy as np
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


def speedy_invert_array2d(spectra_targets, spectra_backgrounds, obs_solar_angles, max_eval=100, x0=np.array([0.5, 0.05, 10, 250]), algorithm=2,
                          bands=None, solar_angles=None, dust_concentrations=None, grain_sizes=None, reflectances=None, interpolator=None):
    """
    Batch inversion of snow reflectance spectra for 2D spatial arrays.

    Processes entire images or 2D grids of observations efficiently using optimized
    C++ implementations. Ideal for processing satellite imagery or gridded data.

    Parameters
    ----------
    spectra_targets : numpy.ndarray
        3D array of mixed spectra to invert with shape (ny, nx, n_bands):
        - dim 0: y spatial dimension
        - dim 1: x spatial dimension
        - dim 2: spectral bands (must match order of `spectra_backgrounds`)
    spectra_backgrounds : numpy.ndarray
        3D array of background (snow-free, R_0) spectra with shape (ny, nx, n_bands):
        - dim 0: y spatial dimension (must match `spectra_targets`)
        - dim 1: x spatial dimension (must match `spectra_targets`)
        - dim 2: spectral bands (must match order of `spectra_targets`)
    obs_solar_angles : numpy.ndarray
        2D array of solar zenith angles (degrees) with shape (ny, nx).
        One angle per spatial location.
    max_eval : int, optional
        Maximum number of optimization iterations per pixel (default: 100).
    x0 : array-like, optional
        Initial guess for [fsca, fshade, dust_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).
        1 = LN_COBYLA (Constrained Optimization BY Linear Approximations),
        2 = LN_NELDERMEAD (Nelder-Mead simplex),
        3 = LD_SLSQP (Sequential Least Squares Programming, not working in C++).
    bands : numpy.ndarray, optional
        Band wavelength coordinates of reflectances. Required if interpolator not provided.
    solar_angles : numpy.ndarray, optional
        Solar angle coordinates of reflectances. Required if interpolator not provided.
    dust_concentrations : numpy.ndarray, optional
        Dust concentration coordinates of reflectances (ppm). Required if interpolator not provided.
    grain_sizes : numpy.ndarray, optional
        Grain size coordinates of reflectances (μm). Required if interpolator not provided.
    reflectances : numpy.ndarray, optional
        4D snow reflectance lookup table with dimensions (bands, solar_angles,
        dust_concentrations, grain_sizes). Required if interpolator not provided.
    interpolator : spires.interpolator.LutInterpolator, optional
        Pre-configured interpolator. If provided, overrides individual LUT parameters.

    Returns
    -------
    numpy.ndarray
        3D array of shape (ny, nx, 4) containing inversion results:
        - results[:, :, 0] : Fractional snow-covered area (0-1)
        - results[:, :, 1] : Fractional shaded area (0-1)
        - results[:, :, 2] : Dust concentration in snow (ppm)
        - results[:, :, 3] : Effective snow grain radius (μm)

    Notes
    -----
    The shade spectrum is automatically set to zeros for all pixels. Future versions
    may support spatially-varying shade spectra.
    """
    
    spectrum_shade = np.zeros(spectra_targets.shape[-1], dtype=np.double)
    
    if spectrum_shade is None:
        spectrum_shade = np.zeros_like(spectra_targets[0])

    if interpolator is not None:
        bands = interpolator.bands
        solar_angles = interpolator.solar_angles
        dust_concentrations = interpolator.dust_concentrations
        grain_sizes = interpolator.grain_sizes
        reflectances = interpolator.reflectances

    results = np.empty((spectra_targets.shape[0], spectra_targets.shape[1], 4), dtype=np.double)


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



def speedy_invert_xarray(spectra_targets, spectra_backgrounds, obs_solar_angles, lut_dataarray,
                          spectrum_shade=None, max_eval=100,
                          x0=np.array([0.5, 0.05, 10, 250]), algorithm=2):
    """
    Batch inversion of snow reflectance spectra using xarray DataArrays.

    Provides a high-level interface for processing geospatial data with coordinate
    information preserved. Automatically handles dimension ordering and metadata.

    Parameters
    ----------
    spectra_targets : xarray.DataArray
        Mixed spectra to invert with dimensions (y, x, band).
        Will be automatically transposed if needed.
    spectra_backgrounds : xarray.DataArray
        Background (snow-free, R_0) spectra with dimensions (y, x, band).
        Must have same spatial dimensions as `spectra_targets`.
    obs_solar_angles : xarray.DataArray
        Solar zenith angles (degrees) with dimensions (y, x).
        One angle per spatial location.
    lut_dataarray : xarray.DataArray
        Lookup table with dimensions (band, solar_angle, dust_concentration, grain_size).
        Coordinates are extracted and used for interpolation.
    spectrum_shade : numpy.ndarray, optional
        1D array representing the ideal shaded spectrum.
        Must have same length as number of bands. If None, uses zeros (default: None).
    max_eval : int, optional
        Maximum number of optimization iterations per pixel (default: 100).
    x0 : array-like, optional
        Initial guess for [fsca, fshade, dust_conc, grain_size].
        Default is [0.5, 0.05, 10, 250].
    algorithm : int, optional
        Optimization algorithm to use (default: 2).
        1 = LN_COBYLA (Constrained Optimization BY Linear Approximations),
        2 = LN_NELDERMEAD (Nelder-Mead simplex),
        3 = LD_SLSQP (Sequential Least Squares Programming, not working in C++).

    Returns
    -------
    numpy.ndarray
        3D array of shape (ny, nx, 4) containing inversion results:
        - results[:, :, 0] : Fractional snow-covered area (0-1)
        - results[:, :, 1] : Fractional shaded area (0-1)
        - results[:, :, 2] : Dust concentration in snow (ppm)
        - results[:, :, 3] : Effective snow grain radius (μm)

    Notes
    -----
    Currently returns a numpy array. Future versions will return an xarray.DataArray
    with appropriate coordinates and metadata (see TODO comment in code).
    """
    
    spectra_targets = spectra_targets.transpose('y', 'x', 'band')
    spectra_backgrounds = spectra_backgrounds.transpose('y', 'x', 'band')
    obs_solar_angles = obs_solar_angles.transpose('y', 'x')
    
    if spectrum_shade is None:
        spectrum_shade = np.zeros(spectra_targets.band.size, dtype=np.double)
   
    bands = lut_dataarray.band
    solar_angles = lut_dataarray.solar_angle
    dust_concentrations = lut_dataarray.dust_concentration
    grain_sizes = lut_dataarray.grain_size
    reflectances = lut_dataarray.transpose('band', 'solar_angle', 'dust_concentration', 'grain_size').values

    results = np.empty((spectra_targets.y.size, spectra_targets.x.size, 4), dtype=np.double)

    spires.core.invert_array2d(spectra_backgrounds=spectra_backgrounds,
                               spectra_targets=spectra_targets,
                               spectrum_shade=spectrum_shade,
                               obs_solar_angles=obs_solar_angles,
                               bands=bands, 
                               solar_angles=solar_angles, 
                               dust_concentrations=dust_concentrations,
                               grain_sizes=grain_sizes, 
                               reflectances=reflectances,
                               results=results,
                               max_eval=max_eval,
                               x0=x0,
                               algorithm=algorithm)
    
    # TODO: bootstrap the returned xarray!
    return results


def speedy_invert_dask(spectra_targets, spectra_backgrounds, obs_solar_angles,
                       interpolator, spectrum_shade=None, max_eval=100,
                       x0=np.array([0.5, 0.05, 10, 250]), algorithm=2,
                       client=None, scatter_lut=True):
    """
    Parallel inversion of snow reflectance spectra using Dask and xarray.

    This method enables distributed processing of large satellite imagery datasets
    by leveraging Dask's parallel computation capabilities through xarray.apply_ufunc.
    It's particularly useful for processing time series of satellite imagery where
    the data is too large to fit in memory.

    Parameters
    ----------
    spectra_targets : xarray.DataArray
        Mixed spectra to invert. Must have a 'band' dimension and can have
        any combination of spatial (x, y) and temporal (time) dimensions.
        Shape: (time, y, x, band) or (y, x, band).
    spectra_backgrounds : xarray.DataArray
        Background (snow-free, R_0) spectra with same dimensions as targets
        except potentially missing the time dimension if using static backgrounds.
        Shape: (y, x, band).
    obs_solar_angles : xarray.DataArray
        Solar zenith angles (degrees) for each observation.
        Shape: (time, y, x) or (y, x).
    interpolator : spires.interpolator.LutInterpolator
        Lookup table interpolator object containing reflectance data
        and coordinate arrays (bands, solar_angles, dust_concentrations, grain_sizes).
    spectrum_shade : numpy.ndarray, optional
        1D array representing the ideal shaded spectrum.
        Must have same length as number of bands. If None, uses zeros (default: None).
    max_eval : int, optional
        Maximum number of optimization iterations per pixel (default: 100).
    x0 : array-like, optional
        Initial guess: [fsca, fshade, dust_conc (ppm), grain_size (μm)].
        Default: [0.5, 0.05, 10, 250].
    algorithm : int, optional
        NLopt algorithm code:
        - 0: LN_NELDERMEAD
        - 1: LN_SBPLX
        - 2: LN_COBYLA (default, recommended)
        - 3: LN_NEWUOA
        - 4: LN_NEWUOA_BOUND
        - 5: LN_BOBYQA
        - 6: LN_PRAXIS
        - 7: LD_MMA
        - 8: LD_SLSQP (not working in C++)
        - 9: LD_LBFGS
    client : dask.distributed.Client, optional
        Dask client for distributed computation. If None, uses default scheduler.
    scatter_lut : bool, optional
        Whether to scatter (broadcast) the LUT to all workers for faster access.
        Recommended for large LUTs that will be reused many times (default: True).

    Returns
    -------
    xarray.Dataset
        Dataset containing inversion results with variables:
        - fsca : Fractional snow-covered area (0-1)
        - fshade : Fractional shaded area (0-1)
        - dust_concentration : Dust concentration in snow (ppm)
        - grain_size : Effective snow grain radius (μm)

        Preserves all input coordinates and dimensions.

    Examples
    --------
    >>> import spires
    >>> import xarray as xr
    >>> from dask.distributed import Client
    >>>
    >>> # Load data
    >>> ds = xr.open_zarr('sentinel2_data.zarr')
    >>> ds_r0 = xr.open_zarr('background_reflectance.zarr')
    >>>
    >>> # Setup Dask client for parallel processing
    >>> client = Client(n_workers=4, threads_per_worker=2)
    >>>
    >>> # Load LUT
    >>> lut = spires.LutInterpolator('sentinel2_lut.mat')
    >>>
    >>> # Run parallel inversion
    >>> results = spires.speedy_invert_dask(
    ...     spectra_targets=ds['reflectance'],
    ...     spectra_backgrounds=ds_r0['reflectance'],
    ...     obs_solar_angles=ds['sun_zenith'],
    ...     interpolator=lut,
    ...     client=client
    ... )
    >>>
    >>> # Compute results (triggers actual computation)
    >>> results = results.compute()

    Notes
    -----
    - Input data should be chunked appropriately for your system's memory.
      Typical chunk sizes: {'time': 1, 'y': 256, 'x': 256, 'band': -1}
    - The 'band' dimension should not be chunked (use band=-1) as the
      inversion operates on full spectra.
    - For time series processing, chunking by time=1 enables parallel
      processing across time steps.
    - Performance scales with number of workers and chunk size.
    - Consider using persist() on frequently accessed data.

    See Also
    --------
    speedy_invert_xarray : Non-parallel xarray version
    speedy_invert_array2d : Core 2D array inversion function
    """
    import xarray

    # Handle optional imports
    try:
        import dask
        import dask.array
        from dask.distributed import Client
    except ImportError:
        raise ImportError(
            "Dask is required for parallel processing. "
            "Install with: conda install -c conda-forge dask distributed"
        )

    # Ensure we have a client
    if client is None:
        try:
            # Try to get existing client
            client = Client.current()
        except ValueError:
            # No client exists, will use default sched
            # uler
            pass

    # Handle spectrum_shade
    if spectrum_shade is None:
        spectrum_shade = np.zeros(len(interpolator.bands))

    # Scatter LUT to workers if requested and client exists
    if scatter_lut and client is not None:
        # Convert LUT reflectances to dask array and scatter
        reflectances_da = dask.array.from_array(interpolator.reflectances)
        scattered = client.scatter(dict(reflectances_da.dask), broadcast=True)
        reflectances_scattered = dask.array.Array(
            scattered,
            name=reflectances_da.name,
            chunks=reflectances_da.chunks,
            dtype=reflectances_da.dtype,
            meta=reflectances_da._meta,
            shape=reflectances_da.shape
        )
    else:
        reflectances_scattered = interpolator.reflectances

    # Define the wrapper function for apply_ufunc
    def _invert_wrapper(spectra_targets, spectra_backgrounds, obs_solar_angles,
                       bands, solar_angles, dust, grain, reflectances):
        """Internal wrapper for speedy_invert_array2d."""
        # Handle potential time dimension
        if spectra_targets.ndim == 4:  # Has time dimension
            # Process each time step
            n_time = spectra_targets.shape[0]
            results = np.empty((n_time,) + spectra_targets.shape[1:3] + (4,))
            for t in range(n_time):
                results[t] = speedy_invert_array2d(
                    spectra_targets=spectra_targets[t],
                    spectra_backgrounds=spectra_backgrounds,
                    obs_solar_angles=obs_solar_angles[t],
                    spectrum_shade=spectrum_shade,
                    bands=bands,
                    solar_angles=solar_angles,
                    dust_concentrations=dust,
                    grain_sizes=grain,
                    reflectances=reflectances,
                    max_eval=max_eval,
                    x0=x0,
                    algorithm=algorithm
                )
        else:  # No time dimension
            results = speedy_invert_array2d(
                spectra_targets=spectra_targets,
                spectra_backgrounds=spectra_backgrounds,
                obs_solar_angles=obs_solar_angles,
                spectrum_shade=spectrum_shade,
                bands=bands,
                solar_angles=solar_angles,
                dust_concentrations=dust,
                grain_sizes=grain,
                reflectances=reflectances,
                max_eval=max_eval,
                x0=x0,
                algorithm=algorithm
            )
        return results

    # Determine input core dimensions based on data structure
    has_time = 'time' in spectra_targets.dims

    if has_time:
        # Time-varying data
        target_core_dims = ['band']
        background_core_dims = ['band']
        angle_core_dims = []
        output_core_dims = [['property']]
    else:
        # Static data
        target_core_dims = ['band']
        background_core_dims = ['band']
        angle_core_dims = []
        output_core_dims = [['property']]

    # Apply parallel inversion using xarray.apply_ufunc
    results = xarray.apply_ufunc(
        _invert_wrapper,
        spectra_targets,
        spectra_backgrounds,
        obs_solar_angles,
        interpolator.bands,
        interpolator.solar_angles,
        interpolator.dust_concentrations,
        interpolator.grain_sizes,
        reflectances_scattered,
        dask='parallelized',
        input_core_dims=[
            target_core_dims,      # spectra_targets
            background_core_dims,  # spectra_backgrounds
            angle_core_dims,       # obs_solar_angles
            ['bands'],             # bands
            ['sz'],                # solar_angles
            ['dust'],              # dust_concentrations
            ['grain'],             # grain_sizes
            ['bands', 'sz', 'dust', 'grain']  # reflectances
        ],
        output_core_dims=output_core_dims,
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={
            'allow_rechunk': False,
            'output_sizes': {'property': 4}
        },
        vectorize=False
    )

    # Convert to Dataset with named variables
    results = results.to_dataset(dim='property')
    results = results.rename({
        0: 'fsca',
        1: 'fshade',
        2: 'dust_concentration',
        3: 'grain_size'
    })

    # Add metadata
    results['fsca'].attrs = {
        'long_name': 'Fractional Snow-Covered Area',
        'units': '1',
        'valid_range': [0, 1]
    }
    results['fshade'].attrs = {
        'long_name': 'Fractional Shaded Area',
        'units': '1',
        'valid_range': [0, 1]
    }
    results['dust_concentration'].attrs = {
        'long_name': 'Dust Concentration in Snow',
        'units': 'ppm',
        'valid_range': [0, 10000]
    }
    results['grain_size'].attrs = {
        'long_name': 'Effective Snow Grain Radius',
        'units': 'μm',
        'valid_range': [10, 2000]
    }

    return results


def snow_diff_4(x, spectrum_target, spectrum_background, solar_angle, interpolator, shade):
    r"""
    Calculate spectral difference for 4-parameter snow model.

    Computes the Euclidean distance between observed and modeled spectra using
    a 4-parameter linear mixing model with snow, shade, and background components.

    .. math::

       \begin{align}
        R_{model}   & = R_{pure snow}( \phi_{sun}, c_{dust}, s_{grain}) * f_{sca}  \\
                    & + R_{shade} * f_{shade} \\
                    & + R_{0} * (1 - f_{sca} - f_{shade})
        \end{align}

    Parameters
    ----------
    x : array-like
        Model parameters:
        - x[0] : f_sca - Fractional snow-covered area (0-1)
        - x[1] : f_shade - Fractional shaded area (0-1)
        - x[2] : dust_concentration - Dust concentration in snow (ppm)
        - x[3] : grain_size - Effective snow grain radius (μm)
    spectrum_target : numpy.ndarray
        The observed mixed spectrum to match.
    spectrum_background : numpy.ndarray
        The background (snow-free, R_0) spectrum.
    solar_angle : float
        Solar zenith angle of the observation (degrees).
    interpolator : spires.interpolator.LutInterpolator
        Callable object that returns modeled snow spectrum given
        solar_angle, dust_concentration, and grain_size.
    shade : numpy.ndarray
        Ideal shade endmember spectrum.

    Returns
    -------
    float
        Euclidean distance between modeled and target spectra.

    Notes
    -----
    If f_sca is within 2%, consider using 3-parameter solution (snow_diff_3)
    to avoid overfitting.

    Examples
    --------
    >>> import spires
    >>> import numpy as np
    >>> interpolator = spires.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> f_sca = 0.482
    >>> f_shade = 0.065
    >>> dust_concentration = 1000  # ppm
    >>> grain_size = 220  # μm
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
    r"""
    Calculate spectral difference for 3-parameter snow model.

    Computes the Euclidean distance between observed and modeled spectra using
    a simplified 3-parameter model where shade fills the non-snow fraction.

    .. math::

        \begin{align}
        R_{model} & = R_{pure snow}( \phi_{sun}, c_{dust}, s_{grain}) * f_{sca} \\
                  & + R_{shade} * (1-f_{sca})
        \end{align}

    Parameters
    ----------
    x : array-like
        Model parameters (note: only first 3 are used):
        - x[0] : f_sca - Fractional snow-covered area (0-1)
        - x[1] : dust_concentration - Dust concentration in snow (ppm)
        - x[2] : grain_size - Effective snow grain radius (μm)
    spectrum_target : numpy.ndarray
        The observed mixed spectrum to match.
    solar_angle : float
        Solar zenith angle of the observation (degrees).
    interpolator : spires.interpolator.LutInterpolator
        Callable object that returns modeled snow spectrum given
        solar_angle, dust_concentration, and grain_size.
    shade : numpy.ndarray
        Ideal shade endmember spectrum.

    Returns
    -------
    float
        Euclidean distance between modeled and target spectra.

    Notes
    -----
    This 3-parameter model assumes the non-snow fraction is entirely shade
    (no background component). Use when f_sca is near 100% to avoid overfitting.

    Examples
    --------
    >>> import spires
    >>> import numpy as np
    >>> interpolator = spires.interpolator.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
    >>> f_sca = 0.482
    >>> dust_concentration = 1000  # ppm
    >>> grain_size = 220  # μm
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
    Invert snow spectra using scipy.optimize.minimize.

    Alternative implementation using SciPy's optimization routines instead of NLopt.
    Provides compatibility with legacy code and additional solver options.

    Parameters
    ----------
    interpolator : spires.interpolator.LutInterpolator
        Interpolator object with:
        - Attributes: `bands`, `solar_angles`, `dust_concentrations`, `grain_sizes`
        - Method: `interpolate_all(solar_angle, dust_concentration, grain_size)`
    spectrum_target : numpy.ndarray
        Target spectrum to be inverted. Must be same shape as `spectrum_background`.
    spectrum_background : numpy.ndarray
        Background (snow-free, R_0) spectrum. Must be same shape as `spectrum_target`.
    solar_angle : float
        Solar zenith angle of observation (degrees).
        Must use same units as interpolator coordinates.
    shade : numpy.ndarray, optional
        Ideal shade endmember spectrum. Must be same shape as `spectrum_target`.
        If None, uses zeros (default: None).
    scipy_options : dict, optional
        SciPy solver options. Default:
        `{'disp': False, 'iprint': 100, 'maxiter': 1000, 'ftol': 1e-9}`
    mode : int, optional
        Number of parameters in model (default: 3).
        3 = Simplified model (f_sca, dust, grain_size).
        4 = Full model (f_sca, f_shade, dust, grain_size).
        Use mode=3 when f_sca is near 100% to avoid overfitting.
    method : str, optional
        SciPy optimization method (default: 'SLSQP').
        Common options: 'SLSQP', 'L-BFGS-B', 'TNC'.

    Returns
    -------
    tuple
        (res, model_refl) where:

        - res : scipy.optimize.OptimizeResult
          Optimization result object. res.x contains:
          [f_sca, f_shade, dust_concentration, grain_size]
        - model_refl : numpy.ndarray
          The optimized modeled reflectance spectrum.

    See Also
    --------
    scipy.optimize.OptimizeResult : Documentation of result object
    speedy_invert : NLopt-based implementation (faster)

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
    Convert normalized index to coordinate value.

    Linearly interpolates between coordinate values based on a normalized
    index in the range [0, 1].

    Parameters
    ----------
    index : float
        Normalized index value between 0 and 1.
    coords : numpy.ndarray
        Array of coordinate values to interpolate between.

    Returns
    -------
    float
        Interpolated coordinate value.

    Notes
    -----
    Used internally by speedy_invert_scipy_normalized to convert
    normalized optimization parameters back to physical units.
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
    Invert snow spectra with normalized parameter space.

    Performs optimization with all parameters scaled to [0, 1] range to improve
    convergence for solvers like COBYLA that don't support parameter-specific
    step sizes.

    Parameters
    ----------
    interpolator : spires.interpolator.LutInterpolator
        Interpolator object with lookup table and coordinate arrays.
    spectrum_target : numpy.ndarray
        Target spectrum to be inverted.
    spectrum_background : numpy.ndarray
        Background (snow-free, R_0) spectrum. Must be same shape as `spectrum_target`.
    solar_angle : float
        Solar zenith angle of observation (degrees).
    spectrum_shade : numpy.ndarray, optional
        Ideal shade endmember spectrum. Must be same shape as `spectrum_target`.
        If None, uses zeros (default: None).
    method : str, optional
        SciPy optimization method (default: 'COBYLA').
        COBYLA is recommended as it handles the normalized space well.

    Returns
    -------
    tuple
        (res, model_refl) where:

        - res : scipy.optimize.OptimizeResult
          Optimization result with res.x containing:
          [f_sca, f_shade, dust_concentration, grain_size]
          (dust and grain_size are converted back to physical units)
        - model_refl : numpy.ndarray
          The optimized modeled reflectance spectrum.

    Notes
    -----
    This function internally normalizes dust_concentration and grain_size
    to [0, 1] for optimization, then converts back to physical units.
    This improves convergence for algorithms that assume similar scales
    across parameters.
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
