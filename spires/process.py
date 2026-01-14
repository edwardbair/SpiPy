import spires
import numpy as np
import xarray
import dask.distributed
import tempfile


def invert_one(spectrum_target, spectrum_background, interpolator, solar_angle, shade, mode=4):
    res, model = spires.speedy_invert(spectrum_target=spectrum_target, spectrum_background=spectrum_background,
                                      solar_angle=solar_angle, interpolator=interpolator)
    return res, model


def invert_vectorized_array(r, r0):
    properties = np.zeros([r.shape[0], r.shape[1], 4])

    for x in range(r.shape[0]):
        for y in range(r.shape[1]):
            r_ = r[x, y, :]
            r0_ = r0[x, y, :]
            res, model_refl = invert_one()
            properties[x, y, :] = res.x

    return properties


def invert_ufunc(r, r0, lut_interpolator, solar_z, shade, mode, cluster):
    res = xarray.apply_ufunc(invert_vectorized_array,
                             r,
                             r0,
                             dask='parallelized',
                             input_core_dims=[['band'], ['band']],
                             output_core_dims=[['property']],
                             dask_gufunc_kwargs={'allow_rechunk': False, 'output_sizes': {'property': 4}},
                             output_dtypes=[float],
                             vectorize=False)
    
    with dask.distributed.Client(cluster) as client:
        res = res.compute()

    properties = ['fsca', 'fshade', 'rg', 'dust']
    res = res.assign_coords(coords={'property': properties})
    res = res.to_dataset(dim='property')


def speedy_invert_dask(spectra_targets, spectra_backgrounds, obs_solar_angles, 
                       bands, solar_angles, dust_concentrations, grain_sizes, reflectances, cluster, chunksize ):
    """
    bands, solar_angles, dust_concentrations, grain_sizes, reflectances are supposed to be numpy arrays!
    """

    tmp_st = tempfile.NamedTemporaryFile(suffix=".nc")
    spectra_targets.to_netcdf(tmp_st.name, engine='netcdf4', format='NETCDF4')

    tmp_sb = tempfile.NamedTemporaryFile(suffix=".nc")
    spectra_backgrounds.to_netcdf(tmp_sb.name, engine='netcdf4', format='NETCDF4')

    tmp_so = tempfile.NamedTemporaryFile(suffix=".nc")
    obs_solar_angles.to_netcdf(tmp_so.name, engine='netcdf4', format='NETCDF4')

    #chunksize = 325
    spectra_targets = xarray.open_dataarray(tmp_st.name, chunks={'x': chunksize, 'y': chunksize, 'band': -1})
    spectra_backgrounds = xarray.open_dataarray(tmp_sb.name, chunks={'x': chunksize, 'y': chunksize, 'band': -1})
    obs_solar_angles = xarray.open_dataarray(tmp_so.name, chunks={'x': chunksize, 'y': chunksize})

    with dask.distributed.Client(cluster) as client:
        # We scatter the interpolator. This is super odd but somehow works        
        a = dask.array.from_array(reflectances)
        dsk = client.scatter(dict(a.dask), broadcast=True)
        a = dask.array.Array(dsk, name=a.name, chunks=a.chunks, dtype=a.dtype, meta=a._meta, shape=a.shape)
        refletance_scattered = xarray.DataArray(a, dims=['bands', 'sz', 'dust', 'grain'])

        results = xarray.apply_ufunc(
            spires.speedy_invert_array2d,
            spectra_targets,
            spectra_backgrounds,
            obs_solar_angles,    
            bands,
            solar_angles,
            dust_concentrations,
            grain_sizes,                    
            refletance_scattered,
            input_core_dims=[['band'], ['band'], [],['bands'], ['solar'], ['dust'], ['grain'], ['bands', 'sz', 'dust', 'grain']],
            output_core_dims=[['property']],
            dask='parallelized',
            dask_gufunc_kwargs={'allow_rechunk': False, 'output_sizes': {'property': 4}},
            output_dtypes=[float],
            vectorize=False
        )

        r = results.compute()
    tmp_st.close()
    tmp_sb.close()
    tmp_so.close()
    
    return r