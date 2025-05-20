import spires
import numpy as np
import xarray
import dask.distributed


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


