#!/usr/bin/env python
"""
Example script demonstrating the new speedy_invert_dask method for parallel snow inversion.

This script shows how to use the refactored Dask parallelization functionality
that was previously implemented inline in the notebooks.
"""

import spires
import xarray as xr
import numpy as np
from dask.distributed import Client, LocalCluster
import dask

def main():
    # Configure Dask
    dask.config.set({'temporary-directory': '/tmp/dask'})
    dask.config.set({'distributed.comm.timeouts.tcp': '3600s'})

    # Setup Dask cluster
    print("Setting up Dask cluster...")
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=2,
        memory_limit='4GB',
        processes=True
    )
    client = Client(cluster)
    print(f"Dashboard available at: {client.dashboard_link}")

    # Load your data (example paths - adjust as needed)
    print("Loading data...")
    region = 'UCSB'

    # Load observations
    ds = xr.open_zarr(f'/scratch/tristate/{region}_sharp.zarr/')
    ds = ds.sel(band=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B11', 'B12', 'B8'])

    # Load background reflectances
    ds_r0 = xr.open_zarr(f'/scratch/tristate/{region}_r0.zarr/')
    ds_r0 = ds_r0.sel(band=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B11', 'B12', 'B8'])

    # Load LUT
    print("Loading lookup table...")
    lut_file = '../tests/data/lut_sentinel2b_b2to12_3um_dust.mat'
    lut_interpolator = spires.LutInterpolator(lut_file=lut_file)

    # Select a subset for processing (e.g., first 10 time steps)
    spectra_targets = ds['reflectance'].isel(time=slice(0, 10))
    spectra_backgrounds = ds_r0['reflectance']
    obs_solar_angles = ds['sun_zenith_grid'].isel(time=slice(0, 10))

    # Chunk the data appropriately
    print("Chunking data...")
    spectra_targets = spectra_targets.chunk({
        'time': 1,      # Process each time step independently
        'y': 256,       # Spatial chunks
        'x': 256,
        'band': -1      # Never chunk bands
    })
    spectra_backgrounds = spectra_backgrounds.chunk({
        'y': 256,
        'x': 256,
        'band': -1
    })
    obs_solar_angles = obs_solar_angles.chunk({
        'time': 1,
        'y': 256,
        'x': 256
    })

    # Set initial guess
    x0 = np.array([0.5, 0.05, 10, 250])  # [fsca, fshade, dust_ppm, grain_size_μm]

    # Run parallel inversion using the new method
    print("Running parallel inversion with speedy_invert_dask...")
    results = spires.speedy_invert_dask(
        spectra_targets=spectra_targets,
        spectra_backgrounds=spectra_backgrounds,
        obs_solar_angles=obs_solar_angles,
        interpolator=lut_interpolator,
        max_eval=100,
        x0=x0,
        algorithm=2,  # LN_COBYLA
        client=client,
        scatter_lut=True  # Broadcast LUT to all workers
    )

    print("Results structure:")
    print(results)

    # Compute the results (triggers actual computation)
    print("Computing results...")
    results = results.compute()

    # Cast to appropriate data types for storage
    print("Casting results for storage...")
    fill_value = -1

    results['fsca'] = xr.where(
        np.isnan(results['fsca']),
        fill_value,
        results['fsca'] * 100
    ).astype(np.int8)

    results['fshade'] = xr.where(
        np.isnan(results['fshade']),
        fill_value,
        results['fshade'] * 100
    ).astype(np.int8)

    results['dust_concentration'] = xr.where(
        np.isnan(results['dust_concentration']),
        fill_value,
        results['dust_concentration']
    ).astype(np.int16)

    results['grain_size'] = xr.where(
        np.isnan(results['grain_size']),
        fill_value,
        results['grain_size']
    ).astype(np.int16)

    # Set fill values
    for var in ['fsca', 'fshade', 'dust_concentration', 'grain_size']:
        results[var].attrs["_FillValue"] = fill_value

    # Add spatial reference
    results['spatial_ref'] = ds['spatial_ref']

    # Save results
    output_path = f'/tmp/{region}_results_dask.zarr'
    print(f"Saving results to {output_path}...")
    results.to_zarr(output_path, mode='w', consolidated=False)

    print("Done!")

    # Clean up
    client.close()
    cluster.close()

if __name__ == "__main__":
    main()