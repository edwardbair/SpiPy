# Getting Started

## Installation

### Prerequisites

**Important:** Use conda-forge for all dependencies. The apt version of `nlopt` does not include required C++ headers.

```bash
# Install build tools and nlopt (required)
conda install -c conda-forge swig gxx gcc nlopt

# Install all dependencies (recommended)
conda install -c conda-forge numpy h5py scipy xarray netCDF4 gdal geopandas matplotlib tox sphinx dask jupyterlab pyproj
```

### Git LFS

This repository uses Git LFS for test data. Install Git LFS before cloning:

```bash
# macOS
brew install git-lfs

# Linux
sudo apt install git-lfs

# Initialize
git lfs install
```

### Build and Install

```bash
# Clone the repository
git clone https://github.com/[username]/SpiPy.git
cd SpiPy

# Build SWIG extensions
python3 setup.py build_ext --inplace

# Install package
pip3 install .

# Or build a wheel
python -m build --wheel
pip3 install dist/*.whl
```

## Quick Start

### Basic Inversion Example

Here's a minimal example of inverting a single snow spectrum:

```python
import spires
import numpy as np

# Load the lookup table
interpolator = spires.interpolator.LutInterpolator(
    lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat'
)

# Define observation and background spectra
spectrum_target = np.array([0.3424, 0.366, 0.3624, 0.3893,
                           0.4162, 0.3957, 0.0704, 0.0627, 0.3792])
spectrum_background = np.array([0.0182, 0.0265, 0.0283, 0.0561,
                               0.0954, 0.1204, 0.1249, 0.0789, 0.1406])
solar_angle = 55.73  # degrees

# Run inversion
fsca, fshade, dust, grain_size = spires.speedy_invert(
    spectrum_target=spectrum_target,
    spectrum_background=spectrum_background,
    solar_angle=solar_angle,
    interpolator=interpolator,
    algorithm=1  # LN_COBYLA
)

print(f"Fractional snow cover: {fsca:.3f}")
print(f"Grain size: {grain_size:.1f} μm")
print(f"Dust concentration: {dust:.1f} ppm")
```

### Batch Processing

For processing multiple pixels or entire images, use the array-based functions:

```python
# Process a 2D array of spectra
results = spires.speedy_invert_array2d(
    spectra_targets=targets,      # shape: (ny, nx, n_bands)
    spectra_backgrounds=backgrounds,  # shape: (ny, nx, n_bands)
    obs_solar_angles=solar_angles,    # shape: (ny, nx)
    interpolator=interpolator,
    max_eval=100,
    algorithm=2  # LN_NELDERMEAD
)

# Extract results
fsca = results[:, :, 0]
fshade = results[:, :, 1]
dust = results[:, :, 2]
grain_size = results[:, :, 3]
```

### Using with xarray

For geospatial data with coordinates:

```python
import xarray as xr

# Load data as xarray DataArrays
targets = xr.open_dataarray('observations.nc')
backgrounds = xr.open_dataarray('background_r0.nc')
solar_angles = xr.open_dataarray('solar_angles.nc')

# Load LUT as xarray
lut = xr.open_dataarray('lut.nc')

# Run inversion
results = spires.speedy_invert_xarray(
    spectra_targets=targets,
    spectra_backgrounds=backgrounds,
    obs_solar_angles=solar_angles,
    lut_dataarray=lut
)
```

## Understanding the Algorithm

SPIRES (SPectral Inversion of REflectance from Snow) retrieves snow properties by:

1. **Loading pre-computed lookup tables (LUTs)** - Generated from Mie scattering theory
2. **Defining a forward model** - Mixed pixel reflectance as a linear combination:
   ```
   R_mixed = fsca * R_snow(dust, grain_size, angle) +
             fshade * R_shade +
             (1 - fsca - fshade) * R_background
   ```
3. **Optimizing parameters** - Minimizes difference between observed and modeled spectra
4. **Returning snow properties** - Fractional snow cover, grain size, dust concentration

## Key Parameters

- **fsca**: Fractional Snow-Covered Area (0-1)
- **grain_size**: Effective snow grain radius (30-1200 μm)
- **dust**: Dust/impurity concentration in snow (0-1000 ppm)
- **R_0** (background): Snow-free reflectance spectrum

## Performance Notes

The C++ optimized version achieves dramatic speedups:
- Interpolation: **3000x faster** (1.07 ms → 309 ns)
- Full optimization: **3000x faster** (165 ms → 43 μs)

This enables processing entire satellite images (millions of pixels) in reasonable time.

## Next Steps

- See [Examples](examples.md) for complete workflow tutorials
- Check the [API Reference](reference.rst) for detailed function documentation
- Read the original paper: [Bair et al. (2021)](https://doi.org/10.1109/TGRS.2020.3040328)
