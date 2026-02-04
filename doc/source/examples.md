# Examples and Tutorials

This page links to example Jupyter notebooks demonstrating the usage of SpiPy for snow property inversion from satellite imagery.

## Main Workflow Examples

### 1. Setup

- **[01_define_regions.ipynb](../../examples/01_define_regions.ipynb)** - Define regions of interest for processing

### 2. Data Preparation (Prerequisites for Inversion)

- **[02_pansharpening.ipynb](../../examples/02_pansharpening.ipynb)** - Pansharpening to improve spatial resolution of Sentinel-2 bands
- **[03_create_background_reflectance.ipynb](../../examples/03_create_background_reflectance.ipynb)** - Generate R0 (background/snow-free) reflectance maps from snow-free periods
- **[04_cloud_masking.ipynb](../../examples/04_cloud_masking.ipynb)** - Cloud detection and masking for satellite imagery

### 3. Core Processing Pipeline

- **[05_sentinel_snow_inversion.ipynb](../../examples/05_sentinel_snow_inversion.ipynb)** - **Main workflow**: Sentinel-2 snow property inversion using SPIRES
  - Loading preprocessed Sentinel-2 data
  - Applying SPIRES algorithm
  - Batch processing with Dask
  - Saving inversion results (fsca, grain size, dust concentration)

### 4. Postprocessing

- **[06_postprocess_clouds.ipynb](../../examples/06_postprocess_clouds.ipynb)** - Interpolate cloud gaps and fix sharpening artifacts in results
- **[07_postprocess_trees.ipynb](../../examples/07_postprocess_trees.ipynb)** - Tree masking and inpainting using deep learning

### 5. Analysis and Visualization

- **[08_create_animations.ipynb](../../examples/08_create_animations.ipynb)** - Generate temporal animations of snow properties

## Test Notebooks

These notebooks demonstrate specific functionality and can be used for testing:

- **[test_interpolator.ipynb](../../examples/test_interpolator.ipynb)** - Test LUT interpolation functionality
- **[test_inversion.ipynb](../../examples/test_inversion.ipynb)** - Test basic inversion on single pixels/images
- **[test_spectrum_diff.ipynb](../../examples/test_spectrum_diff.ipynb)** - Test spectral difference calculations

## Recommended Learning Path

### For New Users

If you're new to SpiPy, we recommend going through the notebooks in this order:

1. **[test_interpolator.ipynb](../../examples/test_interpolator.ipynb)** - Understand the core LUT interpolation
2. **[test_inversion.ipynb](../../examples/test_inversion.ipynb)** - See basic inversion examples on single pixels
3. **[05_sentinel_snow_inversion.ipynb](../../examples/05_sentinel_snow_inversion.ipynb)** - Full workflow with batch processing

### Complete Workflow

For processing your own Sentinel-2 data, follow this sequence:

1. **[01_define_regions.ipynb](../../examples/01_define_regions.ipynb)** - Define your region of interest
2. **[02_pansharpening.ipynb](../../examples/02_pansharpening.ipynb)** - Improve spatial resolution
3. **[03_create_background_reflectance.ipynb](../../examples/03_create_background_reflectance.ipynb)** - Generate R0 maps
4. **[04_cloud_masking.ipynb](../../examples/04_cloud_masking.ipynb)** - Apply cloud detection
5. **[05_sentinel_snow_inversion.ipynb](../../examples/05_sentinel_snow_inversion.ipynb)** - Run SPIRES inversion
6. **[06_postprocess_clouds.ipynb](../../examples/06_postprocess_clouds.ipynb)** - Fill cloud gaps
7. **[07_postprocess_trees.ipynb](../../examples/07_postprocess_trees.ipynb)** - (Optional) Remove tree artifacts
8. **[08_create_animations.ipynb](../../examples/08_create_animations.ipynb)** - Visualize temporal evolution

## Requirements

Most notebooks require:
- SpiPy package installed (see [Getting Started](getting_started.md))
- Access to Sentinel-2 data (zarr format)
- Lookup tables (LUT) for snow reflectance
- Sufficient memory for processing satellite imagery

Some notebooks have additional requirements:
- `postprocess_trees.ipynb` requires PyTorch and simple-lama-inpainting
- Processing notebooks benefit from Dask for parallel computation

## Data Notes

- Notebooks assume data is stored in `/scratch/tristate/` or `/data/sentinel2/` directories
- Most workflows are designed for UCSB region but can be adapted
- Large notebooks (>1MB) contain embedded outputs and visualizations

## Development Notebooks

The `examples/development/` directory contains notebooks used during development, kept for reference:

- `legacy_speedy_invert.ipynb` - Legacy MATLAB-style implementation comparison
- `cobyla.ipynb` - COBYLA optimizer testing
- `invert2d.ipynb` - 2D inversion experiments
- `compress_nc.ipynb` - NetCDF compression utilities
