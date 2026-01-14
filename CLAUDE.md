# CLAUDE.md - AI Assistant Context for SpiPy

## Project Overview
SpiPy is a Python implementation of SPIRES (SPectral Inversion of REflectance from Snow), a spectral unmixing algorithm for analyzing snow reflectance data. The original MATLAB implementation is described in [this IEEE paper](https://ieeexplore.ieee.org/document/9290428).

**Core Purpose**: Invert satellite reflectance spectra to retrieve snow properties (grain size, dust concentration, fractional snow-covered area) using lookup tables generated from Mie-scattering theory.

## Architecture

### Hybrid Python/C++ Design
This is a **performance-critical scientific computing project** with a hybrid architecture:
- **Python layer**: High-level API, data I/O, coordinate transformations
- **C++ layer**: Performance-critical bottlenecks (3000x speedup achieved)
- **SWIG**: Bindings between Python and C++

**Critical files**:
- [spires/spires.cpp](spires/spires.cpp): Core C++ implementations (interpolation, optimization)
- [spires/interpolator.py](spires/interpolator.py): LUT interpolation interface
- [spires/invert.py](spires/invert.py): Main inversion algorithm
- [spires/process.py](spires/process.py): High-level processing pipeline

### Key Components

1. **Lookup Table (LUT) Interpolator**
   - Interpolates pre-computed Mie-scattering reflectance spectra
   - 4D interpolation: (band, solar_angle, dust, grain_size)
   - C++ implementation using SWIG is ~3000x faster than pure Python

2. **Spectral Inversion**
   - Nonlinear optimization to match observed reflectance to modeled reflectance
   - Uses NLopt library (COBYLA algorithm) in C++
   - Note: SLSQP doesn't work in C++ implementation (see issues in README)

3. **Data Processing**
   - Handles MODIS, Sentinel-2, Landsat data
   - Coordinate transformations, cloud masking, temporal analysis
   - Uses xarray, rioxarray, gdal for geospatial operations

## Build System & Dependencies

### Critical Build Requirements
**You MUST use conda-forge for dependencies**, not apt:
- `nlopt`: Optimization library (apt version missing C++ headers!)
- `swig`: Generate Python/C++ bindings
- `gcc`, `g++`: C++ compilation
- See [README.md](README.md) lines 32-48 for full installation

### Python Dependencies
Core: numpy, h5py, scipy, xarray, netCDF4
Optional: gdal, geopandas, matplotlib, dask, rioxarray, pyproj

### Build Commands
```bash
# Build SWIG extension (required after C++ changes)
python3 setup.py build_ext --inplace

# Install package
pip3 install .

# Build wheel
python -m build --wheel
```

## Working with This Codebase

### Before Making Changes
1. **Performance is paramount**: This code has been heavily optimized. Check [README.md](README.md) lines 160-183 for performance benchmarks before/after changes.
2. **SWIG bindings**: Changes to C++ files require rebuilding with `python3 setup.py build_ext --inplace`
3. **Test after C++ changes**: Run `pytest --doctest-modules` and check examples notebooks

### Common Gotchas
- **Dimension ordering**: C++ uses different array ordering than NumPy (see issues in README)
- **Scaling in optimization**: Different coordinate scales require problem scaling (see line 344-346)
- **COBYLA vs SLSQP**: SLSQP doesn't work in C++, using LN_COBYLA instead
- **Git LFS**: Test data stored in LFS, need `git lfs install`

### File Organization
```
spires/
  ├── __init__.py          # Package exports
  ├── interpolator.py      # LUT interpolation wrapper
  ├── invert.py            # Main inversion functions
  ├── process.py           # High-level processing
  ├── core.py              # Core utilities
  ├── spires.cpp           # C++ optimized code
  ├── spires.h             # C++ header
  ├── spires_wrap.cpp      # SWIG-generated (don't edit!)
  └── cobyla.cpp           # COBYLA implementation
tests/
  └── data/                # LUT files, test images (git-lfs)
examples/                  # Jupyter notebooks with use cases
doc/                       # Sphinx documentation
```

## Testing & Documentation

### Running Tests
```bash
pytest --doctest-modules  # Doctests in source files
```

### Building Documentation
```bash
cd doc/
make html
```

### Example Notebooks
See [examples/](examples/) for Jupyter notebooks demonstrating:
- Basic inversion workflows
- Sentinel-2/MODIS data processing
- Cloud masking and postprocessing
- Temporal analysis

## Known Issues & TODOs

### Issues (from README)
- Interpolator behavior differs between SWIG and scipy when coordinates aren't linspace
- SLSQP solver doesn't work in C++, using COBYLA
- NumPy data ownership issues may cause minor performance loss

### TODOs
- Potential optimization: Keep R_0 constant when inverting same location over time
- Switch from MATLAB .mat LUT files to HDF5 or similar
- Accept xarray inputs for interpolator and target/background spectra

## Domain Knowledge

### What is SPIRES?
SPIRES retrieves snow properties from multispectral satellite imagery by:
1. Loading pre-computed reflectance lookup tables (Mie scattering theory)
2. Iterating through each pixel's observed reflectance spectrum
3. Optimizing model parameters (grain size, dust, fsca) to minimize difference from observed
4. Using spatial constraints and background spectra for robustness

### Key Variables
- **fsca**: Fractional Snow-Covered Area (0-1)
- **grain_size**: Effective snow grain radius (μm)
- **dust**: Dust concentration in snow
- **R_0**: Background (snow-free) reflectance spectrum
- **solar_angle**: Solar zenith angle at image acquisition

## Performance Context
The C++ optimizations achieved dramatic speedups:
- Interpolation: 3000x faster (1.07 ms → 309 ns)
- Spectrum difference: 1000x faster (1.1 ms → 1 μs)
- Full optimization: 3000x faster (165 ms → 43 μs)

These speedups enable processing entire satellite images (millions of pixels) in reasonable time.

## Questions to Ask Me
When working on this codebase, feel free to ask:
- "Should this change affect the C++ layer or stay in Python?"
- "Will this impact performance? Should I benchmark?"
- "Are there examples/tests that cover this use case?"
- "Does this need to work with MODIS, Sentinel-2, or both?"
