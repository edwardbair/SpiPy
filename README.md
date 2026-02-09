# SpiPy

[![PyPI version](https://badge.fury.io/py/spires.svg)](https://badge.fury.io/py/spires)
[![Documentation Status](https://readthedocs.org/projects/spipy/badge/?version=latest)](https://spipy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/edwardbair/SpiPy/workflows/Build%20and%20Test/badge.svg)](https://github.com/edwardbair/SpiPy/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

SpiPy is a Python implementation of [SPIRES](https://ieeexplore.ieee.org/document/9290428) (SPectral Inversion of REflectance from Snow), originally implemented in MATLAB ([SPIRES GitHub repository](https://github.com/edwardbair/SPIRES)).

## Overview

SPIRES retrieves snow properties (grain size, dust concentration, fractional snow-covered area) from satellite multispectral imagery by inverting reflectance spectra using lookup tables generated from Mie-scattering theory.

**Key features:**
- Hybrid Python/C++ implementation for performance (3000x speedup over pure Python)
- Support for MODIS, Sentinel-2, and Landsat data
- SWIG bindings for optimized interpolation and optimization routines
- NLopt-based nonlinear optimization

## Installation

### Quick Install (PyPI)

```bash
pip install spires
```

**Note:** Pre-built binary wheels are available for Linux and macOS (Python 3.9-3.12). For other platforms or to build from source, see below.

### Install from Source
  
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
# Build SWIG extensions
python3 setup.py build_ext --inplace

# Install package
pip install .

# Or install with optional dependencies
pip install ".[dev,test,docs]"
```

## Usage

See the `examples/` folder for Jupyter notebooks with detailed use cases.

Basic usage:

```python
import spires

# Load lookup table
interpolator = spires.LutInterpolator(
    lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat'
)

# Process imagery to get fractional snow-covered area
fsca = spires.get_fsca(...)
```

## Development

### Building Wheels

Build a wheel for the active Python interpreter:

```bash
pip install build
python -m build --wheel
```

Build wheels for multiple Python versions using tox:

```bash
tox -e py39,py310,py311,py312
```

**Note:** When using pyenv, wheels for Python 3.9 may incorrectly build for x86 instead of arm64 on M1 Macs. Use a conda environment to build correctly.

### Building C++ Extensions Manually

The setuptools build process handles SWIG bindings automatically. To build manually:

```bash
cd spires
make
```

Or specify paths explicitly:

```bash
NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
g++ -shared -o spires_module.so spires.cpp -I$NUMPY_INCLUDE
```

### Testing

Run doctests:

```bash
pytest --doctest-modules
```

### Documentation

Install documentation dependencies:

```bash
pip install ".[docs]"
```

Build documentation:

```bash
cd doc/
make html
```

## Lookup Tables

Simulated Mie-scattering snow reflectance lookup tables are available at:
- ftp://ftp.snow.ucsb.edu/pub/org/snow/users/nbair/SpiPy

Download example:
```bash
wget "ftp://ftp.snow.ucsb.edu/pub/org/snow/users/nbair/SpiPy/LUT_MODIS.mat"
```

**Note:** Lookup tables are currently available for MODIS and Sentinel-2. Landsat support is planned.

## Performance

The C++ optimizations provide significant speedups over pure Python:

**Interpolation:** 3000x faster (1.07 ms → 309 ns)
- Pure Python RegularGridInterpolator: 1.07 ms
- Vectorized Python: 143 μs
- SWIG C++ (vectorized): 5.58 μs
- SWIG C++ (index lookup): 309 ns

**Spectrum Difference:** 1000x faster (1.1 ms → 1 μs)
- Pure Python: 1.1 ms
- With optimized interpolator: 3.8 μs
- C++ implementation: 1 μs

**Full Optimization:** 3000x faster (165 ms → 43 μs)
- Scipy optimization: 165 ms
- With optimized interpolator: 4.94 ms
- With C++ spectrum difference: 3.5 ms
- NLopt in C++: 43 μs

## Known Issues

- SLSQP solver doesn't work in the C++ implementation; using COBYLA instead
- SWIG interpolator and scipy's RegularGridInterpolator behave differently when coordinates aren't linspace
- COBYLA in scipy can't set `rhobeg` per dimension individually, requiring problem scaling

## Roadmap

- [ ] Optimize inversion for single location over multiple timesteps (keep R_0 constant)
- [ ] Support xarray inputs for interpolator and spectra
- [ ] Add Landsat lookup tables
- [ ] Improve cloud masking workflows

## License

See LICENSE file for details.

## Citation

If you use this software, please cite both the algorithm paper and the software implementation:

**Algorithm:**
```bibtex
@article{bair2021spires,
  title={Snow Property Inversion From Remote Sensing (SPIReS): A Generalized Multispectral Unmixing Approach With Examples From MODIS and Landsat 8 OLI},
  author={Bair, E. H. and Stillinger, T. and Dozier, J.},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={59},
  number={9},
  pages={7270--7284},
  year={2021},
  doi={10.1109/TGRS.2020.3040328}
}
```

**Software:**
```bibtex
@software{bair2024spipy,
  title={SpiPy: Python implementation of SPIRES snow property inversion},
  author={Bair, Ned and Griessbaum, Niklas},
  year={2024},
  url={https://github.com/edwardbair/SpiPy},
  version={0.2.0},
  note={See CITATION.cff for full metadata}
}
```

Alternatively, see [CITATION.cff](CITATION.cff) or use GitHub's "Cite this repository" feature.

## Funding

Development of this software was supported by:

**Contract:** W913E523C0002
**Program:** "Climate and natural hazards, snow-covered and mountain environment sensing research"
**Sponsor:** Broad Agency Announcement Program, Cold Regions Research and Engineering Laboratory
**Monitored by:** U.S. Army Engineer Research and Development Center, Hanover, NH 03755

**Distribution Statement:** Approved for public release; distribution is unlimited.
