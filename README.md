# SpiPy Readme

SpiPy is the python implementation of [SPIRES](https://ieeexplore.ieee.org/document/9290428), originally implemented in
Matlab ([SPIRES Github repository](https://github.com/edwardbair/SPIRES))


## Inputs: Simulated Mie-Scattering snow reflectance Lookup tables
- MODIS: `LUT_MODIS.mat` and a MODIS hdf file are at: ftp://ftp.snow.ucsb.edu/pub/org/snow/users/nbair/SpiPy
  - wget "ftp://ftp.snow.ucsb.edu/pub/org/snow/users/nbair/SpiPy/LUT_MODIS.mat"

I realize these inputs need to be switched to Landsat, 
but for the time being there's just spectra for the SPIRES part, so it doesn't matter

## Online
Use [Google Colab](https://colab.research.google.com/github/edwardbair/SpiPy/blob/master/examples/callSpeedyInvert.ipynb)

## git-lfs
We are using git-LFS for the testing data.

On macos:
```bash
brew install git-lfs
git lfs install
```

## Create Documentation

```bash
pip install Sphinx
pip install sphinx-automodapi
pip install sphinx-markdown-tables
pip install myst-parser
pip install nbsphinx
pip install numpydoc
pip install pydata-sphinx-theme
```

```bash
cd doc/
make html
```

## Setup

```bash
pip3 install .
```

## Usage
the `examples/` folder contains some notebooks with use cases.

A general usecase may look like:

```python
import spires
interpolator = spires.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')

spires.get_fsca()
```

## Testing
Do the doctests

```
pytest --doctest-modules
```



## SpiPy Swig extensions

We offloaded bottleneck functions to C++.
To build the exension and create the swig bindings, run

```bash
python3 setup.py build_ext --inplace
````

How we would build the shared object by itself:
```bash
NUMPY_INCLUDE=/Users/griessban/miniconda3/envs/spipy_swig/lib/python3.12/site-packages/numpy/core/include
NUMPY_LIB=/Users/griessban/miniconda3/envs/spipy_swig/lib/python3.12/site-packages/numpy/core/lib
g++ -shared -o your_module.so spires.cpp -I$NUMPY_INCLUDE -L$NUMPY_LIB -lnumpy
```

or:

```bash
cd spires
make
````

### To-do
- [ ] there might be something to gain when inverting for a single location over multiple timesteps as we can keep
  R_0 constant
- [ ] use Xarrays as inputs
  - [ ] for the interpolator
  - [ ] for target spectra / background spectra

### issues
- [x] swig interpolator and RegularGridInterpolator behave differently when a coordinate is not a linspace.
- [ ] cannot get the SLSQP solver to work in c++; using LN_COBYLA instead leading to slightly different results
- [ ] we can set use COBYLA in scipy, however, we cannot set rhobeg for each dimension individually.
  (even though it accepts a list or array, it only eats the first element). Since the different coordinates are on wahy
 different scales, this leads to suboptimal solutions. --> We have to scale the problem
- [ ] the interpolator speedup ix 3000x, while the spectrum difference computation speedup is only 300x. We must be loosing some cycles during conversions to arrays or, more likely: because numpy doesnt' own the data

### Performance evaluation


#### Interpolator
- [x] Baseline: RegularGridInterpolator with 4 dimensions. 1.07 ms
- [x] One interpolator per band: There is no need to interpolate in band dimension rather than having a 4D interpolator (band, solar_angle, dust, grain_size), 9 3D interpolators inprove performance. 611 micro seconds.
- [x] Vectorized interpolator call: While there is a performance gain for using the 9 invidiual interpolators, we have to call the interpolator 9 times. We get more performance improvements by keeping the 4D interpolator and call it once for all bands: 143 micro seconds
- [x] Swig individual: Moving the interpolator in C++ allows us to implement it as 9 3D interpolators without performance loss. If we call it 9 times for each band: 45.8 micro seconds.
- [x] Swig vectorized: Call the same function once for all bands. 5.58  micro seconds
- [x] Swig vectorized; index lookup in c+: 309 ns

Speedup: 3000x

#### Spectrum difference
- [x] Old spectrum difference: 1.1 ms
- [x] Spectrum difference with new interpolator: 3.8 micro seconds; speedup 250x.  Not sure how we loose 10x
- [x] Spectrum difference in c++: 1 micro seconds. Speedup 1000x

#### Optimization:
- [x] Old method: 165 ms
- [x] With new interpolator: 4.94 ms. Speedup: 30x
- [x] With spectrum difference in c++: 3.5 ms. Speedup 35x
- [x] Optimize in c++ using nlopt: 43 microseconds. Speedup 3000x
- [x] Optimize multiple (iterate in c++): Speedup 3000x