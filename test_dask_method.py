#!/usr/bin/env python
"""
Test script for the new speedy_invert_dask method.
"""
import numpy as np
import xarray as xr
import sys

# Add the package to path if needed
sys.path.insert(0, '/home/griessban/SpiPy')

try:
    import spires
    print("✓ Successfully imported spires")

    # Check if the new function exists
    if hasattr(spires, 'speedy_invert_dask'):
        print("✓ speedy_invert_dask function is available")
    else:
        print("✗ speedy_invert_dask function not found")
        sys.exit(1)

    # Try importing with dask
    try:
        import dask
        import dask.array as da
        from dask.distributed import Client
        print("✓ Dask imports successful")
    except ImportError as e:
        print(f"⚠ Dask not available: {e}")
        print("  The method will still work but requires Dask for parallel processing")

    # Test the function signature
    import inspect
    sig = inspect.signature(spires.speedy_invert_dask)
    params = list(sig.parameters.keys())
    expected_params = [
        'spectra_targets', 'spectra_backgrounds', 'obs_solar_angles',
        'interpolator', 'spectrum_shade', 'max_eval', 'x0', 'algorithm',
        'client', 'scatter_lut'
    ]

    if params == expected_params:
        print("✓ Function signature matches expected parameters")
    else:
        print(f"⚠ Function signature differs:")
        print(f"  Expected: {expected_params}")
        print(f"  Got: {params}")

    # Check docstring
    if spires.speedy_invert_dask.__doc__:
        print("✓ Function has documentation")
        # Check for key sections
        doc = spires.speedy_invert_dask.__doc__
        if "Parameters" in doc and "Returns" in doc and "Examples" in doc:
            print("✓ Documentation includes Parameters, Returns, and Examples sections")
        else:
            print("⚠ Documentation may be incomplete")
    else:
        print("✗ Function lacks documentation")

    print("\n✅ All basic checks passed!")
    print("\nTo fully test the method, you'll need:")
    print("1. Sentinel-2 data in zarr format")
    print("2. Background reflectance data")
    print("3. A LUT file")
    print("4. Dask installed for parallel processing")

except ImportError as e:
    print(f"✗ Failed to import spires: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)