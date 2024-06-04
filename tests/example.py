import xarray
import spires.core
import spires
import numpy as np


r = xarray.load_dataset('tests/data/sentinel_r.nc')
r0 = xarray.load_dataset('tests/data/sentinel_r0.nc')
ts = r.sel(time='2024-02-25').squeeze().drop_vars('time')

interpolator = spires.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
bands = interpolator.bands
solar_angles = interpolator.solar_angles
dust_concentrations = interpolator.dust_concentrations
grain_sizes = interpolator.grain_sizes
reflectances = interpolator.reflectances

spectrum_target = ts.isel(x=0, y=0)['reflectance'].values
spectrum_background = r0.isel(x=0, y=0)['reflectance'].values
print(spectrum_target, spectrum_background)
spectrum_shade = np.zeros_like(spectrum_target)
solar_angle = ts.attrs['sun_zenith_mean']

x0 = np.array([0.5, 0.05, 10, 250])

res = spires.core.invert(spectrum_background=spectrum_background,
                         spectrum_target=spectrum_target,
                         spectrum_shade=spectrum_shade,
                         solar_angle=solar_angle,
                         bands=bands,
                         solar_angles=solar_angles,
                         dust_concentrations=dust_concentrations,
                         grain_sizes=grain_sizes,
                         lut=reflectances,
                         max_eval=100,
                         x0=x0,
                         algorithm=2)

print(res)
