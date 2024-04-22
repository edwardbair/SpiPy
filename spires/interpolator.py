import numpy as np
import netCDF4
import spires.core
import scipy


def get_index(coordinates: np.array, value: float) -> float:
    """
    Returns the interpolated index of a coordinate in a numpy array.

    Parameters
    ----------
    coordinates: array-like
        Array of coordinates.
    value: float
        Value of the coordinate to find the coordinate index for

    Returns
    -------
    interpolated_index: float
        The interpolated index in the coordinates for value

    Examples
    ---------
    >>> import spires
    >>> coordinates = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> value = 1.5
    >>> index = spires.get_index(coordinates, value)
    >>> index == 1.5
    True
    """

    right_idx = np.searchsorted(coordinates, value, side='right')
    left_idx = right_idx - 1
    value_left = coordinates[left_idx]
    value_right = coordinates[right_idx]
    fraction = (value - value_left) / (value_right - value_left)
    interpolated_index = left_idx + fraction
    return interpolated_index


def get_index_linspace(coordinates: np.array, value: float) -> float:
    """
    Returns the interpolated index of a coordinate in a numpy array assuming that coordinates are a linear space.

    Parameters
    ----------
    coordinates: array-like
        Array of coordinates.
    value: float
        Value of the coordinate to find the coordinate index for

    Returns
    -------
    interpolated_index: float
        The interpolated index in the coordinates for value

    """
    interpolated_index = (value - coordinates[0]) / (coordinates[-1] - coordinates[0]) * (coordinates.size - 1)
    return interpolated_index


def is_linspace(array):
    diffs = np.diff(array)
    return np.allclose(diffs, diffs[0])


class LutInterpolator:

    def __init__(self, grid=None, reflectances=None, bands=None, solar_angles=None, dust_concentrations=None, grain_sizes=None, lut_file=None):
        """

        Parameters
        ----------
        grid: xarray.DataArray
            Data array with
                - data variable: reflectances
                - coordinates: bands, solar_angles, dust_concentrations, grain_sizes and
        reflectances: numpy.ndarray:
            4D array of reflectances with
                - dim1: bands
                - dim2: solar_angles
                - dim3: dust_concentrations
                - dim4: grain_sizes
            If reflectances is specified, bands, solar_angles, dust_concentrations,
            and grain_sizes have to be set as well
        bands: numpy.ndarray:
            1D array of band coordinates
        solar_angles: numpy.ndarray:
            1D array of solar angle coordinates
        dust_concentrations: numpy.ndarray:
            1D array of dust concentration coordinates
        grain_sizes: numpy.ndarray
            1D array of grain size coordinates
        lut_file: str
            Specify alternatively to the reflectances file

        """
        if lut_file is not None:
            self.load_mat(lut_file)
        else:
            self.reflectances = reflectances
            self.bands = bands
            self.solar_angles = solar_angles
            self.dust_concentrations = dust_concentrations
            self.grain_sizes = grain_sizes

        self.interpolator_scipy = None
        self.solar_angles_is_linspace = False
        self.dust_concentrations_is_linspace = False
        self.grain_sizes_is_linspace = False
        self.verify_linspace()

    def verify_linspace(self):
        self.solar_angles_is_linspace = is_linspace(self.solar_angles)
        self.dust_concentrations_is_linspace = is_linspace(self.dust_concentrations)
        self.grain_sizes_is_linspace = is_linspace(self.grain_sizes)

    def load_mat(self, lut_file):
        """
        Loads a LutInterpolator from a mat file. The file has to be structured in a fairly idiosyncratic format.

        Parameters
        ----------
        lut_file: string

        """
        with netCDF4.Dataset(lut_file) as lut_nc:
            self.reflectances = np.squeeze(lut_nc['#refs#']["h"][:])
            self.grain_sizes = np.squeeze(lut_nc['#refs#']['d'][:])
            self.dust_concentrations = np.squeeze(lut_nc['#refs#']['e'][:])
            self.solar_angles = np.squeeze(lut_nc['#refs#']['f'][:])
            self.bands = np.squeeze(lut_nc['#refs#']['g'][:])

    def make_scipy_interpolator(self):
        """
        Creates a scipy interpolator object using coordinates and reflectances

        Returns
        -------
        None

        """
        points = [self.bands, self.solar_angles, self.dust_concentrations, self.grain_sizes]
        self.interpolator_scipy = scipy.interpolate.RegularGridInterpolator(points=points, values=self.reflectances)

    def make_scipy_interpolator_legacy(self):
        """
        Creates a scipy interpolator object using coordinates and reflectances.

        In the legacy implementation, the dimensions were (axis 2 and 3 swapped):
        - dim1 = bands
        - dim2 = solar_angles
        - dim3 = grain_sizes
        - dim4 = dust_concentrations

        Returns
        -------
        None

        """
        points = [self.bands, self.solar_angles, self.grain_sizes, self.dust_concentrations]
        reflectances = self.reflectances.swapaxes(2, 3)
        self.interpolator_scipy = scipy.interpolate.RegularGridInterpolator(points=points, values=reflectances)

    def interpolate_scipy(self, band, solar_angle, dust_concentration, grain_size):
        """
        Interpolate values using the scipy RegularGridInterpolator interpolator

        Parameters
        ----------
        band: int
        solar_angle: double
        dust_concentration: double
        grain_size: double

        Returns
        -------
        Interpolated snow reflectance for given band and solar angle, dust concentration and grain size

        """
        pts = np.array([band, solar_angle, dust_concentration, grain_size])
        return self.interpolator_scipy(pts)

    def interpolate_scipy_pts(self, pts):
        """
        Interpolate values using the scipy RegularGridInterpolator interpolator

        Parameters
        ----------
        band: int
        solar_angle: double
        dust_concentration: double
        grain_size: double

        Returns
        -------
        Interpolated snow reflectance for given band and solar angle, dust concentration and grain size

        """
        return self.interpolator_scipy(pts)

    def interpolate(self, band, solar_angle, dust_concentration, grain_size):
        """
        Interpolate values using the c++ interpolator. Index lookup is performed in python.

        Parameters
        ----------
        band: int
            Band to interpolate the reflectance for. Has to be a value in `self.bands`
        solar_angle: double
            solar angle to interpolate the reflectance for.
            Units of `solar_angle` has to match the units in `self.solar_angles` (e.g. degrees).
        dust_concentration: double
            dust concentration to interpolate the reflectance for.
            Units of `dust_concentration` has to match the units in `self.dust_concentrations` (e.g. ppm)
        grain_size: double
            grain size to interpolate the reflectance for.
            Units of `grain_size` has to match the units in `self.grain_sizes` (e.g. micro meters).

        Returns
        -------
        Interpolated snow reflectance for given band and solar angle, dust concentration and grain size

        Examples
        --------
        >>> import spires
        >>> interpolator = spires.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
        >>> reflectance = interpolator.interpolate(band=1, solar_angle=0, dust_concentration=0.1, grain_size=30)
        >>> round(reflectance, 6)
        0.992369
        """
        band_idx = band - 1
        solar_idx = get_index(coordinates=self.solar_angles, value=solar_angle)
        dust_idx = get_index(coordinates=self.dust_concentrations, value=dust_concentration)
        grain_idx = get_index(coordinates=self.grain_sizes, value=grain_size)
        reflectance = spires.core.interpolate_idx(self.reflectances, band_idx, solar_idx, dust_idx, grain_idx)
        return reflectance

    def interpolate_pts(self, pts):
        """
        Compatability function; interpolate values using the c++ interpolator
        Parameters
        ----------
        pts: array_like with
            - pts[0]: band
            - pts[1]: solar_angle
            - pts[2]: dust_concentration
            - pts[3]: grain_size

        Returns
        -------
        Interpolated snow reflectance for given band and solar angle, dust concentration and grain size

        Examples
        --------
        >>> import spires
        >>> interpolator = spires.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
        >>> reflectance = interpolator.interpolate_pts([1, 0, 0.1, 30])
        >>> round(reflectance, 6)
        0.992369

        """
        reflectance = self.interpolate(band=pts[0], solar_angle=pts[1], dust_concentration=pts[2], grain_size=pts[3])
        return reflectance

    def interpolate_all_np_index(self, solar_angle, dust_concentration, grain_size):
        """
        Interpolate spectrum using the c++ interpolator for all bands in self.bands.
        Derive the index values using numpy functions.

        Parameters
        ----------
        solar_angle: double
            solar angle to interpolate the spectrum for.
            Units of `solar_angle` has to match the units in `self.solar_angles` (e.g. degrees).
        dust_concentration: double
            dust concentration to interpolate the spectrum for.
            Units of `dust_concentration` has to match the units in `self.dust_concentrations` (e.g. ppm)
        grain_size: double
            grain size to interpolate the spectrum for.
            Units of `grain_size` has to match the units in `self.grain_sizes` (e.g. micro meter).

        Returns
        -------
        The snow reflectance spectrum

        Examples
        ----------
        >>> import spires
        >>> interpolator = spires.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
        >>> interpolator.interpolate_all_np_index(solar_angle=0, dust_concentration=0.1, grain_size=30)
        array([0.99236893, 0.987902  , 0.97464906, 0.96756319, 0.96042174,
               0.94498655, 0.1533866 , 0.18644477, 0.92160772])

        """
        solar_idx = get_index(coordinates=self.solar_angles, value=solar_angle)
        dust_idx = get_index(coordinates=self.dust_concentrations, value=dust_concentration)
        grain_idx = get_index(coordinates=self.grain_sizes, value=grain_size)
        reflectance = spires.core.interpolate_all_idx(self.reflectances, solar_idx, dust_idx, grain_idx)
        reflectance = np.array(reflectance)
        return reflectance

    def interpolate_all(self, solar_angle, dust_concentration, grain_size):
        """
        Interpolate spectrum using the c++ interpolator for all bands in self.bands.
        Derive the index values using swig C++ functions (about 5x more efficient than `self.interpolate_all_np_index`).

        Parameters
        ----------
        solar_angle: double
            solar angle to interpolate the spectrum for.
            Units of `solar_angle` has to match the units in `self.solar_angles` (e.g. degrees).
        dust_concentration: double
            dust concentration to interpolate the spectrum for.
            Units of `dust_concentration` has to match the units in `self.dust_concentrations` (e.g. ppm)
        grain_size: double
            grain size to interpolate the spectrum for.
            Units of `grain_size` has to match the units in `self.grain_sizes` (e.g. micro meters).

        Returns
        -------
        The snow reflectance spectrum

        Examples
        --------
        >>> import spires
        >>> interpolator = spires.LutInterpolator(lut_file='tests/data/lut_sentinel2b_b2to12_3um_dust.mat')
        >>> interpolator.interpolate_all(solar_angle=0, dust_concentration=0.1, grain_size=30)
        array([0.99236893, 0.987902  , 0.97464906, 0.96756319, 0.96042174,
               0.94498655, 0.1533866 , 0.18644477, 0.92160772])

        """

        return spires.core.interpolate_all_array(self.reflectances,
                                                 self.bands,
                                                 self.solar_angles,
                                                 self.dust_concentrations,
                                                 self.grain_sizes,
                                                 solar_angle,
                                                 dust_concentration,
                                                 grain_size)
