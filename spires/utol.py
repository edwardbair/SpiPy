import numpy as np
import xarray


def unique_elements_spacetime(spectra, spectra_background):
    """
    Parameters
    ----------
    spectra: xarray.DataArray
        A DataArray containing observation spectra with dimensions x, y, time
    spectra_background: xarray.DataArray
        A DataArray containing background spectra with dimensions x, y, time

    Returns
    -------
    labels: xarray.DataArray
    unique: numpy.ndarray

    """
    r_flat = spectra.stack(spacetime=('x', 'y', 'time')).transpose('spacetime', 'band').values
    r0_flat = spectra_background.stack(spacetime=('x', 'y', 'time')).transpose('spacetime', 'band').values

    values = np.concatenate([r_flat, r0_flat], axis=1)
    unique, labels = uniquetol_1d(values)

    labels = labels.reshape(spectra.x.size, spectra.y.size, spectra.time.size)
    labels = xarray.DataArray(data=labels, dims=('x', 'y', 'time'), coords=(spectra.x, spectra.y, spectra.time))

    return labels, unique


def unique_elements_space(spectra, spectra_background):
    """
    Parameters
    ----------
    spectra: xarray.DataArray
        A DataArray containing observation spectra with dimensions x, y
    spectra_background: xarray.DataArray
        A DataArray containing background spectra with dimensions x, y

    Returns
    -------

    """
    r_flat = spectra.stack(space=('x', 'y')).transpose('space', 'band').values
    r0_flat = spectra_background.stack(space=('x', 'y')).transpose('space', 'band').values

    values = np.concatenate([r_flat, r0_flat], axis=1)
    unique, labels = uniquetol_1d(values)

    labels = labels.reshape(spectra.x.size, spectra.y.size)
    labels = xarray.DataArray(data=labels, dims=('x', 'y'), coords=(spectra.x, spectra.y))

    return labels, unique


def uniquetol_1d(array, tol=1e-3):
    """
    Finds unique vectors in an array across dimension 1

    Parameters
    ----------
    array: numpy.ndarray
        A 2D numpy array with
        - dim1 being individual observations
        - dim2 being the components of each observation (e.g. the bands)
    tol: float
        tolerance

    Returns
    -------
    unique_vectors: numpy.ndarray
        A 2D numpy array of unique vectors.
        - dim1 being the unique vectors
        - dim2 being the components of each unique vector (e.g. the bands)

    labels: numpy.ndarray
        1D array with same length dim1 of array

    Examples
    ----------
    >>> import spires.utol
    >>> array = np.array([[0.3696, 0.3844, 0.3676],
    ...                   [0.3312, 0.3452, 0.3444],
    ...                   [0.3228, 0.346 , 0.3272],
    ...                   [0.302 , 0.3168, 0.3152],
    ...                   [0.3068, 0.3124, 0.3164]], dtype=float32)
    >>> unique, labels = spires.utol.uniquetol_1d(array)
    >>> labels
    array([1, 0, 2, 3, 3])
    """

    # Round the array elements to the tolerance level
    rounded_array = np.round(array / tol) * tol

    # Convert array to a structured array to enable unique operation
    structured_array = rounded_array.view(np.dtype((np.void, rounded_array.dtype.itemsize * rounded_array.shape[1])))

    # Find unique rows in the structured array and their corresponding indices
    unique_rows, unique_indices, inverse_indices = np.unique(structured_array, return_index=True, return_inverse=True)

    # Extract unique rows from the original array
    unique_vectors = array[unique_indices]

    # Generate labels for the unique vectors based on their index in the unique_rows array
    labels = np.arange(len(unique_vectors))

    # Assign labels to the original array using inverse indices
    labels = labels[inverse_indices]

    return unique_vectors, labels


