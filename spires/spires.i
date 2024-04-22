%module core

%{
  #define SWIG_FILE_WITH_INIT  /* To import_array() below */
  #include "spires.h"
  #include <vector>
  #include <string>
%}

%include "numpy.i"

%init %{
    import_array();
%}

%include "std_vector.i"
namespace std {
    %template(DoubleVector) vector<double>;
}


%apply (double* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4) { 
    (double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes)
};

%apply (double* IN_ARRAY1, int DIM1) { 
    (double* spectrum_background, int len_background),
    (double* spectrum_target, int len_target),
    (double* x, int len_x),    
    (double* grain_sizes, int len_grain_sizes),
    (double* dust_concentrations, int len_dust_concentrations),
    (double* solar_angles, int len_solar_angles),
    (double* bands, int len_bands),
    (double* obs_solar_angles, int n_obs_solar_angles)   
}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {
       (double* spectra_backgrounds, int n_obs_backgrounds, int n_bands_backgrounds),
       (double* spectra_targets, int n_obs_target, int n_bands_target),
       (double* results, int n_obs, int n_results)
}




%typemap(out) double* {
    // Convert the returned pointer to a NumPy array
    npy_intp dims[1] = {9};  // Define the dimensions of the array
    $result = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)$1);
}


%include spires.h