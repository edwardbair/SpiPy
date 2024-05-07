#include <vector>


// Function declaration for interpolation

double interpolate_idx(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes, 
                       int band_idx, double solar_angle_idx, double dust_concentration_idx, double grain_size_idx);


double interpolate(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                   double* bands, int len_bands,
                   double* solar_angles, int len_solar_angles,
                   double* dust_concentrations, int len_dust_concentrations,
                   double* grain_sizes, int len_grain_sizes,
                   int band,
                   double solar_angle, 
                   double dust_concentration, 
                   double grain_size);

std::vector<double> interpolate_all_idx(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                                        double solar_angle, double dust, double grain_size);

std::vector<double> interpolate_all(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes, 
                                    double* bands, int len_bands,
                                    double* solar_angles, int len_solar_angles,
                                    double* dust_concentrations, int len_dust_concentrations,
                                    double* grain_sizes, int len_grain_sizes,
                                    double solar_angle, 
                                    double dust_size, 
                                    double grain_size);


double* interpolate_all_array(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                              double* bands, int len_bands,
                              double* solar_angles, int len_solar_angles,
                              double* dust_concentrations, int len_dust_concentrations,
                              double* grain_sizes, int len_grain_sizes,
                              double solar_angle,
                              double dust_concentration,
                              double grain_size);

double spectrum_difference(const std::vector<double>& x,
                           double* spectrum_background, int len_background, 
                           double* spectrum_target, int len_target,
                           double* spectrum_shade, int len_shade,
                           double solar_angle,       
                           double* bands, int len_bands,
                           double* solar_angles, int len_solar_angles,
                           double* dust_concentrations, int len_dust_concentrations,
                           double* grain_sizes, int len_grain_sizes,                     
                           double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes);

double spectrum_difference_scaled(const std::vector<double> &x,
                                  double* spectrum_background, int len_background,
                                  double* spectrum_target, int len_target,
                                  double* spectrum_shade, int len_shade,
                                  double solar_angle,
                                  double* bands, int len_bands,
                                  double* solar_angles, int len_solar_angles,
                                  double* dust_concentrations, int len_dust_concentrations,
                                  double* grain_sizes, int len_grain_sizes,
                                  double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes);


std::vector<double>  invert(double* spectrum_background, int len_background,
                           double* spectrum_target, int len_target,
                           double* spectrum_shade, int len_shade,
                           double solar_angle,
                           double* bands, int len_bands,
                           double* solar_angles, int len_solar_angles,
                           double* dust_concentrations, int len_dust_concentrations,
                           double* grain_sizes, int len_grain_sizes,
                           double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,                           
                           int max_eval,
                           std::vector<double> x0,
                           int algorithm);



void invert_array(double* spectra_backgrounds, int n_obs_backgrounds, int n_bands_backgrounds,
                  double* spectra_targets, int n_obs_target, int n_bands_target,
                  double* spectrum_shade, int len_shade,
                  double* obs_solar_angles, int n_obs_solar_angles,
                  double* bands, int len_bands,
                  double* solar_angles, int len_solar_angles,
                  double* dust_concentrations, int len_dust_concentrations,
                  double* grain_sizes, int len_grain_sizes,
                  double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                  double* results, int n_obs, int n_results,
                  int max_eval,
                  std::vector<double> x0,
                  int algorithm) ;