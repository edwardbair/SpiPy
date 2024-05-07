#include <iostream>
#include <vector>
#include "spires.h"


double linearInterpolate(double y1, double y2, double x, double x1, double x2) {
    return y1 + (y2 - y1) * ((x - x1) / (x2 - x1));
}


double interpolate_idx(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                       int band_idx, double solar_angle_idx, double dust_concentration_idx, double grain_size_idx) {

    // Check if the coordinates are within bounds
    if (band_idx < 0 || band_idx > n_bands || 
        solar_angle_idx < 0 || solar_angle_idx > n_solar_angles  || 
        dust_concentration_idx < 0 || dust_concentration_idx > n_dust_concentrations  || 
        grain_size_idx < 0 || grain_size_idx > n_grain_sizes ) {
        std::cerr << "Error: Coordinates out of bounds" << std::endl;
        return -1;
    }

    // We only interpolate in solar_angle, dust_concentration, and grain_size dimension; 
    // therefore, we can just select the 3D cube for the band and interolate on this one
    int band_idx_floor = static_cast<int>(band_idx);
    int start_idx = band_idx_floor * (n_solar_angles * n_dust_concentrations * n_grain_sizes);
    double* cube = lut + start_idx; // Pointing to the beginning of the lut

    int iz1 = static_cast<int>(solar_angle_idx);        // solar_angle_idx floor
    int iz2 = iz1 + 1;                                  // solar_angle_idx ceil
    int id1 = static_cast<int>(dust_concentration_idx); // dust_concentration_idx floor
    int id2 = id1 + 1;                                  // dust_concentration_idx ceil
    int iw1 = static_cast<int>(grain_size_idx);         // grain_size_idx floor
    int iw2 = iw1 + 1;                                  // grain_size_idx ceil

    // Perform linear interpolation along each dimension
    double v000 = cube[n_grain_sizes * (id1 + iz1 * n_dust_concentrations) + iw1];
    double v001 = cube[n_grain_sizes * (id1 + iz1 * n_dust_concentrations) + iw2];
    double v010 = cube[n_grain_sizes * (id2 + iz1 * n_dust_concentrations) + iw1];
    double v011 = cube[n_grain_sizes * (id2 + iz1 * n_dust_concentrations) + iw2];
    double v100 = cube[n_grain_sizes * (id1 + iz2 * n_dust_concentrations) + iw1];
    double v101 = cube[n_grain_sizes * (id1 + iz2 * n_dust_concentrations) + iw2];
    double v110 = cube[n_grain_sizes * (id2 + iz2 * n_dust_concentrations) + iw1];
    double v111 = cube[n_grain_sizes * (id2 + iz2 * n_dust_concentrations) + iw2];

    double value = linearInterpolate(
        linearInterpolate(
            linearInterpolate(v000, v001, grain_size_idx, iw1, iw2),
            linearInterpolate(v010, v011, grain_size_idx, iw1, iw2),
            dust_concentration_idx, id1, id2
        ),
        linearInterpolate(
            linearInterpolate(v100, v101, grain_size_idx, iw1, iw2),
            linearInterpolate(v110, v111, grain_size_idx, iw1, iw2),
            dust_concentration_idx, id1, id2
        ),
        solar_angle_idx, iz1, iz2
    );

    return value;
}


std::vector<double> interpolate_all_idx(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                                        double solar_angle_idx, double dust_concentration_idx, double grain_size_idx) {
        
    std::vector<double> spectrum(n_bands);

    for (int band_idx=0; band_idx<n_bands; band_idx++) {
        spectrum[band_idx] = interpolate_idx(lut, n_bands, n_solar_angles,  n_dust_concentrations,  n_grain_sizes,  band_idx, solar_angle_idx, dust_concentration_idx,  grain_size_idx) ;
    };

    return spectrum;
}

double* interpolate_all_idx_array(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                                  double solar_angle_idx, double dust_concentration_idx, double grain_size_idx) {
        
    double* spectrum = new double[n_bands];

    for (int band_idx=0; band_idx<n_bands; band_idx++) {
        spectrum[band_idx] = interpolate_idx(lut, n_bands, n_solar_angles,  n_dust_concentrations,  n_grain_sizes,  band_idx, solar_angle_idx, dust_concentration_idx,  grain_size_idx) ;
    };

    return spectrum;
}


double get_idx_linspace(double value, double* coordinates, int len_coordinates){
    // assuming that values is a linspace, get the (float) position at which value sits
    return (value - coordinates[0]) / (coordinates[len_coordinates-1] - coordinates[0]) * (len_coordinates - 1);
}


double get_idx(double value, double* coordinates, int len_coordinates){
    size_t left_index = 0;
    while (left_index < len_coordinates - 1 && coordinates[left_index] < value) {
        left_index++;
    }
    size_t right_index = left_index + 1;
    
    // Perform linear interpolation
    double left_coord = coordinates[left_index]; 
    double right_coord = coordinates[right_index]; 
    if (left_coord == value){
        return static_cast<double>(left_index);
    }

    // Calculate the interpolation factor
    double interpolation_factor = (value - left_coord) / (right_coord- left_coord);

    // Calculate the interpolated position
    double idx = left_index + interpolation_factor;    

    return idx;
}


double interpolate(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                   double* bands, int len_bands,
                   double* solar_angles, int len_solar_angles,
                   double* dust_concentrations, int len_dust_concentrations,
                   double* grain_sizes, int len_grain_sizes,
                   int band,
                   double solar_angle, 
                   double dust_concentration, 
                   double grain_size) {
    
    // convert to index space
    double solar_angle_idx = get_idx(solar_angle, solar_angles, len_solar_angles);
    double dust_concentration_idx = get_idx(dust_concentration, dust_concentrations, len_dust_concentrations);
    double grain_size_idx = get_idx(grain_size, grain_sizes, len_grain_sizes);
    int band_idx = band - 1;
    return interpolate_idx(lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes, band_idx, solar_angle_idx, dust_concentration_idx, grain_size_idx);
}


std::vector<double> interpolate_all(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                                    double* bands, int len_bands,
                                    double* solar_angles, int len_solar_angles,
                                    double* dust_concentrations, int len_dust_concentrations,
                                    double* grain_sizes, int len_grain_sizes,
                                    double solar_angle, 
                                    double dust_concentration, 
                                    double grain_size) {
    // convert to index space
    double solar_angle_idx = get_idx(solar_angle, solar_angles, len_solar_angles);
    double dust_concentration_idx = get_idx(dust_concentration, dust_concentrations, len_dust_concentrations);
    double grain_size_idx = get_idx(grain_size, grain_sizes, len_grain_sizes);

    //std::cerr << solar_angle_idx << ' ' << dust_concentration_idx  << ' ' << grain_size_idx  << std::endl;    
        
    std::vector<double> spectrum = interpolate_all_idx(lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes, solar_angle_idx, dust_concentration_idx, grain_size_idx);    
    return spectrum;
}

double* interpolate_all_array(double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                                    double* bands, int len_bands,
                                    double* solar_angles, int len_solar_angles,
                                    double* dust_concentrations, int len_dust_concentrations,
                                    double* grain_sizes, int len_grain_sizes,
                                    double solar_angle, 
                                    double dust_concentration, 
                                    double grain_size) {
    // convert to index space
    double solar_angle_idx = get_idx(solar_angle, solar_angles, len_solar_angles);
    double dust_concentration_idx = get_idx(dust_concentration, dust_concentrations, len_dust_concentrations);
    double grain_size_idx = get_idx(grain_size, grain_sizes, len_grain_sizes);

    //std::cerr << solar_angle_idx << ' ' << dust_concentration_idx  << ' ' << grain_size_idx  << std::endl;
    
    double* spectrum = interpolate_all_idx_array(lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes,
                                                 solar_angle_idx, dust_concentration_idx, grain_size_idx);
    
    return spectrum;
}

class Spectrum{
private:
    std::vector<double> data; // Internal storage for vector elements

public: 
    // Constructor to initialize the array with given elements
    Spectrum(const std::vector<double>& vec) : data(vec) {}

    // Constructor to initialize the array with elements from an array
    template <size_t N>
    Spectrum(const std::array<double, N>& arr) : data(arr.begin(), arr.end()) {}

    // Constructor to initialize the array with elements from a pointer and size
    Spectrum(const double* ptr, size_t size) : data(ptr, ptr + size) {}

    // Overload the * operator for vector multiplication by a constant
    Spectrum operator*(double constant) const {
        std::vector<double> result;
        result.reserve(data.size()); // Reserve memory for efficiency
        for (double value : data) {
            result.push_back(value * constant);
        }
        return Spectrum(result);
    }

    // Overload the * operator for element-wise multiplication of two arrays
    Spectrum operator*(const Spectrum& other) const {
        if (data.size() != other.data.size()) {
            std::cerr << "Error: Arrays must have the same size for element-wise multiplication." << std::endl;
            return Spectrum({});
        }

        std::vector<double> result;
        result.reserve(data.size()); // Reserve memory for efficiency
        for (size_t i = 0; i < data.size(); ++i) {
            result.push_back(data[i] * other.data[i]);
        }
        return Spectrum(result);
    }

    // Overload the + operator for element-wise adding of two arrays
    Spectrum operator+(const Spectrum& other) const {
        if (data.size() != other.data.size()) {
            std::cerr << "Error: Arrays must have the same size for element-wise addition." << std::endl;
            return Spectrum({});
        }

        std::vector<double> result;
        result.reserve(data.size()); // Reserve memory for efficiency
        for (size_t i = 0; i < data.size(); ++i) {
            result.push_back(data[i] + other.data[i]);
        }
        return Spectrum(result);
    }

    // Overload the - operator for element-wise subtracting of two arrays
    Spectrum operator-(const Spectrum& other) const {
        if (data.size() != other.data.size()) {
            std::cerr << "Error: Arrays must have the same size for element-wise addition." << std::endl;
            return Spectrum({});
        }

        std::vector<double> result;
        result.reserve(data.size()); // Reserve memory for efficiency
        for (size_t i = 0; i < data.size(); ++i) {
            result.push_back(data[i] - other.data[i]);
        }
        return Spectrum(result);
    }


    double sum() const {
        double total = 0.0;
        for (double value : data) {
            total += value;
        }
        return total;

    } 

    // Function to output the vector elements
    void print() const {
        for (double value : data) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    
    double euclideanNorm() {
        double sumOfSquares = 0.0;
        for (double value : data) {
            sumOfSquares += value * value;
        }
        return std::sqrt(sumOfSquares);
    }
};



double spectrum_difference(const std::vector<double>& x ,
                           double* spectrum_background, int len_background, 
                           double* spectrum_target, int len_target,
                           double* spectrum_shade, int len_shade,
                           double solar_angle,       
                           double* bands, int len_bands,
                           double* solar_angles, int len_solar_angles,
                           double* dust_concentrations, int len_dust_concentrations,
                           double* grain_sizes, int len_grain_sizes,                     
                           double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes) {
    /*
    Calculate the Euclidean norm of modeled and measured reflectance

    Parameters
    ------------
    x: fsca, fshade, grain size, dust in array of len 4       
    */

    double f_sca = x[0];
    double f_shade = x[1];
    double dust = x[2];
    double grain_size = x[3];

    //std::cerr << ' ' << f_sca << ' ' << f_shade  << ' ' << solar_angle << ' ' << dust << ' ' << grain_size << std::endl;

    Spectrum shade(spectrum_shade, len_shade);
    Spectrum background(spectrum_background, len_background);
    Spectrum target(spectrum_target, len_target);

    // Get the model_reflectances for each band for snow properties i.e. if pixel were pure snow (no f_shade, no f_other)
    std::vector<double> model_reflectance_vector;
    
    model_reflectance_vector = interpolate_all(lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes, 
                                               bands, len_bands,
                                               solar_angles, len_solar_angles,
                                               dust_concentrations, len_dust_concentrations,
                                               grain_sizes, len_grain_sizes,
                                               solar_angle, dust, grain_size);

    Spectrum model_reflectance(model_reflectance_vector);
    model_reflectance = model_reflectance * f_sca + shade * f_shade + background * (1 - f_sca - f_shade);
    //model_reflectance.print();
    
    // Euclidean norm of measured - modeled reflectance    
    Spectrum diff =  target - model_reflectance;
    double distance = diff.euclideanNorm();
    //std::cerr << distance << ' ' << f_sca << ' ' << f_shade  << ' ' << dust << ' ' << grain_size << std::endl;
    
    return distance;
}

double index_to_value(double value, double* coords, int len_coords){
    double idx = value * len_coords;
    int l_idx = static_cast<int>(idx);
    int r_idx = l_idx + 1;
    double diff = coords[r_idx] - coords[l_idx];
    double dist = idx - l_idx;
    return coords[l_idx] + dist * diff;

}

double spectrum_difference_scaled(const std::vector<double> &x ,
                                  double* spectrum_background, int len_background,
                                  double* spectrum_target, int len_target,
                                  double* spectrum_shade, int len_shade,
                                  double solar_angle,
                                  double* bands, int len_bands,
                                  double* solar_angles, int len_solar_angles,
                                  double* dust_concentrations, int len_dust_concentrations,
                                  double* grain_sizes, int len_grain_sizes,
                                  double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes) {

    double dust_scaled =  index_to_value(x[2], dust_concentrations, len_dust_concentrations);
    double grain_scaled = index_to_value(x[3], grain_sizes, len_grain_sizes);

    std::vector<double> x_scaled = {x[0], x[1], dust_scaled, grain_scaled };

    return spectrum_difference(x_scaled,
                               spectrum_background, len_background,
                               spectrum_target, len_target,
                               spectrum_shade, len_shade,
                               solar_angle,
                               bands, len_bands,
                               solar_angles, len_solar_angles,
                               dust_concentrations, len_dust_concentrations,
                               grain_sizes, len_grain_sizes,
                               lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes);

}


#include <nlopt.hpp>

struct ObjectiveData{
    double* lut;
    int n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes;
    double* spectrum_background; int len_background;
    double* spectrum_target; int len_target;
    double* spectrum_shade; int len_shade;
    double solar_angle;
    double* bands; int len_bands;
    double* solar_angles; int len_solar_angles;
    double* dust_concentrations; int len_dust_concentrations;
    double* grain_sizes; int len_grain_sizes;
};


 double spectrum_difference_wrapper(const std::vector<double> &x, std::vector<double> &grad, void *data){
    ObjectiveData *obj_data = reinterpret_cast<ObjectiveData*>(data);

    double* lut = obj_data->lut;
    int n_bands = obj_data->n_bands;
    int n_solar_angles = obj_data->n_solar_angles;
    int n_dust_concentrations = obj_data->n_dust_concentrations;
    int n_grain_sizes = obj_data->n_grain_sizes;
    double* spectrum_background = obj_data->spectrum_background;
    int len_background = obj_data->len_background;
    double* spectrum_target = obj_data->spectrum_target;
    int len_target = obj_data->len_target;
    double* spectrum_shade = obj_data-> spectrum_shade;
    int len_shade = obj_data->len_shade;
    double solar_angle = obj_data->solar_angle;
    double* bands = obj_data->bands;
    int len_bands = obj_data->len_bands;
    double* solar_angles = obj_data->solar_angles;
    int len_solar_angles = obj_data->len_solar_angles;
    double* dust_concentrations = obj_data->dust_concentrations;
    int len_dust_concentrations = obj_data->len_dust_concentrations;
    double* grain_sizes = obj_data->grain_sizes;
    int len_grain_sizes = obj_data->len_grain_sizes;

    return spectrum_difference(x,
                               spectrum_background, len_background, 
                               spectrum_target, len_target,
                               spectrum_shade, len_shade,
                               solar_angle,      
                               bands, len_bands,
                               solar_angles, len_solar_angles,
                               dust_concentrations, len_dust_concentrations,
                               grain_sizes, len_grain_sizes,                     
                               lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes);
 }


#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <chrono>
#include <thread>

double constraint(const std::vector<double> &x, std::vector<double> &grad, void *data) {
    // f_sca + f_shade <= 1
    return x[0] + x[1] - 1;
}



std::vector<double> invert(double* spectrum_background, int len_background,
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
                           int algorithm) {

    //std::cout << "Starting" << std::endl;                
    
    ObjectiveData obj_data;
    obj_data.spectrum_background = spectrum_background;
    obj_data.len_background = len_background;
    obj_data.spectrum_target = spectrum_target;
    obj_data.len_target = len_target;
    obj_data.spectrum_shade = spectrum_shade;
    obj_data.len_shade = len_shade;
    obj_data.solar_angle = solar_angle;
    obj_data.bands = bands;
    obj_data.len_bands = len_bands;
    obj_data.solar_angles = solar_angles;
    obj_data.len_solar_angles = len_solar_angles;
    obj_data.dust_concentrations = dust_concentrations;
    obj_data.len_dust_concentrations = len_dust_concentrations;
    obj_data.grain_sizes = grain_sizes;
    obj_data.len_grain_sizes = len_grain_sizes;
    obj_data.lut = lut;
    obj_data.n_bands = n_bands;
    obj_data.n_solar_angles = n_solar_angles;
    obj_data.n_dust_concentrations = n_dust_concentrations;
    obj_data.n_grain_sizes = n_grain_sizes;

    // Create an instance of the NLopt optimizer
    //"LD" means local optimization, (derivative/gradient-based) "LN" means local optimization, (no derivatives)

    bool constrained_algorithm;
    nlopt::opt opt;




    if (algorithm==1) {
        opt = nlopt::opt(nlopt::LN_COBYLA, 4);
        constrained_algorithm = true;

        // Set the initial step size (rhobeg) for the COBYLA algorithm
        std::vector<double> rhobeg = {0.1, 0.1, 100, 100};
        opt.set_initial_step(rhobeg);

        /*
        nlopt::LN_COBYLA::XTOL_REL: Sets the relative tolerance on optimization parameters.
        nlopt::LN_COBYLA::FTOL_REL: Sets the relative tolerance on the function value.
        nlopt::LN_COBYLA::XTOL_ABS: Sets the absolute tolerance on optimization parameters.
        nlopt::LN_COBYLA::FTOL_ABS: Sets the absolute tolerance on the function value.
        nlopt::LN_COBYLA::MAX_EVAL: Sets the maximum number of function evaluations allowed.
        nlopt::LN_COBYLA::RHOBEGIN: Sets the initial step size.
        nlopt::LN_COBYLA::RHOEND: Sets the final step size.
        nlopt::LN_COBYLA::MAX_ITER: Sets the maximum number of iterations allowed.
        */


    } else if (algorithm==2) {
        opt = nlopt::opt(nlopt::LN_NELDERMEAD, 4);
        constrained_algorithm = false;

        /*
        nlopt::LN_NELDERMEAD::STOPVAL: Sets the stop value for the objective function.
        nlopt::LN_NELDERMEAD::XTOL_REL: Sets the relative tolerance on optimization parameters.
        nlopt::LN_NELDERMEAD::FTOL_REL: Sets the relative tolerance on the function value.
        nlopt::LN_NELDERMEAD::XTOL_ABS: Sets the absolute tolerance on optimization parameters.
        nlopt::LN_NELDERMEAD::FTOL_ABS: Sets the absolute tolerance on the function value.
        nlopt::LN_NELDERMEAD::MAX_EVAL: Sets the maximum number of function evaluations allowed.
        nlopt::LN_NELDERMEAD::MAX_ITER: Sets the maximum number of iterations allowed.
        */


    } else if (algorithm==3) {
        opt = nlopt::opt(nlopt::LD_SLSQP, 4); // Use SLSQP algorithm with 4 dimensions
        constrained_algorithm = true;

        /*
        nlopt::LD_SLSQP::STOPVAL: Sets the stop value for the objective function.
        nlopt::LD_SLSQP::XTOL_REL: Sets the relative tolerance on optimization parameters.
        nlopt::LD_SLSQP::FTOL_REL: Sets the relative tolerance on the function value.
        nlopt::LD_SLSQP::XTOL_ABS: Sets the absolute tolerance on optimization parameters.
        nlopt::LD_SLSQP::FTOL_ABS: Sets the absolute tolerance on the function value.
        nlopt::LD_SLSQP::MAX_EVAL: Sets the maximum number of function evaluations allowed.
        nlopt::LD_SLSQP::MAX_ITER: Sets the maximum number of iterations allowed.
        nlopt::LD_SLSQP::INITIAL_STEP: Sets the initial step size for line search.
        nlopt::LD_SLSQP::DIFF_STEP: Sets the step size used to compute numerical derivatives.
        */
    }

    if (constrained_algorithm ) {
        // Add the constraint function
        opt.add_inequality_constraint(constraint, &obj_data);
        constrained_algorithm = true;
    }

    // Set objective function and gradient function (if available)
    opt.set_min_objective(spectrum_difference_wrapper, &obj_data);

    // Set the stopping criteria (e.g., maximum number of iterations)
    opt.set_maxeval(max_eval);

    // Set the lower and upper bounds for each dimension of x
    double min_dust_concentration = dust_concentrations[0];
    double max_dust_concentration = dust_concentrations[len_dust_concentrations-1];
    double min_grain_size = grain_sizes[0];
    double max_grain_size = grain_sizes[len_grain_sizes-1];

    std::vector<double> lower_bounds = {0.0, 0.0, min_dust_concentration, min_grain_size};   // Lower bounds for each dimension of x
    std::vector<double> upper_bounds = {1.0, 1.0, max_dust_concentration, max_grain_size};   // Upper bounds for each dimension of x

    opt.set_lower_bounds(lower_bounds);
    opt.set_upper_bounds(upper_bounds);

    // Set convergence tolerance
    opt.set_ftol_abs(1e-4);
    opt.set_xtol_rel(1e-2);  // Relative parameter tolerance

    // Optimize
    double minf; // Objective function value at minimum
    //std::vector<double> x = {0.5, 0.05, 10, 250};
    std::vector<double> x = x0;
    nlopt::result result = opt.optimize(x, minf);

    return x;
}


std::vector<double> invert_scaled(double* spectrum_background, int len_background,
                                  double* spectrum_target, int len_target,
                                  double solar_angle,
                                  double* bands, int len_bands,
                                  double* solar_angles, int len_solar_angles,
                                  double* dust_concentrations, int len_dust_concentrations,
                                  double* grain_sizes, int len_grain_sizes,
                                  double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                                  int max_eval,
                                  std::vector<double> x0,
                                  int algorithm) {

    ObjectiveData obj_data;
    obj_data.spectrum_background = spectrum_background;
    obj_data.len_background = len_background;
    obj_data.spectrum_target = spectrum_target;
    obj_data.len_target = len_target;
    obj_data.solar_angle = solar_angle;
    obj_data.bands = bands;
    obj_data.len_bands = len_bands;
    obj_data.solar_angles = solar_angles;
    obj_data.len_solar_angles = len_solar_angles;
    obj_data.dust_concentrations = dust_concentrations;
    obj_data.len_dust_concentrations = len_dust_concentrations;
    obj_data.grain_sizes = grain_sizes;
    obj_data.len_grain_sizes = len_grain_sizes;
    obj_data.lut = lut;
    obj_data.n_bands = n_bands;
    obj_data.n_solar_angles = n_solar_angles;
    obj_data.n_dust_concentrations = n_dust_concentrations;
    obj_data.n_grain_sizes = n_grain_sizes;

    // Create an instance of the NLopt optimizer
    //"LD" means local optimization, (derivative/gradient-based) "LN" means local optimization, (no derivatives)
    nlopt::opt opt(nlopt::LN_COBYLA, 4);
    //nlopt::opt opt(nlopt::LD_SLSQP, 4); // Use SLSQP algorithm with 4 dimensions

    // Set objective function and gradient function (if available)
    opt.set_min_objective(spectrum_difference_wrapper, &obj_data);

    // Set the stopping criteria (e.g., maximum number of iterations)
    opt.set_maxeval(max_eval);

    // Add the constraint function
    opt.add_inequality_constraint(constraint, &obj_data);

    // Set the lower and upper bounds for each dimension of x
    double min_dust_concentration = dust_concentrations[0];
    double max_dust_concentration = dust_concentrations[len_dust_concentrations-1];
    double min_grain_size = grain_sizes[0];
    double max_grain_size = grain_sizes[len_grain_sizes-1];

    std::vector<double> lower_bounds = {0.0, 0.0, min_dust_concentration, min_grain_size};   // Lower bounds for each dimension of x
    std::vector<double> upper_bounds = {1.0, 1.0, max_dust_concentration, max_grain_size};   // Upper bounds for each dimension of x

    opt.set_lower_bounds(lower_bounds);
    opt.set_upper_bounds(upper_bounds);

    // Set the initial step size (rhobeg) for the COBYLA algorithm
    std::vector<double> rhobeg = {0.1, 0.1, 100, 100};
    opt.set_initial_step(rhobeg); //

    // Set convergence tolerance
    opt.set_ftol_abs(1e-4);
    //opt.set_xtol_abs(1e-3);  // Relative parameter tolerance

    // Optimize
    double minf; // Objective function value at minimum

    std::vector<double> x = x0;
    nlopt::result result = opt.optimize(x, minf);


    return x;
}


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
                  int algorithm) {

    // n_obs_backgrounds == n_obs_target == n_obs_solar_angles
    // n_bands_backgrounds == n_bands_target == n_obs_solar_angles == len_bands
    
    for (size_t obs = 0; obs < n_obs_backgrounds; obs++) {
        int n = obs * n_bands_backgrounds;
        double* spectrum_background = &spectra_backgrounds[n];
        double* spectrum_target = &spectra_targets[n];
        double solar_angle = obs_solar_angles[obs];

        //std::cerr << solar_angle << std::endl;

        std::vector<double> x = invert(spectrum_background, len_bands,
                                       spectrum_target, len_bands,
                                       spectrum_shade, len_shade,
                                       solar_angle,
                                       bands, len_bands,
                                       solar_angles,  len_solar_angles,
                                       dust_concentrations,  len_dust_concentrations,
                                       grain_sizes,  len_grain_sizes,
                                       lut, n_bands, n_solar_angles, n_dust_concentrations, n_grain_sizes,
                                       max_eval,
                                       x0,
                                       algorithm
                                       );

        for (size_t i = 0; i < x.size(); ++i) {
            results[obs * n_results + i] = x[i];
        }
    }
}



