
#include <iostream>
#include <vector>
#include <cmath>



 double spectrum_difference_wrapper2(const std::vector<double> &x, const ObjectiveData& obj_data){
    double* lut = obj_data.lut;
    int n_bands = obj_data.n_bands;
    int n_solar_angles = obj_data.n_solar_angles;
    int n_dust_concentrations = obj_data.n_dust_concentrations;
    int n_grain_sizes = obj_data.n_grain_sizes;
    double* spectrum_background = obj_data.spectrum_background;
    int len_background = obj_data.len_background;
    double* spectrum_target = obj_data.spectrum_target;
    int len_target = obj_data.len_target;
    double* spectrum_shade = obj_data.spectrum_shade;
    int len_shade = obj_data.len_shade;
    double solar_angle = obj_data.solar_angle;
    double* bands = obj_data.bands;
    int len_bands = obj_data.len_bands;
    double* solar_angles = obj_data.solar_angles;
    int len_solar_angles = obj_data.len_solar_angles;
    double* dust_concentrations = obj_data.dust_concentrations;
    int len_dust_concentrations = obj_data.len_dust_concentrations;
    double* grain_sizes = obj_data.grain_sizes;
    int len_grain_sizes = obj_data.len_grain_sizes;

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



// COBYLA algorithm
std::vector<double> cobyla(const std::vector<double>& initial_guess,
                           double (*objective)(const std::vector<double>&, const ObjectiveData&),
                           double (*constraint)(const std::vector<double>&),
                           const ObjectiveData& data,
                           const std::vector<double>& lb,
                           const std::vector<double>& ub) {

    std::vector<double> x = initial_guess;
    double rhobeg = 0.01;
    double rho = rhobeg;
    double rhoend = 1e-6;
    int maxiter = 10;

    for (int iter = 0; iter < maxiter; ++iter) {

        std::vector<double> gradient(x.size(), 0.0);
        for (size_t i = 0; i < x.size(); ++i) {
            std::vector<double> dx(x.size(), 0.0);
            dx[i] = rho;
            gradient[i] = (objective({x[0] + dx[0], x[1] + dx[1], x[2] + dx[2], x[3] + dx[3]}, data) - objective(x, data)) / rho;
            //std::cout << gradient[i] << " ";
        }
        //std::cout <<  std::endl;

        if (constraint(x) >= 0) {
            std::vector<double> linear_constraint(x.size(), 0.0);
            for (size_t i = 0; i < x.size(); ++i) {
                std::vector<double> dx(x.size(), 0.0);
                dx[i] = rho;
                linear_constraint[i] = (constraint({x[0] + dx[0], x[1] + dx[1], x[2] + dx[2], x[3] + dx[3]}) - constraint(x)) / rho;
            }
            for (size_t i = 0; i < x.size(); ++i) {
                if (linear_constraint[i] < 0) {
                    gradient[i] = std::max(gradient[i], -linear_constraint[i]);
                }
            }
        }


        // Update x
        std::vector<double> xnew(x.size(), 0.0);
        double step;
        for (size_t i = 0; i < x.size(); ++i) {
            double step = gradient[i] * (constraint(x) / sqrt(gradient[0] * gradient[0] + gradient[1] * gradient[1] + gradient[2] * gradient[2] + gradient[3] * gradient[3]));
            xnew[i] = x[i] - step;
        }

        for (size_t i = 0; i < x.size(); ++i) {
            //std::cout << xnew[i] << " ";
        }

        //std::cout << objective(x, data) << std::endl;


        // Project onto the feasible region defined by bounds
        for (size_t i = 0; i < x.size(); ++i) {
            xnew[i] = std::max(lb[i], std::min(ub[i], xnew[i]));
        }



        // Check convergence
        if (sqrt(pow(objective(xnew, data) - objective(x, data), 2)) < rhoend) {
            break;
        }

        x = xnew;
        rho *= 0.5;  // Reduce the trust region size in each iteration
    }

    return x;

}

double constraint2(const std::vector<double> &x) {
    // f_sca + f_shade <= 1
    return x[0] + x[1] - 1;
}

std::vector<double> invert2(double* spectrum_background, int len_background,
                           double* spectrum_target, int len_target,
                           double* spectrum_shade, int len_shade,
                           double solar_angle,
                           double* bands, int len_bands,
                           double* solar_angles, int len_solar_angles,
                           double* dust_concentrations, int len_dust_concentrations,
                           double* grain_sizes, int len_grain_sizes,
                           double* lut, int n_bands, int n_solar_angles, int n_dust_concentrations, int n_grain_sizes,
                           int max_eval) {

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

    // Initial guess
    std::vector<double> x0 = {0.5, 0.05, 100, 250};

    std::vector<double> lb = {0.0, 0.0, 30, 30}; // Lower bounds
    std::vector<double> ub = {1.0, 1.0, 900, 900}; // Upper bounds

    // Call COBYLA algorithm with bounds and additional data
    std::vector<double> result = cobyla(x0, spectrum_difference_wrapper2, constraint2, obj_data, lb, ub); // 4D problem with bounds


    std::cout << "Optimal solution: ";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return result;
}

