import numpy as np
from scipy.optimize import minimize

# method specific options
scipy_options = {'disp': False, 'iprint': 100, 'maxiter': 1000, 'ftol': 1e-9}


def speedy_invert(F, R, R0, solarZ, shade, mode=3, scipy_options=scipy_options, method='SLSQP'):
    # invert snow spectra
    # inputs: 
    #     F - RT/Mie LUT, gridded interpolant LUT w/ F(band # (1-7),
    #       solar zenith angle (0-90 deg),
    #       dust (0-1000 ppm),grain radius (30-1200 um)
    #     R - target spectra, array of len 7
    #     R0 - background spectra, array of len 7
    #     solarZ - solar zenith angle, deg, scalar
    #     shade - ideal shade endmember, scalar
    # output:
    #     res -  results from chosen solution, dict
    #     model_reflectances - reflectance for chosen solution, array of len 7
    #     res1, res2 - mixed pixel (fsca,fshade,fother) vs snow only (fsca,fshade) solutions.
    #     One of those will be the same as res

    # bounds: fsca, fshade, grain size, dust (note: order for grain radius and dust is switched from F)
    bounds = np.array([[0, 1], [0, 1], [30, 1200], [0, 1000]])

    # initial guesses for fsca, fshade,dust, & grain size
    x0 = [0.5, 0.05, 10, 250]

    # model reflectance preallocation
    model_reflectances = np.zeros(len(R))

    # objective function
    def snow_diff(x):
        nonlocal model_reflectances, mode
        # calc the Euclidean norm of modeled and measured reflectance
        # input:
        # x - parameters: fsca, fshade, grain size, dust array of len 4
        # nonlocal vars from parent function
        # model_reflectances - model_reflectances to be filled out, array of len 7
        # mode -  4 or 3 int
        # mode - 4 variable solution (1-fsca,2-fshade,fother(1-fsca-fshade),3-grain radius,4-dust)
        # mode - 3 variable solution (1-fsca,fshade (1-fsca),2-grain radius,3-dust)
        # fill in model_reflectances for each band for snow properties ie if pixel were pure snow (no fshade, no fother)

        if mode == 4:
            for i in range(0, len(R)):
                # x[2] and x[3] are grain radius and dust
                pts = np.array([i + 1, solarZ, x[3], x[2]])
                model_reflectances[i] = F(pts)
            # now adjust model reflectance for a mixed pixel, with x[0] and x[1]
            # as fsca, fshade, and 1-x[0]-x[1] as fother
            model_reflectances = model_reflectances * x[0] + shade * x[1] + R0 * (1 - x[0] - x[1])

        if mode == 3:
            for i in range(0, len(R)):
                # x[1] and x[2] are grain radius and dust
                pts = np.array([i + 1, solarZ, x[2], x[1]])
                model_reflectances[i] = F(pts)

            model_reflectances = model_reflectances * x[0] + shade * (1 - x[0])

        # Euclidean norm of measured - modeled reflectance
        diffR = np.linalg.norm(R - model_reflectances)
        return diffR

    # construct the bounds in the form of constraints
    # inequality: constraint is => 0
    constraints = [{"type": "ineq",
                    "fun": lambda x: 1 - x[0] + x[1]}]
    #  1-(x[0]+x[1]) >= 0 <-> 1 >= x[0]+x[1]
    #  mixed pixel contraint: 1 >= fsca+fshade <-> 1 = fsca+fshade+(1-fsca)

    # rranges=(slice(0,1,0.1),slice(0,1,0.1),slice(30,1200,30),slice(0,1000,50))
    # resBrute = brute(SnowDiff, rranges, full_output=False, disp=False, finish=None)
    # x0=resBrute
    if mode == 4:
        # run minimization w/ 4 variables to solve
        res = minimize(snow_diff, x0, method=method, options=scipy_options, bounds=bounds, constraints=constraints)
    elif mode == 3:
        # run minimization w/ 3 variables to solve, no constraint needed
        # delete fshade(index 1) guess and bounds
        x0 = np.delete(x0, 1)
        bounds = np.delete(bounds, 1, axis=0)
        res = minimize(snow_diff, x0, method=method, options=scipy_options, bounds=bounds)
        # insert a zero in for consistency for x[2] (fother)
        res.x = np.insert(res.x, 1, 1 - res.x[0])

    # store modeled refl
    model_refl = np.copy(model_reflectances)

    # if fsca is within 2 pct, use 3 variable solution
    # error will be higher, but 4 parameter solution likely overfits
    # if (abs(res1.x[0]-res2.x[0]) < 0.02):
    #     choice=3
    #     res=res2
    #     model_reflectances=model_refl2
    # else:
    #     choice=4
    #     res=res1
    #     model_reflectances=model_refl1

    return res, model_refl
