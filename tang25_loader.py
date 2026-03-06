# ---------------------------------------------------------------
# SWEET loader
# Loads the "Steam Evolution Extended in Temperature" grid and
# creates a function that interpolates intermediate planet radii
# ---------------------------------------------------------------

import numpy as np 
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

# Load the data - change here the path to the folder, and if necessary the name of files
path_folder_grids = Path("../")
name_grid_t25 = Path("Tang2025.dat")
name_grid_boiloff = Path("t24_grids/boil-off.csv")

# Value for extrapolation:
# - np.nan for nan
# - None for extrapolation
fill_value = np.nan

# Make T25 interpolator
dim_met_t25 = np.array([1.0,50.0])
dim_age_t25 = np.log10(np.array([0.01,0.1,1.0,10.0]))
dim_finc_t25= np.array([1.0,10.0,100.0,1000.0])
dim_teq_t25 = 278.0*(dim_finc_t25)**(0.25)
dim_mass_t25= np.array([1,2,3,4,5,6,8,10,13,16,20])
dim_fenv_t25= np.array([0.10,0.20,0.50,1,2,5,10,20])
dim_top_t25 = np.array([0,1,2])
t25_labels = ["RCB", "20 mbar", "1 nbar"]

t25_data_radius = np.zeros((3,2,4,4,11,8))

path_grid_t25 = path_folder_grids / name_grid_t25
data0 = np.genfromtxt(path_grid_t25,filling_values=fill_value,comments='#',skip_header=1,usecols=(5,6,7,8))
t25_data_radius[0,:,:,:,:,:] = data0[:,0].reshape(2,4,4,11,8)
t25_data_radius[1,:,:,:,:,:] = data0[:,1].reshape(2,4,4,11,8)
t25_data_radius[2,:,:,:,:,:] = data0[:,2].reshape(2,4,4,11,8)

interp_t25 = RegularGridInterpolator((dim_top_t25,dim_met_t25, dim_age_t25, dim_teq_t25, dim_mass_t25, dim_fenv_t25), t25_data_radius, method='linear', bounds_error=False, fill_value=fill_value)

# Make the boil-off limit
t25_bolim_fenv = np.zeros((2,8,11))
t25_bolim_radius = np.zeros((3,2,8,11))
path_grid_boiloff = path_folder_grids / name_grid_boiloff
data3 = np.genfromtxt(path_grid_boiloff,delimiter=",",filling_values=20.0,comments='#',skip_header=0)
data3 = data3[:,:12]
t25_bolim_fenv = data3[:,1:].reshape(2,8,11)
t25_bolim_fenv = np.clip(t25_bolim_fenv,a_min=0.0,a_max=20)

dim_finc_t25_bolim= np.array([1.0,3.0,10.0,30.0,100.0,300.0,1000.0,3000.0])
dim_teq_t25_bolim = 278.0*(dim_finc_t25_bolim)**(0.25)

interp_t25_bolim_maxfenv = RegularGridInterpolator((dim_met_t25, dim_teq_t25_bolim, dim_mass_t25), t25_bolim_fenv, method='linear', bounds_error=False, fill_value=0.0)


def radius_t25(top,met,age,teq,mp,fenv,boiloff_cut=True,verbose=True):
    '''
    Computes the radius of a planet, using the T25 model from Tang+2025.

    Physical parameters:
    - chosen top of the radius: 0 for radiative-convective boundary (RCB),
                                1 for 20 mbar (clear atmosphere),
                                2 for 1 nanobar (high altitude aerosols)
    - atmosphere metallicity (met): 1 to 50 (xSolar)
    - system age (age): 0.01 to 10 Gyr
    - planet equilibrium temperature (teq): 500 to 700 K
    - planet mass (mp): 0.2 to 20 Me
    - planet envelope mass fraction (fenv): 0.1 to 20 %

    Input shapes:
    - mp: 1D numpy array of any length
    - everything else: scalar
    '''
    # If all parameters are scalars, evaluate and return a single value
    if isinstance(mp, (float, int)):
        # Get maf fenv from boil-off
        maxfenv = interp_t25_bolim_maxfenv((met,teq,mp)).item()
        if (fenv > maxfenv) and verbose:
            print(f"Warning in 'radius_t25_grid', single value: fenv > maxfenv. maxfenv= {maxfenv:.3f} %")
        
        return interp_t25((top,met,age,teq,mp,fenv)).item()
        
    # If all are arrays of the same length, evaluate for each set of parameters
    elif isinstance(mp, np.ndarray):
        # compute all points
        points = np.array([np.full(len(mp),top),
                           np.full(len(mp),met),
                           np.full(len(mp),age),
                           np.full(len(mp),teq),
                           mp,
                           np.full(len(mp),fenv),
                            ]).T  # Stack the arrays into a 2D array of points
        array_radius = interp_t25(points) # Compute all radii

        # evaluate boil-off
        points_boiloff = np.array([np.full(len(mp),met),
                           np.full(len(mp),teq),
                           mp,
                            ]).T  # Stack the arrays into a 2D array of points
        array_maxfenv = interp_t25_bolim_maxfenv(points_boiloff)
        mask_boiloff = fenv > array_maxfenv
        if boiloff_cut:
            array_radius[mask_boiloff] = np.nan
        return array_radius  # This will return an array of results

    # Handle invalid input types
    else:
        raise ValueError("Inputs must either be scalars or arrays of the same length.")
