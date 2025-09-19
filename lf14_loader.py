# ---------------------------------------------------------------
# LF14 loader
# Loads the Lopez & Fortney 2014 grid and
# creates a function that interpolates intermediate planet radii
# ---------------------------------------------------------------

import numpy as np 
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

# Load the data
path_folder_grids = Path("../")

name_grid_lf14 = Path("LF2014.dat")

# Load grid data
path_grid_lf14 = path_folder_grids / name_grid_lf14
list_met_lf14,list_age_lf14,list_finc_lf14,list_m_lf14,list_fenv_lf14,list_r_lf14 = np.loadtxt(path_grid_lf14,skiprows=11,unpack=True,usecols=(0,1,2,3,4,5))

dim_met_lf14 = np.array([1.0,50.0])
dim_age_lf14 = np.array([0.1,1.0,10.0])
dim_finc_lf14= np.array([0.1,10.0,1000.0])
dim_teq_lf14 = 278.0*(dim_finc_lf14)**(0.25)
dim_mass_lf14= np.array([1,1.5,2.4,3.6,5.5,8.5,13,20])
dim_fenv_lf14= np.array([0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0])

data_r_lf14 = np.reshape(list_r_lf14,(2,3,3,8,11))

interp_lf14 = RegularGridInterpolator((dim_met_lf14, dim_age_lf14, dim_teq_lf14, dim_mass_lf14, dim_fenv_lf14), data_r_lf14, method='linear', bounds_error=False, fill_value=np.inf)

def radius_lf14(met,age,teq,mp,fenv):
    '''
    Computes the radius of a planet, using the Lopez & Fortney 2014 model.

    Physical parameters:
    - atmosphere metallicity (met): 1 to 50 times solar
    - system age (age): 0.1 to 10 Gyr
    - planet equilibrium temperature (teq): 156.33 to 1563.3 K
    - planet mass (mp): 1 to 20 Me
    - planet envelope mass fraction (fenv): 0.01% to 20 %

    Input shapes:
    - mp: 1D numpy array of any length
    - everything else: scalar
    '''
    # If all parameters are scalars, evaluate and return a single value
    if isinstance(mp, (float, int)):
        return interp_lf14((met,age,teq,mp,fenv)).item()
    # If all are arrays of the same length, evaluate for each set of parameters
    elif isinstance(mp, np.ndarray):
        points = np.array([np.full(len(mp),met),
                           np.full(len(mp),age),
                           np.full(len(mp),teq),
                           mp,
                           np.full(len(mp),fenv)]).T  # Stack the arrays into a 2D array of points
        return interp_lf14(points)  # This will return an array of results
    # Handle invalid input types
    else:
        raise ValueError("Inputs must either be scalars or arrays of the same length.")
