# ---------------------------------------------------------------
# SWEET loader
# Loads the "Steam Evolution Extended in Temperature" grid and
# creates a function that interpolates intermediate planet radii
# ---------------------------------------------------------------

import numpy as np 
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

# Load the data
path_folder_grids = Path("../")

name_grid_sweet = Path("Aguichine2025_SWEET_all.dat")

swe_top = np.array([0,1])
swe_labels_top = ["20 mbar", "1 µbar"]
swe_labels_host = ["Type M", "Type G"]
swe_teqs = [400, 500, 700, 900, 1100, 1300, 1500]
#swe_teqs = [500, 600, 700]
swe_wmfs = [0.1,1,10,20,30,40,50,60,70,80,90,100]  # 12 water mass fractions from 0.05 to 0.60
swe_masses = [0.2       ,  0.254855  ,  0.32475535,  0.41382762,  0.52733018,
        0.67196366,  0.85626648,  1.09111896,  1.39038559,  1.77173358,
        2.25767578,  2.87689978,  3.66596142,  4.67144294,  5.95270288,
        7.58538038,  9.66586048, 12.31696422, 15.69519941, 20.        ]  # 20 points in mass from 0.1 to 2.0

swe_ages = np.array([0.001,0.0015,0.002,0.003,0.005,0.01,
                        0.02,0.03,0.05,
                        0.1,0.2,0.5,
                        1.0,2.0,5.0,
                        10,20])

path_grid_sweet = path_folder_grids / name_grid_sweet
listrpfull1 = np.loadtxt(path_grid_sweet,skiprows=36,unpack=True,usecols=(6))
listrpfull2 = np.loadtxt(path_grid_sweet,skiprows=36,unpack=True,usecols=(7))
listrpfull = np.array((listrpfull1,listrpfull2))
listrpfull = np.delete(listrpfull, np.arange(17, listrpfull.size, 18))


# Make SWE interpolator
swe_dim_wmf = np.array(swe_wmfs)/100
swe_dim_teq = np.array(swe_teqs)
swe_dim_mass = np.array(swe_masses)
swe_dim_age = swe_ages
swe_dim_star = np.array([0,1])
swe_dim_top = np.array([0,1])

swe_data_radius = np.reshape(listrpfull,(2,2,12,7,20,17))

# Value for extrapolation:
# - np.nan for nan
# - None for extrapolation
fill_value = None
interp_swe = RegularGridInterpolator((swe_dim_top,
                                      swe_dim_star,
                                      swe_dim_wmf,
                                      swe_dim_teq,
                                      swe_dim_mass,
                                      swe_dim_age), 
                                     swe_data_radius, method='linear', bounds_error=False, fill_value=fill_value)

def radius_swe(top,star,wmf,teq,mp,age):
    '''
    Computes the radius of a planet, using the SWEET model from Aguichine+2025.

    Physical parameters:
    - chosen top of the radius: 0 for 20 mbar (clear atmosphere) or 1 for 1 microbar (high altitude clouds)
    - host star spectral type: 0 for M-dwarf or 1 for G-type star
    - planet water mass fraction (wmf): 0.001 to 1
    - planet equilibrium temperature (teq): 500 to 700 K
    - planet mass (mp): 0.2 to 20 Me
    - system age (age): 0.001 to 20 Gyr

    Input shapes:
    - mp: 1D numpy array of any length
    - everything else: scalar
    '''
    # If all parameters are scalars, evaluate and return a single value
    if isinstance(mp, (float, int)):
        return interp_swe((top,star,wmf,teq,mp,age)).item()
    # If all are arrays of the same length, evaluate for each set of parameters
    elif isinstance(mp, np.ndarray):
        points = np.array([np.full(len(mp),top),
                           np.full(len(mp),star),
                           np.full(len(mp),wmf),
                           np.full(len(mp),teq),
                           mp,
                           np.full(len(mp),age)]).T  # Stack the arrays into a 2D array of points
        return interp_swe(points)  # This will return an array of results
    # Handle invalid input types
    else:
        raise ValueError("Inputs must either be scalars or arrays of the same length.")
