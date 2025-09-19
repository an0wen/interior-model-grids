# ---------------------------------------------------------------
# IOP loader
# Loads the "Irradiated Ocean Planet" grid and
# creates a function that interpolates intermediate planet radii
#
# Two methods are given:
# - pure interpolation in the grid
# - interpolate over the "fit coefficients" (analytical MR curve fitted on the data, at most 2.3% of deviation)
# ---------------------------------------------------------------

import numpy as np 
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

# Load the data
path_folder_grids = Path("../")

name_grid_iop = Path("Aguichine2021_mr_all_2024.dat")
name_fit_iop = Path("Aguichine2021_fit_coefficients_2024.dat")


# ---------------------------------------------------------------
# Pure grid interpolation
# ---------------------------------------------------------------

# Load grid data
path_grid_iop = path_folder_grids / name_grid_iop
listmbfull,listmafull,listrbfull,listrafull,listerrfull = np.loadtxt(path_grid_iop,unpack=True,usecols=(4,5,6,7,8))

dimcmf = np.linspace(0.0,0.9,10)
dimwmf = np.linspace(0.0,1.0,11)
dimteq = np.linspace(400.0,1300.0,10)
dimmass = np.logspace(np.log10(0.2),np.log10(20.0),20)

data_radius = np.transpose( np.reshape(listrbfull+listrafull,(10,10,11,20)) , (0,2,1,3))
data_errcode = np.transpose( np.reshape(listerrfull,(10,10,11,20)) , (0,2,1,3))

interp_radius_iop = RegularGridInterpolator((dimcmf, dimwmf, dimteq, dimmass), data_radius, method='linear', bounds_error=False, fill_value=None)
interp_errcode_iop = RegularGridInterpolator((dimcmf, dimwmf, dimteq, dimmass), data_errcode, method='linear', bounds_error=False, fill_value=None)

def radius_iop_grid(cmf,wmf,teq,mp,verbose=True):
    '''
    Computes the radius of a planet, using the IOP model from Aguichine+2021.
    The original IOP model goes only down to 10% water, this version is extended to 0% with Zeng+2016 model.

    Physical parameters:
    - planet reduced core mass fraction (cmf = M_iron/(M_iron+M_silicate)): 0.325 for an Earth-like rocky part
    - planet water mass fraction (wmf): 0 to 1
    - planet equilibrium temperature (teq): 400 to 1300 K
    - planet mass (mp): 0.2 to 20 Me

    Input shapes:
    - mp: 1D numpy array of any length
    - everything else: scalar
    '''
    # If all parameters are scalars, evaluate and return a single value
    if isinstance(mp, (float, int)):
        return interp_radius_iop((cmf, wmf, teq, mp)).item()
    
    # If all are arrays of the same length, evaluate for each set of parameters
    elif isinstance(mp, np.ndarray):
        points = np.array([np.full(len(mp),cmf),
                           np.full(len(mp),wmf),
                           np.full(len(mp),teq),
                           mp]).T  # Stack the arrays into a 2D array of points
        array_radius = interp_radius_iop(points) # Compute all radii
        array_errcode = interp_errcode_iop(points) # Compute all error codes
        if np.any(array_errcode > 0) and verbose:
            print(f"Warning in 'radius_iop_grid': some planet radii have errcode >0 with parameters {cmf:.3f}, {wmf:.3f}, {teq:.1f} K")
        return array_radius  # This will return an array of results

    # Handle invalid input types
    else:
        raise ValueError("Inputs must either be scalars or arrays of the same length.")


# ---------------------------------------------------------------
# Construct from fit coefficients
# Recommended strategy: get the mass validity limits to avois errcode >0 then build the MR curve
# ---------------------------------------------------------------

# Load fit coefficients
path_grid_iop_fit = path_folder_grids / name_fit_iop
listcmf,listwmf,listteq,lista,listb,listd,listc,listmasslow,listmasshigh \
    = np.loadtxt(path_grid_iop_fit,skiprows=19,unpack=True,usecols=(0,1,2,3,4,5,6,11,12))

dimwmf_f = np.linspace(0.1,1.0,10)

data_a = np.reshape(lista,(10,10,10))
data_b = np.reshape(listb,(10,10,10))
data_c = np.reshape(listc,(10,10,10))
data_d = np.reshape(listd,(10,10,10))

data_masslimlow = np.reshape(listmasslow,(10,10,10))
data_masslimhigh = np.reshape(listmasshigh,(10,10,10))

interp_a = RegularGridInterpolator((dimcmf, dimteq, dimwmf_f), data_a, method='cubic', bounds_error=False, fill_value=None)
interp_b = RegularGridInterpolator((dimcmf, dimteq, dimwmf_f), data_b, method='cubic', bounds_error=False, fill_value=None)
interp_c = RegularGridInterpolator((dimcmf, dimteq, dimwmf_f), data_c, method='cubic', bounds_error=False, fill_value=None)
interp_d = RegularGridInterpolator((dimcmf, dimteq, dimwmf_f), data_d, method='cubic', bounds_error=False, fill_value=None)

interp_masslimlow = RegularGridInterpolator((dimcmf, dimteq, dimwmf_f), data_masslimlow, method='cubic', bounds_error=False, fill_value=None)
interp_masslimhigh = RegularGridInterpolator((dimcmf, dimteq, dimwmf_f), data_masslimhigh, method='cubic', bounds_error=False, fill_value=None)

def radius_iop_fit(cmf,wmf,teq,mp,verbose=True):
    '''
    Constructs the mass-radius curve based on fit on data for the IOP model from Aguichine+2021
    The planet input should contain:
    - planet reduced core mass fraction (cmf = M_iron/(M_iron+M_silicate)): 0.325 for an Earth-like rocky part
    - planet water mass fraction (wmf): 0.1 to 1
    - planet equilibrium temperature (teq): 400 to 1300 K
    - planet mass (mp): 0.2 to 20 Me
    '''
    # Compute fit coefficients
    a,b,c,d = interp_a([cmf,teq,wmf]), \
                interp_b([cmf,teq,wmf]), \
                interp_c([cmf,teq,wmf]), \
                interp_d([cmf,teq,wmf])

    # Compute radius
    return 10**(a*np.log10(mp) + np.exp(-d*(np.log10(mp)+c)) + b)
