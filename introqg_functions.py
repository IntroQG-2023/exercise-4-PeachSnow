"""Functions used in the Introduction to Quantitative Geology course"""

# Import any modules needed in your functions here
import math
import numpy as np
import scipy
from scipy.special import erfc

# Define your new functions below
def mean(numbers):
    return sum(numbers) / len(numbers)

def stddev(numbers):
    mu = mean(numbers)
    variance = sum((x - mu) ** 2 for x in numbers) / len(numbers)
    return math.sqrt(variance)

from math import sqrt
def stderr(data):
    n = len(data)
    if n <= 1:
        raise ValueError("Standard error requires at least two data points")
    std_dev = stddev(data) 
    return std_dev / sqrt(n)

import numpy as np
def gaussian(mean, stddev, x_values):
    x_values = np.array(x_values)  
    constant = 1 / (stddev * np.sqrt(2 * np.pi))
    exponent = np.exp(-((x_values - mean) ** 2) / (2 * stddev ** 2))
    return constant * exponent

#But actually these two functions are not from Exercise 2 or Exercise 1, not like what stated in the Exercise 3, so we need to create new functions (?)
def linregress(x, y):
    """Calculate a linear least-squares regression for two sets of measurements."""
    # Basic statistics
    mean_x, mean_y = mean(x), mean(y)
    variance_x = sum((xi - mean_x) ** 2 for xi in x) / len(x)
    covariance_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / len(x)
    # Coefficients
    B = covariance_xy / variance_x
    A = mean_y - B * mean_x
    return A, B

def chi_squared(observed, expected, errors):
    """Calculate the chi-squared statistic."""
    observed = np.array(observed)
    expected = np.array(expected)
    errors = np.array(errors)  
    # Calculate the chi-squared value
    chi2 = np.sum(((observed - expected) / errors) ** 2)
    return chi2

from math import exp, log
def dodson(cooling_rate, activation_energy, diffusivity_inf, grain_radius, geometry_factor):
    # Define the gas constant in J/(mol K)
    R = 8.314 #R = 8.31446261815324 
    # Convert cooling rate from °C/Myr to K/s (Kelvin per second)
    # 1 Myr = 1e6 years, 1 year = 365.25 days, 1 day = 24 hours, 1 hour = 3600 seconds
    cooling_rate = cooling_rate / (1e6 * 365.25 * 24 * 3600)
    # Convert grain_radius from micrometers to meters
    grain_radius = grain_radius / 1e6
    # Initial guess Tc in Kelvin (modify as needed)
    Tc = 1000
    # Iterate to find the value of Tc that converges
    for i in range(10):  # Iterate up to (xxx) times
        # Calculate tau using Equation 4
        tau = -(R * Tc**2) / (activation_energy * cooling_rate)
        # Calculate Tc using Equation 3
        Tc_new = activation_energy / (R * log(geometry_factor * tau * diffusivity_inf / grain_radius**2))
        # Check if the value converges 
        if abs(Tc - Tc_new) < 0.001:
            break
        Tc = Tc_new 
    # Convert Tc to Celsius before returning
    Tc_celsius = Tc - 273.15
    # Rounding problems for test
    return Tc_celsius

def steady_state_temp(gradient, diffusivity, velocity, depths):# Calculate the steady-state temperature at a given depth
    """
    Calculate the steady-state temperature at a given depth
    """
    # Calculate temperatures
    temperatures = (gradient * diffusivity / velocity) * (1 - np.exp(-(velocity * depths) / diffusivity))
    return temperatures

def transient_temp(initial_gradient, diffusivity, velocity, depths, time):
    """
    Calculates the temperature at given depths and time using the time-dependent heat advection-diffusion equation.
    """
    # Pre-calculate common terms to simplify the equation
    velocity_term = velocity * time
    diffusivity_term = 2 * np.sqrt(diffusivity * time)
    # Convert units for gradient and depths
    # The units are already compatible in this case so no conversion is needed
    G = initial_gradient
    # Initialize the array for temperatures
    temperatures = np.zeros_like(depths)
    # Calculate temperatures at each depth
    for i, z in enumerate(depths):
        term1 = G * (z + velocity_term)
        term2 = ((z - velocity_term) * np.exp(-velocity * z / diffusivity))
        term3 = erfc((z - velocity_term) / diffusivity_term)
        term4 = (z + velocity_term) * erfc((z + velocity_term) / diffusivity_term)
        temperatures[i] = term1 + (G / 2) * (term2 * term3 - term4 )
    return temperatures
