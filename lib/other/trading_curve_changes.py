# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:54:37 2023

@author: awei
"""

import numpy as np
from scipy.interpolate import CubicSpline

def simulate_function(start, end, maximum, minimum, num_points=100):
    # Generate x values
    x_values = np.linspace(0, 1, num_points)

    # Generate y values based on a cubic Hermite spline
    y_values = (1 - 3*x_values**2 + 2*x_values**3) * start + (3*x_values**2 - 2*x_values**3) * end + (x_values - 2*x_values**2 + x_values**3) * maximum + (-x_values**2 + x_values**3) * minimum

    return x_values, y_values

# Example usage:
start_value = 0
end_value = 1
maximum_value = 2
minimum_value = -1

x_vals, y_vals = simulate_function(start_value, end_value, maximum_value, minimum_value)

# Plot the simulated function
import matplotlib.pyplot as plt

plt.plot(x_vals, y_vals)
plt.title('Simulated Function')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
