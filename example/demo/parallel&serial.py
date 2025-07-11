# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 22:40:54 2025

@author: Damian
"""

import numpy as np

# Set random seed for reproducibility (optional)
np.random.seed(42)

# Create matrix1: 5x10 of random floats in [1, 1000], rounded to 3 decimal places
matrix1 = np.round(np.random.uniform(1, 1000, size=(5, 10)), 3)

# Create matrix2: matrix1 multiplied element-wise by random factors in [0.8, 1.2]
multipliers = np.random.uniform(0.8, 1.2, size=(5, 10))
matrix2 = matrix1 * multipliers

# Compute expect1 and expect2
expect1 = matrix2 / matrix1 - 1  # parallel
expect2 = np.log(matrix2 / matrix1)  # serial, risk

# Display results
print("matrix1 (10x20):\n", matrix1)
print("\nmultipliers (10x20):\n", np.round(multipliers, 3))
print("\nmatrix2 (10x20):\n", np.round(matrix2, 3))
print("\nexpect1 = (matrix2 / matrix1 - 1):\n", np.round(expect1, 6))
print("\nexpect2 = log(matrix2 / matrix1):\n", np.round(expect2, 6))


a1 = 100
a2 = 200
a3 = 110
expect2 = np.log(a2 / a1)
expect3 = np.log(a3 / a2)
print(expect2, expect3)

expect22 = a2 / a1 
expect33 = a3 / a2 
print(expect22, expect33)