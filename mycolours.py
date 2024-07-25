#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:17:04 2024

@author: wi11iamk
"""

import matplotlib.pyplot as plt
import numpy as np

def adjust_brightness(color, brightness_factor):
    # Calculate the adjusted color
    if brightness_factor > 0:
        adjusted_color = color + (np.array([1.0, 1.0, 1.0]) - color) * brightness_factor
    else:
        adjusted_color = color * (1 + brightness_factor)
    return np.clip(adjusted_color, 0, 1)  # Ensure the values are within [0, 1]

def custom_colour_list():
    colours = plt.cm.Set1.colors  # Get all colours from a chosen space
    selected_indices = [0, 1, 2, 3, 4, 5, 7]
    base_colors = [colours[i] for i in selected_indices]

    custom_colors = []
    brightness_levels = [-0.2, -0.1, 0, 0.3, 0.6]
    
    for brightness_factor in brightness_levels:
        for color in base_colors:
            adjusted_color = adjust_brightness(np.array(color), brightness_factor)
            custom_colors.append(adjusted_color)
    
    return custom_colors

# Create the custom colourmap
custom_colors = custom_colour_list()
custom_colormap = plt.cm.colors.ListedColormap(custom_colors)

# Example plot to visualise the colourmap
x = np.random.rand(240)
y = np.random.rand(240)
colors = np.linspace(0, 1, 240)  # Use a continuous range of values mapped to the colourmap

plt.scatter(x, y, c=colors, cmap=custom_colormap, alpha=0.6)
plt.colorbar()  # Show the custom colourmap scale
plt.show()
