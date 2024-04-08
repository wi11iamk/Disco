#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:17:04 2024

@author: wi11iamk
"""

import matplotlib.pyplot as plt
import numpy as np

def adjust_brightness(color, brightness_factor):
    # Ensure the brightness_factor is in the range [0, 1] to avoid exceeding RGB bounds
    brightness_factor = min(brightness_factor, 1.0)
    # Calculate the adjusted color
    adjusted_color = color + (np.array([1.0, 1.0, 1.0]) - color) * brightness_factor
    return adjusted_color

def custom_colour_list():
    colours = plt.cm.Set1.colors  # Get all colors from tab10
    selected_indices = [0, 1, 2, 3, 4, 5, 7]
    base_colors = [colours[i] for i in selected_indices]

    custom_colors = []
    for i in range(3):  # Original + 3 levels of brightness increase
        brightness_increase = 0.3 * i
        for color in base_colors:
            brighter_color = adjust_brightness(np.array(color), brightness_increase)
            custom_colors.append(brighter_color)
    
    return custom_colors

# Create the custom colormap
custom_colors = custom_colour_list()
custom_colormap = plt.cm.colors.ListedColormap(custom_colors)

# Example plot to visualize the colormap
x = np.random.rand(240)
y = np.random.rand(240)
colors = np.linspace(0, 1, 240)  # Use a continuous range of values mapped to the colormap

plt.scatter(x, y, c=colors, cmap=custom_colormap, alpha=0.6)
plt.colorbar()  # Show the custom colormap scale
plt.show()
