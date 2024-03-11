#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:55:56 2024

@author: wi11iamk
"""
# Import necessary libraries

import h5py, numpy as np, pandas as pd, umap, matplotlib.pyplot as plt, seaborn as sns

from hubdt import data_loading, behav_session_params, wavelets, hub_utils, hub_analysis, t_sne, hdb_clustering, b_utils

from scipy.signal import find_peaks, peak_prominences

from scipy.stats import gaussian_kde

from scipy.spatial.distance import jensenshannon

#%%

# Initialise the HUB-DT session and import the DLC features and tracking data

mysesh = behav_session_params.load_session_params ('Mine')

features = data_loading.dlc_get_feats (mysesh)

del features[4:7] # Delete features, should you want to

tracking = data_loading.load_tracking (mysesh, dlc=True, feats=features)

#%%

scales, frequencies = wavelets.calculate_scales (0.75, 2, 120, 4)

proj = wavelets.wavelet_transform_np (tracking, scales, frequencies, 120)

#%%

# Transpose the projections array such that the new dimensions match Tracking array

proj = np.transpose(proj)

#%%

# Fit projection array as a UMAP object and store the embeddings into a variable with only 2(D) columns

mapper = umap.UMAP (n_neighbors=30, n_components=2, min_dist=0.1).fit(proj)

embed = mapper.embedding_

#%%

plt.scatter(embed[:, 0], embed[:, 1], s=0.5, c='blue', alpha=0.5)
            
#%%

# Calculate a density grid of the data set using KDE on the whole embedding and then iterate the calculation 
# for any number of slices of the embedding, i.e. trials, and then overlay the densities of the slices atop the whole density grid.
# Plot the density of the whole embedding and then each overlay, per trial, as a subplot

# Define the fixed grid based on the entire dataset
x_min, x_max = embed[:, 0].min(), embed[:, 0].max()
y_min, y_max = embed[:, 1].min(), embed[:, 1].max()
x_grid, y_grid = np.mgrid[x_min:x_max:292j, y_min:y_max:292j]  # 292j for a 292x292 grid
grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])

# Function to calculate density on the fixed grid
def calc_density_on_fixed_grid(data, grid_coords):
    kde = gaussian_kde(data.T)
    density = kde(grid_coords).reshape(x_grid.shape)
    return density

# Function to normalize and flatten a grid to form a probability vector
def normalize_grid(grid):
    flattened = grid.flatten()
    normalized = flattened / np.sum(flattened)
    return normalized

# Calculate the density for the entire dataset
entire_data_density = calc_density_on_fixed_grid(embed, grid_coords)

# Define slices (including an empty tuple for the entire dataset)
slices = [(), (0, 1200), (14400, 15600), (84508, 85828)]

# Initialize container to store the normalized probability vectors
probability_vectors = []

# Plotting
fig, axes = plt.subplots(1, len(slices), figsize=(20, 4))

for i, sl in enumerate(slices):
    ax = axes[i]
    # Plot the entire dataset's density map as background
    pcm = ax.pcolormesh(x_grid, y_grid, entire_data_density, shading='auto', cmap='viridis')
    fig.colorbar(pcm, ax=ax, label='Density')
    
    # Calculate and overlay the slice if specified
    if sl:  # For slices, overlay the slice data and include in legend
        slice_data = embed[sl[0]:sl[1], :]
        slice_density = calc_density_on_fixed_grid(slice_data, grid_coords)
        slice_vector = normalize_grid(slice_density)
        probability_vectors.append(slice_vector)  # Store the probability vector
        
        # Scatter plot for slice data with a label for legend
        ax.scatter(slice_data[:, 0], slice_data[:, 1], color='red', s=1, label=f'Slice {i} Data Points')
        ax.set_title(f'Slice {i}: {sl[0]}-{sl[1]}')
        ax.legend()  # Only call legend when there are labeled artists
    else:
        entire_vector = normalize_grid(entire_data_density)
        probability_vectors.insert(0, entire_vector)  # Ensure the entire dataset vector is first
        ax.set_title('Entire Dataset')
        # No call to ax.legend() here since there's no labeled artist for the entire dataset plot

plt.tight_layout()
plt.show()

#%%

clusterobj = hdb_clustering.hdb_scan (embed, 550, 55, selection='leaf', cluster_selection_epsilon=0.17)

labels = clusterobj.labels_

probabilities = clusterobj.probabilities_

fig = hdb_clustering.plot_hdb_over_tsne(embed, labels, probabilities, noise=False)

fig2 = hdb_clustering.plot_condensed_tree(clusterobj, select_clusts=True, label_clusts=True)

#%%

arraylabels = np.reshape(labels, (-1,1))

#%%

# At this point, determine the synergy of interest and first frame of the synergy using the arraylabels variable
# Enter the Frame value into the variable below

syn_frame_start = 450

fig3 = b_utils.plot_cluster_wav_mags(proj, labels, 13, features, frequencies, wave_correct=True, response_correct=True, mean_response=True, colour='lightblue')

#%%

#fig4 = hdb_clustering.plot_hdb_over_tsne(embed, labels, probabilities, compare_to=True, comp_label=20)

fig5 = b_utils.plot_curr_cluster(embed, density, syn_frame_start, xgrid, ygrid)

#book = b_utils.frame_curr_cluster(embed, density, frames, xi, yi, save_folder, dpi=100, width=800, height=600)(embed, density, xgrid, ygrid, 120, start_f = 740, end_f = 860)

#%%

# Sanity checks: With DLC .h5 file, detect frame-moment of each keypress, calculate onset and offset of each keypress, 
# print keypress counts per channel, and plot time series of pose with keypress onset, offset, and frame-moment markers.

# Path to the .h5 file
h5_path = '/Users/wi11iamk/Documents/GitHub/HUB_DT/sample_data/027_D1DLC_resnet50_keyTest027Jan12shuffle1_400000.h5'

# Assuming an order for the y-values of each channel within the DLC .h5 'values_block_0' array
little_y_idx = 1   
ring_y_idx = 4      
middle_y_idx = 7    
index_y_idx = 10    

with h5py.File(h5_path, 'r') as file:
    # Accessing the structured array from the table
    data = file['df_with_missing/table']['values_block_0'][syn_frame_start:(syn_frame_start+120), :]  # Extracting data for a range of frames
    
    # An array to include only the y-values for the specified channels
    y_values_combined = np.vstack((data[:, little_y_idx], data[:, ring_y_idx], data[:, middle_y_idx], data[:, index_y_idx])).T

keypress_counts = []  # To store the count of keypresses for each channel
fig, ax = plt.subplots(figsize=(14, 6))

# Define colors for each channel to distinguish in the plot
colors = ['blue', 'green', 'red', 'cyan']
channel_names = ['Little', 'Ring', 'Middle', 'Index']

for i, channel_name in enumerate(channel_names):
    # Detect keypresses with the specified prominence
    peaks, properties = find_peaks(y_values_combined[:, i], prominence=25)
    prominences = peak_prominences(y_values_combined[:, i], peaks)[0]
    
    # Store the count of keypresses
    keypress_counts.append(len(peaks))
    
    # Plotting the time series for each channel
    ax.plot(np.arange(syn_frame_start, syn_frame_start + y_values_combined.shape[0]), y_values_combined[:, i], label=channel_name, color=colors[i])
    
    # Adding red markers for the detected keypresses
    ax.plot(peaks + syn_frame_start, y_values_combined[peaks, i], 'r*', markersize=8)

# Properties of the plot
ax.set_title('Time Series of Channels with Detected Keypresses')
ax.set_xlabel('Frame Index')
ax.set_ylabel('Y Position')
ax.legend()

plt.show()

# Printing the count of detected keypresses for each channel
for i, channel_name in enumerate(channel_names):
    print(f"{channel_name} channel detected keypresses: {keypress_counts[i]}")


# Check the first y-value for each channel at 'syn_frame_start' for manual comparison, if desired
print("Y-values (in the order of little, ring, middle, index):")
print(f"Little: {y_values_combined[0, 0]}")  # First column in the combined array
print(f"Ring: {y_values_combined[0, 1]}")   # Second column in the combined array
print(f"Middle: {y_values_combined[0, 2]}") # Third column in the combined array
print(f"Index: {y_values_combined[0, 3]}")  # Fourth column in the combined array

#%%

# Calculate Average Range of Movement, average frame to frame velocity, for each channel, and plot

average_ranges_of_motion = []
average_velocities = []
std_devs_amplitude = []
std_devs_velocity = []

# Calculate Average Range of Motion for each channel and its standard deviation
for i in range(y_values_combined.shape[1]):
    peaks, _ = find_peaks(y_values_combined[:, i], prominence=25)
    if len(peaks) > 0:
        prominences = peak_prominences(y_values_combined[:, i], peaks)[0]
        average_range = np.mean(prominences)
        std_dev_amplitude = np.std(prominences)
    else:
        average_range = 0
        std_dev_amplitude = 0
    average_ranges_of_motion.append(average_range)
    std_devs_amplitude.append(std_dev_amplitude)

# Calculate Average Frame-to-Frame Velocity for each channel and its standard deviation
for i in range(y_values_combined.shape[1]):
    velocities = np.diff(y_values_combined[:, i])
    average_velocity = np.mean(np.abs(velocities))
    std_dev_velocity = np.std(np.abs(velocities))
    average_velocities.append(average_velocity)
    std_devs_velocity.append(std_dev_velocity)

# Create dataframes for plotting
channels = ['Little', 'Ring', 'Middle', 'Index']

# Amplitude Plot Data
df_amplitude = pd.DataFrame({
    'Channel': channels,
    'Average Amplitude': average_ranges_of_motion,
    'STD': std_devs_amplitude
})

# Velocity Plot Data
df_velocity = pd.DataFrame({
    'Channel': channels,
    'Average Velocity': average_velocities,
    'STD': std_devs_velocity
})

# Plotting Average Amplitude for each channel
plt.figure(figsize=(10, 6))
sns.barplot(x='Channel', y='Average Amplitude', data=df_amplitude, capsize=.1, palette='viridis')
plt.errorbar(x=range(len(channels)), y=df_amplitude['Average Amplitude'], yerr=df_amplitude['STD'], fmt='none', c='black', capsize=5)
plt.title('Average Keypress Amplitudes for Each Channel')
plt.ylabel('Average Amplitude')
plt.xlabel('Channel')
plt.show()

# Plotting Average Velocity for each channel
plt.figure(figsize=(10, 6))
sns.barplot(x='Channel', y='Average Velocity', data=df_velocity, capsize=.1, palette='viridis')
plt.errorbar(x=range(len(channels)), y=df_velocity['Average Velocity'], yerr=df_velocity['STD'], fmt='none', c='black', capsize=5)
plt.title('Average Velocities for Each Channel')
plt.ylabel('Average Velocity')
plt.xlabel('Channel')
plt.show()

#%%

# Flatten KDE grids to vectors and normalise such that the vector values sum to 1, 
# then measure distance between vectors using Jensen-Shannon Divergence. 





