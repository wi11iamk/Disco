#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:55:56 2024

@author: wi11iamk
"""
#%%

import h5py

import numpy as np

import pandas as pd

from hubdt import data_loading

from hubdt import behav_session_params

from hubdt import wavelets

import umap

from hubdt import hub_utils, hub_analysis

from hubdt import t_sne

import matplotlib.pyplot as plt

import seaborn as sns

from hubdt import hdb_clustering

from hubdt import b_utils

from scipy.signal import find_peaks, peak_prominences

#%%

mysesh = behav_session_params.load_session_params ('Mine')

features = data_loading.dlc_get_feats (mysesh)

del features[4:7]

tracking = data_loading.load_tracking (mysesh, dlc=True, feats=features)

#%%

scales, frequencies = wavelets.calculate_scales (0.75, 1.5, 120, 4)

proj = wavelets.wavelet_transform_np (tracking, scales, frequencies, 120)

#%%

proj = np.transpose(proj)

#%%

mapper = umap.UMAP (n_neighbors=35, n_components=2, min_dist=0.0).fit(proj)

embed = mapper.embedding_

#%%

density, xgrid, ygrid = t_sne.calc_density(embed)

#%%

fig, ax = plt.subplots()

plt.pcolormesh(xgrid, ygrid, density)

#%%

clusterobj = hdb_clustering.hdb_scan (embed, 550, 55, selection='leaf', cluster_selection_epsilon=0.17)

labels = clusterobj.labels_

probabilities = clusterobj.probabilities_

fig = hdb_clustering.plot_hdb_over_tsne(embed, labels, probabilities, noise=False)

fig2 = hdb_clustering.plot_condensed_tree(clusterobj, select_clusts=True, label_clusts=True)

#%%

arraylabels = np.reshape(labels, (-1,1))

fig3 = b_utils.plot_cluster_wav_mags(proj, labels, 19, features, frequencies, wave_correct=True, response_correct=True, mean_response=True, colour='lightblue')

#fig4 = hdb_clustering.plot_hdb_over_tsne(embed, labels, probabilities, compare_to=True, comp_label=20)

fig5 = b_utils.plot_curr_cluster(embed, density, 400, xgrid, ygrid)

#book = b_utils.frame_curr_cluster(embed, density, frames, xi, yi, save_folder, dpi=100, width=800, height=600)(embed, density, xgrid, ygrid, 120, start_f = 740, end_f = 860)

#%%

# Path to the .h5 file
h5_path = '/Users/wi11iamk/Documents/GitHub/HUB_DT/sample_data/027_D1DLC_resnet50_keyTest027Jan12shuffle1_400000.h5'

# Assuming an order for the y-values of each channel within the 'values_block_0' array
little_y_idx = 1   
ring_y_idx = 4      
middle_y_idx = 7    
index_y_idx = 10    

with h5py.File(h5_path, 'r') as file:
    # Accessing the structured array from the table
    data = file['df_with_missing/table']['values_block_0'][400:520, :]  # Extracting data for a range of frames
    
    # An array to include only the y-values for the specified channels
    y_values_combined = np.vstack((data[:, little_y_idx], data[:, ring_y_idx], data[:, middle_y_idx], data[:, index_y_idx])).T

# Sanity checks: Detect keypresses, calculate average range of motion, standard deviations, 
# print keypress counts, and plot time series with keypress markers.

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
    ax.plot(np.arange(400, 400 + y_values_combined.shape[0]), y_values_combined[:, i], label=channel_name, color=colors[i])
    
    # Adding red markers for the detected keypresses
    ax.plot(peaks + 400, y_values_combined[peaks, i], 'r*', markersize=8)

# Enhancing the plot
ax.set_title('Time Series of Channels with Detected Keypresses')
ax.set_xlabel('Frame Index')
ax.set_ylabel('Y Position')
ax.legend()

plt.show()

# Printing the count of detected keypresses for each channel
for i, channel_name in enumerate(channel_names):
    print(f"{channel_name} channel detected keypresses: {keypress_counts[i]}")


# Check the first y-value for each channel at frame index __ for comparison
print("Y-values at frame index 400 (in the order of little, ring, middle, index):")
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



