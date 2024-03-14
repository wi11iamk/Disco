#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:55:56 2024

@author: wi11iamk
"""
#%%

# Import necessary libraries

import h5py, numpy as np, pandas as pd, umap, matplotlib.pyplot as plt, seaborn as sns

from hubdt import data_loading, behav_session_params, wavelets, hub_utils, hub_analysis, t_sne, hdb_clustering, b_utils

from scipy.signal import find_peaks, peak_prominences

from scipy.stats import gaussian_kde

from scipy.spatial.distance import jensenshannon

#%%

# Initialise the HUB-DT session; import all DLC features and tracking data

mysesh = behav_session_params.load_session_params ('Mine')

features = data_loading.dlc_get_feats (mysesh)

del features[4:7] # Delete features, should you want to

tracking = data_loading.load_tracking (mysesh, dlc=True, feats=features)

#%%

scales, frequencies = wavelets.calculate_scales (0.75, 2.25, 120, 4)

proj = wavelets.wavelet_transform_np (tracking, scales, frequencies, 120)

#%%

# Transpose projection array to match dimensions of tracking array

proj = np.transpose(proj)

#%%

# Fit projection array as a UMAP object and store the embeddings into a variable with only 2(D) columns

mapper = umap.UMAP (n_neighbors=30, n_components=2, min_dist=0.1).fit(proj)

embed = mapper.embedding_

#%%

plt.scatter(embed[:, 0], embed[:, 1], s=0.25, c='blue', alpha=0.25)
            
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
fig, axes = plt.subplots(1, len(slices), figsize=(24, 4))

for i, sl in enumerate(slices):
    ax = axes[i]
    # Plot the entire dataset's density map as background
    pcm = ax.pcolormesh(x_grid, y_grid, entire_data_density, shading='auto', cmap='plasma')
    fig.colorbar(pcm, ax=ax, label='Density')
    
    # Calculate and overlay the slice if specified
    if sl:  # For slices, overlay the slice data and include in legend
        slice_data = embed[sl[0]:sl[1], :]
        slice_density = calc_density_on_fixed_grid(slice_data, grid_coords)
        slice_vector = normalize_grid(slice_density)
        probability_vectors.append(slice_vector)  # Store the probability vector
        
        # Scatter plot for slice data with a label for legend
        ax.scatter(slice_data[:, 0], slice_data[:, 1], color='deepskyblue', s=0.75, alpha=0.33, label=f'Slice {i} Data Points')
        ax.set_title(f'Slice {i}: {sl[0]}-{sl[1]}')
        ax.legend()  # Only call legend when there are labeled artists
    else:
        entire_vector = normalize_grid(entire_data_density)
        probability_vectors.insert(0, entire_vector)  # Ensure the entire dataset vector is first
        ax.set_title('Entire Dataset')

# Ensure that each probability vector stored in the list sums to 1 within a specified tolerance
for i, vector in enumerate(probability_vectors):
    vector_sum = np.sum(vector)
    if np.isclose(vector_sum, 1.0, atol=1e-8):  # atol is the tolerance level
        print(f"Vector {i} sums to 1.0 within tolerance.")
    else:
        print(f"Vector {i} does not sum to 1.0; sum is {vector_sum}.")


plt.tight_layout()
plt.show()

#%%

clusterobj = hdb_clustering.hdb_scan (embed, 500, 50, selection='leaf', cluster_selection_epsilon=0.15)

labels = clusterobj.labels_

probabilities = clusterobj.probabilities_

fig1 = hdb_clustering.plot_condensed_tree(clusterobj, select_clusts=True, label_clusts=True)

fig2 = hdb_clustering.plot_hdb_over_tsne(embed, labels, probabilities, noise=False)

#%%

# Calcuate the average synergy length, min, max, number of uses, total time in use, and percent of time used

arraylabels = np.reshape(labels, (-1,1))

def calculate_label_data(arr):
    continuous_counts = {}
    current_val = None
    count = 0
    start_frame = 0  # To track the starting frame of the current sequence
    
    for idx, val in enumerate(arr.flatten()):  # Iterate with index to track frame numbers
        if val != -1:
            if val == current_val:
                count += 1
            else:
                if current_val is not None:
                    if current_val in continuous_counts:
                        continuous_counts[current_val]['counts'].append((count, start_frame))
                    else:
                        continuous_counts[current_val] = {'counts': [(count, start_frame)], 'total_frames': 0}
                current_val = val
                count = 1
                start_frame = idx  # Update start frame for the new sequence
        else:
            if current_val is not None:
                if current_val in continuous_counts:
                    continuous_counts[current_val]['counts'].append((count, start_frame))
                else:
                    continuous_counts[current_val] = {'counts': [(count, start_frame)], 'total_frames': 0}
            current_val = None
            count = 0

    if current_val is not None and count > 0:
        if current_val in continuous_counts:
            continuous_counts[current_val]['counts'].append((count, start_frame))
        else:
            continuous_counts[current_val] = {'counts': [(count, start_frame)], 'total_frames': 0}

    total_all_frames = 0
    for val, data in continuous_counts.items():
        counts = [c[0] for c in data['counts']]
        start_frames = [c[1] for c in data['counts']]
        total_frames = sum(counts)
        max_count = max(counts)
        max_count_index = counts.index(max_count)
        max_count_start_frame = start_frames[max_count_index]
        continuous_counts[val]['avg'] = np.mean(counts)
        continuous_counts[val]['min'] = min(counts)
        continuous_counts[val]['max'] = max_count
        continuous_counts[val]['max_frame'] = max_count_start_frame  # The frame where the max count starts
        continuous_counts[val]['length'] = len(counts)
        continuous_counts[val]['total_frames'] = total_frames
        total_all_frames += total_frames
    
    # Calculate the percentage of total frames for each int
    for val in continuous_counts:
        continuous_counts[val]['percent'] = (continuous_counts[val]['total_frames'] / total_all_frames) * 100
    
    # Sort the results by int value and prepare the final list of tuples
    sorted_results = sorted([(k, v['avg'], v['min'], v['max'], v['max_frame'], v['length'], v['total_frames'], v['percent']) 
                             for k, v in continuous_counts.items()])
    
    return sorted_results

# Example usage with the 'arraylabels' variable
arraylabels_data = calculate_label_data(arraylabels)

#%%

# At this point, determine your synergy of interest, including the start and end frames
# Consult the arraylabels_data list for information about each synergy

fig3 = b_utils.plot_cluster_wav_mags(proj, labels, 18, features, frequencies, wave_correct=True, response_correct=True, mean_response=True, colour='lightblue')

syn_frame_start = 70030
syn_frame_end = 70167

#%%

#fig4 = hdb_clustering.plot_hdb_over_tsne(embed, labels, probabilities, compare_to=True, comp_label=8)

fig5 = b_utils.plot_curr_cluster(embed, entire_data_density, syn_frame_start, x_grid, y_grid)

#%%

# Calculate and plot the frame of each keypress; calculate onset and offset of each keypress

# Path to the .h5 file
h5_path = '/Users/wi11iamk/Documents/GitHub/HUB_DT/sample_data/027_D1DLC_resnet50_keyTest027Jan12shuffle1_400000.h5'
# Write in the index location of each feature y-value within the 'values_block_0' .h5 table
little_y_idx = 1
ring_y_idx = 4
middle_y_idx = 7
index_y_idx = 10

y_threshold = 460  # A keypress cannot be detected beneath this value

with h5py.File(h5_path, 'r') as file:
    # Accessing the structured array from the table for the specified frame range
    data = file['df_with_missing/table']['values_block_0'][syn_frame_start:syn_frame_end, :]
    # An array to include only the y-values for the specified channels
    y_values_combined = np.vstack((data[:, little_y_idx], data[:, ring_y_idx], data[:, middle_y_idx], data[:, index_y_idx])).T

def find_onset_offset(derivative, peak_index, window=15):
    onset_index = peak_index - window
    offset_index = peak_index + window
    return max(0, onset_index), min(len(derivative) - 1, offset_index)

keypress_counts = []  # To store the count of keypresses for each channel
onset_offset_data = {channel: [] for channel in ['Little', 'Ring', 'Middle', 'Index']}  # To store onset and offset frames

fig, ax = plt.subplots(figsize=(14, 6))
colors = ['blue', 'green', 'red', 'cyan']  # Colors for each channel
channels = ['Little', 'Ring', 'Middle', 'Index']

for i, channel_name in enumerate(channels):
    derivative = np.diff(y_values_combined[:, i], prepend=y_values_combined[0, i])
    peaks, _ = find_peaks(y_values_combined[:, i], prominence=26)
    valid_peaks = [peak for peak in peaks if y_values_combined[peak, i] > y_threshold]  # Filter peaks based on threshold

    keypress_counts.append(len(valid_peaks))
    ax.plot(np.arange(syn_frame_start, syn_frame_start + len(y_values_combined[:, i])), y_values_combined[:, i], label=channel_name, color=colors[i])
    for peak in valid_peaks:
        adjusted_peak = peak + syn_frame_start  # Correctly adjust peak index for plotting
        ax.plot(adjusted_peak, y_values_combined[peak, i], 'r*', markersize=8)
        onset, offset = find_onset_offset(derivative, peak)
        ax.plot(onset + syn_frame_start, y_values_combined[onset, i], 'go')  # Green for onset
        ax.plot(offset + syn_frame_start, y_values_combined[offset, i], 'mo')  # Magenta for offset
        onset_offset_data[channel_name].append((onset + syn_frame_start, offset + syn_frame_start))

ax.set_title('Time Series of Channels with Detected Keypresses')
ax.set_xlabel('Frame Index')
ax.set_ylabel('Y Position')
ax.legend()
plt.show()

for i, channel_name in enumerate(channels):
    print(f"{channel_name} channel detected keypresses: {keypress_counts[i]}")

print("Y-values at frame {}:".format(syn_frame_start))
for i, channel_name in enumerate(channels):
    print(f"{channel_name}: {y_values_combined[0, i]}")

#%%

# Calculate and plot range of movement, average frame to frame velocity, and overlap of movements for each channel

average_velocities = []
std_devs_amplitude = []
std_devs_velocity = []

# Initialize an empty list to store all keypress events
all_keypress_events = []

# Collect keypress events for each channel, applying the threshold
for i, channel in enumerate(channels):
    peaks, _ = find_peaks(y_values_combined[:, i], prominence=26)
    # Filter peaks based on the y-value threshold
    valid_peaks = [peak for peak in peaks if y_values_combined[peak, i] > y_threshold]
    for peak in valid_peaks:
        onset, offset = find_onset_offset(np.diff(y_values_combined[:, i], prepend=y_values_combined[0, i]), peak)
        # Store events with their adjusted frame number
        all_keypress_events.append((onset + syn_frame_start, 'onset', channel, peak))
        all_keypress_events.append((peak + syn_frame_start, 'peak', channel, peak))
        all_keypress_events.append((offset + syn_frame_start, 'offset', channel, peak))

# Sort the events list by peak_frame primarily, and then by frame to maintain the sequence
all_keypress_events_sorted = sorted(all_keypress_events, key=lambda x: (x[3], x[0]))

# Calculate the range of motion for each channel
ranges_of_motion = []
for i, channel in enumerate(channels):
    channel_data = y_values_combined[:, i]
    range_of_motion = np.max(channel_data) - np.min(channel_data)
    ranges_of_motion.append(range_of_motion)

# Calculate average frame-to-frame velocity for each channel and its standard deviation
for i in range(y_values_combined.shape[1]):
    velocities = np.diff(y_values_combined[:, i])
    average_velocity = np.mean(np.abs(velocities))
    std_dev_velocity = np.std(np.abs(velocities))
    average_velocities.append(average_velocity)
    std_devs_velocity.append(std_dev_velocity)
    
# Overlap calculation: Sequentially compare offset to the next onset within the combined and sequential list
# Initialize variable to count total frames involved in overlaps
total_overlap_frames = 0
for i in range(len(all_keypress_events_sorted) - 1):
    current_event = all_keypress_events_sorted[i]
    next_event = all_keypress_events_sorted[i + 1]

    if current_event[1] == 'offset' and next_event[1] == 'onset':
        overlap = current_event[0] - next_event[0]
        if overlap > 0:
            total_overlap_frames += overlap

# Calculate percent overlap
percent_overlap = total_overlap_frames / len(y_values_combined) * 100

# Amplitude plot data
df_range_of_motion = pd.DataFrame({
    'Channel': channels,
    'Range of Motion': ranges_of_motion
})

# Velocity plot data
df_velocity = pd.DataFrame({
    'Channel': channels,
    'Average Velocity': average_velocities,
    'STD': std_devs_velocity
})

# Plotting the range of motion for each channel
plt.figure(figsize=(10, 6))
sns.barplot(x='Channel', y='Range of Motion', data=df_range_of_motion, palette='viridis')
plt.title('Range of Motion for Each Channel')
plt.ylabel('Range of Motion')
plt.xlabel('Channel')
plt.show()

# Plot average velocity for each channel
plt.figure(figsize=(10, 6))
sns.barplot(x='Channel', y='Average Velocity', data=df_velocity, capsize=.1, palette='viridis')
plt.errorbar(x=range(len(channels)), y=df_velocity['Average Velocity'], yerr=df_velocity['STD'], fmt='none', c='black', capsize=5)
plt.title('Average Velocities for Each Channel')
plt.ylabel('Average Velocity')
plt.xlabel('Channel')
plt.show()

# Plot percent of frames detected as overlap
plt.figure(figsize=(6, 4))
sns.barplot(x=['Total Overlap'], y=[percent_overlap], palette='viridis')
plt.title('Total Percent Overlap Among Channels')
plt.ylabel('Percent Overlap')
plt.ylim(0, 100)  # Assuming percent values, adjust if necessary
plt.show()
# Display the result
print(f"Total percent overlap among channels: {percent_overlap:.2f}%")

#%%

# TODO Calculate distance between vectors using Jensen-Shannon Divergence. 
# TODO     Calculate sub-slices of trial embeddings to then calculate divergence within and between trials
# TODO Calculate statistical differences between vectors using Kolomgorov-Smirnov Test
