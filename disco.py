#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:55:56 2024

@author: wi11iamk
"""
#%%

###
# Import necessary libraries
###

import h5py, numpy as np, pandas as pd, umap, matplotlib.pyplot as plt, seaborn as sns

from hubdt import data_loading, behav_session_params, wavelets, hdb_clustering, b_utils

from scipy.integrate import simps

from scipy.signal import find_peaks

from scipy.spatial.distance import jensenshannon

from scipy.stats import gaussian_kde, ks_2samp

#%%

###
# Initialise the HUB-DT session; import all DLC features and tracking data
###

mysesh = behav_session_params.load_session_params ('Mine')

features = data_loading.dlc_get_feats (mysesh)

del features[4:7] # Delete features, should you want to

tracking = data_loading.load_tracking (mysesh, dlc=True, feats=features)

#%%

###
# Generate scales and frequencies for wavelet transform of the tracking data;
# store the wavelet projection into a variable and then transpose the output
###

scales, frequencies = wavelets.calculate_scales (0.75, 2.25, 120, 4)

proj = wavelets.wavelet_transform_np (tracking, scales, frequencies, 120)

proj = np.transpose(proj)

#%%

###
# Fit wavelet projection into two dimensional embedded space (UMAP); plot
###

mapper = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0.1).fit(proj)

embed = mapper.embedding_

plt.scatter(embed[:, 0], embed[:, 1], s=0.25, c='blue', alpha=0.25)
            
#%%

###
# Calculate and plot a gaussian KDE over embedded data; calculate a list of
# slices from the embedding and plot each as an overlay atop the gaussian KDE;
# calculate the Jensen-Shannon divergence and Kolmogorov-Smirnov statistic for
# each slice as a probability vector or data sample, respectively
###

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

# Initialize container to store normalized probability vectors
probability_vectors = []
# Initialize a list to store slice_data segments for KS test
slice_data_segments = []

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
        slice_data_segments.append(slice_data)
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
        
plt.tight_layout()
plt.show()

# Ensure that each probability vector stored in the list sums to 1 within a specified tolerance
for i, vector in enumerate(probability_vectors):
    vector_sum = np.sum(vector)
    if np.isclose(vector_sum, 1.0, atol=1e-8):  # atol is the tolerance level
        print(f"Vector {i} sums to 1.0 within tolerance.")
    else:
        print(f"Vector {i} does not sum to 1.0; sum is {vector_sum}.")
        
# Calculate pairwise JS divergence test between listed probability_vectors
for i in range(1, len(probability_vectors) - 1):  # Start from 1 to skip the first vector and go to len-1 for pairwise
    js_divergence = jensenshannon(probability_vectors[i], probability_vectors[i + 1])
    print(f"Jansen-Shannon Divergence between Vector {i} and Vector {i + 1}: {js_divergence}")

# Calculate pairwise KS tests between listed slice_data_segments
for i in range(len(slice_data_segments) - 1):
    # Perform KS test between consecutive slice_data segments
    ks_stat, p_value = ks_2samp(slice_data_segments[i].ravel(), slice_data_segments[i + 1].ravel())
    
    # Determine whether to reject the null hypothesis
    if p_value <= 0.05:
        decision = "reject the null hypothesis (the distributions are different)"
    else:
        decision = "fail to reject the null hypothesis (no significant difference between the distributions)"
    
    # Print the results, including the decision
    print(f"KS Statistic between Slice {i+1} and Slice {i+2}: {ks_stat}, P-value: {p_value}. We {decision}.")

#%%

###
# Perform HDBSCAN clustering to obtain labels and probabilities from embedded
# data; plot cluster distance tree; plot cluster labels atop embedded data
###

clusterobj = hdb_clustering.hdb_scan(embed, 500, 50, selection='leaf', cluster_selection_epsilon=0.15)

labels = clusterobj.labels_

probabilities = clusterobj.probabilities_

fig1 = hdb_clustering.plot_condensed_tree(clusterobj, select_clusts=True, label_clusts=True)

fig2 = hdb_clustering.plot_hdb_over_tsne(embed, labels, probabilities, noise=False)

#%%

### 
# Calcuate the mean length in frames of each label; store the first frame of
# the first five occurrences of each label in a list of tuples; calculate total
# uses and percent of time in use for each label
###

a_labels = np.reshape(labels, (-1, 1))

# Function to tally continuous label occurrences, allowing for a threshold
def calculate_label_data(arr, threshold=0):
    continuous_counts = {}
    current_val = None
    count = 0
    start_frame = 0
    minus_one_counter = 0  # Counter for -1 values
    
    arr = arr.flatten()  # Ensure the array is flattened for iteration
    
    for idx, val in enumerate(arr):
        if val != -1:
            if val == current_val or minus_one_counter <= threshold and current_val is not None:
                if minus_one_counter <= threshold and current_val is not None:
                    count += minus_one_counter  # Include -1s within threshold in the count
                count += 1
                minus_one_counter = 0  # Reset the -1 counter
            else:
                if current_val is not None:
                    finalize_count(continuous_counts, current_val, count, start_frame)
                current_val = val
                count = 1
                start_frame = idx
        else:
            if minus_one_counter < threshold:
                minus_one_counter += 1  # Tolerate -1 within the threshold
            else:
                if current_val is not None:
                    finalize_count(continuous_counts, current_val, count, start_frame)
                    current_val = None
                    count = 0
                minus_one_counter = 0  # Reset -1 counter
    
    # Finalize the last sequence
    if current_val is not None and count > 0:
        finalize_count(continuous_counts, current_val, count, start_frame)
    
    process_counts(continuous_counts)
    
    # Prepare sorted results
    sorted_results = sorted([(k,) + v['stats'] for k, v in continuous_counts.items()], key=lambda x: x[0])
    
    return sorted_results

# Function to update or initialize label data in 'continuous_counts'
def finalize_count(continuous_counts, val, count, start_frame):
    if val not in continuous_counts:
        continuous_counts[val] = {'counts': [], 'start_frames': []}
    continuous_counts[val]['counts'].append(count)
    continuous_counts[val]['start_frames'].append(start_frame)

# Function to compute statistics for each label based on its counts
def process_counts(continuous_counts):
    total_all_frames = sum(sum(c for c in v['counts']) for v in continuous_counts.values())
    for val, data in continuous_counts.items():
        counts = data['counts']
        start_frames = data['start_frames']
        mean_count = np.mean(counts)
        first_five = start_frames[:5]

        data['stats'] = (
            mean_count,  # Mean length in frames
            *first_five,  # The first frames of the first five occurrences
            len(counts),  # Total number of occurrences
            (sum(counts) / total_all_frames) * 100  # % of total frames
        )

a_labels_data = calculate_label_data(a_labels, threshold=15)

#%%

###
# Consult the a_labels_data list for information about each synergy; determine
# your synergy of interest; enter start and end frames of the synergy; 
# timeseries calculations are based on this range
###

syn_frame_start = 57955
syn_frame_end = (57955+130)

fig3 = b_utils.plot_cluster_wav_mags(proj, labels, 8, features, frequencies, wave_correct=True, response_correct=True, mean_response=True, colour='lightblue')

#fig4 = hdb_clustering.plot_hdb_over_tsne(embed, labels, probabilities, compare_to=True, comp_label=8)

fig5 = b_utils.plot_curr_cluster(embed, entire_data_density, syn_frame_start, x_grid, y_grid)

#%%

###
# Calculate and plot the time series of each channel, the frame of each 
# keypress, the onset and offset of each keypress; calculate and plot each 
# channel to a normalised frame; calculate integrals of each channel
###

h5_path = '/Users/wi11iamk/Documents/GitHub/HUB_DT/sample_data/027_D1DLC_resnet50_keyTest027Jan12shuffle1_400000.h5'

# Define channels and colors for each channel
channels = ['Little', 'Ring', 'Middle', 'Index']
colors = ['blue', 'green', 'red', 'cyan']

# Function to find the onset and offset given a peak
def find_onset_offset(derivative, peak_index, window=15):
    onset_index = max(0, peak_index - window)
    offset_index = min(len(derivative) - 1, peak_index + window)
    return onset_index, offset_index

# Function to normalize peak frequency to Hz for the given window
def normalize_peaks_to_hz(total_peaks, frames_in_window, frame_rate=120):
    duration_seconds = frames_in_window / frame_rate
    frequency_hz = total_peaks / duration_seconds
    return frequency_hz

# Access and process the .h5 data
with h5py.File(h5_path, 'r') as file:
    data = file['df_with_missing/table']['values_block_0'][syn_frame_start:syn_frame_end, :]
    y_values_combined = np.vstack((data[:, 1], data[:, 4], data[:, 7], data[:, 10])).T

normalized_y_values = y_values_combined - y_values_combined[0, :]
positive_y_values = np.abs(normalized_y_values)

area_under_curve = [simps(positive_y_values[:, i]) for i in range(positive_y_values.shape[1])]
total_area = sum(area_under_curve)
normalized_areas_percent = [(area / total_area) * 100 for area in area_under_curve]

total_peaks_across_channels = 0
keypress_counts = [0] * 4
y_threshold = 460 # keypresses cannot be detected beneath this value
onset_offset_data = {channel: [] for channel in channels}
fig, ax = plt.subplots(figsize=(14, 6))
x_range = np.arange(syn_frame_start, syn_frame_start + len(y_values_combined))

for i, channel_name in enumerate(channels):
    derivative = np.diff(y_values_combined[:, i], prepend=y_values_combined[0, i])
    peaks, _ = find_peaks(y_values_combined[:, i], prominence=26)
    valid_peaks = [peak for peak in peaks if y_values_combined[peak, i] > y_threshold]  # Filter peaks based on threshold

    keypress_counts[i] = len(valid_peaks)
    total_peaks_across_channels += len(valid_peaks)
    ax.plot(x_range, y_values_combined[:, i], label=channel_name, color=colors[i])
    for peak in valid_peaks:
        adjusted_peak = peak + syn_frame_start
        ax.plot(adjusted_peak, y_values_combined[peak, i], 'r*', markersize=8)
        onset, offset = find_onset_offset(derivative, peak)
        ax.plot(onset + syn_frame_start, y_values_combined[onset, i], 'go')
        ax.plot(offset + syn_frame_start, y_values_combined[offset, i], 'mo')
        onset_offset_data[channel_name].append((onset + syn_frame_start, offset + syn_frame_start))
        
# Calculate normalized frequency in Hz
normalized_frequency_hz = normalize_peaks_to_hz(total_peaks_across_channels, syn_frame_end - syn_frame_start + 1)

ax.set_title('Time Series of Channels with Detected Keypresses')
ax.set_xlabel('Frame Index')
ax.set_ylabel('Y Position')
ax.text(0.3, 0.1, s=f"Normalized Frequency: {normalized_frequency_hz: .3f} Hz", transform=ax.transAxes, ha='center', va='center', color='red')
ax.legend()
plt.show()

# Plot normalized and non-negative curves
fig2, ax2 = plt.subplots(figsize=(14, 6))
for i, channel_name in enumerate(channels):
    ax2.plot(x_range, positive_y_values[:, i], label=channel_name, color=colors[i])
ax2.set_title('Normalized and Non-negative Curves')
ax2.set_xlabel('Frame Index')
ax2.set_ylabel('Adjusted Y Position')
ax2.legend()
plt.show()

# Plot the normalized areas as a percentage for each channel
df_normalized_areas = pd.DataFrame({
    'Channel': channels,
    'Normalized Area (%)': normalized_areas_percent
})
plt.figure(figsize=(10, 6))
sns.barplot(x='Channel', y='Normalized Area (%)', data=df_normalized_areas, palette='viridis')
plt.title('Normalized Area Contribution of Each Channel')
plt.xlabel('Channel')
plt.ylabel('Normalized Area (%)')
plt.show()

#%%

###
# Calculate and plot average and peak frame to frame velocity and acceleration
# for each channel; calculate and plot overlap of movements over all channels
###

average_velocities = []
std_devs_velocity = []
peak_velocities = []

average_accelerations = []
std_devs_acceleration = []
peak_accelerations = []

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

# Calculate average and peak frame-to-frame velocity and acceleration for each channel
for i in range(y_values_combined.shape[1]):
    velocities = np.diff(y_values_combined[:, i])
    accelerations = np.diff(velocities)
    
    average_velocity = np.mean(np.abs(velocities))
    std_dev_velocity = np.std(np.abs(velocities))
    peak_velocity = np.max(np.abs(velocities))
    
    average_acceleration = np.mean(np.abs(accelerations))
    std_dev_acceleration = np.std(np.abs(accelerations))
    peak_acceleration = np.max(np.abs(accelerations))
    
    average_velocities.append(average_velocity)
    std_devs_velocity.append(std_dev_velocity)
    peak_velocities.append(peak_velocity)
    
    average_accelerations.append(average_acceleration)
    std_devs_acceleration.append(std_dev_acceleration)
    peak_accelerations.append(peak_acceleration)
    
# Calculate overlap sequentially for each offset and subsequent onset pair
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

# Velocity plot data including peak velocities
df_velocity = pd.DataFrame({
    'Channel': channels,
    'Average Velocity': average_velocities,
    'Peak Velocity': peak_velocities,
    'STD': std_devs_velocity
})

# Acceleration plot data including peak accelerations
df_acceleration = pd.DataFrame({
    'Channel': channels,
    'Average Acceleration': average_accelerations,
    'Peak Acceleration': peak_accelerations,
    'STD': std_devs_acceleration
})

# Plotting average and peak velocities for each channel
plt.figure(figsize=(12, 6))
sns.barplot(x='Channel', y='Average Velocity', data=df_velocity, palette='Blues', label='Average Velocity')
plt.errorbar(x=np.arange(len(channels)), y=average_velocities, yerr=std_devs_velocity, fmt='none', c='black', capsize=5, label='STD Velocity')
sns.scatterplot(x=np.arange(len(channels)), y=peak_velocities, color='red', s=100, label='Peak Velocity', zorder=5)
plt.title('Average and Peak Velocities for Each Channel with STD')
plt.ylabel('Velocity')
plt.xticks(ticks=np.arange(len(channels)), labels=channels)
plt.legend()
plt.show()

# Plotting average and peak accelerations for each channel
plt.figure(figsize=(12, 6))
sns.barplot(x='Channel', y='Average Acceleration', data=df_acceleration, palette='Greens', label='Average Acceleration')
plt.errorbar(x=np.arange(len(channels)), y=average_accelerations, yerr=std_devs_acceleration, fmt='none', c='black', capsize=5, label='STD Acceleration')
sns.scatterplot(x=np.arange(len(channels)), y=peak_accelerations, color='darkgreen', s=100, label='Peak Acceleration', zorder=5)
plt.title('Average and Peak Accelerations for Each Channel with STD')
plt.ylabel('Acceleration')
plt.xticks(ticks=np.arange(len(channels)), labels=channels)
plt.legend()
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

# TODO Convert CKPS calculations from Matlab to Python and add
# TODO Convert micro-online and micro-offline code from Matlab to Python and add
