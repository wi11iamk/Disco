#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:30:06 2024

@author: wi11iamk
"""

import os
import numpy as np
from hubdt import data_loading
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt


# Define the directory containing .h5 files
directory = './data/S3/'

# Target trials and conditions
conditions = {
    'C1': ["4 1 3", "2 4 1", "1 4 3", "3 4 2", "3 2 4", "2 3 4"],
    'C2': ["2 4 1", "1 4 3", "2 3 4", "4 1 3", "3 4 2", "3 2 4"],
    'C3': ["1 4 3", "2 3 4", "3 2 4", "2 4 1", "4 1 3", "3 4 2"],
    'C4': ["2 3 4", "3 2 4", "3 4 2", "1 4 3", "2 4 1", "4 1 3"],
    'C5': ["3 2 4", "3 4 2", "4 1 3", "2 3 4", "1 4 3", "2 4 1"],
    'C6': ["3 4 2", "4 1 3", "2 4 1", "3 2 4", "2 3 4", "1 4 3"]
}

target_trials = [
    [3, 4], [13, 14], [31, 32], [45, 46], [55, 56], [73, 74]
]

# Define constants for trial frames
num_samples_per_trial = 1200
frames_per_participant = 100799

# Define which digit each number corresponds to
digit_mapping = {
    '1': 1,  # Little
    '2': 3,  # Ring
    '3': 5,  # Middle
    '4': 7   # Index
}

# Create a dictionary to hold both velocity and acceleration data
data_containers = {sequence.replace(" ", ""): {'velocity_left': [], 'velocity_right': [],
                                               'acceleration_left': [], 'acceleration_right': [],
                                               'overlap_left': [], 'overlap_right': []}
                   for sequence in set([seq for cond in conditions.values() for seq in cond])}

# Function to apply Savitzky-Golay filter and normalize y-coordinates
def preprocess_tracking(tracking):
    tracking_filtered = tracking.copy()  # Create a copy of tracking to filter
    for col in range(1, 8, 2):  # Iterate over y-value columns (1, 3, 5, 7)
        tracking_filtered[:, col] = savgol_filter(tracking_filtered[:, col], 7, 1)
        # Normalize the filtered y-values
        tracking_filtered[:, col] -= tracking_filtered[0, col]
    return tracking_filtered

def process_y_values(data, digit_indices):
    # Ensure digit_indices do not exceed the number of columns in data
    max_columns = data.shape[1]
    valid_digit_indices = [idx for idx in digit_indices if idx < max_columns]
    
    # Now, select only valid digit indices
    y_values = data[:, valid_digit_indices]
    return y_values

# Function to find onset and offset around a peak within a window
def find_onset_offset(derivative, peak_index, window=15):
    onset_index = max(0, peak_index - window)
    offset_index = min(len(derivative) - 1, peak_index + window)
    return onset_index, offset_index

# Function to plot time series and detect key events
def time_series_events(x_range, y_values, channels, colors, y_threshold=10, plot=True):
    total_peaks_across_channels = 0
    onset_offset_data = {channel: [] for channel in channels}
    peak_indices = []

    # Loop through the channels
    for i, channel_name in enumerate(channels):
        # Handle cases where y_values has fewer columns than channels
        if y_values.shape[1] > i:
            derivative = np.diff(y_values[:, i], prepend=y_values[0, i])
            peaks, _ = find_peaks(y_values[:, i], height=y_threshold, prominence=10)
            total_peaks_across_channels += len(peaks)
            for peak in peaks:
                peak_indices.append((peak, channel_name))
            peak_indices.sort(key=lambda x: x[0])
        
    # Process the onset and offset data
    for peak_idx, (peak, channel_name) in enumerate(peak_indices):
        onset, offset = find_onset_offset(derivative, peak)
        onset_offset_data[channel_name].append({'onset': onset, 'peak': peak, 'offset': offset})
            
    if plot:
        fig, ax = plt.subplots(figsize=(14, 6))
        for i, channel_name in enumerate(channels):
            if y_values.shape[1] > i:
                plt.plot(x_range, y_values[:, i], label=channel_name, color=colors[i])
                for event_data in onset_offset_data[channel_name]:
                    peak = event_data['peak']
                    plt.plot(x_range[peak], y_values[peak, i], 'r*')
                    plt.plot(x_range[event_data['onset']], y_values[event_data['onset'], i], 'go')
                    plt.plot(x_range[event_data['offset']], y_values[event_data['offset'], i], 'mo')

        plt.legend()

    return onset_offset_data, y_threshold

# Function to calculate overlaps between movements across channels
# Function to calculate overlaps and return percentage of overlap for a trial
def calculate_overlaps(onset_offset_data, total_frames=1200):
    total_overlap_frames = 0

    # Convert onset_offset_data to a flat list of dictionaries with added channel information
    all_events = []
    for channel, events in onset_offset_data.items():
        for event in events:
            # Append event with channel information
            all_events.append({**event, 'channel': channel})

    # Sort all events by onset time
    all_events_sorted = sorted(all_events, key=lambda x: x['onset'])

    # Iterate through events to find overlaps between offsets and subsequent onsets across channels
    for i in range(len(all_events_sorted) - 1):
        current_event = all_events_sorted[i]
        next_event = all_events_sorted[i + 1]
        
        # Ensure we're comparing an offset to a subsequent onset and they are from different channels
        if current_event['channel'] != next_event['channel']:
            overlap = current_event['offset'] - next_event['onset']
            
            # If overlap is positive, add it to the total overlap
            if overlap > 0:
                total_overlap_frames += overlap

    # Calculate percentage of overlap relative to the trial length
    percent_overlap = round((total_overlap_frames / total_frames) * 100)

    return percent_overlap

# Load and process each .h5 file
for condition, digit_sequences in conditions.items():
    filepath = os.path.join(directory, f"{condition}_NR.h5")
    
    try:
        # Load DLC h5 file, extract features and tracking data
        h5 = data_loading.load_dlc_hdf(filepath)
        h5 = data_loading.dlc_remove_scorer(h5)
        features = list(h5.T.index.get_level_values(0).unique())
        tracking = data_loading.load_c_tracking(pt=condition, dlc=True, feats=features)

        # Preprocess the tracking data (filter and normalize)
        tracking_filtered = preprocess_tracking(tracking)

        # Calculate the number of participants based on total frames per participant
        num_participants = tracking.shape[0] // frames_per_participant

        for participant_index in range(num_participants):
            participant_data = tracking_filtered[participant_index * frames_per_participant:(participant_index + 1) * frames_per_participant]
        
            for seq_idx, digit_sequence in enumerate(digit_sequences):
                digit_sequence_no_spaces = digit_sequence.replace(" ", "")  # Remove spaces from the digit sequence
                target_trials_pair = target_trials[seq_idx]
                
                left_trial_start = (target_trials_pair[0] - 1) * num_samples_per_trial
                left_trial_end = left_trial_start + num_samples_per_trial
                right_trial_start = (target_trials_pair[1] - 1) * num_samples_per_trial
                right_trial_end = right_trial_start + num_samples_per_trial
        
                # Get the indices for the digits in this sequence
                digit_indices = [digit_mapping[digit] for digit in digit_sequence.split()]
        
                # Extract the trial data for the relevant digits
                left_trial_data = participant_data[left_trial_start:left_trial_end, digit_indices]
                right_trial_data = participant_data[right_trial_start:right_trial_end, digit_indices]
                
                # Normalize within each trial
                left_trial_data -= left_trial_data[0, :]
                right_trial_data -= right_trial_data[0, :]
        
                ### Step 1: Calculate Mean Velocity and Acceleration (Already in your code)
                # Compute the mean velocities for left and right trials
                left_mean_velocity = np.mean(np.abs(np.diff(left_trial_data, axis=0)), axis=0)
                right_mean_velocity = np.mean(np.abs(np.diff(right_trial_data, axis=0)), axis=0)
        
                # Compute the mean accelerations for left and right trials
                left_acceleration = np.mean(np.abs(np.diff(left_mean_velocity, axis=0)), axis=0)
                right_acceleration = np.mean(np.abs(np.diff(right_mean_velocity, axis=0)), axis=0)
        
                # Store the velocities and accelerations for each participant in the respective containers
                data_containers[digit_sequence_no_spaces]['velocity_left'].append(left_mean_velocity)
                data_containers[digit_sequence_no_spaces]['velocity_right'].append(right_mean_velocity)
                data_containers[digit_sequence_no_spaces]['acceleration_left'].append(left_acceleration)
                data_containers[digit_sequence_no_spaces]['acceleration_right'].append(right_acceleration)
        
                ### Step 2: Detect Onsets, Offsets, Peaks, and Overlaps (New Code)
                # Run the time series event detection
                # For left and right trials
                left_y_values = process_y_values(left_trial_data, digit_indices)
                right_y_values = process_y_values(right_trial_data, digit_indices)
                
                # Pass the correct number of labels and colors based on the valid digits
                left_onset_offset_data = time_series_events(
                    np.arange(left_trial_start, left_trial_end), 
                    left_trial_data, 
                    ['Little', 'Ring', 'Middle', 'Index'], 
                    ['blue', 'green', 'orange', 'red'], plot=False
                )[0]
                
                right_onset_offset_data = time_series_events(
                    np.arange(right_trial_start, right_trial_end), 
                    right_trial_data, 
                    ['Little', 'Ring', 'Middle', 'Index'], 
                    ['blue', 'green', 'orange', 'red'], plot=False
                )[0]
                # Calculate overlap
                left_overlap = calculate_overlaps(left_onset_offset_data)
                right_overlap = calculate_overlaps(right_onset_offset_data)
        
                # Store the overlap data in the containers
                data_containers[digit_sequence_no_spaces]['overlap_left'].append(left_overlap)
                data_containers[digit_sequence_no_spaces]['overlap_right'].append(right_overlap)

    except FileNotFoundError:
        print(f"File {filepath} not found.")
        
digit_sequences_ns = [seq.replace(' ', '') for seq in digit_sequences]

#%%

import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# Significance thresholds
significance_levels = {
    0.001: "p < 0.001",
    0.01: "p < 0.01",
    0.05: "p < 0.05"
}

# Function to return significance label based on p-value
def get_significance_label(p_value):
    for threshold, label in significance_levels.items():
        if p_value < threshold:
            return label
    return f"p = {p_value:.3e}"

# Function to calculate Cohen's d for paired samples
def cohen_d(x, y):
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1)

# Plott for velocities
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for i, sequence in enumerate(digit_sequences_ns):
    ax = axes[i]
    
    left_means = np.array([np.mean(participant) for participant in data_containers[sequence]['velocity_left']])
    right_means = np.array([np.mean(participant) for participant in data_containers[sequence]['velocity_right']])

    combined_means = np.concatenate([left_means, right_means])
    combined_labels = np.array(['Left'] * len(left_means) + ['Right'] * len(right_means))

    group_left_mean = np.mean(left_means)
    group_right_mean = np.mean(right_means)

    sns.barplot(ax=ax, x=pd.Series(['Left', 'Right']), y=pd.Series([group_left_mean, group_right_mean]), color='grey', alpha=0.6)
    sns.stripplot(ax=ax, x=combined_labels, y=combined_means, color='black', size=8, jitter=True)

    t_stat, p_value = ttest_rel(left_means, right_means)
    df = len(left_means) - 1
    cohen_d_value = cohen_d(left_means, right_means)
    significance_label = get_significance_label(p_value)

    ax.text(0.5, 0.9, f'{significance_label}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='black')
    ax.text(0.5, 0.85, f't = {t_stat:.3f}, df = {df}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='black')
    ax.text(0.5, 0.8, f"Cohen's d = {cohen_d_value:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='black')

    ax.set_title(f'Velocity: {sequence}', fontsize=14)
    ax.set_ylabel('Mean Velocity', fontsize=12)
    ax.set_xlabel('Trials', fontsize=12)

plt.tight_layout()
plt.show()

# Plot for accelerations
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for i, sequence in enumerate(digit_sequences_ns):
    ax = axes[i]
    
    left_accel = np.array([np.mean(participant) for participant in data_containers[sequence]['acceleration_left']])
    right_accel = np.array([np.mean(participant) for participant in data_containers[sequence]['acceleration_right']])

    combined_accel = np.concatenate([left_accel, right_accel])
    combined_labels_accel = np.array(['Left'] * len(left_accel) + ['Right'] * len(right_accel))

    group_left_accel = np.mean(left_accel)
    group_right_accel = np.mean(right_accel)

    sns.barplot(ax=ax, x=pd.Series(['Left', 'Right']), y=pd.Series([group_left_accel, group_right_accel]), color='grey', alpha=0.6)
    sns.stripplot(ax=ax, x=combined_labels_accel, y=combined_accel, color='black', size=8, jitter=True)

    t_stat, p_value = ttest_rel(left_accel, right_accel)
    df = len(left_accel) - 1
    cohen_d_value = cohen_d(left_accel, right_accel)
    significance_label = get_significance_label(p_value)

    ax.text(0.5, 0.9, f'{significance_label}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='black')
    ax.text(0.5, 0.85, f't = {t_stat:.3f}, df = {df}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='black')
    ax.text(0.5, 0.8, f"Cohen's d = {cohen_d_value:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='black')

    ax.set_title(f'Acceleration: {sequence}', fontsize=14)
    ax.set_ylabel('Mean Acceleration', fontsize=12)
    ax.set_xlabel('Trials', fontsize=12)

plt.tight_layout()
plt.show()

# Plot for overlap percentages
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for i, sequence in enumerate(digit_sequences_ns):
    ax = axes[i]
    
    # Get the overlap data for the current sequence
    left_overlap = np.array([participant for participant in data_containers[sequence]['overlap_left']])
    right_overlap = np.array([participant for participant in data_containers[sequence]['overlap_right']])

    # Combine data for left and right for the strip plot
    combined_overlap = np.concatenate([left_overlap, right_overlap])
    combined_labels_overlap = np.array(['Left'] * len(left_overlap) + ['Right'] * len(right_overlap))

    # Calculate group means for left and right
    group_left_overlap = np.mean(left_overlap)
    group_right_overlap = np.mean(right_overlap)

    # Create the bar plot with group means
    sns.barplot(ax=ax, x=pd.Series(['Left', 'Right']), y=pd.Series([group_left_overlap, group_right_overlap]), color='grey', alpha=0.6)
    
    # Overlay the strip plot with participant overlap percentages
    sns.stripplot(ax=ax, x=combined_labels_overlap, y=combined_overlap, color='black', size=8, jitter=True)

    # Perform paired t-test
    t_stat, p_value = ttest_rel(left_overlap, right_overlap)
    df = len(left_overlap) - 1
    cohen_d_value = cohen_d(left_overlap, right_overlap)
    significance_label = get_significance_label(p_value)

    # Display the significance, t-statistic, and Cohen's d on the plot
    ax.text(0.5, 0.9, f'{significance_label}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='black')
    ax.text(0.5, 0.85, f't = {t_stat:.3f}, df = {df}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='black')
    ax.text(0.5, 0.8, f"Cohen's d = {cohen_d_value:.3f}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='black')

    # Set plot title and labels
    ax.set_title(f'Overlap: {sequence}', fontsize=14)
    ax.set_ylabel('Percent Overlap', fontsize=12)
    ax.set_xlabel('Trials', fontsize=12)

# Adjust the layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()