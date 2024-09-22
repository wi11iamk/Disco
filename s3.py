#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:27:18 2024

@author: wi11iamk
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parameters import conditions, transition_pairs, seed_soil_pairs
from itertools import chain

# Set the condition
cond = 'C1'
trial_counts = [3, 3, 7, 7, 11, 11, 3, 3, 7, 7, 11, 11]
target_sequences = conditions[cond]

def parse_sequence(sequence):
    return [int(x) for x in sequence.split()]

def patternDetect(stream, targetSequence):
    target_pairs = [(targetSequence[i], targetSequence[(i+1) % len(targetSequence)]) for i in range(len(targetSequence))]
    stream_pairs = [(stream[i], stream[i+1]) for i in range(len(stream)-1)]
    matching_pairs_count = sum(1 for pair in stream_pairs if pair in target_pairs)
    score = matching_pairs_count
    consecutive_matches = [pair in target_pairs for pair in stream_pairs]
    if consecutive_matches:
        score += 1
    interruptions = sum(1 for i in range(len(consecutive_matches)-1) if not consecutive_matches[i] and consecutive_matches[i+1])
    score += interruptions
    return score

def calculate_combined_transition_speeds(events, timestamps, transition_pairs):
    combined_speeds = []
    for i in range(len(events) - 2):  # Ensure there are at least two transitions
        current_pair = (events[i], events[i + 1])
        if current_pair in transition_pairs:
            next_pair = (events[i + 1], events[i + 2])
            if next_pair in transition_pairs:
                combined_speed = (timestamps[i + 1] - timestamps[i]) + (timestamps[i + 2] - timestamps[i + 1])
                combined_speeds.append(combined_speed)
    return combined_speeds

def parse_and_analyse_data(file_path, target_sequences, trial_counts, transition_pairs):
    parsed_data = {
        'Trial': [], 'KeyID': [], 'EventType': [], 'TimeStamp': [], 'GlobalTimeStamp': [],
        'CorrectKeyPressesPerS': [], 'KeypressSpeed': []
    }
    trial_events = {}
    trial_first_last_timestamps = {}
    combined_transition_speeds_data = {}
    target_sequences = [parse_sequence(seq) for seq in target_sequences]
    ignore_first_row = True

    with open(file_path, 'r') as file:
        for line in file:
            if ignore_first_row:
                ignore_first_row = False
                continue
            parts = line.strip().split()
            if len(parts) < 5 or parts[:5] == ['0', '0', '99', '0', '0']:
                continue
            trial_number, event_count, event_type, timestamp, globaltime = map(int, parts[:5])
            if trial_number not in trial_events:
                trial_events[trial_number] = {'events': [], 'timestamps': [], 'first_line_index': len(parsed_data['Trial'])}
                trial_first_last_timestamps[trial_number] = {'first': timestamp, 'last': timestamp}
            else:
                trial_first_last_timestamps[trial_number]['last'] = timestamp

            trial_events[trial_number]['events'].append(event_type)
            trial_events[trial_number]['timestamps'].append(timestamp)
            parsed_data['Trial'].append(trial_number)
            parsed_data['KeyID'].append(event_count)
            parsed_data['EventType'].append(event_type)
            parsed_data['TimeStamp'].append(timestamp)
            parsed_data['GlobalTimeStamp'].append(globaltime)
            parsed_data['CorrectKeyPressesPerS'].append(None)
            parsed_data['KeypressSpeed'].append(None)

    last_value = None
    trial_index = 1
    for i, count in enumerate(trial_counts):
        target_sequence = target_sequences[i]
        pairs = transition_pairs[cond][i]
        for _ in range(count):
            if trial_index in trial_events:
                data = trial_events[trial_index]
                events, timestamps = data['events'], data['timestamps']
                correct_per_trial = [patternDetect(events, target_sequence) / 10 for _ in range(10)]
                mean_correct_per_trial = np.mean(correct_per_trial)
                parsed_data['CorrectKeyPressesPerS'][data['first_line_index']] = mean_correct_per_trial
                if last_value is not None:
                    delta = mean_correct_per_trial - last_value if mean_correct_per_trial is not None else None
                    parsed_data['KeypressSpeed'][data['first_line_index']] = delta
                last_value = mean_correct_per_trial

                combined_speeds = calculate_combined_transition_speeds(events, timestamps, pairs)
                trial_num = trial_index  # Use trial_index directly
                if trial_num not in combined_transition_speeds_data:
                    combined_transition_speeds_data[trial_num] = []
                combined_transition_speeds_data[trial_num].append(np.mean(combined_speeds) if combined_speeds else np.nan)
            trial_index += 1

    for trial_num, speeds in combined_transition_speeds_data.items():
        parsed_data[f'CombinedTransitionSpeed_{trial_num}'] = speeds + [None] * (len(parsed_data['Trial']) - len(speeds))

    return pd.DataFrame(parsed_data)

def write_data_to_csv(dataframe, output_file_path):
    dataframe.to_csv(output_file_path, index=False)

def process_all_data_files(input_folder, output_folder, target_sequences, trial_counts, transition_pairs, num_trials_to_plot=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    all_correct_presses = []
    all_combined_speeds = {trial: [] for trial in seed_soil_pairs.keys()}  # Initialise with trial numbers
    participant_means = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            output_file_name = filename[:4] + '.csv'
            output_file_path = os.path.join(output_folder, output_file_name)
            
            dataframe = parse_and_analyse_data(file_path, target_sequences, trial_counts, transition_pairs)
            all_correct_presses.append(dataframe['CorrectKeyPressesPerS'].dropna().tolist())
            participant_means.append(dataframe.filter(regex='CombinedTransitionSpeed_').mean())
            for trial in seed_soil_pairs.keys():
                column_name = f'CombinedTransitionSpeed_{trial}'
                if column_name in dataframe.columns:
                    all_combined_speeds[trial].append(dataframe[column_name].dropna().tolist())
            write_data_to_csv(dataframe, output_file_path)

    # Debug: Print the all_combined_speeds keys and lengths
    # print(f"all_combined_speeds keys: {list(all_combined_speeds.keys())}")
    # for key, value in all_combined_speeds.items():
    #     print(f"Key: {key}, Length of values: {len(value)}")

    trial_means, trial_sems = calculate_and_plot_correct_presses(all_correct_presses, num_trials_to_plot, trial_counts, target_sequences)
    plot_combined_transition_speeds(all_combined_speeds, participant_means, target_sequences, trial_counts)

    return trial_means, participant_means

def calculate_and_plot_correct_presses(all_correct_presses, num_trials_to_plot, trial_counts, target_sequences):
    if num_trials_to_plot is None:
        num_trials_to_plot = len(all_correct_presses[0])

    trial_means = []
    trial_sems = []
    x_values = list(range(1, num_trials_to_plot + 1))
    colors = ['blue', 'green', 'red', 'mediumpurple', 'purple', 'cyan', 'magenta', 'gold', 'hotpink', 'springgreen', 'navy', 'crimson']
    plt.figure(figsize=(30, 10))

    plt.subplot(2, 1, 1)
    break_position = 42
    space = 2
    start = 0

    for idx, count in enumerate(trial_counts):
        end = start + count
        color = colors[idx % len(colors)]
        if start >= break_position:
            segment_x_values = [x + space for x in x_values[start:end]]
        else:
            segment_x_values = x_values[start:end]

        segment_means = []
        segment_sems = []

        # for participant_data in all_correct_presses:
        #     if len(participant_data) >= end:
        #         plt.scatter(segment_x_values, participant_data[start:end], color='lightgrey', alpha=0.75)

        for i in range(start, end):
            trial_data = [participant[i] for participant in all_correct_presses if len(participant) > i]
            if trial_data:
                trial_mean = np.mean(trial_data)
                trial_sem = np.std(trial_data, ddof=1) / np.sqrt(len(trial_data))
            else:
                trial_mean, trial_sem = np.nan, np.nan
            segment_means.append(trial_mean)
            segment_sems.append(trial_sem)

        trial_means.extend(segment_means)
        trial_sems.extend(segment_sems)
        plt.errorbar(segment_x_values, segment_means, yerr=segment_sems, fmt='-o', color=color, ecolor=color, alpha=0.75, capsize=5)
        start = end

    plt.axvline(x=break_position + 1.5, color='black', linestyle='--')
    xtick_labels = []
    xtick_positions = []
    start = 0

    for idx, count in enumerate(trial_counts):
        end = start + count
        midpoint = (start + end) // 2
        xtick_labels.append(''.join(target_sequences[idx]))
        xtick_positions.append(midpoint + 1)
        if end == break_position:
            start = end + space
        else:
            start = end

    plt.xticks(xtick_positions, xtick_labels, rotation=45, ha='right')
    plt.title(f'Mean Correct Key Presses per S in {cond}')
    plt.xlabel('Target Sequence')
    plt.ylabel('Correct Key Presses per S')

    return trial_means, trial_sems

def plot_combined_transition_speeds(all_combined_speeds, participant_means, target_sequences, trial_counts):
    plt.subplot(2, 1, 2)
    xtick_labels = []
    xtick_positions = []

    grouped_trials = [
        [3, 4, 5, 6],
        [13, 14, 15, 16],
        [31, 32, 33, 34],
        [45, 46, 47, 48],
        [55, 56, 57, 58],
        [73, 74, 75, 76]
    ]
    
    for group in grouped_trials:
        group_mean_speeds = []
        group_positions = []
        for idx, trial in enumerate(group):
            combined_speeds = list(chain.from_iterable(all_combined_speeds[trial]))
            mean_speed = np.nanmean(combined_speeds)
            sem_speed = np.nanstd(combined_speeds, ddof=1) / np.sqrt(len(combined_speeds)) if combined_speeds else np.nan
            xtick_labels.append(f'Trial {trial}')
            xtick_positions.append(len(xtick_labels) - 1)
            group_mean_speeds.append(mean_speed)
            group_positions.append(len(xtick_labels) - 1)
            
            # Plot participant mean speeds in faded grey
            # plt.scatter([len(xtick_labels) - 1] * len(combined_speeds), combined_speeds, color='lightgrey', alpha=0.75)

            # Plot mean speed in color
            plt.errorbar(len(xtick_labels) - 1, mean_speed, yerr=sem_speed, fmt='o', color=seed_soil_pairs[trial]['color'], alpha=0.75, capsize=5)
        
        # Draw line connecting mean points
        plt.plot(group_positions, group_mean_speeds, color='grey', alpha=0.75)

    plt.xticks(xtick_positions, xtick_labels, rotation=45, ha='right')
    plt.title('Combined Transition Speeds')
    plt.xlabel('Trial')
    plt.ylabel('Speed (ms)')
    plt.tight_layout()
    plt.show()

# Example call with specified number of trials to plot
input_folder = f'/Users/wi11iamk/Desktop/PhD/PsyToolkit/{cond}/data'
output_folder = f'/Users/wi11iamk/Desktop/PhD/csvOutput/S3/{cond}'
trial_means, participant_means = process_all_data_files(input_folder, output_folder, target_sequences, trial_counts, transition_pairs, num_trials_to_plot=84)

#%%

###
# Condition-level .h5 analysis
###

import umap
from hubdt import data_loading, wavelets, hdb_clustering
from scipy.signal import savgol_filter

# Choose the participant number
pt = cond  
total_trials = 84
num_samples_per_trial = 1200
frames_per_participant = 100799
    
# Load DLC h5 file, extract features and tracking data
h5 = data_loading.load_dlc_hdf(f'./data/S3/{pt}_NR.h5')
h5 = data_loading.dlc_remove_scorer(h5)
features = list(h5.T.index.get_level_values(0).unique())

tracking = data_loading.load_c_tracking(pt=pt, dlc=True, feats=features)

# Step 1: Apply Savitzky-Golay filter and normalise first frames to 0
tracking_filtered = tracking.copy()  # Create copy of tracking to filter
for col in range(1, 8, 2):  # Iterate over y-value columns (1, 3, 5, 7)
    # Apply Savitzky-Golay filter
    tracking_filtered[:, col] = savgol_filter(tracking_filtered[:, col], 7, 1)
    # Normalise the filtered y-values
    tracking_filtered[:, col] -= tracking_filtered[0, col]

# Step 2: Remove the x coordinates (columns 0, 2, 4, 6)
tracking_filtered = np.delete(tracking_filtered, [0, 2, 4, 6], axis=1)

# Step 3: Store the filtered y-coordinate positions directly into expanded_features
num_rows = tracking_filtered.shape[0]
num_original_cols = tracking_filtered.shape[1]
expanded_features = np.zeros((num_rows, num_original_cols))  # Only position data

# Step 4: Copy the filtered y-coordinate positions into expanded_features
expanded_features[:, :] = tracking_filtered[:, :]

# Determine the number of participants
total_samples = len(expanded_features)
n_participants = total_samples // frames_per_participant

#%%

###
# Generate scales and frequencies for wavelet transform of the tracking data;
# store the wavelet projection into a variable; fit wavelet projection into two
# dimensional embedded space (UMAP) and plot
###

# Step 1: Calculate wavelet-transformed features
scales, frequencies = wavelets.calculate_scales(0.75, 2.75, 120, 4)
wavelet_transformed = wavelets.wavelet_transform_s3(expanded_features, scales, frequencies, 120)

# Step 2: Adjust the total columns in the final output
num_rows = expanded_features.shape[0]
num_original_cols = 4  # Pose features for little, ring, middle, index digits (y-values)
num_wavelet_cols_per_digit = 4  # 4 wavelet features per pose column
num_final_cols = num_original_cols * num_wavelet_cols_per_digit  # Only wavelet features, no velocity

# Ensure the final number of columns is consistent with the features
assert num_final_cols == num_original_cols * num_wavelet_cols_per_digit, f"Expected {num_original_cols * num_wavelet_cols_per_digit} columns, but got {num_final_cols}"

# Step 3: Initialise the final projection array
proj = np.zeros((num_rows, num_final_cols))  # Now proj has only wavelet features

# Step 4: Copy wavelet-transformed features into proj
# Ensure correct alignment of wavelet features for each digit (little, ring, middle, index)
for i in range(num_original_cols):
    # Determine where to place wavelet features for this digit
    wavelet_start_idx = i * num_wavelet_cols_per_digit
    wavelet_end_idx = wavelet_start_idx + num_wavelet_cols_per_digit

    # Copy the wavelet data for the current digit (i) from wavelet_transformed into proj
    proj[:, wavelet_start_idx:wavelet_end_idx] = wavelet_transformed[i * num_wavelet_cols_per_digit:(i + 1) * num_wavelet_cols_per_digit, :].T

# Step 5: Define trial groups for categorical feature engineering
trial_groups = [
    (1, 3), (4, 6), (7, 13), (14, 20),
    (21, 31), (32, 42), (43, 45), (46, 48),
    (49, 55), (56, 62), (63, 73), (74, 84)
]

# Step 6: Assign one-hot encoding based on group number
n_participants = total_samples // (num_samples_per_trial * total_trials)

# Initialise arrays for one-hot encoded features (12 groups, hence 12 columns) and group labels
one_hot_features = np.zeros((num_rows, 12))  # Twelve columns for individual groups
group_labels = np.zeros(num_rows)  # Array to hold group labels for coloring the UMAP plot

# Assign group number and one-hot encoding
for participant_idx in range(n_participants):
    participant_start_idx = participant_idx * (num_samples_per_trial * total_trials)
    participant_end_idx = participant_start_idx + (num_samples_per_trial * total_trials)
    
    # Assign trial numbers for this participant
    for trial_idx in range(84):  # 84 trials per participant
        trial_start_idx = participant_start_idx + trial_idx * num_samples_per_trial
        trial_end_idx = trial_start_idx + num_samples_per_trial
        
        # Assign trial number (1 to 84 for each participant)
        trial_number = trial_idx + 1
        
        # Determine group for this trial and assign group number
        group_number = None
        for group_idx, group_range in enumerate(trial_groups):
            if group_range[0] <= trial_number <= group_range[1]:
                group_number = group_idx + 1  # Assign group number (1-12)
                break

        # One-hot encode the group number individually (1-12)
        one_hot_features[trial_start_idx:trial_end_idx, group_number - 1] = 1  # Subtract 1 to align with zero-indexing
        
        # Assign group label for plotting (1-12)
        group_labels[trial_start_idx:trial_end_idx] = group_number

# Step 7: Concatenate the one-hot encoded features with the proj array
proj_with_one_hot = np.hstack([proj, one_hot_features])

# Step 8: Use Hamming distance in UMAP (since we now have categorical one-hot features)
mapper = umap.UMAP(n_neighbors=8, n_components=2, min_dist=0.3, metric='hamming', init='random')
embed = mapper.fit_transform(proj_with_one_hot)

# Step 9: Plot the UMAP projection
plt.figure(figsize=(10, 10))
plt.scatter(embed[:, 0], embed[:, 1], s=0.125/4, c='blue', alpha=0.125/2)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()

# Print the final shapes to verify
print("Data Shape:", proj_with_one_hot.shape)

#%%

###
# Perform HDBSCAN clustering to obtain labels and probabilities from embedded
# data; plot cluster labels atop embedded data
###

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cmocean
import numpy as np

# Step 1: Perform HDBSCAN on the entire dataset
clusterobj = hdb_clustering.hdb_scan(embed, 1500, 20, selection='leaf', cluster_selection_epsilon=0.05)

# Extract labels and probabilities from the initial clustering
labels = clusterobj.labels_
probabilities = clusterobj.probabilities_

# Print the number of unique original labels (clusters)
unique_original_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
print(f"Original number of clusters: {len(unique_original_labels)}")

# Step 2: Find the top 5 largest clusters by membership
cluster_counts = Counter(labels[labels >= 0])  # Ignore noise points (-1)
largest_clusters = [cluster for cluster, count in cluster_counts.most_common(10)]

# Initialise new labels for the entire dataset (copy original labels)
final_labels = labels.copy()

# Step 3: Run HDBSCAN recursively on the top 5 largest clusters
new_cluster_id = max(labels) + 1  # Start new cluster IDs after the last original cluster

for cluster_id in largest_clusters:
    # Select points from the current large cluster
    selected_cluster_mask = labels == cluster_id
    selected_points = embed[selected_cluster_mask]

    # Dynamically adjust min_cluster_size based on cluster size
    cluster_size = len(selected_points)
    refined_clusterobj = hdb_clustering.hdb_scan(selected_points, (cluster_size // 35), 10, selection='leaf', cluster_selection_epsilon=0.01)

    # Extract refined labels for the selected cluster points
    refined_labels = refined_clusterobj.labels_

    # Print number of sub-clusters found for debugging
    num_sub_clusters = len(np.unique(refined_labels[refined_labels >= 0]))  # Exclude noise (-1)
    print(f"Cluster {cluster_id}: Number of sub-clusters found = {num_sub_clusters}")

    # Reassign refined labels to the global dataset using global indexing
    selected_indices = np.where(selected_cluster_mask)[0]  # Get the global indices of the selected points
    for i, refined_label in enumerate(refined_labels):
        if refined_label != -1:  # Ignore noise in the sub-clusters
            final_labels[selected_indices[i]] = new_cluster_id + refined_label

    # Update the new_cluster_id for the next sub-cluster (ensure non-overlapping cluster IDs)
    if num_sub_clusters > 0:
        new_cluster_id += refined_labels.max() + 1  # Increment for the next batch of sub-clusters

# Step 4: Print the number of unique final labels (clusters)
unique_final_labels = np.unique(final_labels[final_labels >= 0])  # Exclude noise (-1)
print(f"Final number of clusters (after recursive HDBSCAN): {len(unique_final_labels)}")

# Step 5: Plot all clusters (original and refined)
num_clusters = 80  # Number of colors needed
spacing_factor = 10  # Adjust this to control how spaced apart the colors are

# Generate evenly spaced values between 0 and 1 with a certain spacing
color_steps = np.linspace(0, 1, num_clusters, endpoint=False)

# Select colors from the colormap with the specified spacing
color_palette = [cmocean.cm.phase((i * spacing_factor) % 1) for i in color_steps]

# Plot clusters using the evenly spaced color palette
cluster_colors_dict = hdb_clustering.plot_hdb_over_tsne_s3(embed, final_labels, probabilities, color_palette)

#%%

###
# Calculate the mean length of each label in frames; store the first frame of
# each occurrence of each label
###

a_labels = np.reshape(final_labels, (-1, 1))

# Function to tally continuous label occurrences, allowing for a threshold
def calculate_label_data(arr):
    continuous_counts = {}
    current_val = None
    count = 0
    start_frame = 0

    arr = arr.flatten()  # Ensure the array is flattened for iteration
    # Generate original indices array
    original_indices = np.arange(len(arr))

    # Filter out -1 values
    noiseless_mask = arr != -1
    a_labels_noiseless = arr[noiseless_mask]
    original_indices_noiseless = original_indices[noiseless_mask]

    # Combine the original indices and labels into a single array
    a_labels_combined = np.column_stack((original_indices_noiseless, a_labels_noiseless))

    for idx, (original_idx, val) in enumerate(a_labels_combined):
        if val == current_val:
            count += 1
        else:
            if current_val is not None:
                finalise_count(continuous_counts, current_val, count, start_frame, a_labels_combined)
            current_val = val
            count = 1
            start_frame = idx  # Start index is adjusted due to filtered array

    # Finalise the last sequence if any valid count exists
    if current_val is not None and count > 0:
        finalise_count(continuous_counts, current_val, count, start_frame, a_labels_combined)

    return process_counts(continuous_counts), a_labels_combined

# Function to update or initialise label data in 'continuous_counts'
def finalise_count(continuous_counts, val, count, start_frame, combined_array):
    if val not in continuous_counts:
        continuous_counts[val] = {'counts': [], 'start_frames': []}
    continuous_counts[val]['counts'].append(count)
    continuous_counts[val]['start_frames'].append(combined_array[start_frame, 0])

# Function to compute statistics for each label based on its counts
def process_counts(continuous_counts):
    results = []
    for label, data in continuous_counts.items():
        mean_count = np.mean(data['counts'])
        total_occurrences = len(data['counts'])
        firsts = data['start_frames'][:5]  # The first five start frames
        lasts = data['start_frames'][-5:]  # The last five start frames, if applicable
        results.append((label, mean_count, total_occurrences, firsts, lasts))
    return results

# Usage
a_labels_data, a_labels_combined = calculate_label_data(a_labels)

#%%

###
# Initialise dictionary to store frame values for each trial
###

# Initialise dictionaries to store aggregated counts for each trial
aggregated_trial_counts = {f"{i+1}": {} for i in range(total_trials)}
participant_trial_counts = {f"{i+1}": {} for i in range(total_trials)}

# Populate the dictionaries
for i in range(n_participants):
    for j in range(total_trials):
        trial_key = f"{j+1}"  # Key name (e.g., '1' for trial 1)
        start_index = i * frames_per_participant + j * 1200
        end_index = start_index + 1200
        slice_labels = a_labels[start_index:end_index]
        
        # For non-negative labels
        non_negative_labels = slice_labels[slice_labels >= 0]
        unique_non_neg, counts_non_neg = np.unique(non_negative_labels, return_counts=True)
        
        # Aggregate counts for each label
        for label, count in zip(unique_non_neg, counts_non_neg):
            if label in aggregated_trial_counts[trial_key]:
                aggregated_trial_counts[trial_key][label] += count
            else:
                aggregated_trial_counts[trial_key][label] = count

            if label in participant_trial_counts[trial_key]:
                participant_trial_counts[trial_key][label] += 1
            else:
                participant_trial_counts[trial_key][label] = 1

#%%

###
# Calculate and plot the distribution of labels within each trial; perform 
# observed Jensen-Shannon Divergence (JSD) calculations, then permute to generate 
# the null distribution and test for statistical significance
###

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

# Function to compute the Jensen-Shannon divergence
def compute_js_divergence(p, q):
    # JSD between two probability distributions p and q
    return jensenshannon(p, q)**2

# Function for a permutation test (modified to return all permuted JSDs)
def permutation_test_jsd(p, q, num_permutations=100000):
    # Filter out the labels that are zero in **both distributions**
    non_zero_indices = (p != 0) | (q != 0)
    p_filtered = p[non_zero_indices]
    q_filtered = q[non_zero_indices]
    
    # Ensure both filtered distributions have the same length
    observed_jsd = compute_js_divergence(p_filtered, q_filtered)
    
    # Concatenate and shuffle non-zero distributions
    combined = np.concatenate([p_filtered, q_filtered])
    
    permuted_jsds = []
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_p = combined[:len(p_filtered)]
        perm_q = combined[len(p_filtered):]
        
        perm_jsd = compute_js_divergence(perm_p, perm_q)
        permuted_jsds.append(perm_jsd)
    
    # p-value is the proportion of permuted JSDs greater than or equal to the observed JSD
    count_greater = np.sum(np.array(permuted_jsds) >= observed_jsd)
    p_value = count_greater / num_permutations
    
    return observed_jsd, p_value, permuted_jsds

# Function to determine significance level
def determine_significance(p_value):
    if p_value < 0.001:
        return "p < 0.001"
    elif p_value < 0.01:
        return "p < 0.01"
    elif p_value < 0.05:
        return "p < 0.05"
    else:
        return f"p = {p_value:.3e}"

# Grouped trials
grouped_trials = [
    [3, 4, 5, 6],
    [13, 14, 15, 16],
    [31, 32, 33, 34],
    [45, 46, 47, 48],
    [55, 56, 57, 58],
    [73, 74, 75, 76]
]

# Flatten the list of grouped trials to get a list of trial indices to include
trials_to_include = [trial for group in grouped_trials for trial in group]

# Normalise trial counts
normalised_trial_counts = []
for trial_key, label_counts in aggregated_trial_counts.items():
    total_count = sum(label_counts.values())
    normalised_counts = {label: count / total_count for label, count in label_counts.items()}
    normalised_trial_counts.append(normalised_counts)

# Define the maximum label value
max_label_non_neg = max(max(label_counts.keys()) for label_counts in normalised_trial_counts)

# Prepare plot data for the filtered trials
filtered_normalised_trial_counts = [normalised_trial_counts[i-1] for i in trials_to_include]
plot_matrix_non_neg = np.zeros((len(filtered_normalised_trial_counts), max_label_non_neg + 1))

# Populate plot matrices for the filtered trials
for i, trial_counts in enumerate(filtered_normalised_trial_counts):
    for label, normalised_count in trial_counts.items():
        plot_matrix_non_neg[i, label] = normalised_count

# Plot the first figure for the filtered trials (showing label distributions across grouped trials)
fig, ax = plt.subplots(figsize=(30, 6))
trial_indices = np.arange(1, len(filtered_normalised_trial_counts) + 1)
for label in range(0, max_label_non_neg + 1):
    bottom = np.sum(plot_matrix_non_neg[:, :label], axis=1)
    color = plt.cm.tab20(label % 20)  # Use a colormap to select colors for labels
    ax.bar(trial_indices, plot_matrix_non_neg[:, label], width=0.5, bottom=bottom, color=color, label=f'Label {label}')
ax.set_xticks(trial_indices)
ax.set_xticklabels([str(trial) for trial in trials_to_include])
ax.set_title('Normalised Distribution of Labels Across Grouped Trials')

plt.tight_layout()
plt.show()

# Step 1: Compute JSD between first two trials in each group and perform permutation test
jsd_results = []

# Function to compute the normalised label distribution for a trial
def get_normalised_distribution(trial_counts, max_label):
    total_count = sum(trial_counts.values())
    distribution = np.zeros(max_label + 1)
    for label, count in trial_counts.items():
        distribution[label] = count / total_count
    return distribution

# List of all trials to include
trials_to_include = [trial for group in grouped_trials for trial in group]

# Function to get the trial index in trials_to_include
def get_trial_index(trial_num):
    try:
        return trials_to_include.index(trial_num)  # Find the index of the trial in trials_to_include
    except ValueError:
        print(f"Trial {trial_num} not found in trials_to_include.")
        return None

# Step 1: Compute JSD between first two trials in each group and perform permutation test
jsd_results = []

# Iterate over the groups and perform JSD and permutation test for the first two trials in each group
for group in grouped_trials:
    trial1, trial2 = group[0], group[1]  # Take the first two trials in each group
    
    # Get the index of the trials in trials_to_include
    trial1_index = get_trial_index(trial1)
    trial2_index = get_trial_index(trial2)
    
    if trial1_index is None or trial2_index is None:
        continue  # Skip if trial indices are not found
    
    # Get normalised distributions for the two trials
    trial1_distribution = get_normalised_distribution(filtered_normalised_trial_counts[trial1_index], max_label_non_neg)
    trial2_distribution = get_normalised_distribution(filtered_normalised_trial_counts[trial2_index], max_label_non_neg)
    
    # Compute JSD and run permutation test (using only non-zero labels)
    jsd_value, p_value, permuted_jsds = permutation_test_jsd(trial1_distribution, trial2_distribution)
    
    # Store the results
    jsd_results.append((f"Trial {trial1} vs {trial2}", jsd_value, determine_significance(p_value), permuted_jsds, p_value))

# Print the results in a table
print(f"{'Trial Pair':<20} {'JSD':<10} {'Significance':<15}")
for result in jsd_results:
    print(f"{result[0]:<20} {result[1]:<10.4f} {result[2]:<15}")

# Step 2: Create the 2x3 grid of subplots for the distribution of permuted JSDs (3 columns, 2 rows)
fig, axs = plt.subplots(2, 3, figsize=(18, 8))  # Adjusting for 3 columns and 2 rows

# Flatten the axes array for easier access
axs = axs.flatten()

# Plot the distribution of permuted JSDs for each trial pair
for i, result in enumerate(jsd_results[:6]):  # Limit to 6 trial pairs
    trial_pair, observed_jsd, significance, permuted_jsds, p_value = result
    
    # Plot histogram of permuted JSDs
    axs[i].hist(permuted_jsds, bins=30, alpha=0.75, color='gray', edgecolor='black')
    
    # Add the observed JSD as a red vertical line
    axs[i].axvline(observed_jsd, color='red', linestyle='dashed', linewidth=2, label=f'Observed JSD: {observed_jsd:.4f}')
    
    # Keep the title as the trial pair
    axs[i].set_title(trial_pair)
    
    # Add the p-value to the legend
    axs[i].legend(title=f'{significance} (p = {p_value:.3e})')
    
    # Set labels
    axs[i].set_xlabel('JSD Value')
    axs[i].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()
