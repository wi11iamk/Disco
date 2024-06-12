# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:27:18 2024

@author: wi11iamk
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the conditions and their respective target sequences
conditions = {
    'C1': ["4 1 3", "2 4 1 3 4", "2 4 1", "1 2 4 1 3", "1 4 3", "3 1 4 3 2", "3 4 2", "2 3 4 2 3", "3 2 4", "4 3 2 4 1", "2 3 4", "3 2 3 4 2"],
    'C2': ["2 4 1", "1 2 4 1 3", "1 4 3", "3 1 4 3 2", "2 3 4", "3 2 3 4 2", "4 1 3", "2 4 1 3 4", "3 4 2", "2 3 4 2 3", "3 2 4", "4 3 2 4 1"],
    'C3': ["1 4 3", "3 1 4 3 2", "2 3 4", "3 2 3 4 2", "3 2 4", "4 3 2 4 1", "2 4 1", "1 2 4 1 3", "4 1 3", "2 4 1 3 4", "3 4 2", "2 3 4 2 3"],
    'C4': ["2 3 4", "3 2 3 4 2", "3 2 4", "4 3 2 4 1", "3 4 2", "2 3 4 2 3", "1 4 3", "3 1 4 3 2", "2 4 1", "1 2 4 1 3", "4 1 3", "2 4 1 3 4"],
    'C5': ["3 2 4", "4 3 2 4 1", "3 4 2", "2 3 4 2 3", "4 1 3", "2 4 1 3 4", "2 3 4", "3 2 3 4 2", "1 4 3", "3 1 4 3 2", "2 4 1", "1 2 4 1 3"],
    'C6': ["3 4 2", "2 3 4 2 3", "4 1 3", "2 4 1 3 4", "2 4 1", "1 2 4 1 3", "3 2 4", "4 3 2 4 1", "2 3 4", "3 2 3 4 2", "1 4 3", "3 1 4 3 2"]
}

# Set the condition
cond = 'C1'
trial_counts = [3, 3, 7, 7, 11, 11, 3, 3, 7, 7, 11, 11]
target_sequences = conditions[cond]

def parse_sequence(sequence):
    return [int(x) for x in sequence.split()]

def patternDetect(stream, targetSequence):
    # Generate pairwise tuples for the target sequence, including a 'ghost' element for circularity
    target_pairs = [(targetSequence[i], targetSequence[(i+1) % len(targetSequence)]) for i in range(len(targetSequence))]
    
    # Generate pairwise tuples for the stream
    stream_pairs = [(stream[i], stream[i+1]) for i in range(len(stream)-1)]
    
    # Count matching pairs
    matching_pairs_count = sum(1 for pair in stream_pairs if pair in target_pairs)
    
    # Initialise score based on matching pairs count
    score = matching_pairs_count
    
    # Identify segments of consecutive matches in the target sequence within the stream
    consecutive_matches = [pair in target_pairs for pair in stream_pairs]
    
    # Check for the start of a sequence or interruptions
    if consecutive_matches:
        score += 1  # Add for the start of the sequence
    
    # Adjust for interruptions, considering each False followed by True as the start of a new sequence segment
    interruptions = sum(1 for i in range(len(consecutive_matches)-1) if not consecutive_matches[i] and consecutive_matches[i+1])
    score += interruptions

    return score

# Function to calculate score within a time window
def calculate_correct_keypresses_within_window(events, timestamps, window_start, window_end, target_sequence):
    slice_events = [event for event, timestamp in zip(events, timestamps) if window_start <= timestamp <= window_end]
    return patternDetect(slice_events, target_sequence)

# Function to analyse scoring in trial start/end windows
def analyse_trial_windows(events, timestamps, target_sequence):
    start_ts, end_ts = timestamps[0], timestamps[-1]
    first_window_end = min(start_ts + 1000, end_ts)
    last_window_start = max(end_ts - 1000, start_ts)
    first_window_correct = calculate_correct_keypresses_within_window(events, timestamps, start_ts, first_window_end, target_sequence)
    last_window_correct = calculate_correct_keypresses_within_window(events, timestamps, last_window_start, end_ts, target_sequence)
    return first_window_correct, last_window_correct

# Function to parse stream data and analyse it
def parse_and_analyse_data(file_path, target_sequences, trial_counts):
    parsed_data = {
        'Trial': [], 'KeyID': [], 'EventType': [], 'TimeStamp': [], 'GlobalTimeStamp': [],
        'CorrectKeyPressesPerS': [], 'KeypressSpeed': []
    }
    trial_events = {}
    trial_first_last_timestamps = {}
    target_sequences = [parse_sequence(seq) for seq in target_sequences]

    ignore_first_row = True
    with open(file_path, 'r') as file:
        for line in file:
            if ignore_first_row:
                ignore_first_row = False
                continue  # Skip the first row
            parts = line.strip().split()
            if len(parts) < 5 or parts[:5] == ['0', '0', '99', '0', '0']:
                continue  # Skip lines not matching expected format
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
            parsed_data['CorrectKeyPressesPerS'].append(None)  # Placeholder
            parsed_data['KeypressSpeed'].append(None)  # Placeholder

    last_value = None
    # Calculate metrics
    trial_index = 1
    for i, count in enumerate(trial_counts):
        target_sequence = target_sequences[i]
        for _ in range(count):
            if trial_index in trial_events:
                data = trial_events[trial_index]
                events, timestamps = data['events'], data['timestamps']
                first_window_correct, last_window_correct = analyse_trial_windows(events, timestamps, target_sequence)
                
                # Example multiple readings per trial, simulate or actual implementation as needed
                correct_per_trial = [patternDetect(events, target_sequence) / 10 for _ in range(10)]
                mean_correct_per_trial = np.mean(correct_per_trial)

                parsed_data['CorrectKeyPressesPerS'][data['first_line_index']] = mean_correct_per_trial

                if last_value is not None:
                    delta = mean_correct_per_trial - last_value if mean_correct_per_trial is not None else None
                    parsed_data['KeypressSpeed'][data['first_line_index']] = delta

                last_value = mean_correct_per_trial

            trial_index += 1

    return pd.DataFrame(parsed_data)

# Function to write analysis results to CSV
def write_data_to_csv(dataframe, output_file_path):
    dataframe.to_csv(output_file_path, index=False)

def process_all_data_files(input_folder, output_folder, target_sequences, trial_counts, num_trials_to_plot=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    all_correct_presses = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            output_file_name = filename[:4] + '.csv'
            output_file_path = os.path.join(output_folder, output_file_name)
            
            dataframe = parse_and_analyse_data(file_path, target_sequences, trial_counts)
            all_correct_presses.append(dataframe['CorrectKeyPressesPerS'].dropna().tolist())
            write_data_to_csv(dataframe, output_file_path)

    # Calculate and plot data
    trial_means, trial_sems = calculate_and_plot_correct_presses(all_correct_presses, num_trials_to_plot, trial_counts)

    return trial_means

def calculate_and_plot_correct_presses(all_correct_presses, num_trials_to_plot, trial_counts):
    if num_trials_to_plot is None:
        num_trials_to_plot = len(all_correct_presses[0])  # Default to the number of trials in the first participant's data

    trial_means = []
    trial_sems = []
    x_values = list(range(1, num_trials_to_plot + 1))
    
    # Define colors to alternate
    colors = ['blue', 'green', 'red', 'mediumpurple', 'purple', 'cyan', 'magenta', 'gold', 'hotpink', 'springgreen', 'navy', 'crimson']
    
    # Prepare for plotting
    plt.figure(figsize=(26, 5))
    
    start = 0
    for idx, count in enumerate(trial_counts):
        end = start + count
        color = colors[idx % len(colors)]
        
        segment_x_values = x_values[start:end]
        segment_means = []
        segment_sems = []

        # Scatter plot for individual participant data points
        for participant_data in all_correct_presses:
            if len(participant_data) >= end:
                plt.scatter(segment_x_values, participant_data[start:end], color='lightgrey', alpha=0.75)

        for i in range(start, end):
            trial_data = [participant[i] for participant in all_correct_presses if len(participant) > i]
            trial_mean = np.mean(trial_data)
            trial_sem = np.std(trial_data, ddof=1) / np.sqrt(len(trial_data))
            segment_means.append(trial_mean)
            segment_sems.append(trial_sem)

        # Plot the mean values with SEM error bars, connected with a line
        plt.errorbar(segment_x_values, segment_means, yerr=segment_sems, fmt='-o', color=color, ecolor=color, alpha=0.75, capsize=5)
        
        start = end

    plt.title(f'Mean Correct Key Presses per S in {cond}')
    plt.xlabel('Trial Number')
    plt.ylabel('Correct Key Presses per S')
    plt.xticks(x_values)
    plt.show()

    return trial_means, trial_sems

# Example call with specified number of trials to plot
input_folder = f'/Users/wi11iamk/Desktop/PhD/PsyToolkit/{cond}/data'
output_folder = f'/Users/wi11iamk/Desktop/PhD/csvOutput/UCL/{cond}'
trial_means = process_all_data_files(input_folder, output_folder, target_sequences, trial_counts, num_trials_to_plot=84)
