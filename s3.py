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
from itertools import chain

# Define the conditions and their respective target sequences
conditions = {
    'C1': ["4 1 3", "2 4 1 3 4", "2 4 1", "1 2 4 1 3", "1 4 3", "3 1 4 3 2", 
           "3 4 2", "2 3 4 2 3", "3 2 4", "4 3 2 4 1", "2 3 4", "3 2 3 4 2"],
    'C2': ["2 4 1", "1 2 4 1 3", "1 4 3", "3 1 4 3 2", "2 3 4", "3 2 3 4 2", 
           "4 1 3", "2 4 1 3 4", "3 4 2", "2 3 4 2 3", "3 2 4", "4 3 2 4 1"],
    'C3': ["1 4 3", "3 1 4 3 2", "2 3 4", "3 2 3 4 2", "3 2 4", "4 3 2 4 1", 
           "2 4 1", "1 2 4 1 3", "4 1 3", "2 4 1 3 4", "3 4 2", "2 3 4 2 3"],
    'C4': ["2 3 4", "3 2 3 4 2", "3 2 4", "4 3 2 4 1", "3 4 2", "2 3 4 2 3", 
           "1 4 3", "3 1 4 3 2", "2 4 1", "1 2 4 1 3", "4 1 3", "2 4 1 3 4"],
    'C5': ["3 2 4", "4 3 2 4 1", "3 4 2", "2 3 4 2 3", "4 1 3", "2 4 1 3 4", 
           "2 3 4", "3 2 3 4 2", "1 4 3", "3 1 4 3 2", "2 4 1", "1 2 4 1 3"],
    'C6': ["3 4 2", "2 3 4 2 3", "4 1 3", "2 4 1 3 4", "2 4 1", "1 2 4 1 3", 
           "3 2 4", "4 3 2 4 1", "2 3 4", "3 2 3 4 2", "1 4 3", "3 1 4 3 2"]
}

# Define the transitions of interest for each condition
transition_pairs = {
    'C1': [[(4, 1), (1, 3)], [(4, 1), (1, 3)], [(2, 4), (4, 1)], [(2, 4), (4, 1)], 
           [(1, 4), (4, 3)], [(1, 4), (4, 3)], [(3, 4), (4, 2)], [(3, 4), (4, 2)], 
           [(3, 2), (2, 4)], [(3, 2), (2, 4)], [(2, 3), (3, 4)], [(2, 3), (3, 4)]],
    'C2': [[(2, 4), (4, 1)], [(2, 4), (4, 1)], [(1, 4), (4, 3)], [(1, 4), (4, 3)], 
           [(2, 3), (3, 4)], [(2, 3), (3, 4)], [(4, 1), (1, 3)], [(4, 1), (1, 3)], 
           [(3, 4), (4, 2)], [(3, 4), (4, 2)], [(3, 2), (2, 4)], [(3, 2), (2, 4)]],
    'C3': [[(1, 4), (4, 3)], [(1, 4), (4, 3)], [(2, 3), (3, 4)], [(2, 3), (3, 4)], 
           [(3, 2), (2, 4)], [(3, 2), (2, 4)], [(2, 4), (4, 1)], [(2, 4), (4, 1)], 
           [(4, 1), (1, 3)], [(4, 1), (1, 3)], [(3, 4), (4, 2)], [(3, 4), (4, 2)]],
    'C4': [[(2, 3), (3, 4)], [(2, 3), (3, 4)], [(3, 2), (2, 4)], [(3, 2), (2, 4)], 
           [(3, 4), (4, 2)], [(3, 4), (4, 2)], [(1, 4), (4, 3)], [(1, 4), (4, 3)], 
           [(2, 4), (4, 1)], [(2, 4), (4, 1)], [(4, 1), (1, 3)], [(4, 1), (1, 3)]],
    'C5': [[(3, 2), (2, 4)], [(3, 2), (2, 4)], [(3, 4), (4, 2)], [(3, 4), (4, 2)], 
           [(4, 1), (1, 3)], [(4, 1), (1, 3)], [(2, 3), (3, 4)], [(2, 3), (3, 4)], 
           [(1, 4), (4, 3)], [(1, 4), (4, 3)], [(2, 4), (4, 1)], [(2, 4), (4, 1)]],
    'C6': [[(3, 4), (4, 2)], [(3, 4), (4, 2)], [(4, 1), (1, 3)], [(4, 1), (1, 3)], 
           [(2, 4), (4, 1)], [(2, 4), (4, 1)], [(3, 2), (2, 4)], [(3, 2), (2, 4)], 
           [(2, 3), (3, 4)], [(2, 3), (3, 4)], [(1, 4), (4, 3)], [(1, 4), (4, 3)]]
}

# Define the seed_soil_pairs with assigned colors
seed_soil_pairs = {
    3: {'pairs': [(4, 1), (1, 3)], 'color': 'blue'},
    4: {'pairs': [(4, 1), (1, 3)], 'color': 'green'}, 5: {'pairs': [(4, 1), (1, 3)], 'color': 'green'}, 6: {'pairs': [(4, 1), (1, 3)], 'color': 'green'},
    13: {'pairs': [(2, 4), (4, 1)], 'color': 'red'},
    14: {'pairs': [(2, 4), (4, 1)], 'color': 'mediumpurple'}, 15: {'pairs': [(2, 4), (4, 1)], 'color': 'mediumpurple'}, 16: {'pairs': [(2, 4), (4, 1)], 'color': 'mediumpurple'},
    31: {'pairs': [(1, 4), (4, 3)], 'color': 'purple'},
    32: {'pairs': [(1, 4), (4, 3)], 'color': 'cyan'}, 33: {'pairs': [(1, 4), (4, 3)], 'color': 'cyan'}, 34: {'pairs': [(1, 4), (4, 3)], 'color': 'cyan'},
    45: {'pairs': [(3, 4), (4, 2)], 'color': 'magenta'},
    46: {'pairs': [(3, 4), (4, 2)], 'color': 'gold'}, 47: {'pairs': [(3, 4), (4, 2)], 'color': 'gold'}, 48: {'pairs': [(3, 4), (4, 2)], 'color': 'gold'},
    55: {'pairs': [(3, 2), (2, 4)], 'color': 'hotpink'},
    56: {'pairs': [(3, 2), (2, 4)], 'color': 'springgreen'}, 57: {'pairs': [(3, 2), (2, 4)], 'color': 'springgreen'}, 58: {'pairs': [(3, 2), (2, 4)], 'color': 'springgreen'},
    73: {'pairs': [(2, 3), (3, 4)], 'color': 'navy'},
    74: {'pairs': [(2, 3), (3, 4)], 'color': 'crimson'}, 75: {'pairs': [(2, 3), (3, 4)], 'color': 'crimson'}, 76: {'pairs': [(2, 3), (3, 4)], 'color': 'crimson'}
}

# Set the condition
cond = 'C6'
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
    plt.figure(figsise=(26, 10))

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

        for participant_data in all_correct_presses:
            if len(participant_data) >= end:
                plt.scatter(segment_x_values, participant_data[start:end], color='lightgrey', alpha=0.75)

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
            plt.errorbar(len(xtick_labels) - 1, mean_speed, yerr=sem_speed, fmt='o', color=seed_soil_pairs[trial]['color'], alpha=0.75)
        
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
