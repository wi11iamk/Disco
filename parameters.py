#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 15:03:56 2024

@author: wi11iamk
"""

###
# S1 and S2 Parameters
###

# Initialise a dictionary to store participant specific parameters
D1 = {
    '012': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '014': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.145, 'y_threshold': 15},
    '015': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '016': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '017': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.145, 'y_threshold': 10},
    '018': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '027': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '028': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 0},
    '029': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.150, 'y_threshold': 145},
    '036': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.135, 'y_threshold': 10},
    '037': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.135, 'y_threshold': 10},
    '039': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.155, 'y_threshold': 20},
    '044': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.130, 'y_threshold': 10},
    '049': {'min_cluster_size': 400, 'cluster_selection_epsilon': 0.150, 'y_threshold': 10},
    '051': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.165, 'y_threshold': 22},
    '054': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '056': {'min_cluster_size': 500, 'cluster_selection_epsilon': 0.140, 'y_threshold': 30},
    '058': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '066': {'min_cluster_size': 400, 'cluster_selection_epsilon': 0.165, 'y_threshold': 10},
    '067': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.120, 'y_threshold': 10},
    
    '003': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '004': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '005': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '006': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '007': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '069': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '070': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '071': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '072': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '074': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '075': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '078': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '080': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '082': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '083': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '084': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '088': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '089': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '090': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '091': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10}
    # Add more participants as needed
}

D2 = {
    '036': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.10, 'y_threshold': 10},
    '037': {'min_cluster_size': 210, 'cluster_selection_epsilon': 0.10, 'y_threshold': 10},
    '039': {'min_cluster_size': 265, 'cluster_selection_epsilon': 0.12, 'y_threshold': 10},
    '049': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.13, 'y_threshold': 10},
    '051': {'min_cluster_size': 250, 'cluster_selection_epsilon': 0.12, 'y_threshold': 10},
    '054': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.10, 'y_threshold': 10},
    '056': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.10, 'y_threshold': 35},
    '058': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.10, 'y_threshold': 10},
    '066': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.12, 'y_threshold': 10},
    '067': {'min_cluster_size': 160, 'cluster_selection_epsilon': 0.10, 'y_threshold': 10}
    # Add more participants as needed
}

###
# S3 Parameters
###

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

video_info = {
    "Above": {
        "009": (16, [2.25, 2.25, 3.8], "C3"),
        "010": (36, [1, 1.75, 3.5], "C3"),
        "011": (10, [3.25, 4, 6.75], "C4"),
        "012": (18, [1.75, 2, 4.25], "C4"),
        "013": (36, [2.25, 2.5, 12], "C5"),
        "014": (15, [2.75, 6, 14], "C5"),
        "015": (15, [1.5, 2.5, 10.75], "C6"),
        "016": (19, [2, 5.25, 10.5], "C6"),
        "019": (10, [2.5, 6, 13.125], "C2"),
        "020": (6, [2.33, 5.25, 13.5], "C2"),
        "021": (10, [2.25, 2.5, 7.75], "C3"),
        "022": (19, [1.75, 3, 5], "C3"),
        "023": (19, [1.75, 3.25, 5], "C4"),
        "024": (8, [2.75, 2.33, 4.75], "C4"),
        "025": (10, [2.75, 3, 7], "C5"),
        "026": (14, [2, 5.75, 11.75], "C5"),
        "027": (14, [3.25, 5.25, 10.25], "C6"),
        "028": (27, [1, 2, 4], "C6"),
        "029": (10, [1.25, 2.75, 5.25], "C1"),
        "030": (14, [1.5, 3.25, 6.5], "C1"),
        "031": (15, [1.5, 2, 12], "C2"),
        "032": (15, [4, 9.75, 15], "C2"),
        "033": (13, [4, 4.75, 4.75], "C3"),
        "034": (18, [3, 5.75, 11], "C3"),
        "035": (11, [2.25, 5, 9.5], "C4"),
        "036": (12, [3, 6, 10], "C4"),
        "037": (7, [3.25, 6, 10], "C5"),
        "038": (7, [3, 6.33, 10], "C5"),
        "039": (16, [2, 6.5, 14], "C6"),
        "040": (14, [1.5, 5.8, 12], "C6"),
        "041": (7, [2.5, 6, 10], "C1"),
        "042": (13, [3.25, 5, 9.5], "C1"),
        "043": (20, [2.75, 3.25, 5.25], "C2"),
        "044": (63, [2.25, 3.33, 5], "C2"),
        "045": (12, [1, 2, 4], "C3"),
        "046": (17, [2.5, 3.25, 4.75], "C3"),
        "047": (3, [3, 5.25, 9.5], "C4"),
        "048": (3, [3.75, 6.5, 10], "C4")
    }, 
    
    "Afront": {
        "017": (10, [1.5, 3, 5.5], "C3"),
        "018": (13, [2.5, 3, 5], "C3"),
        "019": (10, [2.5, 4.5, 9.5], "C5"),
        "020": (6, [3, 7, 13.5], "C5"),
        "021": (10, [3, 7, 14], "C6"),
        "022": (8, [7, 13, 22], "C6"),
        "023": (12, [2, 5.5, 13.5], "C1"),
        "024": (13, [2, 6.5, 13], "C1"),
        "025": (11, [1, 2, 4], "C2"),
        "026": (8, [2, 2, 3.25], "C2"),
        "027": (12, [1, 2, 4], "C3"),
        "028": (11, [1, 1.5, 3.5], "C3"),
        "029": (11, [2, 2, 4], "C4"),
        "030": (11, [2, 2, 3], "C4"),
        "032": (11, [3.5, 4, 8], "C6"),
        "033": (12, [2, 2.5, 5.5], "C6"),
        "034": (10, [2.5, 2.5, 4.5], "C1"),
        "035": (11, [2.5, 2.5, 4], "C1"),
        "036": (27, [3, 3.25, 6.5], "C2"),
        "037": (15, [2, 2, 3], "C2"),
        "038": (13, [2, 2, 5], "C3"),
        "039": (11, [3, 3.5, 6.5], "C3"),
        "040": (19, [1.5, 2, 5], "C4"),
        "041": (14, [1.5, 2, 5], "C4"),
        "042": (58, [2, 3, 6], "C5"),
        "043": (10, [3, 3.25, 5], "C5"),
        "044": (11, [2, 6, 11], "C6"),
        "045": (7, [3.75, 4.25, 8.25], "C6"),
    },
    
    "Aside": {
        "008": (14, [2, 5, 9.25], "C5"),
        "009": (10, [1.5, 5.5, 9.5], "C5"),
        "010": (6, [2, 4.5, 8.5], "C6"),
        "011": (15, [2, 5, 10.25], "C6"),
        "012": (12, [2, 4, 7.5], "C1"),
        "013": (12, [1, 2, 4], "C1"),
        "014": (10, [1.5, 5.25, 7], "C2"),
        "015": (9, [3.5, 4.5, 9], "C2"),
        "016": (30, [3, 4.25, 12], "C3"),
        "017": (14, [1.5, 5.5, 9.5], "C3"),
        "018": (15, [4, 4, 8.25], "C4"),
        "019": (11, [2.5, 5.5, 11], "C4"),
        "020": (9, [2, 2.5, 7], "C5"),
        "021": (12, [3.5, 4.25, 7.5], "C5"),
        "022": (11, [3.25, 3, 7.5], "C6"),
        "023": (6, [2, 5.25, 9.5], "C6"),
        "024": (7, [2, 4.5, 8], "C1"),
        "025": (10, [2, 4, 7.75], "C1"),
        "026": (7, [2, 4, 8], "C2"),
        "027": (10, [2.5, 4.5, 8.5], "C2"),
        "028": (21, [2, 3, 6], "C3"),
        "029": (13, [3.25, 5.5, 9.5], "C3"),
        "030": (37, [3, 3.25, 5], "C4"),
        "031": (24, [4, 4.5, 8], "C4"),
        "032": (13, [1, 2.5, 6.5], "C5"),
        "033": (11, [2.25, 5, 8.5], "C5"),
        "034": (35, [2.25, 3, 5], "C6"),
        "035": (15, [3.25, 6.5, 12], "C6"),
    }
}
