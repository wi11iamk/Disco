#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 15:03:56 2024

@author: wi11iamk
"""
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
    '051': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.165, 'y_threshold': 10},
    '054': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '056': {'min_cluster_size': 500, 'cluster_selection_epsilon': 0.140, 'y_threshold': 35},
    '058': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.140, 'y_threshold': 10},
    '066': {'min_cluster_size': 400, 'cluster_selection_epsilon': 0.165, 'y_threshold': 10},
    '067': {'min_cluster_size': 300, 'cluster_selection_epsilon': 0.120, 'y_threshold': 10}
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
