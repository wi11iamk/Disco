#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 15:03:56 2024

@author: wi11iamk
"""
# Initialise a dictionary to store participant specific parameters
participants = {
    '012': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.14, 'y_threshold': 10},
    '014': {'min_cluster_size': 220, 'cluster_selection_epsilon': 0.145, 'y_threshold': 15},
    '015': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.14, 'y_threshold': 10},
    '016': {'min_cluster_size': 170, 'cluster_selection_epsilon': 0.14, 'y_threshold': 10},
    '017': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.145, 'y_threshold': 10},
    '018': {'min_cluster_size': 250, 'cluster_selection_epsilon': 0.15, 'y_threshold': 10},
    '027': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.14, 'y_threshold': 10},
    '028': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.15, 'y_threshold': 10},
    '029': {'min_cluster_size': 250, 'cluster_selection_epsilon': 0.15, 'y_threshold': 145},
    '036': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.135, 'y_threshold': 10},
    '037': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.135, 'y_threshold': 10},
    '039': {'min_cluster_size': 265, 'cluster_selection_epsilon': 0.15, 'y_threshold': 10},
    '044': {'min_cluster_size': 160, 'cluster_selection_epsilon': 0.125, 'y_threshold': 10},
    '049': {'min_cluster_size': 220, 'cluster_selection_epsilon': 0.16, 'y_threshold': 10},
    '051': {'min_cluster_size': 250, 'cluster_selection_epsilon': 0.165, 'y_threshold': 10},
    '054': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.14, 'y_threshold': 10},
    '056': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.14, 'y_threshold': 35},
    '058': {'min_cluster_size': 200, 'cluster_selection_epsilon': 0.14, 'y_threshold': 10},
    '066': {'min_cluster_size': 250, 'cluster_selection_epsilon': 0.165, 'y_threshold': 10},
    '067': {'min_cluster_size': 190, 'cluster_selection_epsilon': 0.135, 'y_threshold': 10}
    # Add more participants as needed
}