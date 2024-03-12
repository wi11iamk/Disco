# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 06:16:31 2024

@author: kistlerwd
"""

import h5py

# Interogate the contents of an .h5 file generated with DeepLabCut

def read_h5_info(file_path):
    """
    Reads an HDF5 file and prints the names of all groups and datasets,
    along with the shapes and data types of datasets.
    """
    with h5py.File(file_path, 'r') as file:
        if dataset_path in file:
            dataset = file[dataset_path]
            # Print a small sample of the data
            print("Sample data (first 5 rows):")
            print(dataset[:5])
        else:
            print(f"Dataset '{file_path}' not found in the file.")
        def explore(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Data Type: {obj.dtype}")
            else:  # Assuming it's a group if not a dataset
                print(f"Group: {name}")
        file.visititems(explore)

# Example usage
file_path = '/Users/wi11iamk/Documents/GitHub/HUB_DT/sample_data/027_D1DLC_resnet50_keyTest027Jan12shuffle1_400000.h5'
dataset_path = 'df_with_missing/table'  # The path to the dataset within the .h5 file
read_h5_info(file_path)

