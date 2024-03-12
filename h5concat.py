# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 06:34:50 2024

@author: kistlerwd
"""

import h5py
import numpy as np

def concatenate_datasets_vertically(file_paths, dataset_path, output_file_path):
    """
    Concatenates datasets from multiple HDF5 files vertically and saves the result in a new HDF5 file.

    :param file_paths: List of paths to the HDF5 files.
    :param dataset_path: Path to the dataset within the HDF5 files to concatenate.
    :param output_file_path: Path where the output HDF5 file will be saved.
    """
    concatenated_data = None
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            # Read the dataset
            data = f[dataset_path][:]
            if concatenated_data is None:
                concatenated_data = data
            else:
                concatenated_data = np.concatenate((concatenated_data, data), axis=0)
    
    # Save the concatenated dataset into a new HDF5 file
    with h5py.File(output_file_path, 'w') as fout:
        fout.create_dataset(dataset_path, data=concatenated_data)

def get_dataset_shape(file_path, dataset_path):
    """
    Returns the shape of a specified dataset within an HDF5 file.

    :param file_path: Path to the HDF5 file.
    :param dataset_path: Path to the dataset within the HDF5 file.
    """
    with h5py.File(file_path, 'r') as file:
        return file[dataset_path].shape

# Example usage
file_paths = ['C:\\Users\\kistlerwd\\Desktop\\concatTest\\012_D1DLC_resnet50_keyTest012Jan06shuffle1_700000.h5', 
              'C:\\Users\\kistlerwd\\Desktop\\concatTest\\027_D1DLC_resnet50_keyTest027Jan12shuffle1_400000.h5', 
              'C:\\Users\\kistlerwd\\Desktop\\concatTest\\049_D1DLC_resnet50_keyTest049Jan27shuffle1_600000.h5']  # Add your file paths here
dataset_path = 'df_with_missing/table'  # Specify the dataset path inside your files
output_file_path = 'C:\\Users\\kistlerwd\\Desktop\\concatTest\\concatTest.h5'  # Define the output file path

# Concatenate the datasets
concatenate_datasets_vertically(file_paths, dataset_path, output_file_path)

# Verify the shape of the concatenated dataset
concatenated_shape = get_dataset_shape(output_file_path, dataset_path)
print(f"Shape of the concatenated dataset: {concatenated_shape}")
