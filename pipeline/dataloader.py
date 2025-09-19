"""
ECG DataLoader for loading ECG records from various sources.
This module supports loading from WFDB format and CSV files.

Input : raw ECG data
Output: List of dictionaries containing:
- signal: The ECG signal data as a numpy array.
- beat_idx_list: List of indices where the R-peaks are located. 

"""

import numpy as np
from numpy import ndarray
from typing import Dict, Union, List
from wfdb import Annotation, Record, MultiRecord, rdann, rdrecord
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.io
import wfdb


def mitdb_labels():
    """
    Output the standard labels used in the MIT-BIH Arrhythmia Database.
    """
    return ['N', 'S', 'V', 'F', 'Q']

def index_beats(signal : ndarray, sample_rate : int = 360) -> List[int]:
    """
    Detects beat locations (R-peaks) in an ECG signal and returns their sample indices.

    Parameters:
    - signal: 1D numpy array of ECG signal
    - sample_rate: Sampling rate of the signal (default: 360 Hz)

    Returns:
    - List of sample indices corresponding to detected beats
    """
    dist = int(0.25 * sample_rate)  # Minimum distance between peaks (250 ms)
    peaks, _ = find_peaks(signal, prominence=0.6, distance=dist)

    return peaks.tolist()

class ECGDataLoader:
    def __init__(self):
        self.sources = ['mitdb, csv']
        self.csv_path = None
    
    def load(self, src) -> List[Dict[str, Union[ndarray, List[int]]]]:
        """
        Interface to load ECG data from various sources.
        :param src: Source of the data. Options include:
            - "mitdb": Load from MIT-BIH Arrhythmia Database (WFDB format)
            - "other": Load from a CSV file (provide file path)
        :return: List of dictionaries containing:
            - signal: The ECG signal data as a numpy array.
            - beat_idx_list: List of indices where the R-peaks are located.
        """
        # if src == "mitdb":
        #     return self.load_mitdb(src=src)
        if src.endswith('.txt') or src.endswith('.csv'):
            return self.upload_csv(file_path=src)
        elif src.endswith('.hea'):
            output_path = self.convert_wfdb_to_plain_signal(input_path=src)
            if output_path:
                return self.upload_csv(file_path=output_path)
            else:
                print("Failed to convert WFDB to plain signal.")
                return []
        else:
            print(f"Unsupported source: {src}")
            return []
    
    def upload_csv(self, file_path : str) -> List[Dict[str, Union[ndarray, List[int]]]]:
        """
        Upload ECG data from a CSV file.
        :param file_path: Path to the CSV file containing ECG signal data.
        :return: List of dictionaries containing:
            - signal: The ECG signal data as a numpy array.
            - beat_idx_list: List of indices where the R-peaks are located.
        """
        if file_path is None:
            return []

        # Ensure the file has no header and is 
        # a single column of signal values.
        # If there are text values in the file,
        # execution will stop here. 
        try:
            signal = np.loadtxt(file_path, delimiter=',')
        except ValueError as e:
            print(f"Error loading CSV file: {e}")
            return []
        
        num_dim = signal.ndim

        # Works only with 1 dimensional signals
        if num_dim == 1:
            beat_idx_list = index_beats(signal=signal)

            return [{
                "signal" : signal,
                "beat_idx_list" : beat_idx_list
                }
            ]
        else:
            print('Signal has unsupported number of dimensions for CSV export.')
            return []
        

    def convert_wfdb_to_plain_signal(self, input_path, output_path=None):
        """
        Converts a WFDB record (.dat/.hea) into a plain CSV/TXT file
        containing only signal values from the first lead. No headers.
        """
        record_name = os.path.splitext(os.path.basename(input_path))[0]
        record_dir = os.path.dirname(input_path)

        # Read WFDB record
        record = wfdb.rdrecord(os.path.join(record_dir, record_name))
        if isinstance(record, Record) and isinstance(record.p_signal, np.ndarray):
            signal = record.p_signal[:, 0]  # First lead only

            # Define output path
            if output_path is None:
                parent_dir = os.path.dirname(input_path)
                second_parent_dir = os.path.dirname(parent_dir)
                output_path = os.path.join(second_parent_dir, f"{record_name}_signal.txt")

            # Save signal values (one per line, no header)
            np.savetxt(output_path, signal, fmt='%.6f')

            return output_path
        else:
            print(f"Error: Unable to read record or signal from {input_path}")

    