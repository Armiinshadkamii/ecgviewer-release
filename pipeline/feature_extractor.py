"""
Feature extraction module for ECG signals.
It needs 2 inputs:
1. beat_idx_list: List of indices where the R-peaks are located.
2. signal: The ECG signal data as a numpy array.
The output will be a set of features extracted from the ECG segments around the R-peaks.
The features include RR intervals, min, max, mean voltages, std voltage, R-peak amplitude
p wave width and QRS duration.
"""

from typing import Dict, Union, List, Optional
from numpy import ndarray
import numpy as np
from pipeline.dataloader import ECGDataLoader
import matplotlib.pyplot as plt
from pandas import DataFrame
from collections import Counter


def moving_average(signal, window_size=5):
    """
    Gets the average of window_size consequitive sample
    values in a signal to make it smooth.
    :param signal - The ECG signal data as a numpy array.
    :param window_size - number of samples the moving avg
    should work on
    :return : smoothed signal as a numpy array
    """
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def balance_save(data : list, filename: str = 'ecg_train.csv') -> None:
    """
    Balance the data using the label of each beat segment.
    Save the extracted features to a CSV file.
    :param data: List of dictionaries containing the features.
    :param filename: Name of the output CSV file.
    """
    if not data:
        print('No data to save.')
        return
    
    df = DataFrame(data)
    scaled_df = (
        df.groupby('label')
        .apply(lambda x: x.sample(n=min(len(x), 500), random_state=42))
        .round(3)
        .reset_index(drop=True)
    )
    
    scaled_df.to_csv(filename, index=False)
    print(f'Features saved to {filename}')

def save(data: List[Dict[str, Union[ndarray, List[str], List[int]]]], filename: str = 'mit-bih-2.csv') -> None:
    """
    Save the extracted features to a CSV file.
    :param data: List of dictionaries containing the features.
    :param filename: Name of the output CSV file.
    """
    if not data:
        print('No data to save.')
        return
    
    df = DataFrame(data)
    df.to_csv(filename, index=False)
    print(f'Features saved to {filename}')

class ECGFeatureExtractor:
    def __init__(self, data_dict : List[Dict[str, Union[ndarray, List[int]]]]):
        self.data_dict_list = data_dict
        self.sample_rate = 360
        self.beat_data = []

    def get_beat_data(self) -> List[Dict[str, Union[ndarray, List[str], List[int]]]]:
        return self.beat_data
    
    def run_extractor(self) -> None:
        """
        Extract features from ECG segments around R-peaks.
        Features include RR intervals, min, max, mean voltages,
        std voltage, R-peak amplitude, p wave width, and QRS duration.
        1. For each beat index in the beat_idx_list, extract a segment of the
           ECG signal around the beat index.
        2. Calculate the features for each segment.
        3. Store the features in a list of dictionaries.
        """
        rows = []
        # Iter over mit-bih records
        for data_dict in self.data_dict_list:

            beat_idx_list : List[str] | List[int] | ndarray = data_dict['beat_idx_list']
            # labels : List[str] | List[int] | ndarray = data_dict['labels']
            signals = data_dict['signal']

            seg_window_size = 100
            
            # Iter over each beat index to extract features
            for idx in range(1, len(beat_idx_list)):
                seg_start = max(int(beat_idx_list[idx]) - seg_window_size, 0)
                seg_end = min(int(beat_idx_list[idx]) + seg_window_size, len(signals))
                
                if isinstance(signals, ndarray):
                    if signals.ndim > 1:
                        segment : ndarray = signals[seg_start:seg_end, 0]
                    else:
                        segment : ndarray = signals[seg_start:seg_end]
                    
                    rr_int = self.rr_interval(beat_idx_list[idx], beat_idx_list[idx - 1])
                    min_v = self.min_v(segment)
                    max_v = self.max_v(segment)
                    mean_v = self.mean_v(segment)
                    std_v = self.std_v(segment)
                    r_peak_amp = self.r_peak_amplitude(int(beat_idx_list[idx]), signals)
                    
                    qrs_info = self.qrs_duration(segment)
                    if not qrs_info:
                        continue
                    
                    pw_width = self.p_wave_width(
                        beat_idx=beat_idx_list[idx],
                        seg_s=seg_start,
                        qrs_s=qrs_info['qrs_start'],
                        sig=signals, seg=segment
                    )

                    if not pw_width:
                        continue

                    row = {
                        "rr_interval": rr_int,
                        "r_peak_amp": r_peak_amp,
                        "qrs_duration": qrs_info['qrs_duration'],
                        "max_voltage": max_v,
                        "min_voltage": min_v,
                        "mean_voltage": mean_v,
                        "std_voltage": std_v,
                        "p_wave_width": pw_width
                    }
                    self.beat_data.append(
                        {
                            "segment": segment,
                            "features" : row,
                            "plot_ready": True
                        }
                    )
                    # rows.append(row)
        
        # balance_save(rows, 'ecg_features.csv')
        # save(rows)

    def rr_interval(self, curr_beat_idx, prev_beat_idx):
        """
        the time interval between two successive R-waves of the QRS complex on the ECG.
        :param curr_beat_idx - index of the current beat
        :param prev_beat_idx - index of the previous beat
        :return - rr interval in seconds
        """
        return (curr_beat_idx - prev_beat_idx) / self.sample_rate
        
    def min_v(self, seg: ndarray):
        """
        minimum voltage in the ECG segment
        """
        return seg.min()

    def max_v(self, seg: ndarray):
        """
        maximum voltage in the ECG segment
        """
        return seg.max()
    
    def mean_v(self, seg: ndarray):
        """
        Average voltage in the ECG segment
        """
        return seg.mean()
    
    def std_v(self, seg: ndarray):
        """
        Standard deviation of voltage in the ECG segment
        """
        return seg.std()
    
    def r_peak_amplitude(self, beat_idx : int, signals: ndarray):
        """
        returns the amplitude of the R-peak at the given beat index
        :param beat_idx - index of the beat
        :param signals - the ECG signal data as a numpy array
        :return - amplitude of the R-peak
        """
        if signals.ndim > 1:
            return signals[beat_idx, 0]
        else:
            return signals[beat_idx]

    def qrs_duration(self, seg: ndarray) -> Optional[Dict]:
        '''
        qrs duration is the electrical signal that triggers
        the ventricles to contract and pump blood to the lungs
        and body.
        1) get slopes
        2) get the steepest slopes based on a threshold
        3) get the start and end indices of the QRS complex
        4) calculate the duration based on the sample rate
        '''
        threshold = 0.2 * self.std_v(seg)
        slopes : ndarray = np.diff(seg)

        steep_slopes_indices : ndarray = np.where(np.abs(slopes) > threshold)[0]
        
        if steep_slopes_indices.size == 0:
            # print('empty steep slopes')
            return None

        qrs_start = steep_slopes_indices[0]
        qrs_end = steep_slopes_indices[-1]

        if qrs_start >= qrs_end:
            # print('invalid QRS start and end indices')
            return None
        indices_distance = qrs_end - qrs_start
        
        qrs_duration = indices_distance / self.sample_rate

        return {
            "qrs_start" : qrs_start,
            "qrs_end" : qrs_end,
            "qrs_duration" : qrs_duration
        }
        
    def p_wave_width(self,beat_idx, seg_s, qrs_s, sig, seg):
        '''
        P wave is the signal that tells the atria: “Time to contract
        and push blood into the ventricles.
        1) calculate the start and end of p wave width's possible range
        2) smooth the signal
        3) get slopes based on a threshold
        4) calculate the start and end of p wavd width
        '''
        poss_range_s = max(qrs_s - 100, 0)
        poss_range_e = qrs_s

        possible_pwave_range = seg[poss_range_s:poss_range_e]
        if not possible_pwave_range.size:
            return None
        smoothed_pwave_range = moving_average(possible_pwave_range, window_size=5)

        grads : ndarray = np.gradient(smoothed_pwave_range)
        # ignore the first and last 10 samples that may be misleading
        ign = 10
        grads[:ign] = 0
        grads[-ign:] = 0

        # Find the value such that 85% of the data points are smaller than it,
        threshold = np.percentile(np.abs(grads), 85)
        # Keep the slopes steepr than the threshold
        high_grad_indices : ndarray = np.where(np.abs(grads) > threshold)[0]
        
        if high_grad_indices.size == 0:
            return None
        
        pw_start = high_grad_indices[0]
        pw_end = high_grad_indices[-1]

        if pw_start >= pw_end:
            return None
        
        dist = pw_end - pw_start
        p_wave_width = dist / self.sample_rate
        
        return p_wave_width