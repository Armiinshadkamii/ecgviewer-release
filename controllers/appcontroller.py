"""
App Controller Module
----------------------
This module contains the AppController class which manages the flow of data
between the ECGClassifier, ECGFeatureExtractor, and the user interface.
"""


from typing import List, Dict, Union
from numpy import ndarray
from pipeline.classifier import ECGClassifier
from pipeline.feature_extractor import ECGFeatureExtractor
from pipeline.dataloader import ECGDataLoader


class AppController:
    """
    This class serves as a controller for the ECG application, managing the flow of data
    between the model, feature extractor, and the user interface.
    """
    def __init__(self, classifier, dataloader):
        """
        Initialize the AppController with a classifier instance.
        
        :param classifier: An instance of ECGClassifier to handle heartbeat classification.
        :param ui: An instance of ECGUI to manage the user interface.
        :param rec_ids: List of record IDs to load ECG data.
        """
        self.classifier: ECGClassifier = classifier
        self.data_loader: ECGDataLoader = dataloader

        self.ecg_beat_data_list = []
        self.idx = 0

    def load_data(self, src):
        data = self.data_loader.load(src)
        
        feature_xtractor = ECGFeatureExtractor(data)
        feature_xtractor.run_extractor()

        self.ecg_beat_data_list: List[Dict[str, Union[ndarray, List[str], List[int]]]] = feature_xtractor.get_beat_data()

    
    def get_segment(self, idx) -> ndarray | None:
        seg_data_dic = self.ecg_beat_data_list[idx]
        segment = seg_data_dic['segment']
        if isinstance(segment, ndarray):
            return segment
        else:
            print("Invalid segment data for plotting")
            return None
    
    def get_segment_features(self, current_idx) -> dict:
        return self.ecg_beat_data_list[current_idx]['features']
    
    def get_segment_data(self, idx):
        assert self.ecg_beat_data_list, "ecg beat data is empty"
        assert len(self.ecg_beat_data_list) - 1 >= idx, "index is out of the range of ecg_beat_data_list"
        
        return self.ecg_beat_data_list[idx]
    
    def get_class(self, idx):
        x = self.ecg_beat_data_list[idx]['features']
        return self.classifier.classify(x)
    
    
    def get_samples_len(self) -> int:
        """
        Get the number of samples available for classification.
        
        :return: The length of the ECG beat data list.
        """
        return len(self.ecg_beat_data_list)
    
    def get_class_description(self, class_label: str) -> str:
        descriptions = {
            "N" : "Normal beat",
            "V" : "Ventricular",
            "S" : "Supraventricular",
            "F" : "Fusion of ventricular and normal",
            "Q" : "Unknown beat",
            "U" : "unknown beat"
        }

        return descriptions.get(class_label, "No description available")