from joblib import load
from pandas import DataFrame
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import  matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from pandas import Series

class ECGClassifier:
    """
    ECG Classifier for detecting heartbeats and classifying them.
    This module uses a pre-trained model to classify heartbeats based on extracted features.
    """
    
    def __init__(self, model_path: str, encoder_path: str, scalar_path: str):
        """
        Initialize the ECGClassifier with a pre-trained model.
        
        :param model_path: Path to the pre-trained model file.
        """
        self.model: KNeighborsClassifier = self.load_model(model_path)
        self.scalar: StandardScaler = self.load_scalar(scalar_path)
        self.encoder: LabelEncoder = self.load_encoder(encoder_path)
    
    def load_model(self, model_path: str):
        """
        Load the pre-trained model from the specified path.
        
        :param model_path: Path to the pre-trained model file.
        :return: Loaded model.
        """

        return load(model_path)
    
    def load_scalar(self, scalar_path):
        """
        Load the scalar for feature scaling.

        :param scalar_path: Path to the scalar file.
        :return: Loaded scalar.
        """

        return load(scalar_path)
    
    def load_encoder(self, encoder_path: str):
        """
        Load the encoder for feature transformation.
        
        :param encoder_path: Path to the encoder file.
        :return: Loaded encoder.
        """

        return load(encoder_path)
    
    def classify(self, x):
        # try:
        x_df = DataFrame([x])
        x_scaled = self.scalar.transform(x_df)
        
        # Get probability estimates
        y_proba = self.model.predict_proba(x_scaled)  # shape: (1, n_classes)
        y_pred_numeric = np.argmax(y_proba, axis=1)   # index of highest probability
        confidence = np.max(y_proba) * 100            # convert to percentage
        
        y_pred_labels = self.encoder.inverse_transform(y_pred_numeric)

        return y_pred_labels[0], confidence
        # except:
        #     print("Error during classification. Ensure the input data is correctly formatted.")
        #     return "Unknown"
    
    def get_score(self, x, y):
        x_scaled = self.scalar.transform(x)
        y_encoded = self.encoder.transform(y)
        return self.model.score(x_scaled, y_encoded)
    
    def display_confusion_matrix(self, x, y):
        x_scaled = self.scalar.transform(x)
        y_encoded = self.encoder.transform(y)
        y_pred = self.model.predict(x_scaled)
        labels = self.encoder.transform(self.encoder.classes_)

        matrix = confusion_matrix(y_encoded, y_pred, labels=labels)
        # Display the matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title("Confusion Matrix: ECG Classification")
        plt.tight_layout()
        plt.show()
