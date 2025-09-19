import tkinter as tk
from tkinter import filedialog, ttk
import ttkbootstrap as tb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import Counter
from pandas import read_csv, DataFrame

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pipeline.classifier import ECGClassifier
from ui.appui import ECGUI
from controllers.appcontroller import AppController
from pipeline.dataloader import ECGDataLoader

def main():
    classif = ECGClassifier(
        model_path='model/model_v5.sav',
        encoder_path='model/label_encoder_v5.sav',
        scalar_path='model/scalar_v5.sav'
    )
    
    loader = ECGDataLoader()
    
    app_controller = AppController(
        classifier=classif,
        dataloader=loader
    )
    
    app = ECGUI(controller=app_controller)

if __name__ == "__main__":
    main()
    tk.mainloop()