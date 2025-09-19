"""
App UI Module
----------------------
This module contains the ECGUI class which manages the graphical user interface
for the ECG application using Tkinter and Matplotlib.
"""

import tkinter as tk
from tkinter import filedialog, ttk
import ttkbootstrap as tb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import threading
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from controllers.appcontroller import AppController
from enum import Enum

class ButtonState(Enum):
    PAUSE = 1
    CONTINUE = 2


class ECGUI:
    def __init__(self, controller: AppController) -> None:
        self.root = tb.Window(themename="darkly")
        self.root.title("ECG Feature Extractor")
        self.root.geometry("800x600")
        self.controller = controller

        self.total_samples = 0
        self.engine_index = 0 # Index used by the classification engine
        self.ui_index = 0 # Index used by the UI for navigating segments
        self.current_btn = ButtonState.PAUSE

        self.pause = False
        self.data_loaded = False
        self.is_engine_done = False # Pause and continue button disabled when done
        self.end_result = []

        self.create_input_frame()
        self.create_side_frames()
        self.create_mtplt_fig()
        self.create_classify_button()
        self.create_next_seg_button()
        self.create_previous_seg_button()
        self.create_pause_continue_btn()

    def load_dataloader(self):
        """
        Load data using the controller's dataloader.
        pass in the selected dataset from the combo box.
        """
        self.controller.load_data(src=self.dataset_combo.get())
    
    def classify_heartbeat(self):
        """
        Classify the heartbeats using the controller and update the UI with results.
        """
        if self.is_engine_done:
            self.is_engine_done = False


        if not self.data_loaded:
            self.btn_classify.config(state='disabled')
            """
            Sends heavy data loading to another thread
            to prevent the ui from behaving strangely.
            since that thread runs asynchronously, you
            don’t want the rest of classify_heartbeat()
            to continue immediately (e.g., trying to 
            process samples before data is loaded).
            hence, the return statement.
            """
            threading.Thread(target=self._load_data_and_start).start()
            return

        # Run the code if pause is false
        if not self.pause:
            self._process_next_sample()

    def _load_data_and_start(self):
        """
        This function loads the data and starts the classification process.
        It is intended to be run in a separate thread to avoid blocking the UI.
        """
        self.load_dataloader()
        self.total_samples = self.controller.get_samples_len()
        self.data_loaded = True
        # Resume classification on the main thread
        self.root.after(0, self.classify_heartbeat)
    
    def _process_next_sample(self):
        """
        this function processes the next sample in the dataset.
        It updates the plot, output, and details info in the UI.
        It uses the controller to get the segment and classification results.
        It schedules itself to run again after a short delay if there are more samples to process.
        """

        # This condition ensures we don't go out of bounds
        if self.engine_index < self.total_samples:
            seg = self.controller.get_segment(idx=self.engine_index)
            pred, conf = self.controller.get_class(idx=self.engine_index)
            
            # Store the prediction result to show summary later
            self.end_result.append(pred[0])
            self.ui_index = self.engine_index

            self.update_plot(seg)
            self.update_output(pred[0], conf)
            self.create_details_info(self.ui_index) # was current_sample

            self.engine_index += 1

            self.root.after(50, self.classify_heartbeat)
        else:
            self.engine_index = 0
            self.ui_index = 0
            self.is_engine_done = True
            self.data_loaded = False
            self.btn_classify.config(state='normal')
            
            counter = Counter(self.end_result)
            summary = "  ".join(f"{k}:{v}" for k, v in counter.items())
            self.status_var.set(summary)

            style = ttk.Style()
            style.configure("Info.TLabel", foreground="lime")
            lbl_status = ttk.Label(
                self.frame_results,
                textvariable=self.status_var,
                wraplength=200,
                style="Info.TLabel"
            )
            lbl_status.pack(fill="x", padx=5, pady=5)
    
    def upload_file(self):
        """
        Method for uploading a CSV file and updating the dataset combo box.
        """
        try:
            # Ensure there's a valid parent window for the dialog
            if self.root.winfo_exists():
                file_path = filedialog.askopenfilename(
                    parent=self.root,
                    filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("WFDB header files", "*.hea")],
                    title="Select ECG CSV File"
                )
                if file_path:
                    self.dataset_combo.set(file_path)
                    self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                else:
                    self.status_var.set("No file selected")
            else:
                self.status_var.set("UI context lost. Please restart the app.")
        except tk.TclError as e:
            self.status_var.set(f"File dialog error: {e}")
    
    def update_plot(self, seg: np.ndarray | None):
        """
        Method to update the ECG plot with the given segment data.
        If seg is None, it clears the plot and shows a message.
        :param seg: The ECG segment data to plot.
        :return: None
        """
        if seg is None:
            self.status_var.set("No segment data to plot")
            return None
        
        self.ax.clear()
        self.ax.plot(seg, color="cyan", label="ECG Voltage (mV)")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Voltage (mV)")
        self.ax.legend()
        self.canvas.draw()
    
    def update_output(self, pred, conf):
        """
        Update the classification results panel with the given prediction and confidence.
        :param pred: The predicted class label.
        :param conf: The confidence score (0-100).
        :return: None
        """
        for widget in self.frame_results.winfo_children():
            widget.destroy()

        desc = self.controller.get_class_description(pred)
        
        self.create_results_info(
            self.frame_results,
            pred_class=pred,
            desc=desc,  # Placeholder for class description
            confidence=conf
        )
    
    def create_input_frame(self):
        # Input Frame where file upload and dataset selection happens
        self.frame_input = ttk.LabelFrame(self.root, text="Input Data", padding=10)
        self.frame_input.pack(fill="x", padx=10, pady=10)

        btn_upload = ttk.Button(self.frame_input, text="Upload File", command=self.upload_file)
        btn_upload.pack(side="left", padx=5)

        style = ttk.Style()
        style.configure("Danger.Button", foreground="black")

        btn_upload = ttk.Button(
            self.frame_input, text="Stop process",
            command=self.stop_here_btn, style="Danger.Button")
        btn_upload.pack(side="right", padx=5)

        self.dataset_combo = ttk.Combobox(self.frame_input, state="readonly")
        self.dataset_combo.set("Path")
        self.dataset_combo.pack(side="left", padx=5)

    def create_side_frames(self):
        # Visualization & Results Frame
        frame_main = ttk.Frame(self.root)
        frame_main.pack(fill="both", expand=True, padx=10, pady=10)

        self.frame_plot = ttk.LabelFrame(frame_main, text="ECG Visualization", padding=10)
        self.frame_plot.pack(side="left", fill="both", expand=True, padx=5)

        self.frame_results = ttk.LabelFrame(frame_main, text="Classification Results", padding=10)
        self.frame_results.pack(side="right", fill="both", expand=True, padx=5)

        self.create_results_info(self.frame_results)

        self.status_var = tk.StringVar(value="No data to classify yet")
        lbl_status = ttk.Label(self.frame_results, textvariable=self.status_var, wraplength=200)
        lbl_status.pack(fill="x", padx=5, pady=5)
    
    def create_mtplt_fig(self):
        # Matplotlib figure
        plt.style.use('dark_background')
        fig, self.ax = plt.subplots(figsize=(4,3))
        self.canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_classify_button(self):
        style = ttk.Style()
        style.configure("TButton", foreground="black")
        style.map("TButton",
            foreground=[('disabled', 'gray')],
            background=[('disabled', '#f0f0f0')]
        )
        # Classify Button
        self.btn_classify = ttk.Button(
            self.root, text="Classify Heartbeat",
            command=self.classify_heartbeat, style='TButton')
        self.btn_classify.pack(pady=10)

    def create_results_info(self, parent, pred_class="?", desc="---", confidence=0.0):
        """
        Create a styled classification results panel inside the given parent frame.
        """
        # Big predicted class letter
        lbl_class = ttk.Label(parent, text=pred_class, font=("Helvetica", 36, "bold"), foreground="deepskyblue")
        lbl_class.pack(pady=(10, 2))

        # Class name / description
        lbl_label = ttk.Label(parent, text=desc, font=("Helvetica", 14))
        lbl_label.pack(pady=(0, 15))

        # Details frame
        self.details_frame = ttk.LabelFrame(parent, text="Classification Details", padding=10)
        self.details_frame.pack(fill="x", padx=5, pady=5)

        # Confidence
        ttk.Label(self.details_frame, text=f"Confidence: {confidence:.0f}%").pack(anchor="w", pady=2)

        ttk.Label(
            self.details_frame,
            text=f"Sample ID: {self.ui_index} / {max(self.total_samples - 1, 0)}"
            ).pack(anchor="w", pady=2)

    def create_details_info(self, idx):
        """
        Destroy and recreate the details info panel with updated features.
        :param idx: The index of the current segment to display features for.
        :return: None
        """
        segment_features = self.controller.get_segment_features(idx)
        for key in list(segment_features.keys()):
            ttk.Label(
                self.details_frame, text=f'{key}: {round(segment_features[key], 3)}'
                ).pack(anchor='w', pady=2)
    
    def create_pause_continue_btn(self):
        """
        If the current button state is PAUSE, create a Pause button.
        If the current button state is CONTINUE, create a Continue button.
        """
        if self.current_btn == ButtonState.PAUSE:
            self.pause_continue_btn = ttk.Button(self.frame_input, text="Pause", command=self.pause_btn)
            self.pause_continue_btn.pack(padx=5, side='right')
        elif self.current_btn == ButtonState.CONTINUE:
            self.pause_continue_btn = ttk.Button(self.frame_input, text="Continue", command=self.continue_btn)
            self.pause_continue_btn.pack(padx=5, side='right')
    
    def create_previous_seg_button(self):
        self.prev_seg_btn = ttk.Button(self.frame_input, text="<--", command=self.prev_seg)
        self.prev_seg_btn.pack(padx=5, side='right')

    def create_next_seg_button(self):
        self.next_seg_btn = ttk.Button(self.frame_input, text="-->", command=self.next_seg)
        self.next_seg_btn.pack(padx=5, side='right')
    
    def pause_btn(self):
        """
        This method is called when the Pause button is clicked.
        It sets the pause flag to True, toggles the button state,
        destroys the current button, and creates a Continue button.
        """
        if self.is_engine_done == False and self.engine_index > 0: # Prevents pausing when done or not started
            self.pause = True
            # Toggles to continue state
            self.toggle_btn_state()
            self.pause_continue_btn.destroy()
            # Creates continue btn
            self.create_pause_continue_btn()
    
    def continue_btn(self):
        """
        This method is called when the Continue button is clicked.
        It sets the pause flag to False, toggles the button state,
        destroys the current button, creates a Pause button,
        updates the engine index if necessary, and continues classification.
        """
        if self.is_engine_done == False and self.engine_index > 0: # Prevents pausing when done or not started
            self.pause = False
            # Toggles to pause state
            self.toggle_btn_state()
            self.pause_continue_btn.destroy()
            # Creates pause btn
            self.create_pause_continue_btn()

            # update engines sample index
            # If the user goes beyond where
            # the engine left off.
            if self.ui_index > self.engine_index:
                self.engine_index = self.ui_index

            # Continues classification
            self.classify_heartbeat()
    
    def toggle_btn_state(self):
        """
        Toggle the current button state between PAUSE and CONTINUE.
        """
        if self.current_btn == ButtonState.PAUSE:
            self.current_btn = ButtonState.CONTINUE
        elif self.current_btn == ButtonState.CONTINUE:
            self.current_btn = ButtonState.PAUSE
    
    def prev_seg(self):
        """
        This method is called when the Previous Segment button is clicked.
        It checks if there is a previous segment available and updates the UI index.
        If the application is currently paused, it simply updates the display.
        If not paused, it pauses the application first before updating the display.
        """
        if self.ui_index > 0:
            if self.pause:
                self.ui_index = self.ui_index - 1
                self.display_seg_data(self.ui_index)
            else:
                self.pause_btn()
                self.ui_index = self.ui_index - 1
                self.display_seg_data(self.ui_index)
    
    def next_seg(self):
        """
        This method is called when the Next Segment button is clicked.
        It checks if there is a next segment available and updates the UI index.
        If the application is currently paused, it simply updates the display.
        If not paused, it pauses the application first before updating the display.
        """
        if self.ui_index + 1 < self.total_samples:
            if self.pause:
                self.ui_index = self.ui_index + 1
                self.display_seg_data(self.ui_index)
            else:
                self.pause_btn()
                self.ui_index = self.ui_index + 1
                self.display_seg_data(self.ui_index)

    def display_seg_data(self, index):
        """
        This method displays the segment data for the given index.
        It retrieves the segment and features from the controller,
        updates the plot, output, and details info in the UI.
        :param index: The index of the segment to display.
        :return: None
        """
        seg_data_dict = self.controller.get_segment_data(index)
        segment = seg_data_dict["segment"]
        self.update_plot(segment)
        x = seg_data_dict['features']
        pred, conf = self.controller.classifier.classify(x)
        self.update_output(pred[0], conf)
        self.create_details_info(index)

    def stop_here_btn(self):
        self.total_samples = 0