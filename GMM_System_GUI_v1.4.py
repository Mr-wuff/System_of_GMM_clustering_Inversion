import sys
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math
import scipy.io
import time
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import psutil
from functools import lru_cache
import logging
from datetime import datetime

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTextEdit, QProgressBar, QGroupBox, 
                             QLineEdit, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
                             QTabWidget, QSplitter, QFrame, QCheckBox, QComboBox, 
                             QFormLayout, QScrollArea)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QTextCursor

# Import your existing GMM code
# from GMM_get_model import (GaussianMixture, startGMM, calculate_optimal_workers, safe_argmax, check_singular_matrix)


def check_singular_matrix(matrix):
    condition_number = np.linalg.cond(matrix)
    print(f"The condition number of the matrix is {condition_number}")
    
    if condition_number > 10e4:
        print("The matrix is likely to be singular.")
        return 0
    else:
        print("The matrix is not singular.")
        return 1
    

def safe_argmax(arr,axis1):
    if arr.size == 0:
        return None # Or throw an exception
    return np.argmax(arr,axis=axis1)

def calculate_optimal_workers(total_tasks, memory_per_task_mb):
    """
    Calculate the optimal number of work processes
    """
    cpu_count = mp.cpu_count()
    print(f"Number of CPU cores: {cpu_count}")
    
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    base_process_memory_mb = 50
    estimated_memory_per_process_mb = base_process_memory_mb + (total_tasks / cpu_count) * memory_per_task_mb
    estimated_memory_per_process_gb = estimated_memory_per_process_mb / 1024
    
    memory_based_workers = int((available_memory_gb * 0.8) / estimated_memory_per_process_gb)
    task_based_workers = min(cpu_count * 4, total_tasks // 1000)
    
    optimal_workers = max(1, min(
        cpu_count,
        memory_based_workers,
        task_based_workers,
        32 if cpu_count > 8 else 16
    ))
    
    print(f"Available memory: {available_memory_gb:.2f} GB")
    print(f"Estimate the amount of resources required for each process: {estimated_memory_per_process_gb:.2f} GB")
    print(f"Memory-based process limit: {memory_based_workers}")
    print(f"Process number limit based on the number of tasks: {task_based_workers}")
    print(f"Suggested number of work processes: {optimal_workers}")
    
    return optimal_workers


## Improved GMM clustering
class GaussianMixture:
    def __init__(self, n_components=2, max_iter=500, tol=1e-4, cov_type='full', random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.cov_type = cov_type
        if random_state is not None:
            np.random.seed(random_state)

    def _init_params(self, X):
        N, D = X.shape
        self.pi = np.ones(self.n_components) / self.n_components
        
        # Use a more efficient random selection method
        idx = np.random.choice(N, self.n_components, replace=True)
        self.mu = X[idx].copy()
        
        # Pre-computed covariance
        if self.cov_type == 'full':
            base_cov = np.cov(X, rowvar=False)
            self.sigma = np.tile(base_cov, (self.n_components, 1, 1))
        elif self.cov_type == 'diag':
            var = np.var(X, axis=0)
            self.sigma = np.array([np.diag(var)] * self.n_components)
        else:
            raise NotImplementedError("Only full, diag supported here.")
        
    def _e_step(self, X):
        N, D = X.shape
        resp = np.zeros((N, self.n_components))
        # Vectorization of probability density calculation
        for k in range(self.n_components):
            if self.cov_type=='full':
                rv = multivariate_normal(mean=self.mu[k], cov=self.sigma[k]) 
            elif self.cov_type=='diag':
                rv = multivariate_normal(mean=self.mu[k], cov=self.sigma[k])
            resp[:,k] = self.pi[k]*rv.pdf(X)
        # Normalization
        resp = resp/(resp.sum(axis=1, keepdims=True) + 1e-16)
        return resp

    def _m_step(self, X, resp):
        N, D = X.shape
        Nk = resp.sum(axis=0)
        self.pi = Nk / N
        self.mu = (resp.T @ X) / Nk[:, None]
        if self.cov_type == 'full':
            for k in range(self.n_components):
                Xc = X - self.mu[k]
                self.sigma[k] = (resp[:,k][:,None]*Xc).T@Xc / Nk[k]
        elif self.cov_type == 'diag':
            for k in range(self.n_components):
                Xc = X - self.mu[k]
                var = (resp[:, k, None] * (Xc ** 2)).sum(axis=0) / Nk[k]
                self.sigma[k] = np.diag(var)

    def _log_likelihood(self, X):
        N, D = X.shape
        ll = 0
        # Calculate the probability density of all components in advance
        pdf_values = np.zeros((N, self.n_components))
        for k in range(self.n_components):
            try:
                if np.linalg.det(self.sigma[k]) == 0:
                    return 0
                rv = multivariate_normal(mean=self.mu[k], cov=self.sigma[k], allow_singular=True)
                pdf_values[:, k] = self.pi[k] * rv.pdf(X)
            except:
                pdf_values[:, k] = 1e-16
        
        # Vectorized computation of log-likelihood
        sum_pdf = pdf_values.sum(axis=1)
        ll = np.log(sum_pdf + 1e-16).sum()
        return ll
    

    def fit(self, X, progress_callback=None):
        self._init_params(X)
        old_ll = -np.inf
        for iter_ in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            ll = self._log_likelihood(X)
            
            # Send progress update
            if progress_callback:
                progress = int((iter_ + 1) / self.max_iter * 100)
                progress_callback(progress, iter_ + 1, self.max_iter, ll)
            
            if ll == 0:
                self.n_components -= 1
            if self.n_components == 0:
                return 0
            if old_ll is not None:
                if np.abs(ll - old_ll) < self.tol:
                    break
            old_ll = ll
            
        # Send completion signal
        if progress_callback:
            progress_callback(100, self.max_iter, self.max_iter, ll)
    
    def predict(self, X):
        resp = self._e_step(X)
        return safe_argmax(resp, 1)


class GMMWorker(QThread):
    """Work thread, used for running GMM calculations"""
    
    # Definition of Signal
    progress_updated = pyqtSignal(int, int, str)  # Current progress, total number of tasks, status information
    log_message = pyqtSignal(str)
    finished_success = pyqtSignal(dict)  # Complete the signal, transmit the result
    error_occurred = pyqtSignal(str)
    fit_progress_updated = pyqtSignal(int, int, int, float)  # Fitting progress, current iteration, maximum iteration, likelihood value
    
    def __init__(self, target_file, lut_file, parameters, field_mapping, 
                 result_save_path, log_save_path):
        super().__init__()
        self.target_file = target_file
        self.lut_file = lut_file
        self.parameters = parameters
        self.field_mapping = field_mapping
        self.result_save_path = result_save_path
        self.log_save_path = log_save_path
        self.is_running = True
        
        # Initialize variables
        self.target_wind_all = []
        self.normal_wind_all = []

    def Improved_startGMM(self, target_nbrcs, target_les, time_load, expect_Convergence, tol_expect, iter_time, X, wind_speed):
        '''
        The nth cycle checks the wind speed range
        '''  
        warnings.filterwarnings("ignore", category=DeprecationWarning) ##Ingnore DeprecationWarning
        # Use GMM clustering
        gmm = GaussianMixture(n_components=time_load, max_iter=iter_time, tol=tol_expect, cov_type='full', random_state=10)
        ## Check whether the parameters for clustering are singular matrices
        try:
            gmm.fit(X)
            labels = gmm.predict(X)
            if labels is None:
                bias_nbrcs = (target_nbrcs - (max(X[:,0])+min(X[:,0]))/2) / ((max(X[:,0]) - min(X[:,0]))/2)
                bias_les = (target_les - (max(X[:,1])+min(X[:,1]))/2) / ((max(X[:,1]) - min(X[:,1]))/2)
                if np.sign(bias_nbrcs)*np.sign(bias_les) > 0:
                    target_wind = (min(wind_speed)+max(wind_speed))/2
                else:
                    target_wind = (min(wind_speed)+max(wind_speed))/2 + (max(wind_speed)-min(wind_speed))/2 * (-1) * (bias_nbrcs+bias_les)/2
                self.target_wind_all.append(target_wind)
                return 0
        except (np.linalg.LinAlgError, ValueError):
            if time_load == 1:
                bias_nbrcs = (target_nbrcs - (max(X[:,0])+min(X[:,0]))/2) / ((max(X[:,0]) - min(X[:,0]))/2)
                bias_les = (target_les - (max(X[:,1])+min(X[:,1]))/2) / ((max(X[:,1]) - min(X[:,1]))/2)
                if np.sign(bias_nbrcs)*np.sign(bias_les) > 0:
                    target_wind = (min(wind_speed)+max(wind_speed))/2
                else:
                    target_wind = (min(wind_speed)+max(wind_speed))/2 + (max(wind_speed)-min(wind_speed))/2 * (-1) * (bias_nbrcs+bias_les)/2
                self.target_wind_all.append(target_wind)
                return 0
            return self.Improved_startGMM(target_nbrcs, target_les, time_load-1, expect_Convergence, tol_expect, iter_time, X, wind_speed)

        # Cluster the data to determine the wind speed range
        id = np.zeros((time_load))
        id[1:time_load] = 0
        wind_part = np.zeros((time_load, len(labels)))
        for i in range(len(labels)-1):
            for k in range(time_load): 
                if labels[i+1] == k:
                    wind_part[k,int(id[k])] = wind_speed[i]
                    id[k] += 1
        part_min = np.zeros((time_load))
        part_max = np.zeros((time_load))
        for k in range(time_load):
            part_min[k] = wind_part[k,0]
            part_max[k] = max(wind_part[k,:])
        for i in range(len(labels)):
            if X[i,0] == target_nbrcs and X[i,1] == target_les:
                target_id = i
        for i in range(len(labels)):
            for k in range(time_load): 
                if wind_part[k,i] != 0 and  wind_part[k,i] < part_min[k]:
                    part_min[k] = wind_part[k,i]

        target_len = 0
        for i in range(len(labels)):
            if labels[i] == labels[target_id]:
                target_len += 1
        nbrcs_les_part = np.zeros((target_len,2))
        wind_target_part = np.zeros((target_len-1))
        nbrcs_les_part[0,0] = target_nbrcs
        nbrcs_les_part[0,1] = target_les
        k = 0
        for i in range(len(labels)-1):
            if labels[i+1] == labels[target_id]:
                nbrcs_les_part[k+1,0] = X[i,0]
                nbrcs_les_part[k+1,1] = X[i,1]
                wind_target_part[k] = wind_speed[i]
                k += 1
        if part_max[labels[target_id]] - part_min[labels[target_id]] <= expect_Convergence:
            bias_nbrcs = (target_nbrcs - (max(nbrcs_les_part[:,0])+min(nbrcs_les_part[:,0]))/2) / ((max(nbrcs_les_part[:,0]) - min(nbrcs_les_part[:,0]))/2)
            bias_les = (target_les - (max(nbrcs_les_part[:,1])+min(nbrcs_les_part[:,1]))/2) / ((max(nbrcs_les_part[:,1]) - min(nbrcs_les_part[:,1]))/2)
            if np.sign(bias_nbrcs)*np.sign(bias_les) > 0:
                target_wind = (min(wind_speed)+max(wind_speed))/2
            else:
                target_wind = (min(wind_speed)+max(wind_speed))/2 + (max(wind_speed)-min(wind_speed))/2 * (-1) * (bias_nbrcs+bias_les)/2
            self.target_wind_all.append(target_wind)
            return 1
        
        return self.Improved_startGMM(target_nbrcs, target_les, time_load, expect_Convergence, tol_expect, iter_time, nbrcs_les_part, wind_target_part)
        
    def run(self):
        try:
            self.log_message.emit("Starting to load data...")
            
            # Load the target data
            target_data = scipy.io.loadmat(self.target_file)
            
            # Obtain data based on field mapping
            measure_wind_speed = target_data[self.field_mapping['measure_wind_speed']][:]
            measure_nbrcs = target_data[self.field_mapping['measure_nbrcs']][:]
            measure_les = target_data[self.field_mapping['measure_les']][:]
            
            # Make sure that the data is one-dimensional
            measure_wind_speed = measure_wind_speed.flatten()
            total_tasks = len(measure_wind_speed)
            
            self.log_message.emit(f"The target data has been successfully loaded, A total of {total_tasks} tasks were processed.")
            
            # Load LUT data
            cygnss_data = scipy.io.loadmat(self.lut_file)
            
            # Obtain LUT data based on field mapping
            nbrcs_les = cygnss_data[self.field_mapping['nbrcs_les']][:]
            wind_speed = cygnss_data[self.field_mapping['wind_speed']][:]
            nbrcs = cygnss_data[self.field_mapping['nbrcs']][:]
            les = cygnss_data[self.field_mapping['les']][:]
            wind_speed_all = cygnss_data[self.field_mapping['wind_speed_all']][:]
            
            # Initialize global variables
            nbrcs_les_all = np.zeros((len(les), 2))
            for i in range(len(wind_speed_all)):
                nbrcs_les_all[i, 0] = nbrcs[i, 0]
                nbrcs_les_all[i, 1] = les[i, 0]
                
            self.log_message.emit("LUT data loading completed")
            
            # Retrieve the settings from the parameters
            time_load = self.parameters['time_load']
            expect_Convergence = self.parameters['expect_Convergence']
            tol_expect = self.parameters['tol_expect']
            iter_time = self.parameters['iter_time']
            Data_Num = self.parameters['Data_Num']
            if Data_Num == 0:
                pass
            else:
                total_tasks = min(Data_Num, total_tasks)
            self.log_message.emit(f"Select {total_tasks} pieces of data for calculation")
            
            self.log_message.emit("Start parallel computing...")
            
            # Calculate the optimal number of working processes
            optimal_workers = calculate_optimal_workers(total_tasks, 1)
            self.log_message.emit(f"Use {optimal_workers} worker processes")
            
            # Pre-computation optimization: Caching frequently used computations
            nbrcs_les_cache = nbrcs_les.copy()
            wind_speed_cache = wind_speed.copy()
            
            # Modify the main function to support progress updates and optimization
            def _main_optimized_with_progress(i):
                if not self.is_running:
                    return
                    
                self.normal_wind_all.append(measure_wind_speed[i])
                
                # Skip the cases where the wind speed is zero
                if measure_wind_speed[i] == 0:
                    self.target_wind_all.append(0)
                    self.progress_updated.emit(len(self.target_wind_all), total_tasks, f"处理任务 {i+1}/{total_tasks}")
                    return 1
                
                target_nbrcs = float(measure_nbrcs[i])
                target_les = float(measure_les[i])
                target_data = [target_nbrcs, target_les]
                
                # Optimization: Avoid redundant stacking operations
                X = np.vstack([target_data, nbrcs_les_cache])

                self.log_message.emit(f"Current GMM parameters: [target_nbrcs:{target_nbrcs}], [target_les:{target_les}], [time_load:{time_load}], [expect_Convergence:{expect_Convergence}], [tol_expect:{tol_expect}], [iter_time:{iter_time}]")
                
                # Calculate using the optimized GMM
                flag = self.Improved_startGMM(target_nbrcs, target_les, time_load, expect_Convergence, 
                              tol_expect, iter_time, X, wind_speed_cache)
                
                # Update progress
                self.progress_updated.emit(len(self.target_wind_all), total_tasks, f"Handle the task {i+1}/{total_tasks}")
                return 1
            
            # Carry out parallel computing
            start_time = time.time()
            
            # Process in batches to avoid memory issues
            batch_size = min(1000, total_tasks // optimal_workers)
            results = []
            
            for batch_start in range(0, total_tasks, batch_size):
                if not self.is_running:
                    break
                    
                batch_end = min(batch_start + batch_size, total_tasks)
                batch_tasks = range(batch_start, batch_end)
                
                self.log_message.emit(f"Processing batch {batch_start//batch_size + 1}/{(total_tasks-1)//batch_size + 1}")
                
                batch_results = Parallel(
                    n_jobs=optimal_workers, 
                    verbose=0,  
                    batch_size=min(100, batch_size // optimal_workers),
                    prefer="processes"  
                )(
                    delayed(_main_optimized_with_progress)(i) for i in batch_tasks
                )
                results.extend(batch_results)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            self.log_message.emit(f"Calculation completed! Total execution time: {total_time:.2f} seconds")
            
            wind_res = {
                "lut_wind": self.target_wind_all, 
                "ini_wind": self.normal_wind_all,
                "total_time": total_time,
                "parameters": self.parameters,
                "field_mapping": self.field_mapping
            }
            
            if self.result_save_path:
                scipy.io.savemat(self.result_save_path, wind_res)
                self.log_message.emit(f"The result has been saved to: {self.result_save_path}")
            
            self.finished_success.emit(wind_res)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.error_occurred.emit(f"An error occurred during the calculation: {str(e)}\n{error_details}")
    
    def stop(self):
        """Stop Calculating"""
        self.is_running = False


class GMMGUI(QMainWindow):
    """Main Window for GMM Clustering Application"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.log_file = None
        self.target_data_fields = []
        self.lut_data_fields = []
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("CYGNSS GMM Wind Speed Inversion System")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        display_area = self.create_display_area()
        main_layout.addWidget(display_area, 2)
        
    def create_control_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.Box)
        panel.setMaximumWidth(500)
        layout = QVBoxLayout(panel)
        
        file_group = QGroupBox("File selection")
        file_layout = QVBoxLayout(file_group)
        
        target_layout = QHBoxLayout()
        self.target_file_edit = QLineEdit()
        self.target_file_edit.setPlaceholderText("Select the target data file (.mat)")
        target_browse_btn = QPushButton("Browse")
        target_browse_btn.clicked.connect(self.browse_target_file)
        target_layout.addWidget(QLabel("Target data:"))
        target_layout.addWidget(self.target_file_edit)
        target_layout.addWidget(target_browse_btn)
        file_layout.addLayout(target_layout)
        
        lut_layout = QHBoxLayout()
        self.lut_file_edit = QLineEdit()
        self.lut_file_edit.setPlaceholderText("Select LUT data file (.mat)")
        lut_browse_btn = QPushButton("Browse")
        lut_browse_btn.clicked.connect(self.browse_lut_file)
        lut_layout.addWidget(QLabel("LUT data:"))
        lut_layout.addWidget(self.lut_file_edit)
        lut_layout.addWidget(lut_browse_btn)
        file_layout.addLayout(lut_layout)
        
        layout.addWidget(file_group)
        
        self.field_mapping_group = QGroupBox("Field mapping settings")
        self.field_mapping_layout = QFormLayout(self.field_mapping_group)
        
        self.target_wind_combo = QComboBox()
        self.target_nbrcs_combo = QComboBox()
        self.target_les_combo = QComboBox()
        
        self.field_mapping_layout.addRow("measure_wind_speed:", self.target_wind_combo)
        self.field_mapping_layout.addRow("measure_nbrcs:", self.target_nbrcs_combo)
        self.field_mapping_layout.addRow("measure_les:", self.target_les_combo)
        
        self.lut_nbrcs_les_combo = QComboBox()
        self.lut_wind_speed_combo = QComboBox()
        self.lut_nbrcs_combo = QComboBox()
        self.lut_les_combo = QComboBox()
        self.lut_wind_speed_all_combo = QComboBox()
        
        self.field_mapping_layout.addRow("nbrcs_les:", self.lut_nbrcs_les_combo)
        self.field_mapping_layout.addRow("wind_speed:", self.lut_wind_speed_combo)
        self.field_mapping_layout.addRow("nbrcs:", self.lut_nbrcs_combo)
        self.field_mapping_layout.addRow("les:", self.lut_les_combo)
        self.field_mapping_layout.addRow("wind_speed_all:", self.lut_wind_speed_all_combo)
        
        layout.addWidget(self.field_mapping_group)
        self.field_mapping_group.setVisible(False)  
        
        output_group = QGroupBox("Output settings")
        output_layout = QVBoxLayout(output_group)
        
        result_layout = QHBoxLayout()
        self.result_save_edit = QLineEdit()
        self.result_save_edit.setPlaceholderText("Select the save path for the result (.mat)")
        result_browse_btn = QPushButton("Browse")
        result_browse_btn.clicked.connect(self.browse_result_save_path)
        result_layout.addWidget(QLabel("Result file:"))
        result_layout.addWidget(self.result_save_edit)
        result_layout.addWidget(result_browse_btn)
        output_layout.addLayout(result_layout)
        
        log_layout = QHBoxLayout()
        self.log_save_edit = QLineEdit()
        self.log_save_edit.setPlaceholderText("Select the log saving path (.txt)")
        log_browse_btn = QPushButton("Browse")
        log_browse_btn.clicked.connect(self.browse_log_save_path)
        log_layout.addWidget(QLabel("Log file:"))
        log_layout.addWidget(self.log_save_edit)
        log_layout.addWidget(log_browse_btn)
        output_layout.addLayout(log_layout)
        
        self.auto_save_check = QCheckBox("The results will be automatically saved after the calculation is completed")
        self.auto_save_check.setChecked(True)
        output_layout.addWidget(self.auto_save_check)
        
        layout.addWidget(output_group)
        
        param_group = QGroupBox("GMM Parameter Settings")
        param_layout = QVBoxLayout(param_group)
        
        time_load_layout = QHBoxLayout()
        time_load_layout.addWidget(QLabel("Clustering hierarchy:"))
        self.time_load_spin = QSpinBox()
        self.time_load_spin.setRange(1, 20)
        self.time_load_spin.setValue(10)
        time_load_layout.addWidget(self.time_load_spin)
        param_layout.addLayout(time_load_layout)
        
        convergence_layout = QHBoxLayout()
        convergence_layout.addWidget(QLabel("Expected convergence value:"))
        self.convergence_spin = QDoubleSpinBox()
        self.convergence_spin.setRange(0.1, 10.0)
        self.convergence_spin.setValue(1.0)
        self.convergence_spin.setSingleStep(0.1)
        convergence_layout.addWidget(self.convergence_spin)
        param_layout.addLayout(convergence_layout)
        
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Tolerance:"))
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(1e-4, 1e-1)
        self.tol_spin.setValue(1e-3)
        self.tol_spin.setDecimals(4)
        tol_layout.addWidget(self.tol_spin)
        param_layout.addLayout(tol_layout)
        
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Number of iterations:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 2000)
        self.iter_spin.setValue(500)
        iter_layout.addWidget(self.iter_spin)
        param_layout.addLayout(iter_layout)

        DataNum_layout = QHBoxLayout()
        DataNum_layout.addWidget(QLabel("Calculate the amount of data:"))
        self.DataNum_spin = QSpinBox()
        self.DataNum_spin.setRange(0, 100000000)
        self.DataNum_spin.setValue(100)
        DataNum_layout.addWidget(self.DataNum_spin)
        param_layout.addLayout(DataNum_layout)
        
        layout.addWidget(param_group)
        
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start calculation")
        self.start_btn.clicked.connect(self.start_calculation)
        self.stop_btn = QPushButton("Stop calculation")
        self.stop_btn.clicked.connect(self.stop_calculation)
        self.stop_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        self.progress_group = QGroupBox("Overall project schedule of the system")
        progress_layout = QVBoxLayout(self.progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(self.progress_group)
        self.progress_group.setVisible(False)
        
        self.fit_progress_group = QGroupBox("Progress of GMM fitting")
        fit_progress_layout = QVBoxLayout(self.fit_progress_group)
        
        self.fit_progress_bar = QProgressBar()
        self.fit_progress_bar.setVisible(False)
        fit_progress_layout.addWidget(self.fit_progress_bar)
        
        self.fit_progress_label = QLabel("Ready")
        fit_progress_layout.addWidget(self.fit_progress_label)
        
        layout.addWidget(self.fit_progress_group)
        self.fit_progress_group.setVisible(False)
        

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        return panel
    
    def create_display_area(self):
        display_widget = QWidget()
        layout = QVBoxLayout(display_widget)
        
        log_group = QGroupBox("Calculate log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        font = QFont("Courier", 9)
        self.log_text.setFont(font)
        log_layout.addWidget(self.log_text)
        
        # 日志控制按钮
        log_control_layout = QHBoxLayout()
        clear_log_btn = QPushButton("Clear logs")
        clear_log_btn.clicked.connect(self.clear_log)
        export_log_btn = QPushButton("Export logs")
        export_log_btn.clicked.connect(self.export_log)
        save_log_btn = QPushButton("Real-time saving logs")
        save_log_btn.clicked.connect(self.toggle_log_saving)
        
        log_control_layout.addWidget(clear_log_btn)
        log_control_layout.addWidget(export_log_btn)
        log_control_layout.addWidget(save_log_btn)
        log_control_layout.addStretch()
        
        log_layout.addLayout(log_control_layout)
        
        layout.addWidget(log_group)
        
        return display_widget
    
    def browse_target_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select the target data file", "", "MAT Files (*.mat)")
        if file_path:
            self.target_file_edit.setText(file_path)
            self.log_message(f"The target data file has been selected: {file_path}")
            self.load_mat_fields(file_path, 'target')
    
    def browse_lut_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select the LUT file", "", "MAT Files (*.mat)")
        if file_path:
            self.lut_file_edit.setText(file_path)
            self.log_message(f"The LUT file has been selected: {file_path}")
            self.load_mat_fields(file_path, 'lut')
    
    def load_mat_fields(self, file_path, file_type):
        try:
            data = scipy.io.loadmat(file_path)
            fields = [key for key in data.keys() if not key.startswith('__')]
            
            if file_type == 'target':
                self.target_data_fields = fields
                self.update_field_combo(self.target_wind_combo, fields)
                self.update_field_combo(self.target_nbrcs_combo, fields)
                self.update_field_combo(self.target_les_combo, fields)
                
                self.set_default_field(self.target_wind_combo, ['cygnss_wind', 'wind_speed'])
                self.set_default_field(self.target_nbrcs_combo, ['nbrcs_mean', 'nbrcs'])
                self.set_default_field(self.target_les_combo, ['les_mean', 'les'])
                
            elif file_type == 'lut':
                self.lut_data_fields = fields
                self.update_field_combo(self.lut_nbrcs_les_combo, fields)
                self.update_field_combo(self.lut_wind_speed_combo, fields)
                self.update_field_combo(self.lut_nbrcs_combo, fields)
                self.update_field_combo(self.lut_les_combo, fields)
                self.update_field_combo(self.lut_wind_speed_all_combo, fields)
                
                self.set_default_field(self.lut_nbrcs_les_combo, ['nbrcs_les_save'])
                self.set_default_field(self.lut_wind_speed_combo, ['wind_speed_save'])
                self.set_default_field(self.lut_nbrcs_combo, ['nbrcs_effective'])
                self.set_default_field(self.lut_les_combo, ['les_effective'])
                self.set_default_field(self.lut_wind_speed_all_combo, ['wind_speed_effective'])
            
            if self.target_file_edit.text() and self.lut_file_edit.text():
                self.field_mapping_group.setVisible(True)
                
            self.log_message(f"The fields of the {file_type} file have been loaded: {', '.join(fields)}")
            
        except Exception as e:
            self.log_message(f"Error occurred while loading the fields of the MAT file: {str(e)}")
    
    def update_field_combo(self, combo, fields):
        combo.clear()
        combo.addItems(fields)
    
    def set_default_field(self, combo, preferred_fields):
        for field in preferred_fields:
            if combo.findText(field) >= 0:
                combo.setCurrentText(field)
                break
    
    def browse_result_save_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select the save path for the result", "wind_result.mat", "MAT Files (*.mat)")
        if file_path:
            self.result_save_edit.setText(file_path)
            self.log_message(f"The result will be saved to: {file_path}")
    
    def browse_log_save_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select the log saving path", "gmm_calculation_log.txt", "Text Files (*.txt)")
        if file_path:
            self.log_save_edit.setText(file_path)
            self.log_message(f"The log will be saved to: {file_path}")
            
            self.start_log_saving(file_path)
    
    def start_log_saving(self, log_path):
        try:
            self.log_file = open(log_path, 'w', encoding='utf-8')
            self.log_message(f"Start to save logs in real time to: {log_path}")
        except Exception as e:
            self.log_message(f"Unable to open the log file: {str(e)}")
    
    def toggle_log_saving(self):
        if self.log_file:
            self.stop_log_saving()
        else:
            log_path = self.log_save_edit.text()
            if log_path:
                self.start_log_saving(log_path)
            else:
                QMessageBox.warning(self, "Warning", "Please first select the path for saving the log file")
    
    def stop_log_saving(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.log_message("Real-time log saving has been stopped.")
    
    def get_field_mapping(self):
        return {
            'measure_wind_speed': self.target_wind_combo.currentText(),
            'measure_nbrcs': self.target_nbrcs_combo.currentText(),
            'measure_les': self.target_les_combo.currentText(),
            'nbrcs_les': self.lut_nbrcs_les_combo.currentText(),
            'wind_speed': self.lut_wind_speed_combo.currentText(),
            'nbrcs': self.lut_nbrcs_combo.currentText(),
            'les': self.lut_les_combo.currentText(),
            'wind_speed_all': self.lut_wind_speed_all_combo.currentText()
        }
    
    def start_calculation(self):
        target_file = self.target_file_edit.text()
        lut_file = self.lut_file_edit.text()
        
        if not target_file or not os.path.exists(target_file):
            QMessageBox.warning(self, "Warning", "Please select a valid target data file")
            return
            
        if not lut_file or not os.path.exists(lut_file):
            QMessageBox.warning(self, "Warning", "Please select a valid LUT data file")
            return
        
        field_mapping = self.get_field_mapping()
        for key, value in field_mapping.items():
            if not value:
                QMessageBox.warning(self, "Warning", f"Please select the field mapping for {key}")
                return
        
        result_save_path = self.result_save_edit.text()
        if not result_save_path and self.auto_save_check.isChecked():
            QMessageBox.warning(self, "Warning", "Please select the path for saving the result file or cancel the automatic saving.")
            return
        
        parameters = {
            'time_load': self.time_load_spin.value(),
            'expect_Convergence': self.convergence_spin.value(),
            'tol_expect': self.tol_spin.value(),
            'iter_time': self.iter_spin.value(),
            'Data_Num': self.DataNum_spin.value()
        }
        
        self.log_message("=" * 50)
        self.log_message("Start GMM wind speed inversion calculation")
        self.log_message(f"Parameter settings: {parameters}")
        self.log_message(f"Field Mapping: {field_mapping}")
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_group.setVisible(True)
        # self.fit_progress_bar.setVisible(True)
        # self.fit_progress_group.setVisible(True)
        
        self.worker = GMMWorker(target_file, lut_file, parameters, field_mapping,
                               result_save_path if self.auto_save_check.isChecked() else None,
                               self.log_save_edit.text())
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.log_message.connect(self.log_message)
        self.worker.finished_success.connect(self.calculation_finished)
        self.worker.error_occurred.connect(self.calculation_error)
        self.worker.fit_progress_updated.connect(self.update_fit_progress)
        
        self.worker.start()
    
    def stop_calculation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)  
            self.log_message("Calculation has stopped")
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.fit_progress_bar.setVisible(False)
        self.status_label.setText("Has been stopped")
    
    def update_progress(self, current, total, status):
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
        if current % max(1, total // 10) == 0 or current == total:
            self.log_message(f"Progress: {current}/{total} ({progress}%) - {status}")
    
    def update_fit_progress(self, progress, current_iter, max_iter, ll):
        self.fit_progress_bar.setValue(progress)
        self.fit_progress_label.setText(f"Iteration: {current_iter}/{max_iter}, Log-Likelihood Value: {ll:.4f}")
        
        if current_iter == max_iter:
            self.log_message(f"This GMM fitting has been completed. Start the next fitting")
        if current_iter % 10 == 0 or current_iter == max_iter:
            self.log_message(f"GMM fitting progress: {current_iter}/{max_iter} ({progress}%), Log-Likelihood value: {ll:.4f}")
    
    def calculation_finished(self, results):
        self.log_message("Calculation completed!")
        self.log_message(f"Total execution time: {results['total_time']:.2f} seconds")
        
        # 如果没有自动保存，询问用户是否保存
        if not self.auto_save_check.isChecked():
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save the results", "wind_result.mat", "MAT Files (*.mat)")
            
            if save_path:
                scipy.io.savemat(save_path, results)
                self.log_message(f"The result has been saved to: {save_path}")
        
        # 恢复UI状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.fit_progress_bar.setVisible(False)
        self.status_label.setText("Calculation completed")
        
        QMessageBox.information(self, "Completed", "The GMM wind speed inversion calculation has been completed!")
    
    def calculation_error(self, error_msg):
        self.log_message(f"False: {error_msg}")
        QMessageBox.critical(self, "False:", error_msg)
        
        # 恢复UI状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.fit_progress_bar.setVisible(False)
        self.status_label.setText("Calculation error")
    
    def log_message(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
        if self.log_file:
            try:
                self.log_file.write(log_entry + '\n')
                self.log_file.flush()
            except Exception as e:
                # 如果写入失败，停止日志保存
                self.stop_log_saving()
                self.log_message(f"Log file writing failed: {str(e)}")
    
    def clear_log(self):
        self.log_text.clear()
    
    def export_log(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export logs", "gmm_calculation_log.txt", "Text Files (*.txt)")
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log_message(f"The log has been exported to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "False", f"Failed to export the log: {str(e)}")
    
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, "Confirm exit", 
                                       "The calculation is still ongoing. Do you want to cancel it?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait(3000)
                self.stop_log_saving()
                event.accept()
            else:
                event.ignore()
        else:
            self.stop_log_saving()
            event.accept()


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    app = QApplication(sys.argv)
    app.setApplicationName("CYGNSS GMM Wind Speed Inversion System")
    
    window = GMMGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()