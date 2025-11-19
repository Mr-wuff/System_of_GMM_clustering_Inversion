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
from GMM_get_model_deepseek2 import (GaussianMixture, startGMM, calculate_optimal_workers, 
                                     safe_argmax, check_singular_matrix)


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
        return None # 或者抛出异常
    return np.argmax(arr,axis=axis1)

def calculate_optimal_workers(total_tasks, memory_per_task_mb):
    """
    计算最优的工作进程数量
    """
    cpu_count = mp.cpu_count()
    print(f"CPU核心数: {cpu_count}")
    
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
    
    print(f"可用内存: {available_memory_gb:.2f} GB")
    print(f"估算每个进程需要: {estimated_memory_per_process_gb:.2f} GB")
    print(f"基于内存的进程数限制: {memory_based_workers}")
    print(f"基于任务数量的进程数限制: {task_based_workers}")
    print(f"建议工作进程数: {optimal_workers}")
    
    return optimal_workers


## 改进后的GMM聚类
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
        
        # 使用更高效的随机选择方法
        idx = np.random.choice(N, self.n_components, replace=True)
        self.mu = X[idx].copy()
        
        # 预计算协方差
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
        # 向量化计算概率密度
        for k in range(self.n_components):
            if self.cov_type=='full':
                rv = multivariate_normal(mean=self.mu[k], cov=self.sigma[k]) 
            elif self.cov_type=='diag':
                rv = multivariate_normal(mean=self.mu[k], cov=self.sigma[k])
            resp[:,k] = self.pi[k]*rv.pdf(X)
        # 归一化
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
        # 预计算所有分量的概率密度
        pdf_values = np.zeros((N, self.n_components))
        for k in range(self.n_components):
            try:
                if np.linalg.det(self.sigma[k]) == 0:
                    return 0
                rv = multivariate_normal(mean=self.mu[k], cov=self.sigma[k], allow_singular=True)
                pdf_values[:, k] = self.pi[k] * rv.pdf(X)
            except:
                pdf_values[:, k] = 1e-16
        
        # 向量化计算对数似然
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
            
            # 发送进度更新
            if progress_callback:
                progress = int((iter_ + 1) / self.max_iter * 100)
                progress_callback(progress, iter_ + 1, self.max_iter, ll)
            
            if ll == 0:
                self.n_components -= 1
            if self.n_components == 0:
                return 0
            if old_ll is not None:
                if np.abs(ll - old_ll) < self.tol:
                    print(f"期望expect={np.abs(ll - old_ll)}")
                    print(f"在迭代时收敛的迭代位次(Converged at iter)={iter_}, ll={ll:.4f}")
                    break
            old_ll = ll
            
        # 发送完成信号
        if progress_callback:
            progress_callback(100, self.max_iter, self.max_iter, ll)
    
    def predict(self, X):
        resp = self._e_step(X)
        return safe_argmax(resp, 1)


class GMMWorker(QThread):
    """工作线程，用于运行GMM计算"""
    
    # 定义信号
    progress_updated = pyqtSignal(int, int, str)  # 当前进度，总任务数，状态信息
    log_message = pyqtSignal(str)
    finished_success = pyqtSignal(dict)  # 完成信号，传递结果
    error_occurred = pyqtSignal(str)
    fit_progress_updated = pyqtSignal(int, int, int, float)  # 拟合进度，当前迭代，最大迭代，似然值
    
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
        
        # 初始化变量
        self.target_wind_all = []
        self.normal_wind_all = []

    def Improved_startGMM(self, target_nbrcs, target_les, time_load, expect_Convergence, tol_expect, iter_time, X, wind_speed):
        print(f"################################开始循环################################")
        '''
        第n次循环判断风速范围
        '''  
        warnings.filterwarnings("ignore", category=DeprecationWarning) ##忽略DeprecationWarning
        # 使用GMM聚类
        gmm = GaussianMixture(n_components=time_load, max_iter=iter_time, tol=tol_expect, cov_type='full', random_state=10)
        ## 判断聚类的参数是否为奇异矩阵
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

        # 将数据聚类判断风速区间
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
            self.log_message.emit("开始加载数据...")
            
            # 加载目标数据
            target_data = scipy.io.loadmat(self.target_file)
            
            # 根据字段映射获取数据
            measure_wind_speed = target_data[self.field_mapping['measure_wind_speed']][:]
            measure_nbrcs = target_data[self.field_mapping['measure_nbrcs']][:]
            measure_les = target_data[self.field_mapping['measure_les']][:]
            
            # 确保数据是一维的
            measure_wind_speed = measure_wind_speed.flatten()
            total_tasks = len(measure_wind_speed)
            
            self.log_message.emit(f"目标数据加载完成，共 {total_tasks} 个任务")
            
            # 加载LUT数据
            cygnss_data = scipy.io.loadmat(self.lut_file)
            
            # 根据字段映射获取LUT数据
            nbrcs_les = cygnss_data[self.field_mapping['nbrcs_les']][:]
            wind_speed = cygnss_data[self.field_mapping['wind_speed']][:]
            nbrcs = cygnss_data[self.field_mapping['nbrcs']][:]
            les = cygnss_data[self.field_mapping['les']][:]
            wind_speed_all = cygnss_data[self.field_mapping['wind_speed_all']][:]
            
            # 初始化全局变量
            nbrcs_les_all = np.zeros((len(les), 2))
            for i in range(len(wind_speed_all)):
                nbrcs_les_all[i, 0] = nbrcs[i, 0]
                nbrcs_les_all[i, 1] = les[i, 0]
                
            self.log_message.emit("LUT数据加载完成")
            
            # 从参数中获取设置
            time_load = self.parameters['time_load']
            expect_Convergence = self.parameters['expect_Convergence']
            tol_expect = self.parameters['tol_expect']
            iter_time = self.parameters['iter_time']
            Data_Num = self.parameters['Data_Num']
            if Data_Num == 0:
                pass
            else:
                total_tasks = min(Data_Num, total_tasks)
            self.log_message.emit(f"选择计算 {total_tasks} 个数据")
            
            self.log_message.emit("开始并行计算...")
            
            # 计算最优工作进程数
            optimal_workers = calculate_optimal_workers(total_tasks, 1)
            self.log_message.emit(f"使用 {optimal_workers} 个工作进程")
            
            # 预计算优化：缓存常用计算
            nbrcs_les_cache = nbrcs_les.copy()
            wind_speed_cache = wind_speed.copy()
            
            # 修改主函数以支持进度更新和优化
            def _main_optimized_with_progress(i):
                if not self.is_running:
                    return
                    
                self.normal_wind_all.append(measure_wind_speed[i])
                
                # 跳过风速为0的情况
                if measure_wind_speed[i] == 0:
                    self.target_wind_all.append(0)
                    self.progress_updated.emit(len(self.target_wind_all), total_tasks, f"处理任务 {i+1}/{total_tasks}")
                    return 1
                
                target_nbrcs = float(measure_nbrcs[i])
                target_les = float(measure_les[i])
                target_data = [target_nbrcs, target_les]
                
                # 优化：避免重复堆叠操作
                X = np.vstack([target_data, nbrcs_les_cache])

                self.log_message.emit(f"当前GMM参数：target_nbrcs:{target_nbrcs}, target_les:{target_les}, time_load:{time_load}, expect_Convergence:{expect_Convergence}, tol_expect:{tol_expect}, iter_time:{iter_time}")
                
                # 使用优化后的GMM计算
                flag = self.Improved_startGMM(target_nbrcs, target_les, time_load, expect_Convergence, 
                              tol_expect, iter_time, X, wind_speed_cache)
                
                # 更新进度
                self.progress_updated.emit(len(self.target_wind_all), total_tasks, f"处理任务 {i+1}/{total_tasks}")
                return 1
            
            # 执行并行计算
            start_time = time.time()
            
            # 分批处理以避免内存问题
            batch_size = min(1000, total_tasks // optimal_workers)
            results = []
            
            for batch_start in range(0, total_tasks, batch_size):
                if not self.is_running:
                    break
                    
                batch_end = min(batch_start + batch_size, total_tasks)
                batch_tasks = range(batch_start, batch_end)
                
                self.log_message.emit(f"处理批次 {batch_start//batch_size + 1}/{(total_tasks-1)//batch_size + 1}")
                
                batch_results = Parallel(
                    n_jobs=optimal_workers, 
                    verbose=0,  # 减少输出噪音
                    batch_size=min(100, batch_size // optimal_workers),
                    prefer="processes"  # 使用进程而不是线程
                )(
                    delayed(_main_optimized_with_progress)(i) for i in batch_tasks
                )
                results.extend(batch_results)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            self.log_message.emit(f"计算完成！总执行时间: {total_time:.2f} 秒")
            
            # 准备结果
            wind_res = {
                "lut_wind": self.target_wind_all, 
                "ini_wind": self.normal_wind_all,
                "total_time": total_time,
                "parameters": self.parameters,
                "field_mapping": self.field_mapping
            }
            
            # 保存结果到指定路径
            if self.result_save_path:
                scipy.io.savemat(self.result_save_path, wind_res)
                self.log_message.emit(f"结果已保存到: {self.result_save_path}")
            
            self.finished_success.emit(wind_res)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.error_occurred.emit(f"计算过程中出现错误: {str(e)}\n{error_details}")
    
    def stop(self):
        """停止计算"""
        self.is_running = False


class GMMGUI(QMainWindow):
    """GMM聚类应用的主窗口"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.log_file = None
        self.target_data_fields = []
        self.lut_data_fields = []
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("CYGNSS GMM 风速反演系统 - 优化版")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # 右侧显示区域
        display_area = self.create_display_area()
        main_layout.addWidget(display_area, 2)
        
    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Box)
        panel.setMaximumWidth(500)
        layout = QVBoxLayout(panel)
        
        # 文件选择组
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout(file_group)
        
        # 目标数据文件
        target_layout = QHBoxLayout()
        self.target_file_edit = QLineEdit()
        self.target_file_edit.setPlaceholderText("选择目标数据文件 (.mat)")
        target_browse_btn = QPushButton("浏览")
        target_browse_btn.clicked.connect(self.browse_target_file)
        target_layout.addWidget(QLabel("目标数据:"))
        target_layout.addWidget(self.target_file_edit)
        target_layout.addWidget(target_browse_btn)
        file_layout.addLayout(target_layout)
        
        # LUT数据文件
        lut_layout = QHBoxLayout()
        self.lut_file_edit = QLineEdit()
        self.lut_file_edit.setPlaceholderText("选择LUT数据文件 (.mat)")
        lut_browse_btn = QPushButton("浏览")
        lut_browse_btn.clicked.connect(self.browse_lut_file)
        lut_layout.addWidget(QLabel("LUT数据:"))
        lut_layout.addWidget(self.lut_file_edit)
        lut_layout.addWidget(lut_browse_btn)
        file_layout.addLayout(lut_layout)
        
        layout.addWidget(file_group)
        
        # 字段映射组
        self.field_mapping_group = QGroupBox("字段映射设置")
        self.field_mapping_layout = QFormLayout(self.field_mapping_group)
        
        # 目标数据字段映射
        self.target_wind_combo = QComboBox()
        self.target_nbrcs_combo = QComboBox()
        self.target_les_combo = QComboBox()
        
        self.field_mapping_layout.addRow("measure_wind_speed:", self.target_wind_combo)
        self.field_mapping_layout.addRow("measure_nbrcs:", self.target_nbrcs_combo)
        self.field_mapping_layout.addRow("measure_les:", self.target_les_combo)
        
        # LUT数据字段映射
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
        self.field_mapping_group.setVisible(False)  # 初始隐藏，加载文件后显示
        
        # 输出设置组
        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout(output_group)
        
        # 结果文件保存路径
        result_layout = QHBoxLayout()
        self.result_save_edit = QLineEdit()
        self.result_save_edit.setPlaceholderText("选择结果保存路径 (.mat)")
        result_browse_btn = QPushButton("浏览")
        result_browse_btn.clicked.connect(self.browse_result_save_path)
        result_layout.addWidget(QLabel("结果文件:"))
        result_layout.addWidget(self.result_save_edit)
        result_layout.addWidget(result_browse_btn)
        output_layout.addLayout(result_layout)
        
        # 日志文件保存路径
        log_layout = QHBoxLayout()
        self.log_save_edit = QLineEdit()
        self.log_save_edit.setPlaceholderText("选择日志保存路径 (.txt)")
        log_browse_btn = QPushButton("浏览")
        log_browse_btn.clicked.connect(self.browse_log_save_path)
        log_layout.addWidget(QLabel("日志文件:"))
        log_layout.addWidget(self.log_save_edit)
        log_layout.addWidget(log_browse_btn)
        output_layout.addLayout(log_layout)
        
        # 自动保存选项
        self.auto_save_check = QCheckBox("计算完成后自动保存结果")
        self.auto_save_check.setChecked(True)
        output_layout.addWidget(self.auto_save_check)
        
        layout.addWidget(output_group)
        
        # 参数设置组
        param_group = QGroupBox("GMM参数设置")
        param_layout = QVBoxLayout(param_group)
        
        # 聚类层次
        time_load_layout = QHBoxLayout()
        time_load_layout.addWidget(QLabel("聚类层次:"))
        self.time_load_spin = QSpinBox()
        self.time_load_spin.setRange(1, 20)
        self.time_load_spin.setValue(10)
        time_load_layout.addWidget(self.time_load_spin)
        param_layout.addLayout(time_load_layout)
        
        # 期望收敛值
        convergence_layout = QHBoxLayout()
        convergence_layout.addWidget(QLabel("期望收敛值:"))
        self.convergence_spin = QDoubleSpinBox()
        self.convergence_spin.setRange(0.1, 10.0)
        self.convergence_spin.setValue(1.0)
        self.convergence_spin.setSingleStep(0.1)
        convergence_layout.addWidget(self.convergence_spin)
        param_layout.addLayout(convergence_layout)
        
        # 容忍度
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("容忍度:"))
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(1e-4, 1e-1)
        self.tol_spin.setValue(1e-3)
        self.tol_spin.setDecimals(4)
        tol_layout.addWidget(self.tol_spin)
        param_layout.addLayout(tol_layout)
        
        # 迭代次数
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("迭代次数:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 2000)
        self.iter_spin.setValue(500)
        iter_layout.addWidget(self.iter_spin)
        param_layout.addLayout(iter_layout)

        # 计算数据量
        DataNum_layout = QHBoxLayout()
        DataNum_layout.addWidget(QLabel("计算数据量:"))
        self.DataNum_spin = QSpinBox()
        self.DataNum_spin.setRange(0, 100000000)
        self.DataNum_spin.setValue(100)
        DataNum_layout.addWidget(self.DataNum_spin)
        param_layout.addLayout(DataNum_layout)
        
        layout.addWidget(param_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始计算")
        self.start_btn.clicked.connect(self.start_calculation)
        self.stop_btn = QPushButton("停止计算")
        self.stop_btn.clicked.connect(self.stop_calculation)
        self.stop_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        # 总体进度条
        self.progress_group = QGroupBox("系统总进度")
        progress_layout = QVBoxLayout(self.progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(self.progress_group)
        self.progress_group.setVisible(False)
        
        # 拟合进度条
        self.fit_progress_group = QGroupBox("GMM拟合进度")
        fit_progress_layout = QVBoxLayout(self.fit_progress_group)
        
        self.fit_progress_bar = QProgressBar()
        self.fit_progress_bar.setVisible(False)
        fit_progress_layout.addWidget(self.fit_progress_bar)
        
        self.fit_progress_label = QLabel("就绪")
        fit_progress_layout.addWidget(self.fit_progress_label)
        
        layout.addWidget(self.fit_progress_group)
        self.fit_progress_group.setVisible(False)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        return panel
    
    def create_display_area(self):
        """创建右侧显示区域"""
        display_widget = QWidget()
        layout = QVBoxLayout(display_widget)
        
        # 日志输出
        log_group = QGroupBox("计算日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        font = QFont("Courier", 9)
        self.log_text.setFont(font)
        log_layout.addWidget(self.log_text)
        
        # 日志控制按钮
        log_control_layout = QHBoxLayout()
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        export_log_btn = QPushButton("导出日志")
        export_log_btn.clicked.connect(self.export_log)
        save_log_btn = QPushButton("实时保存日志")
        save_log_btn.clicked.connect(self.toggle_log_saving)
        
        log_control_layout.addWidget(clear_log_btn)
        log_control_layout.addWidget(export_log_btn)
        log_control_layout.addWidget(save_log_btn)
        log_control_layout.addStretch()
        
        log_layout.addLayout(log_control_layout)
        
        layout.addWidget(log_group)
        
        return display_widget
    
    def browse_target_file(self):
        """浏览目标数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择目标数据文件", "", "MAT Files (*.mat)")
        if file_path:
            self.target_file_edit.setText(file_path)
            self.log_message(f"已选择目标数据文件: {file_path}")
            self.load_mat_fields(file_path, 'target')
    
    def browse_lut_file(self):
        """浏览LUT数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择LUT数据文件", "", "MAT Files (*.mat)")
        if file_path:
            self.lut_file_edit.setText(file_path)
            self.log_message(f"已选择LUT数据文件: {file_path}")
            self.load_mat_fields(file_path, 'lut')
    
    def load_mat_fields(self, file_path, file_type):
        """加载MAT文件的字段名"""
        try:
            data = scipy.io.loadmat(file_path)
            fields = [key for key in data.keys() if not key.startswith('__')]
            
            if file_type == 'target':
                self.target_data_fields = fields
                self.update_field_combo(self.target_wind_combo, fields)
                self.update_field_combo(self.target_nbrcs_combo, fields)
                self.update_field_combo(self.target_les_combo, fields)
                
                # 设置默认值（如果存在）
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
                
                # 设置默认值（如果存在）
                self.set_default_field(self.lut_nbrcs_les_combo, ['nbrcs_les_save'])
                self.set_default_field(self.lut_wind_speed_combo, ['wind_speed_save'])
                self.set_default_field(self.lut_nbrcs_combo, ['nbrcs_effective'])
                self.set_default_field(self.lut_les_combo, ['les_effective'])
                self.set_default_field(self.lut_wind_speed_all_combo, ['wind_speed_effective'])
            
            # 显示字段映射组
            if self.target_file_edit.text() and self.lut_file_edit.text():
                self.field_mapping_group.setVisible(True)
                
            self.log_message(f"已加载 {file_type} 文件的字段: {', '.join(fields)}")
            
        except Exception as e:
            self.log_message(f"加载MAT文件字段时出错: {str(e)}")
    
    def update_field_combo(self, combo, fields):
        """更新字段选择下拉框"""
        combo.clear()
        combo.addItems(fields)
    
    def set_default_field(self, combo, preferred_fields):
        """设置默认字段"""
        for field in preferred_fields:
            if combo.findText(field) >= 0:
                combo.setCurrentText(field)
                break
    
    def browse_result_save_path(self):
        """浏览结果保存路径"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择结果保存路径", "wind_result.mat", "MAT Files (*.mat)")
        if file_path:
            self.result_save_edit.setText(file_path)
            self.log_message(f"结果将保存到: {file_path}")
    
    def browse_log_save_path(self):
        """浏览日志保存路径"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择日志保存路径", "gmm_calculation_log.txt", "Text Files (*.txt)")
        if file_path:
            self.log_save_edit.setText(file_path)
            self.log_message(f"日志将保存到: {file_path}")
            
            # 如果选择了日志文件，开始实时保存
            self.start_log_saving(file_path)
    
    def start_log_saving(self, log_path):
        """开始实时保存日志到文件"""
        try:
            self.log_file = open(log_path, 'w', encoding='utf-8')
            self.log_message(f"开始实时保存日志到: {log_path}")
        except Exception as e:
            self.log_message(f"无法打开日志文件: {str(e)}")
    
    def toggle_log_saving(self):
        """切换日志保存状态"""
        if self.log_file:
            self.stop_log_saving()
        else:
            log_path = self.log_save_edit.text()
            if log_path:
                self.start_log_saving(log_path)
            else:
                QMessageBox.warning(self, "警告", "请先选择日志文件保存路径")
    
    def stop_log_saving(self):
        """停止实时保存日志"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.log_message("已停止实时日志保存")
    
    def get_field_mapping(self):
        """获取字段映射字典"""
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
        """开始计算"""
        # 检查文件是否存在
        target_file = self.target_file_edit.text()
        lut_file = self.lut_file_edit.text()
        
        if not target_file or not os.path.exists(target_file):
            QMessageBox.warning(self, "警告", "请选择有效的目标数据文件")
            return
            
        if not lut_file or not os.path.exists(lut_file):
            QMessageBox.warning(self, "警告", "请选择有效的LUT数据文件")
            return
        
        # 检查字段映射是否完整
        field_mapping = self.get_field_mapping()
        for key, value in field_mapping.items():
            if not value:
                QMessageBox.warning(self, "警告", f"请为 {key} 选择字段映射")
                return
        
        # 检查输出路径
        result_save_path = self.result_save_edit.text()
        if not result_save_path and self.auto_save_check.isChecked():
            QMessageBox.warning(self, "警告", "请选择结果文件保存路径或取消自动保存")
            return
        
        # 收集参数
        parameters = {
            'time_load': self.time_load_spin.value(),
            'expect_Convergence': self.convergence_spin.value(),
            'tol_expect': self.tol_spin.value(),
            'iter_time': self.iter_spin.value(),
            'Data_Num': self.DataNum_spin.value()
        }
        
        self.log_message("=" * 50)
        self.log_message("开始GMM风速反演计算")
        self.log_message(f"参数设置: {parameters}")
        self.log_message(f"字段映射: {field_mapping}")
        
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_group.setVisible(True)
        # self.fit_progress_bar.setVisible(True)
        # self.fit_progress_group.setVisible(True)
        
        # 创建工作线程
        self.worker = GMMWorker(target_file, lut_file, parameters, field_mapping,
                               result_save_path if self.auto_save_check.isChecked() else None,
                               self.log_save_edit.text())
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.log_message.connect(self.log_message)
        self.worker.finished_success.connect(self.calculation_finished)
        self.worker.error_occurred.connect(self.calculation_error)
        self.worker.fit_progress_updated.connect(self.update_fit_progress)
        
        # 启动线程
        self.worker.start()
    
    def stop_calculation(self):
        """停止计算"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)  # 等待5秒让线程结束
            self.log_message("计算已停止")
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.fit_progress_bar.setVisible(False)
        self.status_label.setText("已停止")
    
    def update_progress(self, current, total, status):
        """更新总体进度条"""
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
        # 只在进度变化较大时记录日志，避免日志过多
        if current % max(1, total // 10) == 0 or current == total:
            self.log_message(f"进度: {current}/{total} ({progress}%) - {status}")
    
    def update_fit_progress(self, progress, current_iter, max_iter, ll):
        """更新拟合进度条"""
        self.fit_progress_bar.setValue(progress)
        self.fit_progress_label.setText(f"迭代: {current_iter}/{max_iter}, 似然值: {ll:.4f}")
        
        # 每10次迭代记录一次日志，避免日志过多
        if current_iter == max_iter:
            self.log_message(f"该次GMM拟合结束, 启动下一次拟合")
        if current_iter % 10 == 0 or current_iter == max_iter:
            self.log_message(f"GMM拟合进度: {current_iter}/{max_iter} ({progress}%), 似然值: {ll:.4f}")
    
    def calculation_finished(self, results):
        """计算完成"""
        self.log_message("计算完成！")
        self.log_message(f"总执行时间: {results['total_time']:.2f} 秒")
        
        # 如果没有自动保存，询问用户是否保存
        if not self.auto_save_check.isChecked():
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存结果", "wind_result.mat", "MAT Files (*.mat)")
            
            if save_path:
                scipy.io.savemat(save_path, results)
                self.log_message(f"结果已保存到: {save_path}")
        
        # 恢复UI状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.fit_progress_bar.setVisible(False)
        self.status_label.setText("计算完成")
        
        QMessageBox.information(self, "完成", "GMM风速反演计算已完成！")
    
    def calculation_error(self, error_msg):
        """计算出错"""
        self.log_message(f"错误: {error_msg}")
        QMessageBox.critical(self, "错误", error_msg)
        
        # 恢复UI状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.fit_progress_bar.setVisible(False)
        self.status_label.setText("计算出错")
    
    def log_message(self, message):
        """在日志框中添加消息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        
        # 自动滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
        # 如果启用了日志文件保存，写入文件
        if self.log_file:
            try:
                self.log_file.write(log_entry + '\n')
                self.log_file.flush()
            except Exception as e:
                # 如果写入失败，停止日志保存
                self.stop_log_saving()
                self.log_message(f"日志文件写入失败: {str(e)}")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
    
    def export_log(self):
        """导出日志到文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出日志", "gmm_calculation_log.txt", "Text Files (*.txt)")
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log_message(f"日志已导出到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出日志失败: {str(e)}")
    
    def closeEvent(self, event):
        """关闭应用时的处理"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, "确认退出", 
                                       "计算仍在进行中，确定要退出吗？",
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
    # 忽略警告
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("CYGNSS GMM 风速反演系统 - 优化版")
    
    # 创建主窗口
    window = GMMGUI()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()