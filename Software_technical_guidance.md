# CYGNSS GMM Wind Speed Inversion System User Manual 
## System Overview 
The CYGNSS GMM wind speed inversion system is a satellite wind speed data processing system based on the Gaussian Mixture Model (GMM). It is specifically designed to handle CYGNSS satellite data. This system provides a user-friendly operation experience through a GUI interface, supports large-scale data parallel processing, and has efficient computing performance and stable operation capabilities. 
## System Requirements 
### Hardware Requirements
- **Memory**: Minimum 8GB, recommended 16GB or more
- **Processor**: Multi-core CPU, recommended 4 cores or more
- **Storage Space**: At least 1GB of available space
- **Operating System**: Windows 10/11, Linux, macOS 
### Software Requirements
- **Python**: 3.7+
- **Dependent Libraries** - PyQt5
- NumPy
- SciPy
- scikit-learn
- joblib
- psutil

## Installation and Startup 
### Installation Steps
1. Make sure Python 3.7 or a higher version has been installed.
2. Install the required packages:
```bash
pip install PyQt5 numpy scipy scikit-learn joblib psutil matplotlib
```
### Start the system 
```bash
python GMM_System_GUI_v1.5.py
```

## Usage Process 
### 1. Data Preparation
- **Target Data File**: A MAT file containing measured wind speed, NBRCS, and LES data
- **LUT Data File**: A MAT file containing lookup table data, used for model training 
### 2. File Selection
1. Click the "Browse" button to select the target data file (.mat)
2. Click the "Browse" button to select the LUT data file (.mat) 
### 3. Field Mapping Settings
The system automatically detects the fields in the MAT file. Users need to select the corresponding field names for each parameter: 
| Parameter | Description | Suggested Field Name | 
|------|------|------------|
| measure_wind_speed | Measured wind speed data | `cygnss_wind`, `wind_speed` |
| measure_nbrcs | Measured NBRCS data | `nbrcs_mean`, `nbrcs` |
| measure_les | Measured LES data | `les_mean`, `les` |
| nbrcs_les | LUT-based NBRCS-LES data | `nbrcs_les_save` |
| wind_speed | LUT-based wind speed data | `wind_speed_save` |
| nbrcs | LUT-based NBRCS data | `nbrcs_effective` |
| les | LUT-based LES data | `les_effective` |
| wind_speed_all | Complete wind speed data in LUT | `wind_speed_effective` | 

### 4. Output Settings
- **Result File**: Select the saving path for the wind speed inversion results (.mat)
- **Log File**: Select the saving path for the calculation logs (.txt)
- **Auto Save**: Enable this option to automatically save the results upon completion of the calculation
- 
### 5. Parameter Settings 

#### Explanation of GMM Parameter Settings 

| Parameter | Explanation | Value Range | Recommended Value | Impact Analysis |
|------|------|----------|--------|----------|
| Clustering Level | GMM Clustering Depth | 1-20 | 10 | The greater the value, the finer the classification and the longer the calculation time |
| Expected Convergence Value | Wind Speed Interval Convergence Threshold | 0.1 - 10.0 | 1.0 | Controls the termination condition of clustering, the smaller the value, the higher the accuracy |
| Tolerance | GMM Fitting Convergence Tolerance | 1e-4 - 1e-1 | 1e-3 | Affects the convergence speed of the model, the smaller the value, the higher the accuracy |
| Number of Iterations | Maximum Number of Iterations | 10 - 2000 | 500 | Ensures full convergence of the model |
| Computational Data Volume | Number of Data Points to Process | 0 - Unlimited | 100 - 10,000 | 0 indicates processing all data | 

### 6. Start Calculation
Click the "Start Calculation" button to initiate the wind speed inversion process. The system will:
- Display a progress bar for the overall process
- Show the progress of GMM fitting
- Provide real-time output of calculation logs
- Support the option to stop the calculation at any point

## Parameter Suggestions 

### Parameter configuration for different data sizes 
#### Small-scale data (less than 1000 points) 
```python
Parameter configuration:
Cluster level: 5-8
Expected convergence value: 0.5-1.0
Tolerance: 1e-3
Number of iterations: 200-300
Data for calculation: All data
```

#### Medium-sized data (1000 - 10000 points) 
```python
Parameter configuration:
Cluster level: 8-12
Expected convergence value: 1.0-2.0
Tolerance: 1e-3
Number of iterations: 300-500
Computational data volume: Adjust as needed
```

#### Large-scale data (more than 10,000 points) 
```python
Parameter configuration:
Cluster level: 10 - 15
Expected convergence value: 2.0 - 5.0
Tolerance: 1e-2
Number of iterations: 500 - 800
Data processing method: Batch processing
```

### Performance Optimization Suggestions 
1. **Memory Management**:
- For large datasets, appropriately reduce the "amount of calculated data"
- Process data in batches to avoid memory overflow

2. **Computational Efficiency**:
- Automatically optimize parallel processes based on the number of CPU cores
- Utilize vectorized operations to enhance computing speed

3. **Balance between Accuracy and Speed**:
- Reducing the "clustering level" and "iteration times" can enhance the speed
- Increasing the "tolerance" can accelerate convergence but may reduce the accuracy

## Output Instructions 

### Result File Structure
The results are saved as a MAT format file and include the following fields: 

```matlab
wind_res =
lut_wind: [n×1 double]    % Wind speed result from GMM inversion
ini_wind: [n×1 double]    % Initial wind speed data
total_time: double        % Total computation time (seconds)
parameters: struct        % Parameter settings used
field_mapping: struct     % Field mapping information
```

### Result Interpretation 

1. **lut_wind**: Wind speed values derived from GMM model inversion, the main analysis object
2. **ini_wind**: Original input wind speed data, used for comparison and verification
3. **Statistical Analysis**:
- Mean error: `mean(lut_wind - ini_wind)`
- Root mean square error: `sqrt(mean((lut_wind - ini_wind).^2))`
- Correlation coefficient: `corr(lut_wind, ini_wind)`

### Log File Content
The log file contains detailed operational information:
- Timestamp and operation records
- Parameter setting information
- Progress status of calculations
- Error and warning messages
- Performance statistics information

## Troubleshooting 

### Common Questions and Solutions 

#### 1. Memory shortage error
**Symptoms**: Program crashes or reports memory error
**Solution**:
- Reduce the "data processing volume" parameter
- Close other programs that consume memory
- Increase the system's virtual memory

#### 2. Calculation speed is too slow
**Symptom**: Processing time is extremely long
**Solution**:
- Reduce the "clustering level" and "iteration times"
- Increase the "tolerance" parameter
- Decrease the "amount of calculation data"

#### 3. Field Mapping Error
**Symptom**: Error message during data loading indicating that the field does not exist
**Solution**:
- Verify the actual field names in the MAT file
- Ensure that the fields in the target file and the LUT file are correctly matched

#### 4. Abnormal Results
**Symptom**: The inverted wind speed values are obviously unreasonable.
**Solution**:
- Check the quality of input data
- Adjust the GMM parameters, especially the convergence value
- Verify if the field mapping is correct

## Advanced Features 

### Real-time Monitoring
- Overall System Progress: Displays the overall processing progress
- GMM Fitting Progress: Shows the training progress of each GMM model
- Real-time Log: Monitors detailed information during the calculation process

### Parallel computing optimization
The system automatically optimizes parallel computing based on the following factors:
- Number of CPU cores
- Available memory size
- Data volume size
- Complexity of the task

### Data Caching Mechanism
- LUT Data Caching: Avoid redundant loading to enhance efficiency
- Target Data Caching: Accelerate processing of the same data multiple times

## Technical Support 

In case of technical issues, please provide the following information:
1. The content of the system log file
2. The parameter settings used
3. Basic information of the data file
4. Screenshots of the error message

## Version Update 

### v1.4 Update Contents
- Optimized the GMM algorithm to enhance computing speed
- Improved memory management to support larger data volumes
- Enhanced error handling and logging
- Improved user interface and operational experience 
---

This document is updated along with the program version. Please ensure you are using the latest version of this document.
