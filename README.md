## Overview

The `auto_run_code` package automates job continuation in HPC environments, enabling simulations to extend beyond runtime limits, ensuring smooth operation for long-term SOLEDGE simulations.

## Prerequisites

1.  **Python Environment**: Ensure that `Python 3` is installed, along with required packages: `numpy`, `matplotlib`, `scipy`, `intersect`, `h5py`, `pandas`.
2.  **SOLEDGE Case**: Prepare a SOLEDGE case by following either **Option A** or **Option B** below:
    
    ### A. Provided SOLEDGE-EIRENE Test Case run in Marconi
    
    *   It is WEST LSN case, low input power, full drifts based on updated master branch (**SOLEDGE3X branch**: v1.4.1\_master, **STYX branch**: v1.3.2\_master)
    *   Ensure all paths in `run_case_MARCONI.sh` are updated for the lines marked by `# !!! CHANGE HERE!!!`).
    *   Ensure the correct projet information in job\_MARCONI.slurm
    
    ### B. Custom SOLEDGE Case
    
    *   Verify that your SOLEDGE case runs independently without issues.
    *   Ensure the `NITER` value in `param_raptorX.txt` is not too high to complete SOLEDGE calculation within the `SBATCH --time` limit.
    *   **Launch Script**:
        *   Add the python3 module to your launch script:
            
            ```plaintext
            module load python/3.9.4  # used in Marconi, choose proper python3 module for your HPC environment
            ```
            
        *   Replace `mv plasma_*.h5 Plasma/` with `cp plasma_*.h5 Plasma/`.

## Step-by-Step Guide

### 1\. Prepare the Environment

*   Copy the `auto_run_code` package and `auto_run_S3XE_control_setup.py` to the SOLEDGE case folder.

### 2\. Initialize Automation

*   In the case folder, initialize the process with:
    
    **For the provided test case:**
    
    ```plaintext
    python3 auto_run_code/initialization.py job_MARCONI.slurm setup_soledge.py
    ```
    
    **For a custom SOLEDGE case:**  
    Replace `job_MARCONI.slurm` with the Slurm job script of your case.
    
*   After initialization, a Slurm file (e.g., `auto_run_job_MARCONI.slurm`) will be generated. Modify the following parameters if needed:
    
    *   `running_time_limit_in_hours` (default: 10000 hours)
    *   `iteration_number_limit` (default: 10000 runs)
    
    The calculation will stop when either criterion is met.
    

### 3\. Configure the SOLEDGE Running Setup

*   Review and adjust `auto_run_S3XE_control_setup.py` as needed. No changes are necessary for the provided test case.
    *   **Trace Mode Options**:
        *   `BASIC`: Extends runtime without output data analysis (compatible with most SOLEDGE cases).
        *   `PRO`: Extracts output data at user defined points and lines.
        *   `WEST`: Performs specialized output data analysis for the LSN WEST configuration (default for the test case).

### 4\. Start the Simulation

*   **For the test case:**
    
    ```plaintext
    sbatch auto_run_job_MARCONI.slurm
    ```
    
    or
    
    ```plaintext
    python3 auto_run_code/running_control.py slurm_run auto_run_job_MARCONI.slurm
    ```
    
*   **For a custom SOLEDGE case:**  
    Replace `auto_run_job_MARCONI.slurm` with `auto_run_<slurm job script for your case>`.

### 5\. Stop the Simulation

*   **For the test case:**
    
    To stop the simulation safely, use:
    
    ```plaintext
    python3 auto_run_code/running_control.py safe_stop auto_run_job_MARCONI.slurm
    ```
    
    For a quicker stop:
    
    ```plaintext
    python3 auto_run_code/running_control.py slurm_quick_stop auto_run_job_MARCONI.slurm
    ```
    
    Alternatively, you can manually cancel the Slurm job using:
    
    ```plaintext
    scancel <jobid>
    ```
    
*   **For a custom SOLEDGE case:**
    
    Replace `auto_run_job_MARCONI.slurm` with the `auto_run_<slurm job script for your case>`.
    

## Time Trace Data Analysis

*   You can check the time evolution of input and output parameters by reviewing the data in `auto_run_output/Trace_Data.csv`.

> **Tip**: When changing the `Trace_mode`, it is necessary to remove the `auto_run_output` directory before running.