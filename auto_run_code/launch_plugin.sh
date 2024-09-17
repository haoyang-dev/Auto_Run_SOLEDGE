#!/bin/bash

# Configuration
min_cycle_time=1
keep_running=true

# Check if required variables are set
if [ -z "$execute_commands" ]; then
    echo "Error: execute_commands is empty. Exiting."
    exit 1
fi

if [ -z "$code_folder_path" ]; then
    echo "Error: code_folder_path is not set. Exiting."
    exit 1
fi

if [ -z "$auto_run_setup" ]; then
    echo "Error: auto_run_setup is not set. Exiting."
    exit 1
fi

# Initial setup
python3 "$code_folder_path/main.py" --mode initialize --setup "$auto_run_setup" --time_limit "$running_time_limit_in_hours" --iteration_limit "$iteration_number_limit"
initial_exit_code=$?

# Check if the initial command succeeded and handle exit codes
if [ $initial_exit_code -ne 0 ]; then
    if [ $initial_exit_code -eq 211 ]; then
        echo "Running not started due to manual stop"
        exit 0
    else
        echo "Initial setup failed with exit code $initial_exit_code. Exiting."
        exit $initial_exit_code
    fi
fi

# Main loop
while $keep_running; do
    # Record the start time
    start_time=$(date +%s)

    # Execute custom commands
    eval "$execute_commands"

    if [ $? -ne 0 ]; then
        echo "execute_commands failed, exiting."
        exit 1
    fi

    # Record the end time
    end_time=$(date +%s)

    # Calculate the elapsed time
    elapsed_time=$((end_time - start_time))

    # Calculate remaining sleep time if needed
    if [ $elapsed_time -lt $min_cycle_time ]; then
        remaining_sleep=$((min_cycle_time - elapsed_time))
        echo "------------------------------------------------------------------------------------------------"
        echo "Executed commands ran for a short time ($elapsed_time seconds). Please verify if they are working correctly."
        echo "Sleeping for $remaining_sleep seconds to ensure a minimum cycle time and avoid rapid looping."
        sleep $remaining_sleep
    fi

    # Supervision step
    python3 "$code_folder_path/main.py" --mode supervise --setup "$auto_run_setup" --time_limit "$running_time_limit_in_hours" --iteration_limit "$iteration_number_limit"
    exit_code=$?

    # Handle exit codes
    if [ $exit_code -ne 0 ]; then
        if [ $exit_code -eq 211 ]; then
            echo "Running stopped normally."
        else
            echo "Running failed with exit code $exit_code."
        fi
        keep_running=false
    fi
done

echo "Job finished."