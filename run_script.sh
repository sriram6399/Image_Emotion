#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="gen_images.py"

# Define how long the script should run before killing it (in seconds)
RUN_TIME=$((25 * 60)) # 25 minutes in seconds

# Define how long to wait before restarting (in seconds)
RESTART_DELAY=3 # 3 seconds

# Define the total runtime for the entire script (6 hours in seconds)
TOTAL_RUNTIME=$((5 * 60 * 60)) # 6 hours in seconds

# Initialize elapsed time
elapsed_time=0

# Start looping until the total runtime exceeds 6 hours
while [ $elapsed_time -lt $TOTAL_RUNTIME ]; do
    echo "Starting Python script..."

    # Run the Python script in the background
    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PYTHON_SCRIPT &
    
    # Get the process ID of the background Python script
    PYTHON_PID=$!

    # Let the script run for the specified amount of time
    sleep $RUN_TIME

    # Kill the Python process
    echo "Stopping Python script with PID $PYTHON_PID"
    kill $PYTHON_PID

    # Wait for the process to be killed properly
    wait $PYTHON_PID 2>/dev/null

    # Update the elapsed time
    elapsed_time=$((elapsed_time + RUN_TIME))

    # Check if the total runtime has been reached
    if [ $elapsed_time -ge $TOTAL_RUNTIME ]; then
        echo "Reached total runtime of 6 hours. Exiting."
        break
    fi

    # Wait before restarting the process
    echo "Waiting for $RESTART_DELAY seconds before restarting..."
    sleep $RESTART_DELAY

    # Update the elapsed time for the total period (RUN_TIME + RESTART_DELAY)
    elapsed_time=$((elapsed_time + RESTART_DELAY))
done

echo "Script finished running for 6 hours."
