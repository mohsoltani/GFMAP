#!/bin/bash

# Script to run the framework

# Define variables
TASK=$1
APPROACH=$2

# Check if arguments are provided
if [ -z "$TASK" ] || [ -z "$APPROACH" ]; then
    echo "Usage: ./run.sh <task> <approach>"
    exit 1
fi

# Run the framework with provided arguments
python3 main.py --t "$TASK" --a "$APPROACH"
