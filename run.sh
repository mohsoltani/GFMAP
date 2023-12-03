#!/bin/bash

# Script to run the framework

TASK=$1
APPROACH=$2

if [ -z "$TASK" ] || [ -z "$APPROACH" ]; then
    echo "Usage: ./run.sh <task> <approach>"
    exit 1
fi

python3 main.py --t "$TASK" --a "$APPROACH"
