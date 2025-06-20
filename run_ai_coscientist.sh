#!/bin/bash
# AI Co-Scientist Launcher Script
# This script ensures the correct Python path and runs the AI Co-Scientist system

export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:$PYTHONPATH"

# Run the AI Co-Scientist with all provided arguments
python3 main.py "$@"