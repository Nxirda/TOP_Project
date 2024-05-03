#!/bin/bash

# Set the name for the virtual environment
venv="TOP"

# Create the virtual environment
python3 -m venv "$venv"

# Activate the virtual environment
source "$venv/bin/activate"

# Install Matplotlib and NumPy
pip install matplotlib numpy pandas optuna

# Deactivate the virtual environment
deactivate

echo "Virtual environment $venv_name created and packages installed."

