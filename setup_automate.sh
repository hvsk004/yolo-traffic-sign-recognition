#!/bin/bash

# Update and upgrade the system
echo "Updating and upgrading system..."
sudo apt update && sudo apt upgrade -y

# Install Git, Python3, pip3, and python3-venv
echo "Installing Git, Python3, pip3, and python3-venv..."
sudo apt install -y git python3 python3-pip python3.12-venv

# Create aliases for python and pip
echo "Creating aliases for python and pip..."
echo "alias python='python3'" >> ~/.bashrc
echo "alias pip='pip3'" >> ~/.bashrc
source ~/.bashrc

# Create a new folder AI-ML and navigate into it
echo "Creating directory AI-ML and navigating into it..."
mkdir ~/AI-ML
cd ~/AI-ML

# Create and activate a virtual environment named AI-ML
echo "Creating and activating virtual environment..."
python3 -m venv AI-ML
source AI-ML/bin/activate

# Install the required Python packages
echo "Installing required Python packages..."
pip install kagglehub ultralytics torch torchvision torchaudio

# Clone your GitHub repository
echo "Cloning the YOLO Traffic Sign Recognition repository..."
git clone https://github.com/hvsk004/yolo-traffic-sign-recognition.git

# Navigate to the cloned repo directory
cd yolo-traffic-sign-recognition

# Final message
echo "Setup complete! You can now start working on your project."
