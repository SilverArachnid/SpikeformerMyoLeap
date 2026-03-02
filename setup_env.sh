#!/bin/bash
set -e

# Get the script's directory and project roots
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

cd "$PROJECT_DIR"

# Ensure Ultraleap system dependencies are installed
echo "Checking and installing Ultraleap system dependencies (requires sudo)..."
if ! dpkg -l | grep -q ultraleap-hand-tracking; then
    echo "Adding Ultraleap GPG Key and APT repository..."
    wget -qO - https://repo.ultraleap.com/keys/apt/gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/ultraleap.gpg >/dev/null
    echo 'deb [arch=amd64] https://repo.ultraleap.com/apt stable main' | sudo tee /etc/apt/sources.list.d/ultraleap.list >/dev/null
    sudo apt update
    sudo apt install -y ultraleap-hand-tracking
else
    echo "Ultraleap hand tracking is already installed."
fi

# Initialize uv environment
echo "Creating uv virtual environment..."
export LEAPSDK_INSTALL_LOCATION="/usr/lib/ultraleap-hand-tracking-service"
export LEAPC_HEADER_OVERRIDE="/usr/include/LeapC.h"
export LEAPC_LIB_OVERRIDE="/usr/lib/x86_64-linux-gnu/libLeapC.so"

uv venv --python 3.10
source .venv/bin/activate

# Install main project dependencies
echo "Installing project dependencies..."
uv pip install -e .
uv pip install build

# Setup LeapC python bindings from local clone
echo "Building and installing leapc-python-bindings..."
LEAPC_DIR="$PROJECT_DIR/vendor/leapc-python-bindings"
if [ ! -d "$LEAPC_DIR" ]; then
    echo "Cloning leapc-python-bindings..."
    git clone https://github.com/ultraleap/leapc-python-bindings.git "$LEAPC_DIR"
fi

cd "$LEAPC_DIR"
uv pip install -r requirements.txt
python -m build leapc-cffi
uv pip install leapc-cffi/dist/*.tar.gz
uv pip install -e leapc-python-api

echo ""
echo "======================================"
echo "Environment setup complete!"
echo "Run 'source .venv/bin/activate' to enter the environment."
echo "======================================"
