#!/bin/bash
# Setup script for SelfIE-compatible environment using venv

echo "🔧 Setting up SelfIE-compatible environment with venv..."

# Create a separate venv environment for SelfIE
python -m venv selfie_env

# Activate the environment
source selfie_env/bin/activate

# Install compatible transformers version (SelfIE was tested with 4.34.0)
echo "📦 Installing transformers==4.34.0..."
pip install transformers==4.34.0

# Install other required packages
echo "📦 Installing PyTorch and dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas tqdm
pip install accelerate

# Install SelfIE
echo "📦 Installing SelfIE..."
cd ../selfie
pip install -e .
cd ../selfie_gemma_integration

echo "✅ SelfIE environment setup complete!"
echo ""
echo "To use the SelfIE integration:"
echo "  source selfie_gemma_integration/selfie_env/bin/activate"
echo "  python gemma_selfie_adapter.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"