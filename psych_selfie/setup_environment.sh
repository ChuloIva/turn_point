#!/bin/bash
# Setup script for SelfIE Psychology Environment

echo "🧠 Setting up SelfIE Psychology Environment..."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment 'selfie_psych_env'..."
python3 -m venv selfie_psych_env

# Activate environment
echo "🔄 Activating environment..."
source selfie_psych_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing requirements..."
pip install -r requirements.txt

# Install SelfIE library
echo "🔧 Installing SelfIE library..."
cd ../third_party/selfie
pip install -e .
cd ../../psych_selfie

# Install Jupyter kernel
echo "📓 Installing Jupyter kernel..."
pip install ipykernel
python -m ipykernel install --user --name=selfie_psych --display-name="SelfIE Psychology"

echo "✅ Setup complete!"
echo ""
echo "🚀 To get started:"
echo "1. source psych_selfie/selfie_psych_env/bin/activate"
echo "2. jupyter notebook selfie_experiment_notebook.ipynb"
echo ""
echo "📝 Note: This environment uses transformers==4.34.0 (required for SelfIE)"
echo "    Keep it separate from your main environment to avoid conflicts."