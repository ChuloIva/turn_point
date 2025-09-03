#!/bin/bash
# Wrapper script to run SelfIE integration with compatible environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SELFIE_ENV="$SCRIPT_DIR/selfie_env"

# Check if SelfIE environment exists
if [ ! -d "$SELFIE_ENV" ]; then
    echo "❌ SelfIE environment not found!"
    echo "Run: ./setup_selfie_env.sh"
    exit 1
fi

# Activate SelfIE environment
echo "🔧 Activating SelfIE environment..."
source "$SELFIE_ENV/bin/activate"

# Run the requested script
if [ $# -eq 0 ]; then
    echo "🚀 Running SelfIE adapter demo..."
    python gemma_selfie_adapter.py
else
    echo "🚀 Running: $@"
    python "$@"
fi

# Deactivate environment
deactivate
echo "✅ Done!"