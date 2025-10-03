#!/bin/bash

# Streamlit App Startup Script for Codespaces
# This script ensures proper environment setup before starting the app

echo "════════════════════════════════════════════════════════════════"
echo "  Starting Resume Analysis Agent with Streamlit"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Navigate to project root
cd /workspaces/RAG-Resume || exit 1

# Set PYTHONPATH to ensure Ray workers can find local modules
export PYTHONPATH=/workspaces/RAG-Resume:$PYTHONPATH
echo "✅ PYTHONPATH set to: $PYTHONPATH"
echo ""

# Check Ray status from .env
USE_RAY=$(grep "^USE_RAY=" .env | cut -d'=' -f2)
echo "📊 Ray Status: $USE_RAY"

if [ "$USE_RAY" = "true" ]; then
    echo "⚠️  WARNING: Ray is enabled in .env"
    echo "   This may cause disconnections in Codespaces."
    echo "   Set USE_RAY=false in .env for stability."
    echo ""
else
    echo "✅ Ray is disabled (recommended for Codespaces)"
    echo "   Processing will use ThreadPoolExecutor"
    echo ""
fi

# Kill any existing Streamlit processes
pkill -f streamlit 2>/dev/null
sleep 1

# Check if port 8501 is available
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8501 is in use, attempting to free it..."
    lsof -ti:8501 | xargs kill -9 2>/dev/null
    sleep 2
fi

echo ""
echo "🚀 Starting Streamlit app..."
echo "   URL: http://0.0.0.0:8501"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Start Streamlit with proper configuration
python -m streamlit run app/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
