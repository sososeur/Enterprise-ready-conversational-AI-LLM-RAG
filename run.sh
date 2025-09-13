#!/bin/bash

# A more robust script to run the backend and frontend, with cleanup.

# Function to clean up background processes on exit
cleanup() {
    echo "Shutting down backend API server..."
    # Kill the process running on port 8000
    kill $(lsof -t -i:8000)
    echo "Cleanup complete."
}

# Trap the EXIT signal to run the cleanup function when the script ends
trap cleanup EXIT

echo "Starting Backend API Server on port 8000..."
# Start the FastAPI server in the background
uvicorn backend.api:app --host 0.0.0.0 --port 8000 &

# Give the server a moment to start up
echo "Waiting for backend to be ready..."
sleep 5

echo "Starting Frontend Streamlit App..."
# Start the Streamlit app. The script will wait here until you close Streamlit.
streamlit run frontend/app.py

# When you press Ctrl+C in the terminal, the script will exit,
# triggering the 'trap' command above, which calls the 'cleanup' function.