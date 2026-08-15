"""
run_frontend.py
Simple wrapper to run Streamlit frontend with proper configuration
"""
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Run streamlit
if __name__ == "__main__":
    os.system("streamlit run frontend/app.py --server.fileWatcherType none")
