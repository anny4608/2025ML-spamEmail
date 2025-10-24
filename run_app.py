"""Wrapper script to run the Streamlit app from the project root."""
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# Now run the Streamlit app
if __name__ == "__main__":
    import streamlit.web.bootstrap
    from app_streamlit import main
    
    streamlit.web.bootstrap.run(main, "", [], [])