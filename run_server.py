#!/usr/bin/env python3
"""
Server runner for MEGA-RAG.
"""
import uvicorn
import sys
import os

if __name__ == "__main__":
    # Ensure current directory is in path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting MEGA-RAG Server on http://localhost:8000")
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
