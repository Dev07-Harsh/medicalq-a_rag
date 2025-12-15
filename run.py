#!/usr/bin/env python3
"""
MEGA-RAG Quick Start Script
Simplified runner for the MEGA-RAG system.
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mega_rag.main import main

if __name__ == "__main__":
    main()
