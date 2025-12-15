"""
MEGA-RAG Setup Script
"""
from setuptools import setup, find_packages

setup(
    name="mega-rag",
    version="1.0.0",
    description="Medical Evidence-Guided Augmented Retrieval-Augmented Generation",
    author="HARSH SINGH",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "google-generativeai>=0.8.0",
        "langchain>=0.3.0",
        "langchain-google-genai>=2.0.0",
        "langgraph>=0.2.0",
        "chromadb>=0.5.0",
        "sentence-transformers>=3.0.0",
        "rank-bm25>=0.2.2",
        "networkx>=3.2",
        "pypdf>=4.0.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "numpy>=1.24.0",
        "ragas>=0.1.0",
        "datasets>=2.14.0",
        "transformers>=4.35.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "pydantic>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "mega-rag=mega_rag.main:main",
        ],
    },
)
