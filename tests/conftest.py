"""
Pytest configuration and fixtures for MEGA-RAG tests.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def pubmedqa_dir(project_root):
    """Get PubMedQA directory."""
    return project_root / "pubmedQA"


@pytest.fixture(scope="module")
def data_manager():
    """Create DatasetManager instance."""
    from mega_rag.utils.data_loader import DatasetManager
    return DatasetManager()


@pytest.fixture(scope="module")
def pubmedqa_evaluator():
    """Create PubMedQAEvaluator instance without workflow."""
    from mega_rag.utils.pubmedqa_evaluator import PubMedQAEvaluator
    return PubMedQAEvaluator()


@pytest.fixture(scope="session")
def sample_questions():
    """Sample medical questions for testing."""
    return [
        "Is aspirin effective for preventing heart attacks?",
        "Does exercise reduce blood pressure?",
        "Are statins safe for long-term use?",
        "Can diet alone control type 2 diabetes?",
        "Is metformin the first-line treatment for diabetes?",
    ]


@pytest.fixture(scope="session")
def sample_contexts():
    """Sample medical contexts for testing."""
    return [
        "Studies have shown that regular aspirin use can reduce the risk of heart attacks in high-risk patients.",
        "Clinical trials demonstrate that moderate exercise leads to significant reductions in systolic blood pressure.",
        "Long-term statin therapy has been associated with improved cardiovascular outcomes and acceptable safety profiles.",
    ]


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: marks tests that require Ollama to be running"
    )
