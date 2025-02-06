import pytest
import os

@pytest.fixture
def outdir():
    return os.path.join(os.path.dirname(__file__), 'test_output')

