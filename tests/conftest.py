import pytest
import os
from generate_data import save_chirps, create_data_iterators, generate_iterator

@pytest.fixture
def outdir():
    return os.path.join(os.path.dirname(__file__), 'test_output')



@pytest.fixture
def data_iterators():
    return generate_iterator(4096, 32), generate_iterator(1024, 32)