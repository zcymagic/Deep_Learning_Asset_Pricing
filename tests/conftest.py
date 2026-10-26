import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glassbox.synthetic import SyntheticSpec, make_synthetic_panel


@pytest.fixture(scope='session')
def small_panel():
	return make_synthetic_panel(SyntheticSpec(T=60, N=150, seed=3))


@pytest.fixture(scope='session')
def full_panel():
	return make_synthetic_panel(SyntheticSpec(seed=0))


@pytest.fixture(scope='session')
def null_panel():
	return make_synthetic_panel(SyntheticSpec(seed=11, mispricing=0.0, missing_factor=0.0))
