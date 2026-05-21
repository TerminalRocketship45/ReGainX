import numpy as np
import pytest


def test_compute_severity_min():
    from utils import compute_severity
    # min impairment: force_scale=0.9, slowdown=1.1, avg_mf=0
    assert compute_severity(0.9, 1.1, 0.0) == pytest.approx(0.0, abs=1e-6)


def test_compute_severity_max():
    from utils import compute_severity
    # max impairment: force_scale=0.6, slowdown=1.4, avg_mf=1.0
    assert compute_severity(0.6, 1.4, 1.0) == pytest.approx(1.0, abs=1e-6)


def test_compute_severity_mid():
    from utils import compute_severity
    # midpoint of each component = 0.5
    assert compute_severity(0.75, 1.25, 0.5) == pytest.approx(0.5, abs=1e-6)


def test_get_angle_bin_returns_valid_index():
    from utils import get_angle_bin
    edges = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    assert get_angle_bin(0.6, edges) == 0
    assert get_angle_bin(1.2, edges) == 1
    assert get_angle_bin(2.4, edges) == 3


def test_get_severity_quartile():
    from utils import get_severity_quartile
    edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert get_severity_quartile(0.1, edges) == 0
    assert get_severity_quartile(0.6, edges) == 2
    assert get_severity_quartile(0.9, edges) == 3
