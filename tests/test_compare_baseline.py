import numpy as np
import pytest


def test_build_matrix_fills_correct_cells():
    from compare_baseline import build_matrix

    severity_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    angle_edges    = np.linspace(0.0, 2.27, 5)  # 4 bins

    trials = [
        {"angle_bin": 0, "severity": 0.1, "test_corr": 0.8},
        {"angle_bin": 0, "severity": 0.1, "test_corr": 0.6},
        {"angle_bin": 1, "severity": 0.6, "test_corr": 0.5},
    ]

    matrix = build_matrix(trials, "test_corr", severity_edges)

    assert matrix.shape == (4, 4)
    # Cell (0,0): mean of 0.8 and 0.6
    assert abs(matrix[0, 0] - 0.7) < 1e-9
    # Cell (1,2): 0.5
    assert abs(matrix[1, 2] - 0.5) < 1e-9
    # Unfilled cells are NaN
    assert np.isnan(matrix[0, 1])


def test_build_matrix_empty_returns_all_nan():
    from compare_baseline import build_matrix

    severity_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    angle_edges    = np.linspace(0.0, 2.27, 5)
    matrix = build_matrix([], "corr", severity_edges)
    assert matrix.shape == (4, 4)
    assert np.all(np.isnan(matrix))


def test_compute_boost_pct():
    from compare_baseline import compute_boost_pct

    # 50% of the remaining gap from 0.4 floor to 1.0 ceiling
    assert abs(compute_boost_pct(acc=0.7, floor=0.4) - 50.0) < 1e-9
    # At floor — 0% boost
    assert compute_boost_pct(acc=0.4, floor=0.4) == 0.0
    # At ceiling — 100% boost
    assert abs(compute_boost_pct(acc=1.0, floor=0.4) - 100.0) < 1e-9
