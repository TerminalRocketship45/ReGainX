# tests/test_algo_compare.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
from utils import plot_confusion_matrix

def test_plot_confusion_matrix_pct_labels(tmp_path):
    matrix = np.array([[0.8, 0.6], [0.4, np.nan]])
    angle_labels = ["0.5-1.0", "1.0-1.5"]
    sev_labels   = ["Q1 mild", "Q2"]
    out = str(tmp_path / "cm.png")
    plot_confusion_matrix(matrix, angle_labels, sev_labels, "Test", out, pct=True)
    assert os.path.exists(out)

def test_plot_confusion_matrix_default_no_pct(tmp_path):
    matrix = np.array([[0.8, 0.6], [0.4, 0.2]])
    out = str(tmp_path / "cm.png")
    plot_confusion_matrix(matrix, ["a", "b"], ["c", "d"], "T", out)
    assert os.path.exists(out)
