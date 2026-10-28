import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model_plots import plot_pr_roc_curves

FOLD_COUNT = 3


def test_plot_pr_roc_curves_draws_each_fold_plus_pooled_and_reference_lines():
    rng = np.random.default_rng(42)
    y_true = np.tile([0, 0, 0, 1], 30)
    proba = np.clip(y_true * 0.3 + rng.random(len(y_true)) * 0.7, 0, 1)
    fold_ids = np.arange(len(y_true)) % FOLD_COUNT

    fig = plot_pr_roc_curves(y_true, proba, fold_ids, "Model")

    pr_ax, roc_ax = fig.axes
    assert len(pr_ax.lines) == FOLD_COUNT + 2
    assert len(roc_ax.lines) == FOLD_COUNT + 2
    reference_level = pr_ax.lines[-1].get_ydata()[0]
    assert reference_level == y_true.mean()
    plt.close(fig)
