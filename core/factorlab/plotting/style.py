"""统一报告视觉风格的绘图预设。"""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 4.8),
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )
