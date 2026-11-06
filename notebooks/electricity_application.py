"""Electricity exploration and Oventi figures; no loading or experiment engine."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

OVENTI = {
    "light": "#3737AC",
    "medium": "#252673",
    "dark": "#131439",
    "teal": "#009C9E",
    "orange": "#F79021",
}


def describe_data(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Describe every client, exposing missingness and invalid values."""
    finite = frame.where(np.isfinite(frame))
    description = finite.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    description = description.rename(columns={"std": "std_sample"})
    description["std_population"] = finite.std(ddof=0)
    description["variance_sample"] = finite.var(ddof=1)
    description["range"] = finite.max() - finite.min()
    description["iqr"] = finite.quantile(0.75) - finite.quantile(0.25)
    description["skewness"] = finite.skew()
    description["excess_kurtosis"] = finite.kurt()
    description["zeros"] = frame.eq(0).sum()
    description["zero_fraction"] = description["zeros"] / len(frame)
    description["negative"] = frame.lt(0).sum()
    description["missing"] = frame.isna().sum()
    description["infinite"] = np.isinf(frame).sum()
    description["n_unique"] = frame.nunique()
    quality = pd.Series(
        {
            "observations": len(frame),
            "client_columns": frame.shape[1],
            "numeric_columns": len(frame.select_dtypes(include="number").columns),
            "missing_cells": int(frame.isna().sum().sum()),
            "infinite_cells": int(np.isinf(frame).sum().sum()),
            "constant_clients": int((finite.nunique() <= 1).sum()),
            "repeated_rows_not_removed": int(frame.duplicated().sum()),
            "index_unique": frame.index.is_unique,
            "index_increasing": frame.index.is_monotonic_increasing,
            "source_dates_available": False,
        },
        name="Data quality",
    )
    return quality, description


class ExperimentReporter:
    """Save figures lazily; show observed fold means with temporal bootstrap caps."""

    def __init__(self, level: float = 0.9, output: Path | None = None):
        self.level, self.output = level, output

    def save(self, fig: Figure, name: str) -> None:
        """Create the destination only when a figure is actually written."""
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output / f"{name}.png", bbox_inches="tight", facecolor="white")

    @staticmethod
    def _caps(
        ax: Axes,
        positions: np.ndarray,
        observed: np.ndarray,
        summary: pd.DataFrame,
        half_width: float,
    ) -> None:
        """Draw empirical endpoints directly, even when they exclude the observed bar."""
        lower, upper = summary[["lower", "upper"]].to_numpy(dtype=float).T
        available = summary.status.eq("ok").to_numpy() & np.isfinite(lower) & np.isfinite(upper)
        x = positions[available]
        ax.vlines(x, lower[available], upper[available], color=OVENTI["dark"], lw=1.4, zorder=4)
        for endpoints in (lower, upper):
            ax.hlines(
                endpoints[available],
                x - half_width,
                x + half_width,
                color=OVENTI["dark"],
                lw=1.4,
                zorder=4,
            )
        for x, value in zip(positions[~available], observed[~available], strict=True):
            if np.isfinite(value):
                ax.annotate(
                    "×",
                    (x, value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    color=OVENTI["dark"],
                    fontsize=15,
                    fontweight="bold",
                )
            else:
                ax.text(x, 0.04, "×", transform=ax.get_xaxis_transform(), ha="center", fontsize=15)

    def plot_metrics(
        self,
        point_folds: pd.DataFrame,
        interval_folds: pd.DataFrame,
        point_metrics: tuple[str, ...],
        interval_metrics: tuple[str, ...],
        *,
        point_summary: pd.DataFrame,
        interval_summary: pd.DataFrame,
        confidence: float = 0.95,
    ) -> Figure:
        """Three unique point panels over two wide interval panels and a shared legend."""
        points = point_folds.groupby("family", sort=False)[list(point_metrics)].mean()
        intervals = interval_folds.groupby(["family", "method"], sort=False)[
            list(interval_metrics)
        ].mean()
        families = points.index.tolist()
        methods = interval_summary.method.drop_duplicates().tolist()
        labels = {
            "Moving average": "Moving\naverage",
            "Trend-season MSTL": "Trend/season\nMSTL",
            "SARIMA": "SARIMA",
            "HistGradientBoosting": "Histogram\nboosting",
            "Online GD": "Online\nGD",
        }
        colors = (OVENTI["light"], OVENTI["teal"], OVENTI["orange"], OVENTI["medium"])
        hatches = ("", "//", "..", "xx")
        positions = np.arange(len(families))
        fig = plt.figure(figsize=(15, 12), layout="constrained")
        grid = fig.add_gridspec(3, 6, height_ratios=(1, 1.08, 0.58), hspace=0.08, wspace=0.1)
        axes = [fig.add_subplot(grid[0, i : i + 2]) for i in (0, 2, 4)]
        axes += [fig.add_subplot(grid[1, :3]), fig.add_subplot(grid[1, 3:])]
        legend_ax = fig.add_subplot(grid[2, :])
        legend_ax.set_axis_off()
        for ax, metric in zip(axes[:3], point_metrics, strict=True):
            values = points[metric].to_numpy()
            ax.bar(
                positions,
                np.where(np.isfinite(values), values, np.nan),
                width=0.64,
                color=OVENTI["light"],
                edgecolor=OVENTI["dark"],
                linewidth=0.6,
                zorder=2,
            )
            rows = point_summary.loc[point_summary.metric.eq(metric)].set_index("family")
            self._caps(ax, positions, values, rows.reindex(families), half_width=0.12)
        bar_width = 0.8 / len(methods)
        for ax, metric in zip(axes[3:], interval_metrics, strict=True):
            for i, method in enumerate(methods):
                values = intervals.xs(method, level="method").reindex(families)[metric].to_numpy()
                x = positions + (i - (len(methods) - 1) / 2) * bar_width
                ax.bar(
                    x,
                    np.where(np.isfinite(values), values, np.nan),
                    width=bar_width,
                    color=colors[i],
                    hatch=hatches[i],
                    edgecolor=OVENTI["dark"],
                    linewidth=0.6,
                    zorder=2,
                )
                rows = interval_summary.loc[
                    interval_summary.metric.eq(metric) & interval_summary.method.eq(method)
                ].set_index("family")
                self._caps(ax, x, values, rows.reindex(families), half_width=bar_width * 0.3)
        titles = ("RMSE", "MAE", "Bias (forecast − actual)", "Coverage", "Interval width")
        for ax, metric, title in zip(
            axes, (*point_metrics, *interval_metrics), titles, strict=True
        ):
            ax.set_title(title, fontsize=15, color=OVENTI["medium"], pad=13)
            panel_labels = dict(labels)
            if ax in axes[:3]:
                panel_labels["Trend-season MSTL"] = "Trend/\nseason\nMSTL"
            ax.set_xticks(
                positions,
                [panel_labels.get(family, family) for family in families],
                fontsize=11 if ax in axes[:3] else 12,
            )
            ax.tick_params(axis="x", length=0, pad=9)
            ax.tick_params(axis="y", labelsize=11)
            ax.set_ylabel(
                "Target share" if metric == "coverage" else "Published load scale",
                fontsize=12,
                labelpad=8,
            )
            ax.set_axisbelow(True)
            ax.grid(axis="x", visible=False)
            ax.grid(axis="y", alpha=0.3)
            ax.margins(x=0.06, y=0.12)
            ax.spines[["top", "right"]].set_visible(False)
        axes[2].axhline(0, color=OVENTI["dark"], lw=0.8)
        axes[3].axhline(self.level, color=OVENTI["dark"], ls="--", lw=1.3, zorder=3)
        axes[3].set_ylim(0, 1.06)
        axes[3].yaxis.set_major_formatter(PercentFormatter(1))
        method_legend = legend_ax.legend(
            handles=[
                Patch(facecolor=colors[i], edgecolor=OVENTI["dark"], hatch=hatches[i], label=method)
                for i, method in enumerate(methods)
            ],
            loc="upper center",
            title="Calibrators · interval panels only",
            title_fontsize=12,
            ncol=len(methods),
            fontsize=12,
            frameon=False,
            handlelength=2,
            columnspacing=2,
        )
        legend_ax.add_artist(method_legend)
        legend_ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=OVENTI["dark"],
                    marker="_",
                    label=f"{confidence:.0%} metric interval",
                ),
                Line2D(
                    [0],
                    [0],
                    color=OVENTI["dark"],
                    ls="--",
                    label=f"{self.level:.0%} nominal prediction coverage",
                ),
                Line2D(
                    [0],
                    [0],
                    color=OVENTI["dark"],
                    ls="",
                    marker="x",
                    label="Metric interval unavailable",
                ),
            ],
            loc="center",
            ncol=3,
            fontsize=11,
            frameon=False,
        )
        legend_ax.text(
            0.5,
            0.04,
            "Bars: observed fold means (mean fold RMSE, not pooled RMSE).\n"
            "Caps: pointwise temporal bootstrap of fixed losses; "
            "conditional sensitivity only. See README.",
            ha="center",
            va="bottom",
            transform=legend_ax.transAxes,
            fontsize=11,
            color=OVENTI["dark"],
            linespacing=1.5,
        )
        fig.suptitle(
            "OVENTI | Mean fold metrics with temporal bootstrap",
            fontsize=18,
            fontweight="bold",
            color=OVENTI["dark"],
        )
        self.save(fig, "04_mean_fold_metrics")
        return fig
