from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import Tensor


_MAX_TRACE_PANELS = 6


def save_time_series_plot(
    time_series: Sequence[Tensor],
    descriptions: Sequence[str],
    path: Path,
    *,
    patient_id: str,
) -> Path:
    if not time_series:
        raise ValueError("At least one time series is required for plotting.")
    if len(time_series) != len(descriptions):
        raise ValueError("Time-series descriptions must match the plotted series.")

    arrays = [
        series.detach().cpu().float().numpy().reshape(-1)
        for series in time_series
    ]
    maximum_length = max(len(values) for values in arrays)
    matrix = np.full((len(arrays), maximum_length), np.nan, dtype=np.float32)
    for index, values in enumerate(arrays):
        matrix[index, : len(values)] = values

    shown = min(len(arrays), _MAX_TRACE_PANELS)
    figure = Figure(figsize=(14, 8), layout="constrained")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, 1, height_ratios=(1.35, 1.0))
    traces_axis = figure.add_subplot(grid[0])
    heatmap_axis = figure.add_subplot(grid[1])

    offsets = np.arange(shown, dtype=np.float32) * 7.0
    colors = colormaps["tab10"]
    for index, (values, offset) in enumerate(zip(arrays[:shown], offsets)):
        traces_axis.plot(
            np.arange(len(values)),
            values + offset,
            color=colors(index % 10),
            linewidth=0.8,
        )
    traces_axis.set_yticks(offsets, [_short_label(value) for value in descriptions[:shown]])
    traces_axis.set_xlabel("Sample index")
    traces_axis.set_title(
        f"Representative standardized model inputs ({shown} of {len(arrays)})",
        loc="left",
        fontweight="bold",
    )
    traces_axis.grid(axis="x", color="#d1d5db", linewidth=0.6, alpha=0.7)
    traces_axis.spines[["top", "right", "left"]].set_visible(False)

    image = heatmap_axis.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-3.0,
        vmax=3.0,
    )
    heatmap_axis.set_title(
        f"All standardized input series for patient {patient_id}",
        loc="left",
        fontweight="bold",
    )
    heatmap_axis.set_xlabel("Sample index")
    heatmap_axis.set_ylabel("Input series")
    _set_heatmap_ticks(heatmap_axis, len(arrays))
    colorbar = figure.colorbar(image, ax=heatmap_axis, pad=0.01)
    colorbar.set_label("Standardized amplitude")

    path = path.resolve()
    if not path.suffix:
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=160, facecolor="white")
    temporary.replace(path)
    return path


def _short_label(description: str) -> str:
    label = description.split(", contains", maxsplit=1)[0]
    return label if len(label) <= 76 else f"{label[:73]}..."


def _set_heatmap_ticks(axis: Axes, count: int) -> None:
    if count <= 12:
        positions = np.arange(count)
    else:
        positions = np.unique(np.linspace(0, count - 1, num=7, dtype=int))
    axis.set_yticks(positions, [str(int(value) + 1) for value in positions])
